#include "GLTFManager.h"

GLTFLoader::GLTFLoader() = default;

GLTFLoader::~GLTFLoader() {
    clear();
}

bool GLTFLoader::load(const std::string& filename) {
    clear();

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    // Try loading as binary first, then ASCII
    bool success = loader.LoadBinaryFromFile(&model, &err, &warn, filename) ||
        loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "GLTF Error: " << err << std::endl;
    }
    if (!success) {
        std::cerr << "Failed to load GLTF file: " << filename << std::endl;
        return false;
    }

    if (!processModel(model)) {
        std::cerr << "Failed to process GLTF model" << std::endl;
        return false;
    }

    if (!createCUDATextures()) {
        std::cerr << "Failed to create CUDA textures" << std::endl;
        return false;
    }

    std::cout << "Loaded GLTF: " << meshes.size() << " meshes, "
        << materials.size() << " materials, "
        << textures.size() << " textures" << std::endl;

    return true;
}

void GLTFLoader::clear() {
    // Cleanup CUDA textures
    for (auto& tex_obj : texture_objects) {
        if (tex_obj != 0) {
            cudaDestroyTextureObject(tex_obj);
        }
    }
    for (auto& array : texture_arrays) {
        if (array != nullptr) {
            cudaFreeArray(array);
        }
    }

    texture_objects.clear();
    texture_arrays.clear();
    meshes.clear();
    materials.clear();
    textures.clear();
}

bool GLTFLoader::processModel(const tinygltf::Model& model) {
    // Process materials first
    for (const auto& mat : model.materials) {
        materials.push_back(processMaterial(mat, model));
    }

    // Add default material if none exist
    if (materials.empty()) {
        materials.push_back(MaterialData());
    }

    // Process meshes
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
                meshes.push_back(processPrimitive(primitive, model));
            }
        }
    }

    // Process textures
    for (const auto& image : model.images) {
        textures.push_back(processTexture(image));
    }

    return true;
}

GLTFLoader::MaterialData GLTFLoader::processMaterial(const tinygltf::Material& mat,
    const tinygltf::Model& model) {
    MaterialData material;

    // Base color factor
    if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
        for (int i = 0; i < 4; i++) {
            material.base_color[i] = static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[i]);
        }
    }

    // Metallic/roughness factors
    material.metallic = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
    material.roughness = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);

    // Texture indices
    material.base_color_tex = mat.pbrMetallicRoughness.baseColorTexture.index;
    material.metallic_roughness_tex = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
    material.normal_tex = mat.normalTexture.index;
    material.emissive_tex = mat.emissiveTexture.index;

    // Emissive factor
    if (mat.emissiveFactor.size() == 3) {
        for (int i = 0; i < 3; i++) {
            material.emissive_factor[i] = static_cast<float>(mat.emissiveFactor[i]);
        }
    }

    return material;
}

GLTFLoader::MeshData GLTFLoader::processPrimitive(const tinygltf::Primitive& primitive,
    const tinygltf::Model& model) {
    MeshData mesh;
    mesh.material_id = primitive.material >= 0 ? primitive.material : 0;

    // Extract vertex attributes
    extractAttribute(primitive, model, "POSITION", mesh.vertices, 3);
    extractAttribute(primitive, model, "NORMAL", mesh.normals, 3);
    extractAttribute(primitive, model, "TEXCOORD_0", mesh.texcoords, 2);

    // Extract indices
    if (primitive.indices >= 0) {
        const auto& accessor = model.accessors[primitive.indices];
        const auto& bufferView = model.bufferViews[accessor.bufferView];
        const auto& buffer = model.buffers[bufferView.buffer];

        const unsigned char* data = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

        if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            const uint32_t* indices = reinterpret_cast<const uint32_t*>(data);
            mesh.indices.assign(indices, indices + accessor.count);
        }
        else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            const uint16_t* indices = reinterpret_cast<const uint16_t*>(data);
            for (size_t i = 0; i < accessor.count; i++) {
                mesh.indices.push_back(static_cast<uint32_t>(indices[i]));
            }
        }
        else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            const uint8_t* indices = reinterpret_cast<const uint8_t*>(data);
            for (size_t i = 0; i < accessor.count; i++) {
                mesh.indices.push_back(static_cast<uint32_t>(indices[i]));
            }
        }
    }

    return mesh;
}

void GLTFLoader::extractAttribute(const tinygltf::Primitive& primitive,
    const tinygltf::Model& model,
    const std::string& attribute,
    std::vector<float>& output, int components) {
    auto it = primitive.attributes.find(attribute);
    if (it == primitive.attributes.end()) return;

    int accessor_index = it->second;
    const auto& accessor = model.accessors[accessor_index];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    const float* data = reinterpret_cast<const float*>(
        buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);

    output.assign(data, data + accessor.count * components);
}

GLTFLoader::TextureData GLTFLoader::processTexture(const tinygltf::Image& image) {
    TextureData tex;
    tex.width = image.width;
    tex.height = image.height;
    tex.channels = image.component;
    tex.data = image.image;
    return tex;
}

bool GLTFLoader::createCUDATextures() {
    texture_objects.resize(textures.size(), 0);
    texture_arrays.resize(textures.size(), nullptr);

    for (size_t i = 0; i < textures.size(); i++) {
        const auto& tex = textures[i];

        if (tex.data.empty()) continue;

        // Convert to RGBA format for CUDA textures
        std::vector<unsigned char> rgba_data;
        const unsigned char* source_data = tex.data.data();
        int actual_channels = tex.channels;

        if (tex.channels == 1) {
            // Grayscale to RGBA
            rgba_data.resize(tex.width * tex.height * 4);
            for (size_t j = 0; j < tex.width * tex.height; j++) {
                unsigned char value = tex.data[j];
                rgba_data[j * 4] = value;
                rgba_data[j * 4 + 1] = value;
                rgba_data[j * 4 + 2] = value;
                rgba_data[j * 4 + 3] = 255;
            }
            source_data = rgba_data.data();
            actual_channels = 4;
        }
        else if (tex.channels == 3) {
            // RGB to RGBA
            rgba_data.resize(tex.width * tex.height * 4);
            for (size_t j = 0; j < tex.width * tex.height; j++) {
                rgba_data[j * 4] = tex.data[j * 3];
                rgba_data[j * 4 + 1] = tex.data[j * 3 + 1];
                rgba_data[j * 4 + 2] = tex.data[j * 3 + 2];
                rgba_data[j * 4 + 3] = 255;
            }
            source_data = rgba_data.data();
            actual_channels = 4;
        }

        texture_objects[i] = createTextureObject(source_data, tex.width, tex.height, actual_channels);
        if (texture_objects[i] == 0) {
            std::cerr << "Failed to create CUDA texture for texture " << i << std::endl;
            return false;
        }
    }

    return true;
}

cudaTextureObject_t GLTFLoader::createTextureObject(const unsigned char* data,
    int width, int height, int channels) {
    cudaTextureObject_t tex_obj = 0;
    cudaArray_t cu_array = nullptr;

    cudaChannelFormatDesc channel_desc;
    if (channels == 4) {
        channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else if (channels == 3) {
        channel_desc = cudaCreateChannelDesc<uchar3>();
    }
    else if (channels == 1) {
        channel_desc = cudaCreateChannelDesc<unsigned char>();
    }
    else {
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
        return 0;
    }

    cudaError_t err = cudaMallocArray(&cu_array, &channel_desc, width, height);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA array: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    size_t pitch = width * channels * sizeof(unsigned char);
    err = cudaMemcpy2DToArray(cu_array, 0, 0, data, pitch,
        width * channels * sizeof(unsigned char), height,
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy texture data: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cu_array);
        return 0;
    }

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.mipmapLevelBias = 0;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.maxMipmapLevelClamp = 0;

    err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create texture object: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cu_array);
        return 0;
    }

    texture_arrays.push_back(cu_array);
    return tex_obj;
}

GLTFManager::GLTFManager() = default;

GLTFManager::~GLTFManager() {
    cleanup();
}

bool GLTFManager::uploadToGPU(const GLTFLoader& loader) {
    cleanup();

    uploadTriangles(loader.getMeshes());
    uploadTextureObjects(loader.getTextureObjects());
    uploadMaterials(loader.getMaterials(), loader.getTextureObjects());

    std::cout << "Uploaded to GPU: " << num_triangles << " triangles, "
        << num_materials << " materials, "
        << num_textures << " textures" << std::endl;

    return true;
}

void GLTFManager::cleanup() {
    if (dev_triangles) {
        cudaFree(dev_triangles);
        dev_triangles = nullptr;
    }
    if (dev_materials) {
        cudaFree(dev_materials);
        dev_materials = nullptr;
    }
    if (dev_texture_objects) {
        cudaFree(dev_texture_objects);
        dev_texture_objects = nullptr;
    }

    num_triangles = 0;
    num_materials = 0;
    num_textures = 0;
}

void GLTFManager::uploadTriangles(const std::vector<GLTFLoader::MeshData>& meshes) {
    std::vector<Triangle> host_triangles;

    for (const auto& mesh : meshes) {
        if (mesh.indices.size() % 3 != 0) continue;

        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            Triangle tri;
            uint32_t idx0 = mesh.indices[i];
            uint32_t idx1 = mesh.indices[i + 1];
            uint32_t idx2 = mesh.indices[i + 2];

            // Vertices
            tri.v0 = glm::vec3(mesh.vertices[idx0 * 3], mesh.vertices[idx0 * 3 + 1], mesh.vertices[idx0 * 3 + 2]);
            tri.v1 = glm::vec3(mesh.vertices[idx1 * 3], mesh.vertices[idx1 * 3 + 1], mesh.vertices[idx1 * 3 + 2]);
            tri.v2 = glm::vec3(mesh.vertices[idx2 * 3], mesh.vertices[idx2 * 3 + 1], mesh.vertices[idx2 * 3 + 2]);

            // Normals
            if (!mesh.normals.empty()) {
                tri.n0 = glm::vec3(mesh.normals[idx0 * 3], mesh.normals[idx0 * 3 + 1], mesh.normals[idx0 * 3 + 2]);
                tri.n1 = glm::vec3(mesh.normals[idx1 * 3], mesh.normals[idx1 * 3 + 1], mesh.normals[idx1 * 3 + 2]);
                tri.n2 = glm::vec3(mesh.normals[idx2 * 3], mesh.normals[idx2 * 3 + 1], mesh.normals[idx2 * 3 + 2]);
            }
            else {
                // Compute face normal if no normals provided
                glm::vec3 edge1 = glm::vec3(tri.v1.x - tri.v0.x, tri.v1.y - tri.v0.y, tri.v1.z - tri.v0.z);
                glm::vec3 edge2 = glm::vec3(tri.v2.x - tri.v0.x, tri.v2.y - tri.v0.y, tri.v2.z - tri.v0.z);
                glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
                tri.n0 = tri.n1 = tri.n2 = normal;
            }

            // UV coordinates
            if (!mesh.texcoords.empty()) {
                tri.uv0 = glm::vec2(mesh.texcoords[idx0 * 2], mesh.texcoords[idx0 * 2 + 1]);
                tri.uv1 = glm::vec2(mesh.texcoords[idx1 * 2], mesh.texcoords[idx1 * 2 + 1]);
                tri.uv2 = glm::vec2(mesh.texcoords[idx2 * 2], mesh.texcoords[idx2 * 2 + 1]);
            }
            else {
                tri.uv0 = tri.uv1 = tri.uv2 = glm::vec2(0.0f, 0.0f);
            }

            tri.material_id = mesh.material_id;
            host_triangles.push_back(tri);
        }
    }

    num_triangles = host_triangles.size();
    if (num_triangles > 0) {
        cudaMalloc(&dev_triangles, num_triangles * sizeof(Triangle));
        cudaMemcpy(dev_triangles, host_triangles.data(),
            num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    }
}

void GLTFManager::uploadMaterials(const std::vector<GLTFLoader::MaterialData>& materials,
    const std::vector<cudaTextureObject_t>& texture_objects) {
    std::vector<PBRMaterial> host_materials;

    for (const auto& mat : materials) {
        PBRMaterial cuda_mat;
        cuda_mat.base_color = glm::vec3(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
        cuda_mat.metallic = mat.metallic;
        cuda_mat.roughness = mat.roughness;
        cuda_mat.emissive_factor = glm::vec3(mat.emissive_factor[0], mat.emissive_factor[1], mat.emissive_factor[2]);

        // Assign texture objects
        cuda_mat.base_color_tex = (mat.base_color_tex >= 0 && mat.base_color_tex < texture_objects.size())
            ? texture_objects[mat.base_color_tex] : 0;
        cuda_mat.metallic_roughness_tex = (mat.metallic_roughness_tex >= 0 && mat.metallic_roughness_tex < texture_objects.size())
            ? texture_objects[mat.metallic_roughness_tex] : 0;
        cuda_mat.normal_tex = (mat.normal_tex >= 0 && mat.normal_tex < texture_objects.size())
            ? texture_objects[mat.normal_tex] : 0;
        cuda_mat.emissive_tex = (mat.emissive_tex >= 0 && mat.emissive_tex < texture_objects.size())
            ? texture_objects[mat.emissive_tex] : 0;

        host_materials.push_back(cuda_mat);
    }

    num_materials = host_materials.size();
    if (num_materials > 0) {
        cudaMalloc(&dev_materials, num_materials * sizeof(PBRMaterial));
        cudaMemcpy(dev_materials, host_materials.data(),
            num_materials * sizeof(PBRMaterial), cudaMemcpyHostToDevice);
    }
}

void GLTFManager::uploadTextureObjects(const std::vector<cudaTextureObject_t>& texture_objects) {
    num_textures = texture_objects.size();
    if (num_textures > 0) {
        cudaMalloc(&dev_texture_objects, num_textures * sizeof(cudaTextureObject_t));
        cudaMemcpy(dev_texture_objects, texture_objects.data(),
            num_textures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }
}