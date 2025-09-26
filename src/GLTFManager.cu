#include "GLTFManager.h"


TextureLoader::TextureLoader() = default;

TextureLoader::~TextureLoader() {
    cleanup();
}

cudaTextureObject_t TextureLoader::loadTexture(const std::string& filename) {
    // Check cache first
    auto it = texture_cache.find(filename);
    if (it != texture_cache.end()) {
        return it->second;
    }

    std::cout << "Loading texture: " << filename << std::endl;

    // Load PNG using stb_image
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 4); // Force 4 channels

    if (!data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        return 0;
    }

    std::cout << "Loaded: " << width << "x" << height << ", channels: " << channels << std::endl;

    // Create CUDA texture
    cudaTextureObject_t tex_obj = createTextureFromData(data, width, height, 4);

    // Free CPU data
    stbi_image_free(data);

    if (tex_obj != 0) {
        texture_cache[filename] = tex_obj;
        std::cout << "Created CUDA texture object: " << tex_obj << std::endl;
    }

    return tex_obj;
}

cudaTextureObject_t TextureLoader::getTexture(const std::string& filename) {
    return loadTexture(filename); // Load if not cached
}

cudaTextureObject_t TextureLoader::createTextureFromData(const unsigned char* data,
    int width, int height, int channels) {
    cudaTextureObject_t tex_obj = 0;
    cudaArray_t cu_array = nullptr;

    // Create channel description
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

    // Create CUDA array
    cudaError_t err = cudaMallocArray(&cu_array, &channel_desc, width, height);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA array: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    // Copy data to array
    size_t pitch = width * channels * sizeof(unsigned char);
    err = cudaMemcpy2DToArray(cu_array, 0, 0, data, pitch,
        width * channels * sizeof(unsigned char), height,
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy texture data: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cu_array);
        return 0;
    }

    // Create resource description
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    // Create texture description
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat; // Convert to [0,1]
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;

    // Create texture object
    err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create texture object: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cu_array);
        return 0;
    }

    texture_arrays.push_back(cu_array);
    return tex_obj;
}

void TextureLoader::cleanup() {
    for (auto& pair : texture_cache) {
        if (pair.second != 0) {
            cudaDestroyTextureObject(pair.second);
        }
    }
    texture_cache.clear();

    for (auto& array : texture_arrays) {
        if (array != nullptr) {
            cudaFreeArray(array);
        }
    }
    texture_arrays.clear();
}

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

    if (!processModel(filename, model)) {
        std::cerr << "Failed to process GLTF model" << std::endl;
        return false;
    }

    std::cout << "Loaded GLTF: " << meshes.size() << " meshes, "
        << materials.size() << " materials, " << std::endl;

    return true;
}

void GLTFLoader::clear() {

    meshes.clear();
    materials.clear();
}

bool GLTFLoader::processModel(const std::string& filename, const tinygltf::Model& model) {
    // Extract directory for relative texture paths
    std::string dir = filename.substr(0, filename.find_last_of("/\\")) + "/";
    // Process materials first
    for (const auto& mat : model.materials) {
        materials.push_back(processMaterial(mat, model, dir));
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

    // Print texture paths for debugging
    for (size_t i = 0; i < materials.size(); i++) {
        if (!materials[i].base_color_texture_path.empty()) {
            std::cout << "Material " << i << " base color texture: "
                << materials[i].base_color_texture_path << std::endl;
        }
    }

    return true;
}

GLTFLoader::MaterialData GLTFLoader::processMaterial(const tinygltf::Material& mat,
    const tinygltf::Model& model, const std::string& dir) {
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

    if (mat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
        int tex_index = mat.pbrMetallicRoughness.baseColorTexture.index;
        if (tex_index < model.textures.size()) {
            const auto& texture = model.textures[tex_index];
            if (texture.source >= 0 && texture.source < model.images.size()) {
                const auto& image = model.images[texture.source];
                if (!image.uri.empty() && image.uri.find("data:") == std::string::npos) {
                    material.base_color_texture_path = dir + image.uri;
                }
            }
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

GLTFManager::GLTFManager() = default;

GLTFManager::~GLTFManager() {
    cleanup();
}

bool GLTFManager::uploadToGPU(const GLTFLoader& loader, TextureLoader& text_loader) {
    cleanup();

    uploadTriangles(loader.getMeshes());
    uploadMaterials(loader.getMaterials(), text_loader);

    std::cout << "Uploaded to GPU: " << num_triangles << " triangles, "
        << num_PBRmaterials << " materials" << std::endl;

    // delete loader as we will no longer need it
    loader.~GLTFLoader();
    return true;
}

void GLTFManager::cleanup() {
    if (dev_triangles) {
        cudaFree(dev_triangles);
        dev_triangles = nullptr;
    }
    if (dev_PBRmaterials) {
        cudaFree(dev_PBRmaterials);
        dev_PBRmaterials = nullptr;
    }

    num_triangles = 0;
    num_PBRmaterials = 0;
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
    TextureLoader& text_loader) {
    std::vector<Material> host_materials;
    for (size_t i = 0; i < materials.size(); i++) {
        const auto& mat = materials[i];
        Material cuda_mat;

        cuda_mat.color = glm::vec3(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
        cuda_mat.metallic = mat.metallic;
        cuda_mat.roughness = mat.roughness;

        // Load textures and get direct CUDA texture handles
        cuda_mat.base_color_tex = text_loader.getTexture(mat.base_color_texture_path);
        cuda_mat.metallic_roughness_tex = text_loader.getTexture(mat.metallic_roughness_texture_path);
        cuda_mat.normal_tex = text_loader.getTexture(mat.normal_texture_path);
        cuda_mat.emissive_tex = text_loader.getTexture(mat.emissive_texture_path);

        // Debug output
        std::cout << "Material " << i << ":" << std::endl;
        std::cout << "  Base color: (" << cuda_mat.color.x << ", "
            << cuda_mat.color.y << ", " << cuda_mat.color.z << ")" << std::endl;
        std::cout << "  Metallic: " << cuda_mat.metallic << ", Roughness: " << cuda_mat.roughness << std::endl;
        std::cout << "  Base color texture: " << (cuda_mat.base_color_tex != 0 ? "LOADED" : "MISSING")
            << " (handle: " << cuda_mat.base_color_tex << ")" << std::endl;
        if (!mat.base_color_texture_path.empty()) {
            std::cout << "  Texture path: " << mat.base_color_texture_path << std::endl;
        }

        host_materials.push_back(cuda_mat);
    }

    // Add a default material if no materials exist
    if (host_materials.empty()) {
        std::cout << "No materials found, creating default material" << std::endl;
        Material default_mat;
        default_mat.color = glm::vec3(0.8f, 0.8f, 0.8f);
        default_mat.metallic = 0.0f;
        default_mat.roughness = 0.5f;
        default_mat.emissive_factor = glm::vec3(0.0f);
        default_mat.base_color_tex = 0;  // No texture
        default_mat.metallic_roughness_tex = 0;
        default_mat.normal_tex = 0;
        default_mat.emissive_tex = 0;
        host_materials.push_back(default_mat);
    }

    num_PBRmaterials = host_materials.size();
    if (num_PBRmaterials > 0) {
        cudaMalloc(&dev_PBRmaterials, num_PBRmaterials * sizeof(Material));
        cudaMemcpy(dev_PBRmaterials, host_materials.data(),
            num_PBRmaterials * sizeof(Material), cudaMemcpyHostToDevice);
        
    }
}