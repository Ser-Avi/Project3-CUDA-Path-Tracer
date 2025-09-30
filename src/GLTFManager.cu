#include "GLTFManager.h"

#define BalancedBVH true
#define BuildFastBVH true

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

EnvMap TextureLoader::loadEnvMap(const std::string& filename) {
    // Check cache first
    auto it = hdr_cache.find(filename);
    if (it != hdr_cache.end()) {
        return it->second;
    }

    std::cout << "Loading HDR environment map: " << filename << std::endl;

    // Load HDR using stb_image
    int width, height, channels;
    float* hdr_data = stbi_loadf(filename.c_str(), &width, &height, &channels, 0);

    if (!hdr_data) {
        std::cerr << "Failed to load HDR file: " << filename << std::endl;
        return EnvMap{ 0, "", 0, 0, nullptr};
    }

    std::cout << "HDR loaded: " << width << "x" << height << ", channels: " << channels << std::endl;

    // Create CUDA texture from HDR data
    cudaTextureObject_t tex_obj = createTextureFromFloatData(hdr_data, width, height, channels);

    EnvMap env_map;
    env_map.texture = tex_obj;
    env_map.width = width;
    env_map.height = height;
    env_map.data = hdr_data;  // Keep CPU data for importance sampling

    if (tex_obj != 0) {
        hdr_cache[filename] = env_map;
        std::cout << "Created HDR environment map texture: " << tex_obj << std::endl;
    }
    else {
        stbi_image_free(hdr_data);
    }

    return env_map;
}

EnvMap TextureLoader::getEnvMap(const std::string& filename) {
    return loadEnvMap(filename);
}

cudaTextureObject_t TextureLoader::createTextureFromFloatData(const float* data,
    int width, int height, int channels) {
    cudaTextureObject_t tex_obj = 0;
    cudaArray_t cu_array = nullptr;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    // Create CUDA array
    cudaError_t err = cudaMallocArray(&cu_array, &channel_desc, width, height);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA array for HDR: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    // If 3 channels, force a 4th
    if (channels == 3) {
        std::vector<float4> rgba_data(width * height);
        for (int i = 0; i < width * height; i++) {
            rgba_data[i] = make_float4(
                data[i * 3], 
                data[i * 3 + 1],  
                data[i * 3 + 2],
                1.0f          // force alpha to 1
            );
        }

        // Copy data
        size_t pitch = width * sizeof(float4);
        err = cudaMemcpy2DToArray(cu_array, 0, 0, rgba_data.data(), pitch,
            width * sizeof(float4), height,
            cudaMemcpyHostToDevice);
    }
    else if (channels == 4) {
        // Already 4 channels, copy directly
        size_t pitch = width * sizeof(float4);
        err = cudaMemcpy2DToArray(cu_array, 0, 0, data, pitch,
            width * sizeof(float4), height,
            cudaMemcpyHostToDevice);
    }
    else {
        std::cerr << "Unsupported HDR channel count: " << channels << std::endl;
        cudaFreeArray(cu_array);
        return 0;
    }

    if (err != cudaSuccess) {
        std::cerr << "Failed to copy HDR data: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cu_array);
        return 0;
    }

    // Create resource description
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    // Create texture description for HDR (different from regular textures!)
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;       // Repeat for environment maps
    tex_desc.addressMode[1] = cudaAddressModeClamp;      // Clamp in vertical direction
    tex_desc.filterMode = cudaFilterModeLinear;          // Bilinear filtering
    tex_desc.readMode = cudaReadModeElementType;         // Keep as float - not normalized
    tex_desc.normalizedCoords = 1;                       // Use [0,1] coordinates
    tex_desc.maxAnisotropy = 1;

    // Create texture object
    err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create HDR texture object: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cu_array);
        return 0;
    }

    texture_arrays.push_back(cu_array);
    return tex_obj;
}

void TextureLoader::cleanupEnvMap() {
    for (auto& pair : hdr_cache) {
        if (pair.second.texture != 0) {
            cudaDestroyTextureObject(pair.second.texture);
        }
        if (pair.second.data != nullptr) {
            stbi_image_free(pair.second.data);
        }
    }
    hdr_cache.clear();
}

void TextureLoader::cleanup() {
    for (auto& pair : texture_cache) {
        if (pair.second != 0) {
            cudaDestroyTextureObject(pair.second);
        }
    }
    texture_cache.clear();

    cleanupEnvMap();

    // Cleanup arrays
    for (auto& array : texture_arrays) {
        if (array != nullptr) {
            cudaFreeArray(array);
        }
    }
    texture_arrays.clear();
}

namespace BVH
{
    void BuildBVH(int N, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode, int& nodesUsed)
    {
        // populate triangle index array
        for (int i = 0; i < N; i++) triIdx[i] = i;
        // calculate triangle centroids for partitioning
        //for (int i = 0; i < N; i++)
        //{
        //    tri[i].centroid = glm::vec3(tri[i].v0 + tri[i].v1 + tri[i].v2) * 0.3333f;
        //}
        // assign all triangles to root node
        BVHNode& root = bvhNode[0];
        root.leftFirst = 0, root.triCount = N;
        UpdateNodeBounds(0, triIdx, tri, bvhNode);
        // subdivide recursively
#if BuildFastBVH
        Subdivide_Fast(0, triIdx, tri, bvhNode, nodesUsed);
#else
        Subdivide(0, triIdx, tri, bvhNode, nodesUsed);
#endif
    }

    void UpdateNodeBounds(uint32_t nodeIdx, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode)
    {
        BVHNode& node = bvhNode[nodeIdx];
        node.aabbMin = glm::vec3(1e30f);
        node.aabbMax = glm::vec3(-1e30f);
        for (uint32_t first = node.leftFirst, i = 0; i < node.triCount; i++)
        {
            uint32_t leafTriIdx = triIdx[first + i];
            Triangle& leafTri = tri[leafTriIdx];
            node.aabbMin = glm::min(node.aabbMin, leafTri.v0);
            node.aabbMin = glm::min(node.aabbMin, leafTri.v1);
            node.aabbMin = glm::min(node.aabbMin, leafTri.v2);
            node.aabbMax = glm::max(node.aabbMax, leafTri.v0);
            node.aabbMax = glm::max(node.aabbMax, leafTri.v1);
            node.aabbMax = glm::max(node.aabbMax, leafTri.v2);
        }
    }

    float EvaluateSAH(BVHNode& node, int axis, float pos,
        std::vector<int>& triIdx, std::vector<Triangle>& tri)
    {
        // determine triangle counts and bounds for this split candidate
        aabb leftBox, rightBox;
        int leftCount = 0, rightCount = 0;
        for (uint32_t i = 0; i < node.triCount; i++)
        {
            Triangle& triangle = tri[triIdx[node.leftFirst + i]];
            glm::vec3 centroid = glm::vec3(triangle.v0 + triangle.v1 + triangle.v2) * 0.3333f;
            if (centroid[axis] < pos)
            {
                leftCount++;
                leftBox.grow(triangle.v0);
                leftBox.grow(triangle.v1);
                leftBox.grow(triangle.v2);
            }
            else
            {
                rightCount++;
                rightBox.grow(triangle.v0);
                rightBox.grow(triangle.v1);
                rightBox.grow(triangle.v2);
            }
        }
        float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
        return cost > 0 ? cost : 1e30f;
    }

    void Subdivide(uint32_t nodeIdx, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode, int& nodesUsed)
    {
        // terminate recursion
        BVHNode& node = bvhNode[nodeIdx];
#if BalancedBVH
        // Balanced split axis and pos using surface area heuristic (SAH)
        int bestAxis = -1;
        float bestPos = 0, bestCost = 1e30f;
        for (int axis = 0; axis < 3; axis++) for (uint32_t i = 0; i < node.triCount; i++)
        {
            Triangle& triangle = tri[triIdx[node.leftFirst + i]];
            glm::vec3 centroid = glm::vec3(triangle.v0 + triangle.v1 + triangle.v2) * 0.3333f;
            float candidatePos = centroid[axis];
            float cost = EvaluateSAH(node, axis, candidatePos, triIdx, tri);
            if (cost < bestCost)
                bestPos = candidatePos, bestAxis = axis, bestCost = cost;
        }
        int axis = bestAxis;
        float splitPos = bestPos;
        // new exit calculations
        glm::vec3 e = node.aabbMax - node.aabbMin; // extent of parent
        float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
        float parentCost = node.triCount * parentArea;
        if (bestCost >= parentCost) return;
#else
        // this check is only for naive
        // since SAH method knows that stopping before this can sometimes be better
        // or stopping later also (?)
        if (node.triCount <= 2)
        {
            std::cout << "less than 2 at idx: " << nodeIdx << std::endl;
            return;
        }
        // Naive split axis and pos
        glm::vec3 extent = node.aabbMax - node.aabbMin;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent.y) axis = 2;
        float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
#endif
        // in-place partition
        int i = node.leftFirst;
        int j = i + node.triCount - 1;
        while (i <= j)
        {
            Triangle triangle = tri[triIdx[i]];
            glm::vec3 centroid = glm::vec3(triangle.v0 + triangle.v1 + triangle.v2) * 0.3333f;
            if (centroid[axis] < splitPos - EPSILON)
            {
                i++;
            }
            else
            {
                std::swap(triIdx[i], triIdx[j]);
                --j;
            }
        }
        // abort split if one of the sides is empty
        int leftCount = i - node.leftFirst;
        //printf("Left Count: %d, triangles: %d\n", leftCount, node.triCount);
        if (leftCount == 0 || leftCount == node.triCount) return;
        // create child nodes
        int leftChildIdx = nodesUsed++;
        int rightChildIdx = nodesUsed++;
        //std::cout << "Nodes used: " << nodesUsed << std::endl;
        bvhNode[leftChildIdx].leftFirst = node.leftFirst;
        bvhNode[leftChildIdx].triCount = leftCount;
        bvhNode[rightChildIdx].leftFirst = i;
        bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
        node.leftFirst = leftChildIdx;
        node.triCount = 0;
        UpdateNodeBounds(leftChildIdx, triIdx, tri, bvhNode);
        UpdateNodeBounds(rightChildIdx, triIdx, tri, bvhNode);
        // recurse
        Subdivide(leftChildIdx, triIdx, tri, bvhNode, nodesUsed);
        Subdivide(rightChildIdx, triIdx, tri, bvhNode, nodesUsed);
    }


    void Subdivide_Fast(uint32_t nodeIdx, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode, int& nodesUsed)
    {
        // terminate recursion
        BVHNode& node = bvhNode[nodeIdx];
        // determine split axis using SAH
        int axis;
        float splitPos;
        float splitCost = FindBestSplitPlane(node, axis, splitPos, triIdx, tri);
        float nosplitCost = CalculateNodeCost(node);
        if (splitCost >= nosplitCost) return;
        // in-place partition
        int i = node.leftFirst;
        int j = i + node.triCount - 1;
        while (i <= j)
        {
            Triangle& triangle = tri[triIdx[i]];
            glm::vec3 centroid = glm::vec3(triangle.v0 + triangle.v1 + triangle.v2) * 0.3333f;
            if (centroid[axis] < splitPos)
                i++;
            else
                std::swap(triIdx[i], triIdx[j--]);
        }
        // abort split if one of the sides is empty
        int leftCount = i - node.leftFirst;
        if (leftCount == 0 || leftCount == node.triCount) return;
        // create child nodes
        int leftChildIdx = nodesUsed++;
        int rightChildIdx = nodesUsed++;
        bvhNode[leftChildIdx].leftFirst = node.leftFirst;
        bvhNode[leftChildIdx].triCount = leftCount;
        bvhNode[rightChildIdx].leftFirst = i;
        bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
        node.leftFirst = leftChildIdx;
        node.triCount = 0;
        UpdateNodeBounds(leftChildIdx, triIdx, tri, bvhNode);
        UpdateNodeBounds(rightChildIdx, triIdx, tri, bvhNode);
        // recurse
        Subdivide_Fast(leftChildIdx, triIdx, tri, bvhNode, nodesUsed);
        Subdivide_Fast(rightChildIdx, triIdx, tri, bvhNode, nodesUsed);
    }

    float CalculateNodeCost(BVHNode& node)
    {
        glm::vec3 e = node.aabbMax - node.aabbMin; // extent of the node
        float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
        return node.triCount * surfaceArea;
    }

#define BINS 8
    float FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos, std::vector<int>& triIdx, std::vector<Triangle>& tri)
    {
        float bestCost = 1e30f;
        for (int a = 0; a < 3; a++)
        {
            float boundsMin = 1e30f, boundsMax = -1e30f;
            for (uint32_t i = 0; i < node.triCount; i++)
            {
                Triangle& triangle = tri[triIdx[node.leftFirst + i]];
                glm::vec3 centroid = glm::vec3(triangle.v0 + triangle.v1 + triangle.v2) * 0.3333f;
                boundsMin = glm::min(boundsMin, centroid[a]);
                boundsMax = glm::max(boundsMax, centroid[a]);
            }
            if (boundsMin == boundsMax) continue;
            // populate the bins
            Bin bin[BINS];
            float scale = BINS / (boundsMax - boundsMin);
            for (uint32_t i = 0; i < node.triCount; i++)
            {
                Triangle& triangle = tri[triIdx[node.leftFirst + i]];
                glm::vec3 centroid = glm::vec3(triangle.v0 + triangle.v1 + triangle.v2) * 0.3333f;
                int binIdx = glm::min(BINS - 1, (int)((centroid[a] - boundsMin) * scale));
                bin[binIdx].triCount++;
                bin[binIdx].bounds.grow(triangle.v0);
                bin[binIdx].bounds.grow(triangle.v1);
                bin[binIdx].bounds.grow(triangle.v2);
            }
            // gather data for the 7 planes between the 8 bins
            float leftArea[BINS - 1], rightArea[BINS - 1];
            int leftCount[BINS - 1], rightCount[BINS - 1];
            aabb leftBox, rightBox;
            int leftSum = 0, rightSum = 0;
            for (int i = 0; i < BINS - 1; i++)
            {
                leftSum += bin[i].triCount;
                leftCount[i] = leftSum;
                leftBox.grow(bin[i].bounds);
                leftArea[i] = leftBox.area();
                rightSum += bin[BINS - 1 - i].triCount;
                rightCount[BINS - 2 - i] = rightSum;
                rightBox.grow(bin[BINS - 1 - i].bounds);
                rightArea[BINS - 2 - i] = rightBox.area();
            }
            // calculate SAH cost for the 7 planes
            scale = (boundsMax - boundsMin) / BINS;
            for (int i = 0; i < BINS - 1; i++)
            {
                float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
                if (planeCost < bestCost)
                    axis = a, splitPos = boundsMin + scale * (i + 1), bestCost = planeCost;
            }
        }
        return bestCost;
    }
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
    if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4)
    {
        for (int i = 0; i < 4; i++) {
            material.base_color[i] = static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[i]);
        }
    }

    // Metallic/roughness factors
    material.metallic = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
    material.roughness = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);

    // base col
    if (mat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
        int tex_index = mat.pbrMetallicRoughness.baseColorTexture.index;
        if (tex_index < model.textures.size()) {
            const auto& texture = model.textures[tex_index];
            if (texture.source >= 0 && texture.source < model.images.size()) {
                const auto& image = model.images[texture.source];
                if (!image.uri.empty() && image.uri.find("data:") == std::string::npos) {
                    material.base_color_texture_path = dir + image.uri;
                    std::cout << "  Found base color texture: " << material.base_color_texture_path << std::endl;
                }
            }
        }
    }

    // ao rough metal -> rgb
    if (mat.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
        int tex_index = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
        if (tex_index < model.textures.size()) {
            const auto& texture = model.textures[tex_index];
            if (texture.source >= 0 && texture.source < model.images.size()) {
                const auto& image = model.images[texture.source];
                if (!image.uri.empty() && image.uri.find("data:") == std::string::npos) {
                    material.metallic_roughness_texture_path = dir + image.uri;
                    std::cout << "  Found metallic-roughness texture: " << material.metallic_roughness_texture_path << std::endl;
                }
            }
        }
    }

    // normal
    if (mat.normalTexture.index >= 0) {
        int tex_index = mat.normalTexture.index;
        if (tex_index < model.textures.size()) {
            const auto& texture = model.textures[tex_index];
            if (texture.source >= 0 && texture.source < model.images.size()) {
                const auto& image = model.images[texture.source];
                if (!image.uri.empty() && image.uri.find("data:") == std::string::npos) {
                    material.normal_texture_path = dir + image.uri;
                    std::cout << "  Found normal texture: " << material.normal_texture_path << std::endl;
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

void GLTFManager::beginSequentialUpload() {
    cleanup();  // Clear any existing data
    host_triangles.clear();
    host_materials.clear();
    host_bvhNodes.clear();
    host_triangleIndices.clear();
    std::cout << "Beginning sequential upload..." << std::endl;
}

bool GLTFManager::addScene(const GLTFLoader& loader, TextureLoader& texture_loader) {
    // Get meshes and materials from the loader
    const auto& meshes = loader.getMeshes();
    const auto& materials = loader.getMaterials();

    std::cout << "Adding scene with " << meshes.size() << " meshes and "
        << materials.size() << " materials" << std::endl;

    // Process meshes into triangles
    for (const auto& mesh : meshes) {
        if (mesh.indices.size() % 3 != 0) continue;

        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            Triangle tri;
            uint32_t idx0 = mesh.indices[i];
            uint32_t idx1 = mesh.indices[i + 1];
            uint32_t idx2 = mesh.indices[i + 2];

            // vertex processing
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
                // Compute face normal
                glm::vec3 edge1 = glm::vec3(tri.v1.x - tri.v0.x, tri.v1.y - tri.v0.y, tri.v1.z - tri.v0.z);
                glm::vec3 edge2 = glm::vec3(tri.v2.x - tri.v0.x, tri.v2.y - tri.v0.y, tri.v2.z - tri.v0.z);
                glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
                tri.n0 = tri.n1 = tri.n2 = normal;
            }

            // UVs
            if (!mesh.texcoords.empty()) {
                tri.uv0 = glm::vec2(mesh.texcoords[idx0 * 2], mesh.texcoords[idx0 * 2 + 1]);
                tri.uv1 = glm::vec2(mesh.texcoords[idx1 * 2], mesh.texcoords[idx1 * 2 + 1]);
                tri.uv2 = glm::vec2(mesh.texcoords[idx2 * 2], mesh.texcoords[idx2 * 2 + 1]);
            }
            else {
                tri.uv0 = tri.uv1 = tri.uv2 = glm::vec2(0.0f, 0.0f);
            }

            // Adjust material ID for combined scene
            tri.material_id = mesh.material_id + host_materials.size();
            host_triangles.push_back(tri);
        }
    }

    // Process materials
    for (const auto& mat : materials) {
        Material cuda_mat;
        cuda_mat.color = glm::vec3(mat.base_color[0], mat.base_color[1], mat.base_color[2]);
        cuda_mat.metallic = mat.metallic;
        cuda_mat.roughness = mat.roughness;
        cuda_mat.ao = mat.ao;

        // Load textures
        cuda_mat.base_color_tex = texture_loader.getTexture(mat.base_color_texture_path);
        cuda_mat.metallic_roughness_tex = texture_loader.getTexture(mat.metallic_roughness_texture_path);
        cuda_mat.normal_tex = texture_loader.getTexture(mat.normal_texture_path);

        host_materials.push_back(cuda_mat);
    }

    std::cout << "Scene added. Total so far: " << host_triangles.size() << " triangles, "
        << host_materials.size() << " materials" << std::endl;

    return true;
}

void GLTFManager::finishSequentialUpload() {
    // Upload accumulated data to GPU
    num_triangles = host_triangles.size();
    num_PBRmaterials = host_materials.size();
    nodes_used = host_bvhNodes.size();

    if (num_triangles > 0) {
        cudaMalloc(&dev_triangles, num_triangles * sizeof(Triangle));
        cudaMemcpy(dev_triangles, host_triangles.data(),
            num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);

        // malloc bvh stuff
        cudaMalloc(&dev_triIndices, num_triangles * sizeof(int));
        cudaMemcpy(dev_triIndices, host_triangleIndices.data(), num_triangles * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_bvh, nodes_used * sizeof(BVHNode));
        cudaMemcpy(dev_bvh, host_bvhNodes.data(), nodes_used * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }

    if (num_PBRmaterials > 0) {
        cudaMalloc(&dev_PBRmaterials, num_PBRmaterials * sizeof(Material));
        cudaMemcpy(dev_PBRmaterials, host_materials.data(),
            num_PBRmaterials * sizeof(Material), cudaMemcpyHostToDevice);
    }

    std::cout << "Sequential upload completed: " << num_triangles << " triangles, "
        << num_PBRmaterials << " materials" << std::endl;

    // Clear host data to save memory
    host_triangles.clear();
    host_materials.clear();
    host_bvhNodes.clear();
    host_triangleIndices.clear();
}

void GLTFManager::clearCurrData() {
    cleanup();
    host_triangles.clear();
    host_materials.clear();
    host_bvhNodes.clear();
    host_triangleIndices.clear();
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
    if (dev_bvh) {
        cudaFree(dev_bvh);
        dev_bvh = nullptr;
    }
    if (dev_triIndices) {
        cudaFree(dev_triIndices);
        dev_triIndices = nullptr;
    }

    num_triangles = 0;
    num_PBRmaterials = 0;
    nodes_used = 1;
}