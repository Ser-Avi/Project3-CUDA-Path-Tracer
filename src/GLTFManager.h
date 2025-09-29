#pragma once

#include "tiny_gltf.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sceneStructs.h"
#include "intersections.h"
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "glm/common.hpp"
#include "glm/glm.hpp"
#include <utility>
#include <algorithm>

class TextureLoader {
public:
    TextureLoader();
    ~TextureLoader();

    // Load a PNG texture and create CUDA texture object
    cudaTextureObject_t loadTexture(const std::string& filename);

    // Get texture by filename (cached)
    cudaTextureObject_t getTexture(const std::string& filename);

    // Cleanup all textures
    void cleanup();

private:
    cudaTextureObject_t createTextureFromData(const unsigned char* data,
        int width, int height, int channels);

    std::map<std::string, cudaTextureObject_t> texture_cache;
    std::vector<cudaArray_t> texture_arrays; // For cleanup
};

/// <summary>
/// This BVH method comes from the following blog, parts 1 through 3
/// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
/// </summary>
namespace BVH
{
    struct Bin { aabb bounds; int triCount = 0; };
    void BuildBVH(int N, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode, int& nodesUsed);
    void UpdateNodeBounds(uint32_t nodeIdx, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode);
    void Subdivide(uint32_t nodeIdx, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode, int& nodesUsed);
    void Subdivide_Fast(uint32_t nodeIdx, std::vector<int>& triIdx, std::vector<Triangle>& tri, std::vector<BVHNode>& bvhNode, int& nodesUsed);
    float FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos, std::vector<int>& triIdx, std::vector<Triangle>& tri);
    float CalculateNodeCost(BVHNode& node);
}

class GLTFLoader {
public:
    struct MeshData {
        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> texcoords;
        std::vector<uint32_t> indices;
        int material_id;
    };

    struct MaterialData {
        float base_color[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        float metallic = 0.0f;
        float roughness = 1.0f;
        float ao;
        // Paths to external textures
        std::string base_color_texture_path;
        std::string metallic_roughness_texture_path;
        std::string normal_texture_path;
    };

    struct TextureData {
        int width, height, channels;
        std::vector<unsigned char> data;
    };

    GLTFLoader();
    ~GLTFLoader();

    bool load(const std::string& filename);
    void clear();

    // Getters
    const std::vector<MeshData>& getMeshes() const { return meshes; }
    const std::vector<MaterialData>& getMaterials() const { return materials; }

private:
    bool processModel(const std::string& filename, const tinygltf::Model& model);
    MaterialData processMaterial(const tinygltf::Material& mat, const tinygltf::Model& model, const std::string& dir);
    MeshData processPrimitive(const tinygltf::Primitive& primitive, const tinygltf::Model& model);

    void extractAttribute(const tinygltf::Primitive& primitive,
        const tinygltf::Model& model,
        const std::string& attribute,
        std::vector<float>& output, int components);

    std::vector<MeshData> meshes;
    std::vector<MaterialData> materials;
};

class GLTFManager {
public:
    GLTFManager();
    ~GLTFManager();

    void beginSequentialUpload();       // for multiple gltfs
    bool addScene(const GLTFLoader& loader, TextureLoader& texture_loader);
    void finishSequentialUpload();

    void clearCurrData();
    void cleanup();

    // Device data accessors
    Triangle* getTrianglesDevice() const { return dev_triangles; }
    std::vector<Triangle>* getTrianglesHost() { return &host_triangles; }
    Material* getPBRMaterialsDevice() const { return dev_PBRmaterials; }
    int* getTriIntDevice() const { return dev_triIndices; }
    BVHNode* getBVHDevice() const { return dev_bvh; }
    std::vector<BVHNode>* getBVHHost() { return &host_bvhNodes; }
    std::vector<int>* getTriIntHost() { return &host_triangleIndices; }

    int getNumTriangles() const { return num_triangles; }
    int getNumMaterials() const { return num_PBRmaterials; }
    int nodes_used = 1;

private:

    // Host vals
    std::vector<Triangle> host_triangles;
    std::vector<Material> host_materials;
    std::vector<BVHNode> host_bvhNodes;
    std::vector<int> host_triangleIndices;

    // Device pointers
    Triangle* dev_triangles = nullptr;
    Material* dev_PBRmaterials = nullptr;
    BVHNode* dev_bvh = nullptr;
    int* dev_triIndices = nullptr;

    int num_triangles = 0;
    int num_PBRmaterials = 0;
};