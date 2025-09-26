#pragma once

#include "tiny_gltf.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sceneStructs.h"
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "glm/common.hpp"
#include "glm/glm.hpp"

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
        // Paths to external textures
        std::string base_color_texture_path;
        std::string metallic_roughness_texture_path;
        std::string normal_texture_path;
        std::string emissive_texture_path;
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

    bool uploadToGPU(const GLTFLoader& loader, TextureLoader& text_loader);
    void cleanup();

    // Device data accessors
    Triangle* getTrianglesDevice() const { return dev_triangles; }
    Material* getPBRMaterialsDevice() const { return dev_PBRmaterials; }

    int getNumTriangles() const { return num_triangles; }
    int getNumMaterials() const { return num_PBRmaterials; }

private:
    void uploadTriangles(const std::vector<GLTFLoader::MeshData>& meshes);
    void uploadMaterials(const std::vector<GLTFLoader::MaterialData>& materials,
        TextureLoader& text_loader);
 
    // Device pointers
    Triangle* dev_triangles = nullptr;
    Material* dev_PBRmaterials = nullptr;

    int num_triangles = 0;
    int num_PBRmaterials = 0;
};