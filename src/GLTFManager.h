#pragma once

#include "tiny_gltf.h"
#include "sceneStructs.h"
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "glm/common.hpp"
#include "glm/glm.hpp"

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
        int base_color_tex = -1;
        int metallic_roughness_tex = -1;
        int normal_tex = -1;
        int emissive_tex = -1;
        float emissive_factor[3] = { 0.0f, 0.0f, 0.0f };
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
    const std::vector<TextureData>& getTextures() const { return textures; }
    const std::vector<cudaTextureObject_t>& getTextureObjects() const { return texture_objects; }

private:
    bool processModel(const tinygltf::Model& model);
    MaterialData processMaterial(const tinygltf::Material& mat, const tinygltf::Model& model);
    MeshData processPrimitive(const tinygltf::Primitive& primitive, const tinygltf::Model& model);
    TextureData processTexture(const tinygltf::Image& image);

    void extractAttribute(const tinygltf::Primitive& primitive,
        const tinygltf::Model& model,
        const std::string& attribute,
        std::vector<float>& output, int components);

    cudaTextureObject_t createTextureObject(const unsigned char* data, int width, int height, int channels);
    bool createCUDATextures();

    std::vector<MeshData> meshes;
    std::vector<MaterialData> materials;
    std::vector<TextureData> textures;
    std::vector<cudaTextureObject_t> texture_objects;
    std::vector<cudaArray_t> texture_arrays; // Keep track for cleanup
};

class GLTFManager {
public:
    GLTFManager();
    ~GLTFManager();

    bool uploadToGPU(const GLTFLoader& loader);
    void cleanup();

    // Device data accessors
    Triangle* getTrianglesDevice() const { return dev_triangles; }
    Material* getPBRMaterialsDevice() const { return dev_PBRmaterials; }
    cudaTextureObject_t* getTextureObjectsDevice() const { return dev_texture_objects; }

    int getNumTriangles() const { return num_triangles; }
    int getNumMaterials() const { return num_PBRmaterials; }
    int getNumTextures() const { return num_textures; }

private:
    void uploadTriangles(const std::vector<GLTFLoader::MeshData>& meshes);
    void uploadMaterials(const std::vector<GLTFLoader::MaterialData>& materials,
        const std::vector<cudaTextureObject_t>& texture_objects);
    void uploadTextureObjects(const std::vector<cudaTextureObject_t>& texture_objects);

    // Device pointers
    Triangle* dev_triangles = nullptr;
    Material* dev_PBRmaterials = nullptr;
    cudaTextureObject_t* dev_texture_objects = nullptr;

    int num_triangles = 0;
    int num_PBRmaterials = 0;
    int num_textures = 0;
};