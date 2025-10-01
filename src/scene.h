#pragma once

#include "sceneStructs.h"
#include "intersections.h"
#include "GLTFManager.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);
    void loadFromGLTF();
    std::vector<std::string> gltfs;
    std::vector<glm::mat4> gltfMatrices;
    GLTFManager gltfManager;
    TextureLoader textLoader;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // Env Map funcs
    EnvMap curr_env_map;
    bool loadEnvironmentMap(const std::string& hdr_filename);
    void clearEnvironmentMap();
    uint32_t numTriangles;
    uint32_t numBVHnodes;
};
