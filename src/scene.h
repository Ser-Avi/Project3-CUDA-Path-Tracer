#pragma once

#include "sceneStructs.h"
#include "GLTFManager.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);
    void loadFromGLTF();
    std::string gltfName = "";
    GLTFManager gltfManager;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    void initGPU();
};
