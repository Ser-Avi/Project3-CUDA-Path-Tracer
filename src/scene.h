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
    GLTFManager gltfManager;
    TextureLoader textLoader;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
