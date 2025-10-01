#include "scene.h"

#include "utilities.h"
#include "pathtrace.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "GLTFManager.h"

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    gltfManager = GLTFManager();

    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        //return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"].get<float>();
            newMaterial.type = EMISSIVE;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = SPECULAR_REFL;
        }
        else if (p["TYPE"] == "Transmissive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = SPECULAR_TRANS;
            newMaterial.indexOfRefraction = p["IOR"].get<float>();
        }
        else if (p["TYPE"] == "Dielectric")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = DIELECTRIC;
            newMaterial.probReflVTrans = p["REFLECTIONODDS"].get<float>();
            newMaterial.indexOfRefraction = p["IOR"].get<float>();
        }
        else if (p["TYPE"] == "Pbr")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = PBR_MAT;
            newMaterial.metallic = p["METALLIC"].get<float>();
            newMaterial.roughness = p["ROUGHNESS"].get<float>();
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        newGeom.material = materials.at(newGeom.materialid).type;
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    // loading in GLTF paths and constructing their matrices
    if (data.contains("GLTF")) {
        const auto& gltfArray = data["GLTF"];
        for (const auto& gltfData : gltfArray) {
            // They need to have a path, we skip if they don't
            if (!gltfData.contains("Path")) continue;
            gltfs.push_back(gltfData["Path"].get<std::string>());

            // transformations with defaults
            const auto& rot = gltfData.value("Rot", json::array({ 0.0f, 0.0f, 0.0f }));
            glm::vec3 rotVec = glm::vec3(rot[0], rot[1], rot[2]);
            const auto& trans = gltfData.value("Trans", json::array({ 0.0f, 0.0f, 0.0f }));
            glm::vec3 transVec = glm::vec3(trans[0], trans[1], trans[2]);
            const auto& scale = gltfData.value("Scale", json::array({ 1.0f, 1.0f, 1.0f }));
            glm::vec3 scaleVec = glm::vec3(scale[0], scale[1], scale[2]);

            glm::mat4 transMat = utilityCore::buildTransformationMatrix(transVec, rotVec, scaleVec);
            gltfMatrices.push_back(transMat);
        }
    }
}

void Scene::loadFromGLTF()
{
    GLTFLoader loader;
    gltfManager.beginSequentialUpload();
    for (std::string gltfName : gltfs)
    {
        if (!loader.load(gltfName)) {
            std::cerr << "Failed to load GLTF file:" << gltfName << std::endl;
            return;
        }
        else
        {
            gltfManager.addScene(loader, textLoader, gltfMatrices);
            loader.clear();
        }
    }

    // after we loaded in all the triangles we can construct the BVH
    numTriangles = gltfManager.getTrianglesHost()->size();
    gltfManager.getBVHHost()->resize(numTriangles * 2);
    // initializing just to be extra safe
    for (auto& node : *gltfManager.getBVHHost())
    {
        node.leftFirst = -1;
        node.triCount = 0;
    }
    gltfManager.nodes_used = 1;
    int nimBVHinit = gltfManager.getBVHHost()->size();
    gltfManager.getTriIntHost()->resize(numTriangles);
    BVH::BuildBVH(numTriangles, *gltfManager.getTriIntHost(), *gltfManager.getTrianglesHost(), *gltfManager.getBVHHost(), gltfManager.nodes_used);

    numBVHnodes = gltfManager.nodes_used;

    gltfManager.finishSequentialUpload();

    std::cout << "All scenes loaded successfully!" << std::endl;
    std::cout << "Total triangles: " << gltfManager.getNumTriangles() << std::endl;
    std::cout << "Total materials: " << gltfManager.getNumMaterials() << std::endl;
    std::cout << "BVH Triangles: " << numTriangles << std::endl;
    std::cout << "BVH Nodes inittialized: " << nimBVHinit << std::endl;
    std::cout << "Nodes used: " << numBVHnodes << std::endl;
}

bool Scene::loadEnvironmentMap(const std::string& hdr_filename) {
    if (curr_env_map.name != "")
    {
        // if we have one already loaded -> clear it
        clearEnvironmentMap();
    }

    curr_env_map = textLoader.loadEnvMap(hdr_filename);
    if (curr_env_map.texture == 0) {
        std::cerr << "Failed to load environment map: " << hdr_filename << std::endl;
        return false;
    }
    curr_env_map.name = hdr_filename;
    printf("Environment map %s loaded: %d x %d\n", curr_env_map.name.c_str(), curr_env_map.width, curr_env_map.height);
    return true;
}

void Scene::clearEnvironmentMap() {
    if (curr_env_map.texture != 0) {
        cudaDestroyTextureObject(curr_env_map.texture);
        if (curr_env_map.data) {
            stbi_image_free(curr_env_map.data);
        }
    }
    curr_env_map = EnvMap{ 0, "", 0, 0, nullptr};
}
