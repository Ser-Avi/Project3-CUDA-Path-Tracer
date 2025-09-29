#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE
};

struct alignas(32) BVHNode
{
    glm::vec3 aabbMin, aabbMax;
    uint32_t leftFirst, triCount;
};

struct Triangle {
    glm::vec3 v0, v1, v2;
    //glm::vec3 centroid;
    glm::vec3 n0, n1, n2;
    glm::vec2 uv0, uv1, uv2;
    int material_id;
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

enum MaterialType
{
    NONE = 0,
    EMISSIVE,
    DIFFUSE,
    SPECULAR_REFL,
    SPECULAR_TRANS,
    DIELECTRIC,
    PBR_MAT
};

struct Geom
{
    enum GeomType type;
    MaterialType material;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    MaterialType type;
    glm::vec3 color = glm::vec3(1.);    // default to white
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float probReflVTrans; // the probability of it reflecting vs transmitting
    float indexOfRefraction;
    float emittance = 0.;   // default to non-emissive
    float roughness = 1.;   // default to rough plastic for pbr
    float metallic = 0.;
    float ao = 1.;
    glm::vec3 emissive_factor;
    cudaTextureObject_t base_color_tex;
    cudaTextureObject_t metallic_roughness_tex;
    cudaTextureObject_t normal_tex;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 thp;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  int materialId;
  MaterialType materialType;
};

struct aabb
{
    glm::vec3 bmin = glm::vec3(1e30f);
    glm::vec3 bmax = glm::vec3(-1e30f);
    void grow(glm::vec3 p) { bmin = glm::min(bmin, p), bmax = glm::max(bmax, p); }
    void grow(aabb& b) { if (b.bmin.x != 1e30f) { grow(b.bmin); grow(b.bmax); } }
    float area()
    {
        glm::vec3 e = bmax - bmin; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};
