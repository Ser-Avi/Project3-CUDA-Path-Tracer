#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <iostream>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <functional>

#include "sceneStructs.h"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define DEV_INLINE __device__ __forceinline__

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

namespace Utils
{
    __global__ void kernResetIntBuffer(int N, int* intBuffer, int value);

    __global__ void kernIdentifyStartEnd(int N, ShadeableIntersection* intSects,
        int* materialStartIndices, int* materialEndIndices);

    __device__ glm::vec4 sampleTexture(cudaTextureObject_t tex, glm::vec2 uv);

    DEV_INLINE float Cos2Theta(glm::vec3 w) { return w.z * w.z; }
    DEV_INLINE float CosTheta(glm::vec3 w) { return w.z; }
    DEV_INLINE float Sin2Theta(glm::vec3 w) { return glm::max(0.f, 1.f - Cos2Theta(w)); }
    DEV_INLINE float SinTheta(glm::vec3 w) { return sqrt(Sin2Theta(w)); }
    DEV_INLINE float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }
    DEV_INLINE float Tan2Theta(glm::vec3 w) { return Sin2Theta(w) / Cos2Theta(w); }
    DEV_INLINE float CosPhi(glm::vec3 w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
    }
    DEV_INLINE float SinPhi(glm::vec3 w) {
        float sinTheta = SinTheta(w);
        return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
    }
    DEV_INLINE float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }
    DEV_INLINE float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }


    /// <summary>
    /// Creates the basis vectors based on a given normal n.
    /// I got this from https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    /// Can easily be used to get local->world and world->local transform matrices
    /// </summary>
    /// <param name="n">normal input</param>
    /// <param name="b1">basis vec1 output</param>
    /// <param name="b2">basis vec2 output</param>
    /// <returns></returns>
    DEV_INLINE void branchlessONB(const glm::vec3& n, glm::vec3& b1, glm::vec3& b2)
    {
        float sign = copysignf(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;
        b1 = glm::vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        b2 = glm::vec3(b, sign + n.y * n.y * a, -n.y);
    }

    /// <summary>
    /// Using the branchlessONB above, this gives the local version of a vector
    /// </summary>
    /// <param name="vector"></param>
    /// <param name="normal"></param>
    /// <returns></returns>
    DEV_INLINE glm::vec3 WorldToLocal(const glm::vec3& vector, const glm::vec3& normal)
    {
        glm::vec3 b1, b2;
        branchlessONB(normal, b1, b2);

        return glm::vec3(
            glm::dot(vector, b1),
            glm::dot(vector, b2),
            glm::dot(vector, normal)
        );
    }
    
    /// <summary>
    /// Using the branchlessONB above, this gives the global version of a vector
    /// </summary>
    /// <param name="localVector"></param>
    /// <returns></returns>
    DEV_INLINE glm::vec3 LocalToWorld(const glm::vec3& localVector, const glm::vec3& normal)
    {
        glm::vec3 b1, b2;
        branchlessONB(normal, b1, b2);

        // For local-to-world: linear combination of basis vectors
        return localVector.x * b1 + localVector.y * b2 + localVector.z * normal;
    }

    DEV_INLINE glm::vec3 getWH(glm::vec3 wo, glm::vec2 xi, float roughness)
    {
        glm::vec3 wh;

        float cosTheta = 0;
        float phi = TWO_PI * xi[1];
        // We'll only handle isotropic microfacet materials
        float tanTheta2 = roughness * roughness * xi[0] / (1.0f - xi[0]);
        cosTheta = 1 / sqrt(1 + tanTheta2);

        float sinTheta = glm::sqrt(glm::max(0.f, 1.f - cosTheta * cosTheta));

        wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        if (!(wo.z * wh.z > 0)) { wh = -wh; }

        return wh;
    }

    __device__ float Lambda(const glm::vec3& w, const float roughness);
    __device__ float TrowbridgeReitzD(const glm::vec3& wh, const float roughness);
    __device__ float TrowbridgeReitzG(const glm::vec3& wo, const glm::vec3& wi, const float roughness);

    __device__ float TrowbridgeReitzPdf(const glm::vec3& wo, const glm::vec3& wh, const float roughness);

    DEV_INLINE glm::vec2 dirToUV(glm::vec3 dir) {
        glm::vec2 uv = glm::vec2(glm::atan(dir.z, dir.x), glm::asin(dir.y));
        glm::vec2 normalize_uv = glm::vec2(0.1591, 0.3183);
        uv *= normalize_uv;
        uv += 0.5;
        uv.y = 1.f - uv.y;  // env map was flipped for some reason, so this flips it back
        return uv;
    }

    __device__ glm::vec3 sampleEnvMap(cudaTextureObject_t env_map, glm::vec3 dir);
}

namespace PBR
{
    __host__ __device__ __inline__
        thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
    {
        int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
        return thrust::default_random_engine(h);
    }

    DEV_INLINE void handleMaterialMaps(Material* mat, const glm::vec2 uv, glm::vec3& albedo, float& metallic, float& rough,
        float& ao, glm::vec3 nor)
    {
        if (mat->metallic_roughness_tex != 0)
        {
            glm::vec4 metRough = Utils::sampleTexture(mat->metallic_roughness_tex, uv);
            ao = metRough.r < EPSILON ? 1. : metRough.r;    // ao is not necessarily in this texture
            metallic = metRough.b;
            rough = metRough.g;
        }
        if (mat->base_color_tex != 0)
        {
            albedo = glm::vec3(Utils::sampleTexture(mat->base_color_tex, uv));
        }
        if (mat->normal_tex != 0)
        {
            nor = glm::vec3(Utils::sampleTexture(mat->normal_tex, uv));
        }
    }

    DEV_INLINE glm::vec3 fresnelShlickRoughness(float cosTheta, glm::vec3 R, float rough)
    {
        return R + (max(glm::vec3(1.0f - rough), R) - R) * pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
    }

    DEV_INLINE void inlineShadeNosect(
        PathSegment* seg,
        cudaTextureObject_t envMap)
    {
        if (seg->remainingBounces < 1) return;
        // sample env map
        glm::vec3 envCol = Utils::sampleEnvMap(envMap, seg->ray.direction);
        //seg->color = (seg->ray.direction + glm::vec3(1.f)) * 0.5f;
        seg->color *= envCol;
        seg->remainingBounces = 0;
    }

    DEV_INLINE void inlineShadeDiffuse(int iter, int idx,
        ShadeableIntersection* intersection,
        PathSegment* seg,
        Material* materials)
    {
        if (intersection->t > 0.0f && seg->remainingBounces > 0) // if the intersection exists...
        {
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, seg->remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);
                

            Material material = materials[intersection->materialId];
            glm::vec3 materialColor = material.color;

            //// If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                seg->color *= (materialColor * material.emittance);
                seg->remainingBounces = 0;
                printf("EMITTANCE IN DIFFUSE???? %d \n", idx);
                return;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            else {
                glm::vec3 wi = calculateRandomDirectionInHemisphere(intersection->surfaceNormal, rng);
                seg->color *= materialColor;

                // then we bounce a new ray
                seg->remainingBounces--;
                glm::vec3 p = seg->ray.origin + seg->ray.direction * intersection->t;
                seg->ray.origin = p + intersection->surfaceNormal * EPSILON;
                seg->ray.direction = wi;
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.

        }
        else {
            seg->remainingBounces = 0;          // tag it as a segment that we can terminate
        }
    }

    DEV_INLINE void inlineShadeEmissive(
        ShadeableIntersection* intersection,
        PathSegment* seg,
        Material* materials)
    {
        if (intersection->t > 0.0 && seg->remainingBounces > 0) // if the intersection exists...
        {
            Material material = materials[intersection->materialId];
            glm::vec3 materialColor = material.color;
            seg->color *= (materialColor * material.emittance);
            seg->remainingBounces = 0;
        }
        else
        {
            seg->remainingBounces = 0;  // tag it as a segment that we can terminate
        }
    }

    DEV_INLINE void inlineShadeSpecularRefl(
        ShadeableIntersection* intersection,
        PathSegment* seg,
        Material* materials)
    {
        Material material = materials[intersection->materialId];
        glm::vec3 materialColor = material.color;

        if (seg->remainingBounces < 1) return;

        // Proper specular reflection
        glm::vec3 wi = glm::reflect(seg->ray.direction, intersection->surfaceNormal);
        //float cos_theta = abs(glm::dot(wi, intersection.surfaceNormal));

        //seg->color *= materialColor * cos_theta;
        //seg->color = glm::clamp(seg->color, 0.0f, 1.0f);

        seg->remainingBounces--;
        glm::vec3 p = seg->ray.origin + seg->ray.direction * intersection->t;
        seg->ray.origin = p + intersection->surfaceNormal * EPSILON;
        seg->ray.direction = wi;
    }

    DEV_INLINE void inlineShadeSpecularTrans(
        ShadeableIntersection* intersection,
        PathSegment* seg,
        Material* materials)
    {
        Material material = materials[intersection->materialId];
        glm::vec3 materialColor = material.color;

        if (seg->remainingBounces < 1) return;

        glm::vec3 nor = intersection->surfaceNormal;
        glm::vec3 wo = glm::normalize(-seg->ray.direction);

        float cosThetaI = glm::dot(nor, wo);
        bool entering = cosThetaI > 0.f;

        // either air or material -> not supporting two dielectrics
        float etaA = 1.;
        float etaB = material.indexOfRefraction;

        float eta = etaA / etaB;

        float iorRatio = entering ? eta : 1.f / eta;

        glm::vec3 wi = glm::refract(-wo, (entering) ? nor : -nor, iorRatio);

        // glm::refract returns vec3(0) when total internal reflection occurs
        if (length(wi) < 0.01) {
            // To kill or reflect? That is the question
            seg->color *= glm::vec3(0.);
            seg->remainingBounces = 0;
            return;
        }

        seg->remainingBounces--;
        glm::vec3 p = seg->ray.origin + seg->ray.direction * intersection->t;
        seg->ray.origin = p + wi * EPSILON;// *500.f;   // this 500 is making this work somewhat, but it is still a bit buggy
        seg->ray.direction = wi;
            seg->color *= materialColor;
    }

    DEV_INLINE void inlineShadePBR(int iter, int idx,
        ShadeableIntersection* intSect,
        PathSegment* seg,
        Material* materials)
    {
        glm::vec2 uv = intSect->uv;
        Material mat = materials[intSect->materialId];
        glm::vec3 albedo = mat.color;
        glm::vec3 norW = intSect->surfaceNormal;
        float metallic = mat.metallic;
        float roughness = mat.roughness;
        float ao = 1.;
        handleMaterialMaps(&mat, uv, albedo, metallic, roughness, ao, norW);
        glm::vec3 woWorld = -seg->ray.direction;
        //glm::vec3 wo = Utils::WorldToLocal(woWorld, norW);
        //glm::vec3 up = glm::vec3(0, 1, 0);
            
        // Actual PBR stuff
        glm::vec3 R = glm::mix(glm::vec3(0.04f), albedo, metallic);
        glm::vec3 kS = fresnelShlickRoughness(abs(dot(norW, woWorld)), R, roughness);
        glm::vec3 kD = 1.0f - kS;
        kD *= 1.0 - metallic;

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, seg->remainingBounces);
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 wiSpec = glm::reflect(-woWorld, norW);
        glm::vec3 wiDiff = calculateRandomDirectionInHemisphere(intSect->surfaceNormal, rng);
        // random direction
        //glm::vec3 xi = glm::vec3(u01(rng), u01(rng), u01(rng));
        float xi = u01(rng);
        glm::vec3 wiFin;

        // in practice, we use kS as a lerp between specular and diffuse
        // however, we have to decide here, so I will be using its max value
        // to decide which way to go and multiply by 2 to offset this distinct split.
        // hoping it works

        // now we do some branching and pray we did it all right
        if(glm::max(glm::max(kS.x, kS.y), kS.z) > xi)
        {
            if (roughness > EPSILON)
            {
                // if roughness is less than 1, then we use it, otherwise we just do perfectly specular
                glm::vec3 wh = glm::normalize(wiSpec + woWorld);
                float D = Utils::TrowbridgeReitzD(wh, roughness);
                float G = Utils::TrowbridgeReitzG(woWorld, wiSpec, roughness);
                albedo = glm::vec3(1.);
                // albedo is changed with the D and G terms -> incorporates roughness
                albedo *= kS * D + G;
            }
            else
            {
                albedo = glm::vec3(1.);   // we don't want to add our color if we're perfectly specular
            }

            /*wiFin = Utils::LocalToWorld(wiSpec);*/
            wiFin = wiSpec;

            //albedo /= kS;             // attenuate?
        }
        else
        {
            wiFin = wiDiff;
            //albedo *= glm::max(glm::dot(norW, wiFin), 0.f);
            //wiFin = Utils::LocalToWorld(wiDiff);
            // NOTE: maybe should have irradiance term somehow?
            // also, maybe shouldn't have absdot term
            albedo *= kD;
            //albedo /= glm::vec3(1.f) - kS;    // attenuate?
        }
        albedo *= ao;
        // then we bounce a new ray
        //wiFin = Utils::LocalToWorld(wiFin, norW);
        seg->remainingBounces--;
        glm::vec3 p = seg->ray.origin + seg->ray.direction * intSect->t;
        seg->ray.origin = p + norW * EPSILON;
        seg->ray.direction = wiFin;
        seg->color *= albedo;
    }

    __global__ void kernShadeAll(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials, Material* pbr_materials, cudaTextureObject_t envMap);

    __global__ void kernShadeNosect(int num_paths,
        PathSegment* pathSegments, cudaTextureObject_t envMap);

    __global__ void kernShadeDiffuse(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeEmissive(
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeSpecularRefl(
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeSpecularTrans(
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeDielectric(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadePBR(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __device__ glm::vec3 FresnelDielectricEval(float cosThetaI, float ior);
}

