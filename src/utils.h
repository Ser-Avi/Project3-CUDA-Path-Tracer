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
}

namespace PBR
{
    __host__ __device__ __inline__
        thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
    {
        int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
        return thrust::default_random_engine(h);
    }

    DEV_INLINE void inlineShadeNosect(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_paths)
        {
            PathSegment* seg = &pathSegments[idx];
            if (seg->remainingBounces < 1) return;
            seg->color = glm::vec3(0.);
            seg->remainingBounces = 0;
        }
    }

    DEV_INLINE void inlineShadeDiffuse(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_paths)
        {
            ShadeableIntersection intersection = shadeableIntersections[idx];
            PathSegment* seg = &pathSegments[idx];

            //if (!seg->keepLooping) return;

            if (intersection.t > 0.0f && seg->remainingBounces > 0) // if the intersection exists...
            {
                // Set up the RNG
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, seg->remainingBounces);
                thrust::uniform_real_distribution<float> u01(0, 1);
                

                Material material = materials[intersection.materialId];
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
                    glm::vec3 wi = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
                    seg->color *= materialColor;

                    // then we bounce a new ray
                    seg->remainingBounces--;
                    glm::vec3 p = seg->ray.origin + seg->ray.direction * intersection.t;
                    seg->ray.origin = p + intersection.surfaceNormal * EPSILON;
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
    }

    DEV_INLINE void inlineShadeEmissive(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_paths)
        {
            ShadeableIntersection intersection = shadeableIntersections[idx];
            PathSegment* seg = &pathSegments[idx];
            if (intersection.t > 0.0 && seg->remainingBounces > 0) // if the intersection exists...
            {
                Material material = materials[intersection.materialId];
                glm::vec3 materialColor = material.color;
                seg->color *= (materialColor * material.emittance);
                seg->remainingBounces = 0;
            }
            else
            {
                seg->remainingBounces = 0;  // tag it as a segment that we can terminate
            }
        }
    }

    DEV_INLINE void inlineShadeSpecularRefl(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_paths)
        {
            ShadeableIntersection intersection = shadeableIntersections[idx];
            PathSegment* seg = &pathSegments[idx];
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (seg->remainingBounces < 1) return;

            // Proper specular reflection
            glm::vec3 wi = glm::reflect(seg->ray.direction, intersection.surfaceNormal);
            //float cos_theta = abs(glm::dot(wi, intersection.surfaceNormal));

            //seg->color *= materialColor * cos_theta;
            //seg->color = glm::clamp(seg->color, 0.0f, 1.0f);

            seg->remainingBounces--;
            glm::vec3 p = seg->ray.origin + seg->ray.direction * intersection.t;
            seg->ray.origin = p + intersection.surfaceNormal * EPSILON;
            seg->ray.direction = wi;
        }
    }

    DEV_INLINE void inlineShadeSpecularTrans(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_paths)
        {
            ShadeableIntersection intersection = shadeableIntersections[idx];
            PathSegment* seg = &pathSegments[idx];
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (seg->remainingBounces < 1) return;

            glm::vec3 nor = intersection.surfaceNormal;

            // either air or material -> not supporting two dielectrics
            float etaA = 1.;
            float etaB = material.indexOfRefraction;

            glm::vec3 wi;
            // checking if ray is coming from inside the house O.O
            bool entering = glm::dot(-seg->ray.direction, nor) > 0.;
            // setting values appropriately
            float etaI = entering ? etaA : etaB;
            float etaT = entering ? etaB : etaA;
            // Computing ray direction
            glm::vec3 normal = -glm::faceforward(nor, -seg->ray.direction, nor); //entering ? nor : -nor;// -glm::faceforward(nor, -seg->ray.direction, nor);
            float etaRatio = etaI / etaT;

            wi = glm::refract(-seg->ray.direction, normal, etaRatio);

            // glm::refract returns vec3(0) when total internal reflection occurs
            if (length(wi) < 0.01) {
                // To kill or reflect? That is the question
                /*seg->color *= glm::vec3(0.);
                seg->keepLooping = false;*/
                wi = glm::reflect(seg->ray.direction, normal);
            }

            seg->remainingBounces--;
            glm::vec3 p = seg->ray.origin + seg->ray.direction * intersection.t;
            seg->ray.origin = p + wi * EPSILON * 5.f;   // this 5 is creating an inside band but without it my rays can get really stuck :(
            seg->ray.direction = wi;

            seg->color *= materialColor;
        }
    }

    __global__ void kernShadeAll(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeNosect(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeDiffuse(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeEmissive(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeSpecularRefl(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeSpecularTrans(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __global__ void kernShadeDielectric(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials);

    __device__ glm::vec3 FresnelDielectricEval(float cosThetaI, float ior);
}

