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
            seg->color = glm::vec3(0.);
            seg->keepLooping = false;
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

            if (!seg->keepLooping) return;

            if (intersection.t > 0.0f) // if the intersection exists...
            {
                // Set up the RNG
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, seg->remainingBounces);
                thrust::uniform_real_distribution<float> u01(0, 1);
                

                Material material = materials[intersection.materialId];
                glm::vec3 materialColor = material.color;

                //// If the material indicates that the object was a light, "light" the ray
                if (material.emittance > 0.0f) {
                    seg->color *= (materialColor * material.emittance);
                    seg->keepLooping = false;
                    printf("EMITTANCE IN MAH DIFFUSE???? %d \n", idx);
                    return;
                }
                // Otherwise, do some pseudo-lighting computation. This is actually more
                // like what you would expect from shading in a rasterizer like OpenGL.
                else {
                    glm::vec3 wi = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
                    //seg->color *= glm::clamp(materialColor * abs(dot(-wi, intersection.surfaceNormal)), 0.0f, 1.0f);
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
                //seg->keepLooping = false;   // tag it as a segment that we can terminate
                //seg->remainingBounces = 0;
                seg->color *= glm::vec3(0.0f);
            }
            if (seg->remainingBounces < 1)
            {
                seg->keepLooping = false;
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
            if (!seg->keepLooping) return;
            if (intersection.t > 0.0) // if the intersection exists...
            {
                Material material = materials[intersection.materialId];
                glm::vec3 materialColor = material.color;
                seg->color *= (materialColor * material.emittance);
                seg->keepLooping = false;
                seg->remainingBounces = 0;
            }
            else
            {
                seg->keepLooping = false;   // tag it as a segment that we can terminate
                seg->remainingBounces = 0;
                //seg->color *= glm::vec3(0.0f);
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

    __global__ __inline__ void kernShadeNosect(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeNosect(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __global__ __inline__ void kernShadeDiffuse(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeDiffuse(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __global__ __inline__ void kernShadeEmissive(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeEmissive(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __global__ __inline__ void kernShadeSpecularRefl(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeSpecularRefl(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }
}

//namespace StreamCompaction {
//    __global__ void kernMapToBoolean(int n, int* bools, const PathSegment* idata, std::function<bool(const PathSegment&)> predicate);
//
//    __global__ void kernScatter(int n, PathSegment* odata,
//        const PathSegment* idata, const int* bools, const int* indices);
//
//    __global__ void kernUpSweep(int n, int* data, int d);
//    __global__ void kernDownSweep(int n, int* data, int d);
//    __global__ void kernChangeOneVal(int index, int* data, int val);
//
//    int compact(int n, PathSegment* odata, const PathSegment* idata, std::function<bool(const PathSegment&)> predicate);
//}

