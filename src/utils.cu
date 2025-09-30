#include "utils.cuh"

namespace Utils
{
    __global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < N) {
            intBuffer[index] = value;
        }
    }

    __global__ void kernIdentifyStartEnd(int N, ShadeableIntersection* intSects,
        int* materialStartIndices, int* materialEndIndices) {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index > N - 1) return;
        MaterialType thisMat = intSects[index].materialType;
        if (index == 0)
        {
            materialStartIndices[thisMat] = index;
            return;
        }
        MaterialType prevMat = intSects[index - 1].materialType;
        if (index > 0 && index < N - 1 && thisMat != prevMat)
        {
            materialStartIndices[thisMat] = index;
            materialEndIndices[prevMat] = index - 1;
        }
        else if (index == N - 1)
        {
            materialEndIndices[thisMat] = index;
        }
    }

    __device__ glm::vec4 sampleTexture(cudaTextureObject_t tex, glm::vec2 uv) {
        if (tex == 0) return glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        float4 text = tex2D<float4>(tex, uv.x, uv.y);
        return glm::vec4(text.x, text.y, text.z, text.w);
    }

    __device__ glm::vec3 sampleEnvMap(cudaTextureObject_t env_map,
        glm::vec3 dir) {
        if (env_map < EPSILON || env_map > 1000) {
            return glm::vec3(0.0f, 0.0f, 0.0f); // black if no map
        }
        glm::vec2 uv = dirToUV(dir);
        if (uv.x > 1.f || uv.y > 1.f)
        {
            printf("ERROR: UV out of bounds! X: %f, Y: %f\n", uv.x, uv.y);
            return glm::vec3(1.f, 0.f, 0.f);
        }
        float4 sample = tex2D<float4>(env_map, uv.x, uv.y);
        return glm::vec3(sample.x, sample.y, sample.z);
    }

    __device__ float Lambda(const glm::vec3& w, const float roughness)
    {
        float absTanTheta = abs(TanTheta(w));
        if (isinf(absTanTheta)) return 0.;

        // Compute alpha for direction w
        float alpha =
            sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
        float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
        return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
    }

    __device__ float TrowbridgeReitzD(const glm::vec3& wh, const float roughness)
    {
        float tan2Theta = Tan2Theta(wh);
        if (isinf(tan2Theta)) return 0.f;

        float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

        float e = (Cos2Phi(wh) / (roughness * roughness) + Sin2Phi(wh) / (roughness * roughness)) * tan2Theta;
        return 1 / (PI * roughness * roughness * cos4Theta * (1 + e) * (1 + e));
    }

    __device__ float TrowbridgeReitzG(const glm::vec3& wo, const glm::vec3& wi, const float roughness)
    {
        return 1 / (1 + Lambda(wo, roughness) + Lambda(wi, roughness));
    }

    __device__ float TrowbridgeReitzPdf(const glm::vec3& wo, const glm::vec3& wh, const float roughness)
    {
        return TrowbridgeReitzD(wh, roughness) * glm::abs(wh.z);
    }
}

namespace PBR
{

    __global__ void kernShadeAll(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials, cudaTextureObject_t envMap)
    {
        int idx =  blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > num_paths) return;

        PathSegment* seg = &pathSegments[idx];
        if (seg->remainingBounces < 1) return;
        ShadeableIntersection* intsect = &shadeableIntersections[idx];
        MaterialType matType = intsect->materialType;

        switch (matType)
        {
        case NONE:
            inlineShadeNosect(iter, num_paths, shadeableIntersections, pathSegments, materials, envMap);
            break;
        case DIFFUSE:
            inlineShadeDiffuse(iter, num_paths, shadeableIntersections, pathSegments, materials);
            break;
        case EMISSIVE:
            inlineShadeEmissive(iter, num_paths, shadeableIntersections, pathSegments, materials);
            break;
        case SPECULAR_REFL:
            inlineShadeSpecularRefl(iter, num_paths, shadeableIntersections, pathSegments, materials);
            break;
        case SPECULAR_TRANS:
            inlineShadeSpecularTrans(iter, num_paths, shadeableIntersections, pathSegments, materials);
            break;
        case DIELECTRIC:
            Material mat = materials[intsect->materialId];
            thrust::default_random_engine rng = PBR::makeSeededRandomEngine(iter, idx, seg->remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);
            float ior = mat.indexOfRefraction;  // NOTE: this can cause weird results if it is incorrectly set
            // use rng to get a random val between [0, 1] -> if its less than prob of refl, then we do reflection
            // otherwise we will transmissive
            if (u01(rng) < mat.probReflVTrans)
            {
                inlineShadeSpecularRefl(iter, num_paths, shadeableIntersections, pathSegments, materials);
                float cosThetaI = glm::dot(intsect->surfaceNormal, glm::normalize(seg->ray.direction));
                seg->color *= 2.f * FresnelDielectricEval(cosThetaI, ior);
            }
            else
            {
                inlineShadeSpecularTrans(iter, num_paths, shadeableIntersections, pathSegments, materials);
                float cosThetaI = glm::dot(intsect->surfaceNormal, glm::normalize(seg->ray.direction));
                seg->color *= 2.f * (glm::vec3(1.f) - FresnelDielectricEval(cosThetaI, ior));
            }
            break;
        case PBR_MAT:
            inlineShadePBR(iter, num_paths, shadeableIntersections, pathSegments, materials);   // I think this should have pbr mats
            break;
        default:
            // No material tag-> how did we even get here?
            printf("ERROR: No mat type in kernShadeAll at index: %d", idx);
            break;
        }

    }
    __global__ void kernShadeNosect(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials, cudaTextureObject_t envMap)
    {
        inlineShadeNosect(iter, num_paths, shadeableIntersections, pathSegments, materials, envMap);
    }

    __global__ void kernShadeDiffuse(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeDiffuse(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __global__ void kernShadeEmissive(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeEmissive(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __global__ void kernShadeSpecularRefl(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeSpecularRefl(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __global__ void kernShadeSpecularTrans(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeSpecularTrans(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __global__ void kernShadeDielectric(int iter,
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
            thrust::default_random_engine rng = PBR::makeSeededRandomEngine(iter, idx, seg->remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);
            float ior = material.indexOfRefraction;// ? material.indexOfRefraction : 1.;
            // use rng to get a random val between [0, 1] -> if its less than prob of refl, then we do reflection
            // otherwise we will transmissive
            if (u01(rng) < material.probReflVTrans)
            {
                inlineShadeSpecularRefl(iter, num_paths, shadeableIntersections, pathSegments, materials);
                float cosThetaI = glm::dot(intersection.surfaceNormal, glm::normalize(seg->ray.direction));
                seg->color *= 2.f * FresnelDielectricEval(cosThetaI, ior);
            }
            else
            {
                inlineShadeSpecularTrans(iter, num_paths, shadeableIntersections, pathSegments, materials);
                float cosThetaI = glm::dot(intersection.surfaceNormal, glm::normalize(seg->ray.direction));
                seg->color *= 2.f * (glm::vec3(1.f) - FresnelDielectricEval(cosThetaI, ior));
            }
        }
    }

    __global__ void kernShadePBR(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadePBR(iter, num_paths, shadeableIntersections, pathSegments, materials);
    }

    __device__ glm::vec3 FresnelDielectricEval(float cosThetaI, float ior)
    {
        // Assuming we are always either entering or exiting air
        float etaI = 1.;
        float etaT = ior;
        cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

        // check if we're entering or exiting and swap accordingly
        bool entering = cosThetaI > 0.f;
        if (!entering) {
            // swap etaI and etaT
            float temp = etaI;
            etaI = etaT;
            etaT = temp;
            cosThetaI = abs(cosThetaI);
        }
        // Snell's Law
        float sinThetaI = sqrt(glm::max(0.0f, 1 - cosThetaI * cosThetaI));
        float sinThetaT = etaI / etaT * sinThetaI;
        // total internal reflection
        if (sinThetaT >= 1) {
            return glm::vec3(1.);
        }

        float cosThetaT = sqrt(glm::max(0.0f, 1 - sinThetaT * sinThetaT));
        float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
            ((etaT * cosThetaI) + (etaI * cosThetaT));
        float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
            ((etaI * cosThetaI) + (etaT * cosThetaT));
        float ret = (Rparl * Rparl + Rperp * Rperp) * 0.5;

        return glm::vec3(ret);
    }
}