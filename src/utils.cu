#include "utils.h"

//__global__ void kernComputeIndices(int N, int* indices, int* gridIndices) {
//    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//    if (index < N)
//    {
//        gridIndices[index] = ;// gridIndex3Dto1D(relativePos.x, relativePos.y, relativePos.z, gridResolution);
//    }
//}

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
        MaterialType prevMat = intSects[index - 1].materialType;

        if (index > 0 && index < N - 1 && thisMat != prevMat)
        {
            materialStartIndices[thisMat] = index;
            materialEndIndices[prevMat] = index - 1;
        }
        else if (index == 0)
        {
            materialStartIndices[thisMat] = index;
        }
        else if (index == N - 1)
        {
            materialEndIndices[thisMat] = index;
        }
    }
}

namespace PBR
{
    __global__ void kernShadeNosect(int iter,
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material* materials)
    {
        inlineShadeNosect(iter, num_paths, shadeableIntersections, pathSegments, materials);
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



//namespace StreamCompaction {
//
//    /**
//        * Maps an array to an array of 0s and 1s for stream compaction. Elements
//        * which map to 0 will be removed, and elements which map to 1 will be kept.
//        */
//    __global__ void kernMapToBoolean(int n, int* bools, const PathSegment* idata, std::function<bool(const PathSegment&)> predicate) {
//        int idx = blockIdx.x * blockDim.x + threadIdx.x;
//        if (idx > n - 1 || idx < 0) return;
//        bools[idx] = predicate(idata[idx]);
//    }
//
//    /**
//     * Performs scatter on an array. That is, for each element in idata,
//     * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
//     */
//    __global__ void kernScatter(int n, PathSegment* odata,
//        const PathSegment* idata, const int* bools, const int* indices) {
//        // TODO
//        int idx = blockIdx.x * blockDim.x + threadIdx.x;
//        if (idx > n - 1 || idx < 0) return;
//        if (bools[idx] == 1)
//        {
//            odata[indices[idx]] = idata[idx];
//        }
//    }
//
//    /// <summary>
//    /// Resets buffer to the set value - used for padding with 0s
//    /// </summary>
//    /// <param name="N"></param>
//    /// <param name="intBuffer"></param>
//    /// <param name="value"></param>
//    /// <returns></returns>
//    __global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
//        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//        if (index < N && index > -1) {
//            intBuffer[index] = value;
//        }
//    }
//
//    int blockSize = 32;
//
//    __global__ void kernUpSweep(int n, int* data, int d)
//    {
//        int idx = blockIdx.x * blockDim.x + threadIdx.x;
//        idx = idx * d - 1;
//        if (idx > n || idx < 0) return;
//        data[idx] += data[idx - (d >> 1)];
//    }
//
//    __global__ void kernChangeOneVal(int index, int* data, int val)
//    {
//        data[index] = val;
//    }
//
//    __global__ void kernDownSweep(int n, int* data, int d)
//    {
//        int idx = blockIdx.x * blockDim.x + threadIdx.x;
//        idx = idx * d - 1;
//        if (idx > n || idx < 0) return;
//        // Left child will become copy of parent
//        // Right child will be sum of left and parent
//        int left = idx - (d >> 1);
//        int t = data[left];
//        data[left] = data[idx];
//        data[idx] += t;
//    }
//
//    /**
//    * Performs stream compaction on idata, storing the result into odata.
//    * All zeroes are discarded.
//    *
//    * @param n      The number of elements in idata.
//    * @param odata  The array into which to store elements.
//    * @param idata  The array of elements to compact.
//    * @returns      The number of elements remaining after compaction.
//    */
//    int compact(int n, PathSegment* odata, const PathSegment* idata, std::function<bool(const PathSegment&)> predicate) {
//        int reqSize = ilog2ceil(n);
//        int ceil = 1 << reqSize;
//
//        dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
//        dim3 scanBlocksPerGrid((ceil + blockSize - 1) / blockSize);
//
//        PathSegment* dev_iArray;            // input array buffer
//        int* dev_boolArray;         // boolean array buffer
//        int* dev_boolScan;           // scan of the boolean array buffer
//        int* temp_Array;
//        PathSegment* dev_outArray;          // the output array buffer
//        cudaMalloc((void**)&dev_boolArray, sizeof(int) * n);
//        cudaMalloc((void**)&dev_iArray, sizeof(PathSegment) * n);
//        cudaMalloc((void**)&dev_boolScan, sizeof(int) * ceil);
//        cudaMalloc((void**)&dev_outArray, sizeof(PathSegment) * n);
//
//        cudaMemcpy(dev_iArray, idata, sizeof(PathSegment) * n, cudaMemcpyHostToDevice);
//
//        // Populate bool array
//        kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_boolArray, dev_iArray, predicate);
//
//        // Scan on bool array
//        cudaMemcpy(dev_boolScan, dev_boolArray, sizeof(int) * n, cudaMemcpyDeviceToDevice);
//        for (int d = 1; d < reqSize + 1; ++d)
//        {
//            scanBlocksPerGrid = dim3(((ceil >> (d - 1)) + blockSize - 1) / blockSize);
//            kernUpSweep << <scanBlocksPerGrid, blockSize >> > (ceil - 1, dev_boolScan, 1 << d);
//        }
//        kernChangeOneVal << <1, 1 >> > (ceil - 1, dev_boolScan, 0);
//        for (int d = reqSize; d > 0; --d)
//        {
//            scanBlocksPerGrid = dim3(((ceil >> (d - 1)) + blockSize - 1) / blockSize);
//            kernDownSweep << <scanBlocksPerGrid, blockSize >> > (ceil - 1, dev_boolScan, 1 << d);
//        }
//
//        // the resulting scan value was at a different location depending on padding or not, so the extra arithmetic here adjusts for that
//        cudaMemcpy(temp_Array, dev_boolScan, sizeof(int) * (n + (n % 2)), cudaMemcpyDeviceToHost);
//        int size = temp_Array[n - ((n + 1) % 2)];
//
//        // Compact
//        kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_outArray, dev_iArray, dev_boolArray, dev_boolScan);
//
//        cudaMemcpy(odata, dev_outArray, sizeof(PathSegment) * n, cudaMemcpyDeviceToHost);
//
//        cudaFree(dev_boolArray);
//        cudaFree(dev_iArray);
//        cudaFree(dev_boolScan);
//        cudaFree(dev_outArray);
//        return size;
//    }
//}