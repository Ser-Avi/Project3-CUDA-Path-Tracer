#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <functional>

#include "sceneStructs.h"
#include "utilities.h"

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

