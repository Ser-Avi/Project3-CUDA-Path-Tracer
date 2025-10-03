#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <iostream>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define MATERIAL_NUM 8
#define isBVH true
#define visBVH true

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialStartIndices = NULL;
static int* dev_materialEndIndices = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

/// <summary>
/// Struct for thrust to use for stream compaction after checking intersections
/// </summary>
struct is_continue
{
    __device__ __host__
        bool operator()(const PathSegment& seg)
    {
        return seg.remainingBounces > 0;
    }
};

struct CompareByKey {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialType > b.materialType;
    } 
};

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene, const std::string& envMapPath)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_materialStartIndices, sizeof(int) * MATERIAL_NUM);
    cudaMemset(dev_materialStartIndices, 0, sizeof(int) * MATERIAL_NUM);
    cudaMalloc(&dev_materialEndIndices, sizeof(int) * MATERIAL_NUM);
    cudaMemset(dev_materialEndIndices, 0, sizeof(int) * MATERIAL_NUM);

    if (scene->gltfs.size() > 0 && scene->gltfManager.getNumTriangles() < 1)
    {
        scene->loadFromGLTF();
    }
    if (envMapPath != "" && scene->curr_env_map.name != envMapPath)
    {
        printf("Trynna load envMap from pathtrace.cu\n");
        scene->loadEnvironmentMap(envMapPath);
    }
    checkCUDAError("pathtraceInit");
}

void pathtraceFree(bool camChange)
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_materialStartIndices);
    cudaFree(dev_materialEndIndices);
    cudaFree(dev_materials);
    if (hst_scene && !camChange)
    {
        hst_scene->gltfManager.cleanup();
        hst_scene->clearEnvironmentMap();
    }
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool isStochastic)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        thrust::default_random_engine rng = PBR::makeSeededRandomEngine(iter, index, index);

        // TODO: implement antialiasing by jittering the ray
        // creating random seed
        if (isStochastic)
        {
            thrust::uniform_real_distribution<float> jitter(-.5f, 0.5f);
            // offset
            float jitterX = jitter(rng);
            float jitterY = jitter(rng);

            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f));
        }
        else
        {
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
        }

        segment.ray.origin = cam.position;

        // depth of field
        if (cam.lensRadius > 0)
        {
            // sample
            thrust::uniform_real_distribution<float> u01(0, 1);
            glm::vec2 lensPoint = glm::vec2(u01(rng), u01(rng));
            // point on plane of focus
            lensPoint = cam.lensRadius * Utils::SampleUniformDiskConcentric(lensPoint);
            float ft = cam.focalLength / glm::max(0.01f, glm::length(segment.ray.direction.z));
            glm::vec3 pFocus = segment.ray.origin + segment.ray.direction * ft;
            //update ray
            segment.ray.origin +=  cam.right* lensPoint.x + cam.up * lensPoint.y; //glm::vec3(lensPoint.x, lensPoint.y, 0.f);
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
        }


        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.color = glm::vec3(1.f);

    }
}

__global__ void kernDrawBVH(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    ShadeableIntersection* intersections, BVHNode* BVHs, float color)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > num_paths - 1) { return; }
    PathSegment* pathSegment = &pathSegments[idx];
    float t;
    BVHNode nodeStack[64]; // max recursion depth is 64 for now, should be enough for most
    int stackIdx = 0;
    nodeStack[stackIdx++] = BVHs[0];
    pathSegment->color = glm::vec3(0.f);        // set base color to black

    while (stackIdx > 0)
    {
        BVHNode node = nodeStack[--stackIdx];
        t = IntersectAABB_Dist(pathSegment->ray, node.aabbMin, node.aabbMax, -1.f);
        if (t < 1e29f) // intersected a bounding box
        {
            // we will simply add some white and continue for each bvh intersection
            pathSegment->color += glm::vec3(color);
            // If this is an internal node, push children
            if (node.triCount < 1)
            {
                // if stack has space, push children
                if (stackIdx + 2 < 64)
                {
                    nodeStack[stackIdx++] = BVHs[node.leftFirst];
                    nodeStack[stackIdx++] = BVHs[node.leftFirst + 1];
                }
            }
        }
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    Triangle* triangles, int num_triangles,
    int* triIndices, BVHNode* BVHs)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths)
    {
        PathSegment* pathSegment = &pathSegments[path_index];

        if (pathSegment->remainingBounces < 1) return;

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        bool triangle = false;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment->ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment->ray, tmp_intersect, tmp_normal, outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        // next we check for triangles
        glm::vec2 uv; // tag uv to negative so we can check if we hit a triangle or not
        glm::vec2 tmp_uv;
        int idx;

        t = FLT_MAX;
#if isBVH
        if (num_triangles > 0)
        {
            t = IntersectBVH_Naive(pathSegment->ray, 0, BVHs, triIndices, triangles, t, tmp_intersect, tmp_normal, tmp_uv, outside, idx);

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = idx;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                triangle = true;
            }
        }
#else
       for (int i = 0; i < num_triangles; ++i)
        {
            t = triangleIntersectionTest(triangles[i], pathSegment->ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                triangle = true;
            }
        }
#endif
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialType = NONE;
            //pathSegment->color = glm::vec3(0.f);
            pathSegment->remainingBounces = 1;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surfaceNormal = normal;
            // if we didn't hit triangle:
            if (!triangle)
            {
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
                intersections[path_index].materialType = geoms[hit_geom_index].material;
            }
            else
            {
                intersections[path_index].materialType = PBR_GLTF;
                intersections[path_index].materialId = triangles[hit_geom_index].material_id;
                intersections[path_index].uv = uv;
            }
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, bool isCompact, bool isMatSort, bool isStochastic, bool isBVHvis)
{
    //std::cout << "PT start" << std::endl;
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    int* hst_materialStartIndices = new (std::nothrow) int[MATERIAL_NUM];
    int* hst_materialEndIndices = new (std::nothrow) int[MATERIAL_NUM];

    Triangle* dev_triangles = hst_scene->gltfManager.getTrianglesDevice();
    Material* dev_PBRmaterials = hst_scene->gltfManager.getPBRMaterialsDevice();
    int* dev_triIndices = hst_scene->gltfManager.getTriIntDevice();
    BVHNode* dev_bvh = hst_scene->gltfManager.getBVHDevice();
    int triangle_num = hst_scene->gltfManager.getNumTriangles();
    //std::cout << triangle_num << std::endl;
    checkCUDAError("setup");
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaTextureObject_t envMap = hst_scene->curr_env_map.texture;
    checkCUDAError("Enviroment Map Loading");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, isStochastic);
    cudaDeviceSynchronize();
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    if (num_paths != pixelcount) {
        printf("Pointer arithmetic error!\n");
    }

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        checkCUDAError("start");
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel Error2: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        if (!isBVHvis)
        {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections,
                dev_triangles,
                triangle_num,
                dev_triIndices,
                dev_bvh
                );
            checkCUDAError("trace one bounce");
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
            }

            if (isMatSort)
            {
                //std::cout << "Depth: " << depth << std::endl;
                thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, CompareByKey());
                Utils::kernResetIntBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (MATERIAL_NUM, dev_materialStartIndices, -1);
                Utils::kernResetIntBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (MATERIAL_NUM, dev_materialEndIndices, -1);
                Utils::kernIdentifyStartEnd << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_materialStartIndices, dev_materialEndIndices);
                cudaMemcpy(hst_materialStartIndices, dev_materialStartIndices, sizeof(int) * MATERIAL_NUM, cudaMemcpyDeviceToHost);
                cudaMemcpy(hst_materialEndIndices, dev_materialEndIndices, sizeof(int) * MATERIAL_NUM, cudaMemcpyDeviceToHost);

                // we can cull NONE materials here by simply setting num_paths to their start index (since they come last)
                // however this trick only works if there are NONE materials, hence the conditional
                num_paths = hst_materialStartIndices[0] > 0 ? hst_materialStartIndices[0] : num_paths;

                for (int mat = 0; mat < MATERIAL_NUM; ++mat)
                {
                    int start = hst_materialStartIndices[mat];
                    int end = hst_materialEndIndices[mat];

                    //printf("Mat: %d, Start: %d, End: %d\n", mat, start, end);

                    if (start < 0 || end < start)
                    {
                        continue;
                    }
                    const int count = end - start + 1;

                    const dim3 numblocks = (count + blockSize1d - 1) / blockSize1d;

                    switch (static_cast<MaterialType>(mat))
                    {
                    case NONE:
                        PBR::kernShadeNosect << <numblocks, blockSize1d >> > (count, dev_paths + start, envMap);
                        checkCUDAError("none");
                        break;
                    case EMISSIVE:
                        PBR::kernShadeEmissive << <numblocks, blockSize1d >> > (count, dev_intersections + start, dev_paths + start, dev_materials);
                        checkCUDAError("emissive");
                        break;
                    case DIFFUSE:
                        PBR::kernShadeDiffuse << <numblocks, blockSize1d >> > (iter, count, dev_intersections + start, dev_paths + start, dev_materials);
                        checkCUDAError("diffuse");
                        break;
                    case SPECULAR_REFL:
                        PBR::kernShadeSpecularRefl << <numblocks, blockSize1d >> > (count, dev_intersections + start, dev_paths + start, dev_materials);
                        checkCUDAError("specular");
                        break;
                    case SPECULAR_TRANS:
                        PBR::kernShadeSpecularTrans << <numblocks, blockSize1d >> > (count, dev_intersections + start, dev_paths + start, dev_materials);
                        checkCUDAError("transmissive");
                        break;
                    case DIELECTRIC:
                        PBR::kernShadeDielectric << <numblocks, blockSize1d >> > (iter, count, dev_intersections + start, dev_paths + start, dev_materials);
                        checkCUDAError("dielectric");
                        break;
                    case PBR_MAT:
                        PBR::kernShadePBR << <numblocks, blockSize1d >> > (iter, count, dev_intersections + start, dev_paths + start, dev_materials);
                        checkCUDAError("PBR");
                        break;
                    case PBR_GLTF:
                        PBR::kernShadePBR << <numblocks, blockSize1d >> > (iter, count, dev_intersections + start, dev_paths + start, dev_PBRmaterials);
                        checkCUDAError("PBR gltf");
                        break;
                    default:
                        std::cout << "ERROR: no material found at loop kern launch" << std::endl;
                        PBR::kernShadeAll << <numblocks, blockSize1d >> > (
                            iter,
                            count,
                            dev_intersections + count,
                            dev_paths + count,
                            dev_materials, dev_PBRmaterials, envMap
                            );
                        break;
                    }
                }
            }
            else
            {
                PBR::kernShadeAll << <numblocksPathSegmentTracing, blockSize1d >> > (
                    iter,
                    num_paths,
                    dev_intersections,
                    dev_paths,
                    dev_materials, dev_PBRmaterials, envMap
                    );
            }

            if (isCompact)
            {
                PathSegment* mid = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_continue());
                num_paths = static_cast<int>(mid - dev_paths);
            }

            if (++depth > traceDepth - 1 || num_paths < 1) iterationComplete = true;
        }
        else
        {
            float col = glm::clamp(hst_scene->numBVHnodes / 1000000.f, 0.001f, 0.2f);
            kernDrawBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_intersections,
                dev_bvh,
                col);
            iterationComplete = true;
        }


        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
