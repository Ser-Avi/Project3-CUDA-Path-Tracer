#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

// The body of this function is adapted from this paper:
// https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
// backfaces are not culled atm
__host__ __device__ float triangleIntersectionTest(
    Triangle triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    glm::vec3 edge1 = triangle.v1 - triangle.v0;
    glm::vec3 edge2 = triangle.v2 - triangle.v0;

    glm::vec3 pvec = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, pvec);

    if (det > -EPSILON && det < EPSILON) return -1;
    
    float inv_det = 1.f / det;
    glm::vec3 tvec = r.origin - triangle.v0;
    float u = glm::dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f)
    {
        return -1;
    }

    glm::vec3 qvec = glm::cross(tvec, edge1);

    float v = glm::dot(r.direction, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
    {
        return -1;
    }

    float t = glm::dot(edge2, qvec) * inv_det;
    intersectionPoint = getPointOnRay(r, t);
    normal = glm::cross(edge1, edge2);
    outside = glm::dot(normal, r.direction) < EPSILON;

    float w = 1.0f - u - v;  // barycentric coordinate for vertex 0
    uv = w * triangle.uv0 + u * triangle.uv1 + v * triangle.uv2;
    uv = glm::fract(uv);    // enforced wrapping
    return t;
}

__device__ bool IntersectAABB(const Ray& ray, const glm::vec3 bmin, const glm::vec3 bmax, float temp_T)
{
    float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
    float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
    float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
    tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
    float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
    tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
    return tmax >= tmin && tmin < temp_T && tmax > 0;
}

// In order for this function to be non-recursive, I used Sebastial Lague's method, as seen in this
// youtube video: https://www.youtube.com/watch?v=C1H4zIiCOaI&t=932s
__device__ float IntersectBVH(Ray& ray, const uint32_t nodeIdx, BVHNode* bvhNode, int* triIdx, Triangle* tri, float temp_t,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside, int& idx)
{
    BVHNode nodeStack[16]; // max recursion depth is 16 for now.
    int stackIdx = 0;
    nodeStack[stackIdx++] = bvhNode[0];

    glm::vec3 tmp_p, tmp_nor;
    glm::vec2 tmp_uv;
    float min_t = FLT_MAX;

    while (stackIdx > 0)
    {
        BVHNode node = nodeStack[--stackIdx];

        if (IntersectAABB(ray, node.aabbMin, node.aabbMax, temp_t))
        {
            if (node.triCount > 0)  // i.e. is leaf
            {
                for (uint32_t i = node.leftFirst; i < node.leftFirst + node.triCount; ++i)
                {
                    temp_t = triangleIntersectionTest(tri[triIdx[node.leftFirst + i]], ray, tmp_p, tmp_nor, tmp_uv, outside);
                    if (temp_t < min_t && temp_t > 0.f)
                    {
                        intersectionPoint = tmp_p;
                        normal = tmp_nor;
                        uv = tmp_uv;
                        min_t = temp_t;
                        idx = i;
                    }
                }
            }
            else
            {
                nodeStack[stackIdx++] = bvhNode[node.leftFirst + 1];
                nodeStack[stackIdx++] = bvhNode[node.leftFirst + 0];
            }
        }
    }

    return min_t;
}
