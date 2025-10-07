# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

&nbsp;Avi Serebrenik

&nbsp; \* [LinkedIn](https://www.linkedin.com/in/avi-serebrenik-a7386426b), [Personal Website](https://aviserebrenik.wixsite.com/cvsite)

\* Tested on: Windows 11, i7-13620H @ 2.4GHz 32GB, RTX 4070 8GB (Laptop)

## Overview

<p align="center">
  <img width="30%" alt="image" src="img/time.png" />
  <br>
  <em>"Time" rendered at 2000x1600 with 5000 iterations and a bounce depth of 12</em>
</p>

This Path Tracer was built with C++ and CUDA, with the code having the following top-level structure, where the first two and the last are done once per scene; 3, 4, and 9 are called once per iteration; and 4-8 are called once per bounce:
 1. Import scene information from a JSON file, which includes basic materials, their positions and transformation matrices, and the paths for any external gLTF meshes with attached transformation matrices.
 2. Convert this scene information into CUDA-readable formats and structs.
 3. Call any needed information in the main path tracing function.
 4. Start the path tracing loop by first calculating a ray from the camera through each pixel via a kernel.
 5. Using a computeIntersect kernel, check for the closest intersection from that ray and set the intersection information accordingly.
 6. If enabled, sort intersections and rays by material information.
 7. Calculate the color information and the next bounce direction of each ray in separate or one unified kernel.
 8. If enabled, use stream compaction to do away with rays that have terminated their bounces and repeat the path tracing loop if we have bounces and rays left.
 9. Post loop, use this loop's information to add to the information in the scene.
 10. Post tracing, clean up the memory, and save the final image to a file.

Beyond these features, I also added the following, which I will describe and analyze the results of below:
 * Supporting diffuse, specular, transmissive, dielectric, and PBR (using a GGX Microfacet model).
 * Stochastic Sampled anti-aliasing.
 * Supporting JSON primitives (spheres and boxes), and gLTF mesh imports with external (non-binary) textures using the tinygLTF library.
 * Depth of Field
 * .hdr environment maps
 * Bounding Volume Hierarchies using a Surface Area Heuristic for much faster runtime improvements and binned construction for much faster pre-process building.
 * Various ImGui features to toggle all of these above, display runtime information, and toggle channels for swapping between the scene, its albedo, its metal-roughness, its BVHs, its normals, or its depth.

## Methods

### Material Evaluation
<p align="center">
  <img width="30%" alt="image" src="img/matsBeauty.png" />
  <br>
  <em>"Cornell Showcase" showing off supported materials. Rendered at 800x800 with 5000 iterations and a bounce depth of 8.</em>
</p>
This one is based on my earlier path tracer in OpenGL, and uses much of the same code ported over to CUDA, so I will be focusing on their implementation over the theory.
In order to easily support the sorted and unsorted versions, each material has a kernel, but the kernels all just call their respective inline functions, so the one big unsorted kernel can share that too.
These kernels and their inline functions can be found in utils.cuh and utils.cu.
**Diffuse** uses cosine weighted hemisphere sampling to give us a random bounce direction and easy calculations.\
**Specular** simply gives us perfect refraction along the intersection's surface normal and doesn't add material color to the path.\
**Transmissive** took a while to fully port here, which turned out to be mainly the fault of a given sphere intersection code that I had to then modify, but relies on Snell's Law by using glm::refract. One of the indices of reflection is presumed to be air aka 1.0, and the other needs to be encoded in the material. This means I don't support transmission between two non-air materials. I made this choice because this would only be a small fraction of all cases and would need some extra information to be stored for each ray, slowing down everything else.\
**Dielectric** combines specular and transmissive materials. I decide which material to use via a random number generator and a probability of chosing specular over transmissive which is encoded in the material information. I then attenuate the results by 2 to account for this, and a Fresnel Dielectric Evaluation helper function.\
**PBR** materials use a GGX Microfacet model. This alone took up multiple times the work as the others combined, as simply "porting it over to CUDA" as I mentioned above didn't work... After getting all the encoded material information and checking for material maps, I first determine the incoming ray direction (wi) by using a similar probability-based split between purely diffuse materials and the rest.
