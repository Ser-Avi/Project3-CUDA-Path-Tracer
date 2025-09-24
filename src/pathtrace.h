#pragma once

#include "scene.h"
#include "utilities.h"
#include "utils.cuh"
#include "GLTFManager.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree(bool camChange);
void pathtrace(uchar4 *pbo, int frame, int iteration, bool isCompact, bool isMatSort, bool isStochastic);
