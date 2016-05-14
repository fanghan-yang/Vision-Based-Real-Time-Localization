#ifndef MULTIVIEWGEOMETRYUTILITY_H
#define MULTIVIEWGEOMETRYUTILITY_H
#include "MathUtility.h"
//#include "Classifier.h"
#include "DataUtility.h"
#include "epnp.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include "sba.h"
//#include "levmar.h"

using namespace std;
#define POINT_AT_INFINITY_ZERO 1e-2
#define PI 3.14159265

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierIndex);

#endif //MULTIVIEWGEOMETRYUTILITY_H
