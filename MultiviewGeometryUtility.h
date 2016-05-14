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

int EPNP_ExtrinsicCameraParamEstimation(CvMat *X, CvMat *x, CvMat *K, CvMat *P);
void BilinearCameraPoseEstimation(vector<Feature> vFeature, int initialFrame1, int initialFrame2, double ransacThreshold, int ransacMaxIter, CvMat *K, CvMat &P, CvMat &X, vector<int> &visibleStructureID);
void BilinearCameraPoseEstimation_OPENCV(vector<Feature> vFeature, int initialFrame1, int initialFrame2, double ransacThreshold, int ransacMaxIter, CvMat *K, CvMat &P, CvMat &X, vector<int> &visibleStructureID);
int BilinearCameraPoseEstimation_OPENCV(vector<Feature> vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat &P, CvMat &X, vector<int> &visibleStructureID);
void VisibleIntersection(vector<Feature> vFeature, int frame1, int frame2, CvMat &cx1, CvMat &cx2, vector<int> &visibleFeatureID);
int VisibleIntersection23(vector<Feature> vFeature, int frame1, CvMat *X, vector<int> visibleStructureID, CvMat &cx, CvMat &cX, vector<int> &visibleID);
int VisibleIntersectionXOR3(vector<Feature> vFeature, int frame1, int frame2, vector<int> visibleStructureID, CvMat &cx1, CvMat &cx2, vector<int> &visibleID);
void NonlinearTriangulation(CvMat *x1, CvMat *x2, CvMat *F, CvMat &xhat1, CvMat &xhat2);
void LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, CvMat &X);
int LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> featureID, CvMat &X, vector<int> &filteredFeatureID);
void GetExtrinsicParameterFromE(CvMat *E, CvMat *x1, CvMat *x2, CvMat &P);
void GetExtrinsicParameterFromE(CvMat *E, CvMat *x1, CvMat *x2, CvMat *P);
int DLT_ExtrinsicCameraParamEstimation(CvMat *X, CvMat *x, CvMat *K, CvMat *P);
int DLT_ExtrinsicCameraParamEstimationWRansac(CvMat *X, CvMat *x, CvMat *K, CvMat &P, double ransacThreshold, int ransacMaxIter);
int ExcludeOutliers(CvMat *cx1, CvMat *cx2, double ransacThreshold, double ransacMaxIter, vector<int> visibleID, CvMat &ex1, CvMat &ex2, vector<int> &eVisibleID);
int ExcludeOutliers(CvMat *cx1, CvMat *P1, CvMat *cx2, CvMat *P2, CvMat *K, double threshold, vector<int> visibleID, CvMat &ex1, CvMat &ex2, vector<int> &eVisibleID);
void GetParameterForSBA(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, CvMat *K, vector<int> visibleStructureID, vector<double> &dCameraParameter, vector<double> &dFeature2DParameter, vector<char> &dVMask);
void GetParameterForSBA(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask);
void SparseBundleAdjustment_MOT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, CvMat *K, vector<int> visibleStructureID);
void SparseBundleAdjustment_MOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, CvMat *K, vector<int> visibleStructureID);
void SparseBundleAdjustment_MOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID);
void SparseBundleAdjustment_KMOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID);
void Projection3Donto2D_MOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void Projection3Donto2D_MOT(int j, int i, double *rt, double *xij, void *adata);
//void GetCameraParameter(CvMat *P, CvMat *K, CvMat &R, CvMat &C);
//void GetCameraParameter(CvMat *P, CvMat *K, CvMat *R, CvMat *C);
void RetrieveParameterFromSBA(double *dCameraParameter, CvMat *K, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID);
void CreateCameraMatrix(CvMat *R, CvMat *C, CvMat *K, CvMat &P);
void CreateCameraMatrix(CvMat *R, CvMat *C, CvMat *K, CvMat *P);
void PrintSBAInfo(double *info);
int ExcludePointBehindCamera(CvMat *X, CvMat *P1, CvMat *P2, vector<int> featureID, vector<int> &excludedFeatureID, CvMat &cX);

void GetParameterForGBA(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask);
void RetrieveParameterFromGBA(double *dCameraParameter, CvMat *K, vector<Camera> &vCamera, vector<Theta> &vTheta, int nFrames);
void ProjectionThetaonto2D_MOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void GlobalBundleAdjustment(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames, int nBase, int nFeatures_static);
void ProjectionThetaonto2D_MOTSTR_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata);
void GetParameterForGBA(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, CvMat &visibilityMask);
void ProjectionThetaonto2D_MOTSTR_LEVMAR(double *p, double *hx, int m, int n, void *adata);
void GlobalBundleAdjustment_LEVMAR(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames, int nBase, int nFeatures_static);
void ProjectionThetaonto2D_MOTSTR_LEVMAR(int j, int i, double *rt, double *xyz, double &x, double &y, void *adata);
void ProjectionThetaonto2D_MOTSTR_LEVMAR(float *p, float *hx, int m, int n, void *adata_);
void RetrieveParameterFromSBA(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames);
void BilinearCameraPoseEstimation(vector<Feature> vFeature, int initialFrame1, int initialFrame2, double ransacThreshold, int ransacMaxIter, int max_nFrames, vector<Camera> vCamera, CvMat &P, CvMat &X, vector<int> &visibleStructureID);
void GetParameterForSBA_KRT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
							vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask);
void Projection3Donto2D_KMOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void RetrieveParameterFromSBA_KRT(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames);
int DLT_ExtrinsicCameraParamEstimationWRansac_KRT(CvMat *X, CvMat *x, CvMat *K, CvMat &P, double ransacThreshold, int ransacMaxIter);
int DLT_ExtrinsicCameraParamEstimation_KRT(CvMat *X, CvMat *x, CvMat *K, CvMat *P);

void GetParameterForSBA_KDRT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
							 vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask);
void SparseBundleAdjustment_KDMOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID);
void RetrieveParameterFromSBA_KDRT(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames);
void Projection3Donto2D_KDMOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata);
int ExcludePointAtInfinity(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K1, CvMat *K2, vector<int> featureID, vector<int> &excludedFeatureID, CvMat &cX);
int VisibleIntersection23_Simple(vector<Feature> &vFeature, int frame1, vector<int> visibleStructureID, vector<int> &visibleID);
void Projection3Donto2D_KDMOT(int j, int i, double *rt, double *xij, void *adata);
void SparseBundleAdjustment_KDMOT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID);
void VisibleIntersection_Simple(vector<Feature> vFeature, int frame1, int frame2, vector<int> &visibleFeatureID);

void ProjectionThetaonto2D_TEMPORAL(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void RetrieveParameterFromSBA_TEMPORAL(double *dCameraParameter, vector<Camera> &vCamera, vector<Theta> &vTheta);
void SparseBundleAdjustment_TEMPORAL(vector<Feature> vFeature, vector<Theta> &vTheta, vector<Camera> &vCamera);
void GetParameterForSBA_TEMPORAL(vector<Feature> vFeature, vector<Theta> vTheta, vector<Camera> vCamera, int max_nFrames,
								 vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask);
void ProjectionThetaonto2D_TEMPORAL_LEVMAR(double *p, double *hx, int m, int n, void *adata);
void SparseBundleAdjustment_TEMPORAL_LEVMAR(vector<Feature> vFeature, vector<Theta> &vTheta, vector<Camera> &vCamera);
void GetParameterForSBA_TEMPORAL_LEVMAR(vector<Feature> vFeature, vector<Theta> vTheta, vector<Camera> vCamera, int max_nFrames,
										vector<double> &cameraParameter, vector<double> &feature2DParameter);
int DLT_ExtrinsicCameraParamEstimation_KRT(CvMat *X, CvMat *x, CvMat *K, CvMat &P);
void EightPointAlgorithm(CvMat *x1_8, CvMat *x2_8, CvMat *F_8);
void ExcludePointHighReprojectionError(vector<Feature> vFeature, vector<CvMat *> cP, vector<int> vUsedFrame, vector<int> &visibleStrucrtureID, CvMat *X_tot);
bool ExcludePointHighReprojectionError_AddingFrame(vector<Feature> vFeature, vector<CvMat *> cP, vector<int> vUsedFrame
												   , vector<int> &visibleStrucrtureID, CvMat &X_tot
												   , vector<int> &visibleStrucrtureID_new, CvMat &X_tot_new);

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP(CvMat *X, CvMat *x, CvMat *K, CvMat &P, double ransacThreshold, int ransacMaxIter);

void PrintSBAInfo(double *info, int nVisiblePoints);
int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_face(CvMat *X, CvMat *x, CvMat *K, CvMat &P);

void VisibleIntersection_mem(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleFeatureID);
int BilinearCameraPoseEstimation_OPENCV_mem(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X, vector<int> &visibleStructureID);
void SetCvMatFromVectors(vector<vector<double> > x, CvMat *X);
void LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, CvMat *X);
int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter);
int VisibleIntersectionXOR3_mem(vector<Feature> &vFeature, int frame1, int frame2, vector<int> visibleStructureID, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID);
int LinearTriangulation_mem(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> featureID, vector<vector<double> > &X, vector<int> &filteredFeatureID);
int ExcludePointBehindCamera_mem(CvMat *X, CvMat *P1, CvMat *P2, vector<int> featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX);
int ExcludePointAtInfinity_mem(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K1, CvMat *K2, vector<int> featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX);
bool ExcludePointHighReprojectionError_AddingFrame_mem(vector<Feature> &vFeature, vector<CvMat *> cP, vector<int> vUsedFrame
												   , vector<int> &visibleStrucrtureID, CvMat *X_tot
												   , vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new);

void ExcludePointHighReprojectionError_mem(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, vector<int> &visibleStrucrtureID, CvMat *X_tot);
int VisibleIntersection23_mem(vector<Feature> vFeature, int frame1, CvMat *X, vector<int> visibleStructureID, vector<vector<double> > &cx, vector<vector<double> > &cX, vector<int> &visibleID);
int ExcludeOutliers_mem(CvMat *cx1, CvMat *P1, CvMat *cx2, CvMat *P2, CvMat *K, double threshold, vector<int> visibleID, vector<vector<double> > &ex1, vector<vector<double> > &ex2, vector<int> &eVisibleID);
void SparseBundleAdjustment_MOTSTR_mem(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID);
void RetrieveParameterFromSBA_mem(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames);
void Projection3Donto2D_MOTSTR_fast(int j, int i, double *rt, double *xyz, double *xij, void *adata);
int BilinearCameraPoseEstimation_OPENCV_mem_fast(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X);
void SparseBundleAdjustment_MOTSTR_mem_fast(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera);
int VisibleIntersection23_Simple_fast(vector<Feature> &vFeature, int frame1);
int VisibleIntersection23_mem_fast(vector<Feature> &vFeature, int frame1, CvMat *X, vector<vector<double> > &cx, vector<vector<double> > &cX, vector<int> &visibleID);
int VisibleIntersectionXOR3_mem_fast(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID);
int ExcludeOutliers_mem_fast(CvMat *cx1, CvMat *P1, CvMat *cx2, CvMat *P2, CvMat *K, double threshold, vector<int> visibleID, vector<vector<double> > &ex1, vector<vector<double> > &ex2, vector<int> &eVisibleID);
int LinearTriangulation_mem_fast(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> &featureID, vector<vector<double> > &X, vector<int> &filteredFeatureID);
int ExcludePointAtInfinity_mem_fast(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K1, CvMat *K2, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX);
bool ExcludePointHighReprojectionError_AddingFrame_mem_fast(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new);
int ExcludePointHighReprojectionError_mem_fast(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot);
int ExcludePointBehindCamera_mem_fast(CvMat *X, CvMat *P1, CvMat *P2, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX);
int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierIndex);
int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs_global(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierIndex);
void AbsoluteCameraPoseRefinement(CvMat *X, CvMat *x, CvMat *P, CvMat *K);
void Projection3Donto2D_MOT_fast(double *rt, double *hx, int m, int n, void *adata);
void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, double omega);
void Projection3Donto2D_MOTSTR_fast_Distortion(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void GetParameterForSBA_Distortion(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
								   vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask);
int ExcludePointHighReprojectionError_mem_fast_Distortion(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, CvMat *K);
bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
																	   , vector<int> &visibleStrucrtureID, CvMat *X_tot
																	   , vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
																	   double omega, CvMat *K);
int BilinearCameraPoseEstimation_OPENCV_OrientationRefinement(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, 
															  vector<Camera> vCamera, CvMat *M, CvMat *m, vector<int> &vVisibleID);
void ObjectiveOrientationRefinement(double *rt, double *hx, int m, int n, void *adata);
void OrientationRefinement(CvMat *R_1, CvMat *R_F, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat*> &vM, vector<CvMat*> &vm, vector<CvMat *> &vx1, vector<CvMat *> &vx2);
void OrientationRefinement1(CvMat *R_1, CvMat *R_F, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat*> &vM, vector<CvMat*> &vm, vector<CvMat *> &vx1, vector<CvMat *> &vx2, 
							vector<int> vFrame1_r, vector<int> vFrame2_r, vector<CvMat*> &vM_r, vector<CvMat*> &vm_r, vector<CvMat *> &vx1_r, vector<CvMat *> &vx2_r);
//void OrientationRefinement_sba(CvMat *R_1, CvMat *R_F, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat*> &vM, vector<CvMat*> &vm, vector<CvMat *> &vx1, vector<CvMat *> &vx2, 
//							   vector<int> vFrame1_r, vector<int> vFrame2_r, vector<CvMat*> &vM_r, vector<CvMat*> &vm_r, vector<CvMat *> &vx1_r, vector<CvMat *> &vx2_r);
void CameraCenterInterpolationWithDegree(CvMat *R_1, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat *> vM, vector<CvMat *> vm,
										 vector<int> vFrame1_r, vector<int> vFrame2_r, vector<CvMat *> vM_r, vector<CvMat *> vm_r, 
										 vector<CvMat *> vC_c, vector<CvMat *> vR_c, vector<int> vFrame_c, vector<CvMat *> &vC, vector<CvMat *> &vR, double weight);
void DistanceBetweenRayAndPoint(double p1, double p2, double p3, double v1, double v2, double v3, double x1, double x2, double x3,
								double &d, double &g1, double &g2, double &g3);
void MeanShift_Gaussian_Ray(double &y1, double &y2, double &y3, vector<CvMat *> vP, vector<CvMat *> vV, double bandwidth, vector<double> &vWeight);
//void DetectPOI(vector<CvMat *> vP, vector<CvMat *> vV, int nSegments, double range, double merging_threshold, vector<double> vBandwidth, vector<CvMat *> &vPOI,
//			   vector<CvMat *> &vU, vector<CvMat *> &vRadius);
void DetectPOI(vector<CvMat *> vP, vector<CvMat *> vV, int nSegments, double range, double merging_threshold, vector<double> vBandwidth, vector<CvMat *> &vPOI,
			   double epsilon_cov, int nSegments_cov, vector<CvMat *> &v_a_cov, vector<CvMat *> &v_b_cov, vector<CvMat *> &v_l_cov, vector<double> &vf);
void DistanceBetweenConeAndPoint(double p1, double p2, double p3, double v1, double v2, double v3, double x1, double x2, double x3,
								 double &d);
void MeanShift_Gaussian_Cone(double &y1, double &y2, double &y3, vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, vector<double> &vWeight);
void POICovariance(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vWeight, CvMat *U, CvMat *Radius);
bool POICovariance(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, CvMat *poi, double epsilon, int nSegments, vector<double> &v_a, vector<double> &v_b, vector<double> &v_l);
void DetectPOI(vector<CvMat *> vP, vector<CvMat *> vV, int nSegments, double range, double merging_threshold, vector<double> vBandwidth, vector<CvMat *> &vPOI,
			   double epsilon_cov, int nSegments_cov, vector<CvMat *> &v_a_cov, vector<CvMat *> &v_b_cov, vector<CvMat *> &v_l_cov, vector<double> &vf, 
			   vector<vector<CvMat *> > &vvMeanTrajectory);
void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera,
	double omega, double princ_x1, double princ_y1);
void Projection3Donto2D_MOTSTR_fast_Distortion_ObstacleDetection(int j, int i, double *rt, double *xyz, double *xij, void *adata);
bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
															double omega, double princ_x1, double princ_y1, CvMat *K);
int ExcludePointHighReprojectionError_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, double princ_x1, double princ_y1, CvMat *K);
int BilinearCameraPoseEstimation_OPENCV_mem_fast_AD(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X, vector<vector<int> > &vvPointIndex);
int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_AD(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlier);
int ExcludePointHighReprojectionError_mem_fast_Distortion_AD(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, CvMat *K, vector<vector<int> > &vvPointIndex);
bool LinearTriangulation(vector<CvMat *> vP, vector<double> vx, vector<double> vy, double &X, double &Y, double &Z);
void Projection3Donto2D_STR_fast_SO(double *rt, double *hx, int m, int n, void *adata);
void TriangulationRefinement(CvMat *K, double omega, vector<CvMat *> vP, vector<double> vx, vector<double> vy, double &X, double &Y, double &Z);
int ExcludePointHighReprojectionError_mem_fast_AD(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, CvMat *K, vector<vector<int> > &vvPointIndex);
void Projection3Donto2D_STR_fast_SO_NoDistortion(double *rt, double *hx, int m, int n, void *adata);
void TriangulationRefinement_NoDistortion(CvMat *K, vector<CvMat *> vP, vector<double> vx, vector<double> vy, double &X, double &Y, double &Z);
void PatchTriangulationRefinement(CvMat *K, vector<CvMat *> vC, vector<CvMat *> vR, 
								  vector<double> vx11, vector<double> vy11, 
								  vector<double> vx12, vector<double> vy12,
								  vector<double> vx21, vector<double> vy21,
								  vector<double> vx22, vector<double> vy22,
								  double X, double Y, double Z, 
								  CvMat *X11, CvMat *X12, CvMat *X21, CvMat *X22, CvMat *pi);
void Projection3Donto2D_Patch(double *rt, double *hx, int m, int n, void *adata);
void ObjectiveFunction_poi_trajectory(double *rt, double *hx, int m, int n, void *adata);
void POIRefinement(POI_Matches &poi, vector<int> vFrame_PV, vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, double lambda);
double EvaulateDensityFunction(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, CvMat *x);
void POIRefinement_Seg(POI_Matches &poi, vector<int> vFrame_PV, vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, double lambda);
double EvaulateDensityFunction(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, CvMat *x, vector<double> &vWeight);
void OptimizePOI(vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth,
				 vector<double> &vx, vector<double> &vy, vector<double> &vz, double lambda);
void POIRefinement_Seg_KillFrame(POI_Matches &poi, vector<int> vFrame_PV, vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, double lambda);
void OptimizePOI1(vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth,
				 vector<double> &vx, vector<double> &vy, vector<double> &vz, double lambda);
void ObjectiveFunction_poi_trajectory1(double *rt, double *hx, int m, int n, void *adata);
void TriangulationRefinement_NoDistortion(vector<vector<CvMat *> > vvP, vector<vector<double> > vvx, vector<vector<double> > vvy, vector<double> &vX, vector<double> &vY, vector<double> &vZ);
void Projection3Donto2D_STR_fast_SO_NoDistortion_All(double *rt, double *hx, int m, int n, void *adata);
void CfMRefinement(Camera &virtualCamera, vector<Camera> &vCamera, vector<CvMat *> &vT);
void Projection3Donto2D_CfM(double *rt, double *hx, int m, int n, void *adata);
void Projection3Donto2D_MOTSTR_fast_Distortion_GoPro(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_GoPro(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera,
	vector<CvMat *> vK, vector<double> vOmega, vector<double> vpx, vector<double> vpy);
int ExcludePointHighReprojectionError_mem_fast_GoPro(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot,
	vector<double> vOmega, vector<CvMat *> vK, vector<double> vpx1, vector<double> vpy1);
void GetParameterForSBA_Distortion(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask,
						vector<CvMat *> vK);
void Projection3Donto2D_MOTSTR_fast_Dome(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void RetrieveParameterFromSBA_mem_Dome(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames);
void SparseBundleAdjustment_MOTSTR_mem_fast_Dome(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera);
void GetParameterForSBA_Dome(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask);
int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_Dome(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlier_dome);
void AbsoluteCameraPoseRefinement_Dome(CvMat *X, CvMat *x, CvMat *P, CvMat *K);
void Projection3Donto2D_MOT_fast_Dome(double *rt, double *hx, int m, int n, void *adata);

int BilinearCameraPoseEstimation_OPENCV_mem_fast_Dome(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X);
int ExcludePointHighReprojectionError_mem_fast_Dome(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot);
void Projection3Donto2D_STR_Ray_ICCV(double *rt, double *hx, int m, int n, void *adata);
void ObjectiveFunction_poi_trajectory1_weight(double *rt, double *hx, int m, int n, void *adata);
void OptimizePOI1_Weight(vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, vector<double> vWeight,
				 vector<int> vFrame, vector<double> &vx, vector<double> &vy, vector<double> &vz, double lambda);
void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera,
	vector<double> &vOmega, vector<double> &vprinc_x1, vector<double> &vprinc_y1, vector<CvMat *> &vK);
void Projection3Donto2D_MOTSTR_fast_Distortion_ObstacleDetection1(int j, int i, double *rt, double *xyz, double *xij, void *adata);
//void POI_TriangulationRefinement_NoDistortion(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vWeight, double &X, double &Y, double &Z);
bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
															vector<double> vOmega, vector<double> vPrinc_x1, vector<double> vPrinc_y1, vector<CvMat *> vK);
int ExcludePointHighReprojectionError_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, vector<double> vOmega, vector<double> vprinc_x1, vector<double> vprinc_y1, vector<CvMat *> vK);
void GetParameterForSBA_Distortion_Each(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask, vector<CvMat *> vK, 
						vector<double> vOmega, vector<double> vpx, vector<double> vpy);
void RetrieveParameterFromSBA_mem_Each(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames
	, vector<double> &vOmega, vector<double> &vpx, vector<double> &vpy, vector<CvMat *> &vK);
int VisibleIntersectionXOR3_mem_fast(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID,
		double omega1, double omega2, double px1, double px2, double py1, double py2);
void GetParameterForSBA_Distortion_Each_iPhone(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask, vector<CvMat *> vK, 
						vector<double> vk1);
void Projection3Donto2D_MOTSTR_fast_Distortion_iPhone(int j, int i, double *rt, double *xyz, double *xij, void *adata);
void RetrieveParameterFromSBA_mem_Each_iPhone(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames
	, vector<double> &vk1, vector<CvMat *> &vK);
void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_iPhone(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera,
	vector<double> &vk1, vector<CvMat *> &vK);
void Undistort_iPhone(double fx, double fy, double px, double py, double k1, double u, double v, double &u1, double &v1);
int VisibleIntersectionXOR3_mem_fast_iPhone(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID,
		CvMat *K1, CvMat *K2, double k11, double k12);
bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion_iPhone(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
															vector<double> vk1, vector<CvMat *> vK);
int ExcludePointHighReprojectionError_mem_fast_Distortion_iPhone(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, vector<double> vk1, vector<CvMat *> vK);
void Projection3Donto2D_CfM_corner(double *rt, double *hx, int m, int n, void *adata);
void CfMRefinement_corner(vector<CvMat *> &vH0i, vector<Camera> vCamera, vector<double> &vRepro_x, vector<double> &vReproj_y);
void Projection3Donto2D_CfM_ECCM(double *rt, double *hx, int m, int n, void *adata);
void CfMRefinement_ECCM_Each(CvMat *H0i, vector<double> vx1, vector<double> vy1, vector<double> vx2, vector<double> vy2);
void CfMRefinement_ECCM_Each(CvMat *H0i, vector<double> vx1, vector<double> vy1, vector<double> vx2, vector<double> vy2, double lambda);
void CfMRefinement_ECCM_Final(vector<CvMat *> vH_inv_r, vector<vector<double> > vVx1, vector<vector<double> > vVy1, vector<vector<double> > vVx2, vector<vector<double> > vVy2);
void Projection3Donto2D_CfM_ECCM_Final(double *rt, double *hx, int m, int n, void *adata);
void CfMRefinement_ECCM_Each(CvMat *H0i, vector<double> vx1, vector<double> vy1, vector<double> vx2, vector<double> vy2, vector<double> vx3, vector<double> vy3);
void Projection3Donto2D_CfM_ECCM1(double *rt, double *hx, int m, int n, void *adata);

void FaceRefinement(vector<CvMat *> &vX, vector<CvMat *> vP, vector<CvMat *> vx, vector<int> vIdx);
void Projection3Donto2D_FaceRefinement(double *rt, double *hx, int m, int n, void *adata);
#endif //MULTIVIEWGEOMETRYUTILITY_H
