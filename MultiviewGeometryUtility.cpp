#include "MultiviewGeometryUtility.h"
#include <assert.h>
#include <sys/time.h>
using namespace std;

double measureTime1(timeval t1, timeval t2){
	time_t sec = t2.tv_sec-t1.tv_sec;
	suseconds_t usec = t2.tv_usec-t1.tv_usec;
	return (sec*1000000+usec)/1000000.0;
}

void AbsoluteCameraPoseRefinement_Jacobian(CvMat *X, CvMat *x, CvMat *P, CvMat *K, int nIters)
{
	//PrintAlgorithm("Absolute Camera Pose Refinement Jacobian");
	double f = cvGetReal2D(K, 0, 0);
	double px = cvGetReal2D(K, 0, 2);
	double py = cvGetReal2D(K, 1, 2);
	
	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	CvMat *t = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR = cvCreateMat(3,3,CV_32FC1);
	GetSubMatColwise(P, 0, 2, R);
	GetSubMatColwise(P, 3, 3, t);
	cvInvert(K, invK);
	cvMatMul(invK, R, R);
	cvInvert(R, invR);
	cvMatMul(invK, t, t);
	cvMatMul(invR, t, t);
	ScalarMul(t, -1, t);
	
	CvMat *uvw = cvCreateMat(3,1,CV_32FC1);
	CvMat *X_ = cvCreateMat(4,1,CV_32FC1);
	CvMat *fR = cvCreateMat(2, 9, CV_32FC1);
	CvMat *Rq = cvCreateMat(9, 4, CV_32FC1);
	
	CvMat *fq_i = cvCreateMat(2, 4, CV_32FC1);
	CvMat *fc = cvCreateMat(2*x->rows,3,CV_32FC1);
	CvMat *fq = cvCreateMat(2*x->rows, 4, CV_32FC1);
	CvMat *J = cvCreateMat(2*x->rows, 7, CV_32FC1);
	CvMat *df = cvCreateMat(2*x->rows, 1, CV_32FC1);
	CvMat *dx = cvCreateMat(7, 1, CV_32FC1);
	
	
	//double p_err = 1e+5;
	for (int iter = 0; iter < nIters; iter++)
	{
		Rotation2Quaternion(R, q);
		double r11 = cvGetReal2D(R, 0, 0);
		double r12 = cvGetReal2D(R, 0, 1);
		double r13 = cvGetReal2D(R, 0, 2);
		
		double r21 = cvGetReal2D(R, 1, 0);
		double r22 = cvGetReal2D(R, 1, 1);
		double r23 = cvGetReal2D(R, 1, 2);
		
		double r31 = cvGetReal2D(R, 2, 0);
		double r32 = cvGetReal2D(R, 2, 1);
		double r33 = cvGetReal2D(R, 2, 2);		
		
		double uc1 = -(f*r11+px*r31);
		double uc2 = -(f*r12+px*r32);
		double uc3 = -(f*r13+px*r33);
		
		double vc1 = -(f*r21+py*r31);
		double vc2 = -(f*r22+py*r32);
		double vc3 = -(f*r23+py*r33);
		
		double wc1 = -r31;
		double wc2 = -r32;
		double wc3 = -r33;		
		
		double qw = cvGetReal2D(q, 0, 0);
		double qx = cvGetReal2D(q, 1, 0);
		double qy = cvGetReal2D(q, 2, 0);
		double qz = cvGetReal2D(q, 3, 0);
		cvSetZero(Rq);
		cvSetReal2D(Rq, 0, 0, 0);		cvSetReal2D(Rq, 0, 1, -4*qy);	cvSetReal2D(Rq, 0, 2, -4*qz);	cvSetReal2D(Rq, 0, 3, 0);
		cvSetReal2D(Rq, 1, 0, 2*qy);	cvSetReal2D(Rq, 1, 1, 2*qx);	cvSetReal2D(Rq, 1, 2, -2*qw);	cvSetReal2D(Rq, 1, 3, -2*qz);
		cvSetReal2D(Rq, 2, 0, 2*qz);	cvSetReal2D(Rq, 2, 1, 2*qw);	cvSetReal2D(Rq, 2, 2, 2*qx);	cvSetReal2D(Rq, 2, 3, 2*qy);
		cvSetReal2D(Rq, 3, 0, 2*qy);	cvSetReal2D(Rq, 3, 1, 2*qx);	cvSetReal2D(Rq, 3, 2, 2*qw);	cvSetReal2D(Rq, 3, 3, 2*qz);
		cvSetReal2D(Rq, 4, 0, -4*qx);	cvSetReal2D(Rq, 4, 1, 0);		cvSetReal2D(Rq, 4, 2, -4*qz);	cvSetReal2D(Rq, 4, 3, 0);
		cvSetReal2D(Rq, 5, 0, -2*qw);	cvSetReal2D(Rq, 5, 1, 2*qz);	cvSetReal2D(Rq, 5, 2, 2*qy);	cvSetReal2D(Rq, 5, 3, -2*qx);
		cvSetReal2D(Rq, 6, 0, 2*qz);	cvSetReal2D(Rq, 6, 1, -2*qw);	cvSetReal2D(Rq, 6, 2, 2*qx);	cvSetReal2D(Rq, 6, 3, -2*qy);
		cvSetReal2D(Rq, 7, 0, 2*qw);	cvSetReal2D(Rq, 7, 1, 2*qz);	cvSetReal2D(Rq, 7, 2, 2*qy);	cvSetReal2D(Rq, 7, 3, 2*qx);
		cvSetReal2D(Rq, 8, 0, -4*qx);	cvSetReal2D(Rq, 8, 1, -4*qy);	cvSetReal2D(Rq, 8, 2, 0);		cvSetReal2D(Rq, 8, 3, 0);
		
		double err = 0;
		for (int i = 0; i < x->rows; i++)
		{
			cvSetReal2D(X_, 0, 0, cvGetReal2D(X, i, 0));
			cvSetReal2D(X_, 1, 0, cvGetReal2D(X, i, 1));
			cvSetReal2D(X_, 2, 0, cvGetReal2D(X, i, 2));
			cvSetReal2D(X_, 3, 0, 1);
			
			cvMatMul(P, X_, uvw);
			double u = cvGetReal2D(uvw, 0, 0);
			double v = cvGetReal2D(uvw, 1, 0);
			double w = cvGetReal2D(uvw, 2, 0);
			
			//PrintMat(uvw, "uvw");
			double w_2 = w*w;
			cvSetReal2D(fc, i*2, 0, (w*uc1-u*wc1)/w_2);
			cvSetReal2D(fc, i*2, 1, (w*uc2-u*wc2)/w_2);
			cvSetReal2D(fc, i*2, 2, (w*uc3-u*wc3)/w_2);
			
			cvSetReal2D(fc, i*2+1, 0, (w*vc1-v*wc1)/w_2);
			cvSetReal2D(fc, i*2+1, 1, (w*vc2-v*wc2)/w_2);
			cvSetReal2D(fc, i*2+1, 2, (w*vc3-v*wc3)/w_2);
			
			//PrintMat(fc, "fc");
			
			double dX1 = cvGetReal2D(X, i, 0)-cvGetReal2D(t, 0, 0);
			double dX2 = cvGetReal2D(X, i, 1)-cvGetReal2D(t, 1, 0);
			double dX3 = cvGetReal2D(X, i, 2)-cvGetReal2D(t, 2, 0);
			
			double uR11 = f*dX1;	double uR12 = f*dX2;	double uR13 = f*dX3;
			double uR21 = 0;		double uR22 = 0; 		double uR23 = 0;
			double uR31 = px*dX1;	double uR32 = px*dX2;	double uR33 = px*dX3;
			
			double vR11 = 0;		double vR12 = 0; 		double vR13 = 0;
			double vR21 = f*dX1;	double vR22 = f*dX2;	double vR23 = f*dX3;
			double vR31 = py*dX1;	double vR32 = py*dX2;	double vR33 = py*dX3;
			
			double wR11 = 0; 		double wR12 = 0;		double wR13 = 0;
			double wR21 = 0;		double wR22 = 0;		double wR23 = 0;
			double wR31 = dX1;		double wR32 = dX2;		double wR33 = dX3;
			
			cvSetReal2D(fR, 0, 0, (w*uR11-u*wR11)/w_2);		cvSetReal2D(fR, 0, 1, (w*uR12-u*wR12)/w_2);		cvSetReal2D(fR, 0, 2, (w*uR13-u*wR13)/w_2);
			cvSetReal2D(fR, 0, 3, (w*uR21-u*wR21)/w_2);		cvSetReal2D(fR, 0, 4, (w*uR22-u*wR22)/w_2);		cvSetReal2D(fR, 0, 5, (w*uR23-u*wR23)/w_2);
			cvSetReal2D(fR, 0, 6, (w*uR31-u*wR31)/w_2);		cvSetReal2D(fR, 0, 7, (w*uR32-u*wR32)/w_2);		cvSetReal2D(fR, 0, 8, (w*uR33-u*wR33)/w_2);
			
			cvSetReal2D(fR, 1, 0, (w*vR11-v*wR11)/w_2);		cvSetReal2D(fR, 1, 1, (w*vR12-v*wR12)/w_2);		cvSetReal2D(fR, 1, 2, (w*vR13-v*wR13)/w_2);
			cvSetReal2D(fR, 1, 3, (w*vR21-v*wR21)/w_2);		cvSetReal2D(fR, 1, 4, (w*vR22-v*wR22)/w_2);		cvSetReal2D(fR, 1, 5, (w*vR23-v*wR23)/w_2);
			cvSetReal2D(fR, 1, 6, (w*vR31-v*wR31)/w_2);		cvSetReal2D(fR, 1, 7, (w*vR32-v*wR32)/w_2);		cvSetReal2D(fR, 1, 8, (w*vR33-v*wR33)/w_2);

			
			cvMatMul(fR, Rq, fq_i);
			SetSubMat(fq, 2*i, 0, fq_i);	

			cvSetReal2D(df, 2*i, 0, cvGetReal2D(x, i, 0) - u/w);
			cvSetReal2D(df, 2*i+1, 0, cvGetReal2D(x, i, 1) - v/w);
			
			err += (cvGetReal2D(x, i, 0) - u/w)*(cvGetReal2D(x, i, 0) - u/w)+(cvGetReal2D(x, i, 1) - v/w)*(cvGetReal2D(x, i, 1) - v/w);
		}

		cvSetZero(J);
		SetSubMat(J, 0, 0, fc);
		SetSubMat(J, 0, 3, fq);		
		
		cvSolve(J, df, dx);

		cvSetReal2D(t, 0, 0, cvGetReal2D(t, 0, 0)+0.03*cvGetReal2D(dx, 0, 0));
		cvSetReal2D(t, 1, 0, cvGetReal2D(t, 1, 0)+0.03*cvGetReal2D(dx, 1, 0));
		cvSetReal2D(t, 2, 0, cvGetReal2D(t, 2, 0)+0.03*cvGetReal2D(dx, 2, 0));
		
		cvSetReal2D(q, 0, 0, cvGetReal2D(q, 0, 0)+0.03*cvGetReal2D(dx, 6, 0));
		cvSetReal2D(q, 1, 0, cvGetReal2D(q, 1, 0)+0.03*cvGetReal2D(dx, 3, 0));
		cvSetReal2D(q, 2, 0, cvGetReal2D(q, 2, 0)+0.03*cvGetReal2D(dx, 4, 0));
		cvSetReal2D(q, 3, 0, cvGetReal2D(q, 3, 0)+0.03*cvGetReal2D(dx, 5, 0));
				
		Quaternion2Rotation1(q, R);
		
		GetCameraMatrix(K, R, t, P);
	}
	cvReleaseMat(&dx);
	cvReleaseMat(&df);
	cvReleaseMat(&J);
	cvReleaseMat(&fq);
	cvReleaseMat(&fc);
	cvReleaseMat(&fq_i);
	cvReleaseMat(&uvw);
	cvReleaseMat(&X_);
	cvReleaseMat(&fR);
	cvReleaseMat(&Rq);

	cvReleaseMat(&R);
	cvReleaseMat(&t);
	cvReleaseMat(&q);
	cvReleaseMat(&invK);
	cvReleaseMat(&invR);
}

//void BilinearCameraPoseEstimation(vector<Feature> vFeature, int initialFrame1, int initialFrame2, double ransacThreshold, int ransacMaxIter, CvMat *K, CvMat &P, CvMat &X, vector<int> &visibleStructureID)
//{
//	PrintAlgorithm("Bilinear Camera Pose Estimation");
//	CvMat cx1, cx2, nx1, nx2;
//	vector<int> visibleFeatureID;
//	X = *cvCreateMat(vFeature.size(), 3, CV_32FC1);
//	VisibleIntersection(vFeature, initialFrame1, initialFrame2, cx1, cx2, visibleFeatureID);
//	assert(visibleFeatureID.size() > 7);
//	CvMat *F = cvCreateMat(3,3,CV_32FC1);
//
//	Classifier classifier;
//	vector<int> visibleID;
//	classifier.SetRansacParam(ransacThreshold, ransacMaxIter);
//	classifier.SetCorrespondance(&cx1, &cx2, visibleFeatureID);
//	classifier.Classify();
//	vector<int> vInlierID, vOutlierID;
//	classifier.GetClassificationResultByFeatureID(vInlierID, vOutlierID);
//	visibleFeatureID = vInlierID;
//	F = cvCloneMat(classifier.F);
//	cx1 = *cvCreateMat(classifier.inlier1->rows, classifier.inlier1->cols, CV_32FC1);
//	cx2 = *cvCreateMat(classifier.inlier2->rows, classifier.inlier2->cols, CV_32FC1);
//	cx1 = *cvCloneMat(classifier.inlier1);
//	cx2 = *cvCloneMat(classifier.inlier2);
//
//	//vector<int> vInlierID;
//	//CvMat *status = cvCreateMat(1,cx1.rows,CV_8UC1);
//	//int n = cvFindFundamentalMat(&cx1, &cx2, F, CV_FM_LMEDS , 1, 0.99, status);
//	//for (int i = 0; i < cx1.rows; i++)
//	//{
//	//	if (cvGetReal2D(status, 0, i) == 1)
//	//	{
//	//		vInlierID.push_back(visibleFeatureID[i]);
//	//	}
//	//}
//	//visibleFeatureID = vInlierID;
//	//CvMat *tempCx1 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
//	//CvMat *tempCx2 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
//	//int temprow = 0;
//	//for (int i = 0; i < cx1.rows; i++)
//	//{
//	//	if (cvGetReal2D(status, 0, i) == 1)
//	//	{
//	//		cvSetReal2D(tempCx1, temprow, 0, cvGetReal2D(&cx1, i, 0));
//	//		cvSetReal2D(tempCx1, temprow, 1, cvGetReal2D(&cx1, i, 1));
//	//		cvSetReal2D(tempCx2, temprow, 0, cvGetReal2D(&cx2, i, 0));
//	//		cvSetReal2D(tempCx2, temprow, 1, cvGetReal2D(&cx2, i, 1));
//	//		temprow++;
//	//	}
//	//}
//	//cx1 = *cvCreateMat(tempCx1->rows, tempCx1->cols, CV_32FC1);
//	//cx2 = *cvCreateMat(tempCx2->rows, tempCx2->cols, CV_32FC1);
//	//cx1 = *cvCloneMat(tempCx1);
//	//cx2 = *cvCloneMat(tempCx2);
//	//cvReleaseMat(&status);
//	//cvReleaseMat(&tempCx1);
//	//cvReleaseMat(&tempCx2);
//
//	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
//	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
//	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
//
//	cvTranspose(K, temp33);
//	cvMatMul(temp33, F, temp33);
//	cvMatMul(temp33, K, E);
//
//	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
//	cvInvert(K, invK);
//	Pxx_inhomo(invK, &cx1, nx1);
//	Pxx_inhomo(invK, &cx2, nx2);
//
//	GetExtrinsicParameterFromE(E, &nx1, &nx2, P);
//	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
//	cvSetIdentity(P0);
//	CvMat cX;
//
//	LinearTriangulation(&nx1, P0, &nx2, &P, cX);
//	cvSetZero(&X);
//	SetIndexedMatRowwise(&X, visibleFeatureID, &cX);
//	cvMatMul(K, &P, temp34);
//	P = *cvCloneMat(temp34);
//
//	visibleStructureID = visibleFeatureID;
//
//	cvReleaseMat(&F);
//	cvReleaseMat(&E);
//	cvReleaseMat(&temp33);
//	cvReleaseMat(&temp34);
//	cvReleaseMat(&invK);
//	cvReleaseMat(&P0);	
//}

void BilinearCameraPoseEstimation_OPENCV(vector<Feature> vFeature, int initialFrame1, int initialFrame2, double ransacThreshold, int ransacMaxIter, CvMat *K, CvMat &P, CvMat &X, vector<int> &visibleStructureID)
{
	PrintAlgorithm("Bilinear Camera Pose Estimation");
	CvMat cx1, cx2, nx1, nx2;
	vector<int> visibleFeatureID;
	X = *cvCreateMat(vFeature.size(), 3, CV_32FC1);
	VisibleIntersection(vFeature, initialFrame1, initialFrame2, cx1, cx2, visibleFeatureID);
	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);

	//Classifier classifier;
	//vector<int> visibleID;
	//classifier.SetRansacParam(ransacThreshold, ransacMaxIter);
	//classifier.SetCorrespondance(&cx1, &cx2, visibleFeatureID);
	//classifier.Classify();
	//vector<int> vInlierID, vOutlierID;
	//classifier.GetClassificationResultByFeatureID(vInlierID, vOutlierID);
	//visibleFeatureID = vInlierID;
	//F = cvCloneMat(classifier.F);
	//cx1 = *cvCreateMat(classifier.inlier1->rows, classifier.inlier1->cols, CV_32FC1);
	//cx2 = *cvCreateMat(classifier.inlier2->rows, classifier.inlier2->cols, CV_32FC1);
	//cx1 = *cvCloneMat(classifier.inlier1);
	//cx2 = *cvCloneMat(classifier.inlier2);

	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1,cx1.rows,CV_8UC1);
	int n = cvFindFundamentalMat(&cx1, &cx2, F, CV_FM_RANSAC , 1, 0.99, status);
	for (int i = 0; i < cx1.rows; i++)
	{
		if (cvGetReal2D(status, 0, i) == 1)
		{
			vInlierID.push_back(visibleFeatureID[i]);
		}
	}
	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
	int temprow = 0;
	for (int i = 0; i < cx1.rows; i++)
	{
		if (cvGetReal2D(status, 0, i) == 1)
		{
			cvSetReal2D(tempCx1, temprow, 0, cvGetReal2D(&cx1, i, 0));
			cvSetReal2D(tempCx1, temprow, 1, cvGetReal2D(&cx1, i, 1));
			cvSetReal2D(tempCx2, temprow, 0, cvGetReal2D(&cx2, i, 0));
			cvSetReal2D(tempCx2, temprow, 1, cvGetReal2D(&cx2, i, 1));
			temprow++;
		}
	}
	cx1 = *cvCreateMat(tempCx1->rows, tempCx1->cols, CV_32FC1);
	cx2 = *cvCreateMat(tempCx2->rows, tempCx2->cols, CV_32FC1);
	cx1 = *cvCloneMat(tempCx1);
	cx2 = *cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);

	cvTranspose(K, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K, E);

	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K, invK);
	Pxx_inhomo(invK, &cx1, nx1);
	Pxx_inhomo(invK, &cx2, nx2);

	GetExtrinsicParameterFromE(E, &nx1, &nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	CvMat cX;

	LinearTriangulation(&nx1, P0, &nx2, &P, cX);
	cvSetZero(&X);
	SetIndexedMatRowwise(&X, visibleFeatureID, &cX);
	cvMatMul(K, &P, temp34);
	P = *cvCloneMat(temp34);

	visibleStructureID = visibleFeatureID;

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&invK);
	cvReleaseMat(&P0);	
}
/*
void POIRefinement_Seg_KillFrame(POI_Matches &poi, vector<int> vFrame_PV, vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, double lambda)
{
	int stride = 5;
	int window = 20;
	PrintAlgorithm("POI Refinement");	
	vector<double> vx_t, vy_t, vz_t;
	int poi_idx=0;
	vector<int> vFrame_t;
	for (int iFrame = poi.vFrame[0]; iFrame <= poi.vFrame[poi.vFrame.size()-1]; iFrame++)
	{
		vector<int>::const_iterator it = find(poi.vFrame.begin(), poi.vFrame.end(), iFrame);

		CvMat *x = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(x, 0, 0, poi.vx[poi_idx]);
		cvSetReal2D(x, 1, 0, poi.vy[poi_idx]);
		cvSetReal2D(x, 2, 0, poi.vz[poi_idx]);

		vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
		int idx = (int)(it1-vFrame_PV.begin());

		vector<double> vWeight;
		double obj = EvaulateDensityFunction(vvP[idx], vvV[idx], vvBandwidth[idx], x, vWeight);
		cvReleaseMat(&x);

		sort(vWeight.begin(), vWeight.end());

		if ((it!=poi.vFrame.end()) && (vWeight[vWeight.size()-2]/vWeight[vWeight.size()-1] > 0.01))
		{
			vx_t.push_back(poi.vx[poi_idx]);
			vy_t.push_back(poi.vy[poi_idx]);
			vz_t.push_back(poi.vz[poi_idx]);
			poi_idx++;	
			vFrame_t.push_back(iFrame);
			//cout << obj << endl;
		}
		else if ((obj > 0.5) && (vWeight[vWeight.size()-2]/vWeight[vWeight.size()-1] > 0.01) && (iFrame-vFrame_t[vFrame_t.size()-1] < 20))
		{
			bool isOK = false;
			int start_i;
			for (int i = 0; i < poi.vFrame.size()-1; i++)
			{
				if ((poi.vFrame[i] < iFrame) && (iFrame < poi.vFrame[i+1]))
				{
					if (poi.vFrame[i+1]-poi.vFrame[i] <= 20)
					{
						isOK = true;
						start_i = i;
					}
					break;
				}
			}

			if (isOK)
			{
				double x = (poi.vx[start_i+1]-poi.vx[start_i])/(poi.vFrame[start_i+1]-poi.vFrame[start_i])*(iFrame-poi.vFrame[start_i]) + poi.vx[start_i];
				double y = (poi.vy[start_i+1]-poi.vy[start_i])/(poi.vFrame[start_i+1]-poi.vFrame[start_i])*(iFrame-poi.vFrame[start_i]) + poi.vy[start_i];
				double z = (poi.vz[start_i+1]-poi.vz[start_i])/(poi.vFrame[start_i+1]-poi.vFrame[start_i])*(iFrame-poi.vFrame[start_i]) + poi.vz[start_i];

				vx_t.push_back(x);
				vy_t.push_back(y);
				vz_t.push_back(z);
				//poi_idx++;
				vFrame_t.push_back(iFrame);
			}			
		}
	}


	vector<double> vx1, vy1, vz1;
	vector<int> vFrame1;
	vx1.push_back(vx_t[0]);
	vy1.push_back(vy_t[0]);
	vz1.push_back(vz_t[0]);
	vFrame1.push_back(vFrame_t[0]);
	for (int ix = 1; ix < vx_t.size()-1; ix++)
	{
		if ((vFrame_t[ix]-1 == vFrame_t[ix-1]) || (vFrame_t[ix]+1 == vFrame_t[ix+1]))
		{
			vx1.push_back(vx_t[ix]);
			vy1.push_back(vy_t[ix]);
			vz1.push_back(vz_t[ix]);
			vFrame1.push_back(vFrame_t[ix]);
		}
	}
	if (vFrame_t[vFrame_t.size()-1]-1 == vFrame_t[vFrame_t.size()-2])
	{
		vx1.push_back(vx_t[vFrame_t.size()-1]);
		vy1.push_back(vy_t[vFrame_t.size()-1]);
		vz1.push_back(vz_t[vFrame_t.size()-1]);
		vFrame1.push_back(vFrame_t[vFrame_t.size()-1]);
	}

	vx_t = vx1;	vy_t = vy1;	vz_t = vz1;	vFrame_t = vFrame1;

	vector<vector<int> > vvFrame_seg;
	vector<vector<double> > vvx_seg, vvy_seg, vvz_seg;

	int idx_t = 0;
	while (vFrame_t.size() != idx_t)
	{
		vector<int> vFrame_seg1;
		vector<double> vx_seg1, vy_seg1, vz_seg1;
		
		while (vFrame_t[idx_t]+1 == vFrame_t[idx_t+1])
		{
			vFrame_seg1.push_back(vFrame_t[idx_t]);
			vx_seg1.push_back(vx_t[idx_t]);
			vy_seg1.push_back(vy_t[idx_t]);
			vz_seg1.push_back(vz_t[idx_t]);
			idx_t++;
			if (vFrame_t.size() == idx_t)
				break;
		}

		vFrame_seg1.push_back(vFrame_t[idx_t]);
		vx_seg1.push_back(vx_t[idx_t]);
		vy_seg1.push_back(vy_t[idx_t]);
		vz_seg1.push_back(vz_t[idx_t]);

		//if (vFrame_seg1.size() < 3)
		//{
		//	vFrame_seg1.push_back(vFrame_t[idx_t]+1);
		//	vx_seg1.push_back(vx_t[idx_t]);
		//	vy_seg1.push_back(vy_t[idx_t]);
		//	vz_seg1.push_back(vz_t[idx_t]);
		//}

		idx_t++;

		if (vFrame_seg1.size() < 5)
			continue;
		vvFrame_seg.push_back(vFrame_seg1);
		vvx_seg.push_back(vx_seg1);
		vvy_seg.push_back(vy_seg1);
		vvz_seg.push_back(vz_seg1);
	}

	

	for (int iSeg = 0; iSeg < vvFrame_seg.size(); iSeg++)
	{
		bool isWhole = false;
		if (vvFrame_seg[iSeg].size()/2 < window)
		{
			vector<vector<CvMat *> > vvP_seg1, vvV_seg1;
			vector<vector<double> > vvBandwidth_seg1;
			vector<int> vFrame_seg1;
			vector<double> vx, vy, vz;
			cout << vvFrame_seg[iSeg][0] << " " << vvFrame_seg[iSeg][vvFrame_seg[iSeg].size()-1] << endl;
			for (int iFrame = vvFrame_seg[iSeg][0]; iFrame <= vvFrame_seg[iSeg][vvFrame_seg[iSeg].size()-1]; iFrame++)
			{
				vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
				int idx = (int) (it1 - vFrame_PV.begin());
				vvP_seg1.push_back(vvP[idx]);
				vvV_seg1.push_back(vvV[idx]);
				vvBandwidth_seg1.push_back(vvBandwidth[idx]);
				vFrame_seg1.push_back(iFrame);

				vector<int>::const_iterator it2 = find(vvFrame_seg[iSeg].begin(), vvFrame_seg[iSeg].end(), iFrame);
				int idx1 = (int) (it2 - vvFrame_seg[iSeg].begin());

				vx.push_back(vvx_seg[iSeg][idx1]);
				vy.push_back(vvy_seg[iSeg][idx1]);
				vz.push_back(vvz_seg[iSeg][idx1]);	
				//cout << iFrame <<" ";
			}

			//cout << endl;
			if (vx.size() > 6)
				OptimizePOI(vvP_seg1, vvV_seg1, vvBandwidth_seg1, vx, vy, vz, lambda);
			else
				OptimizePOI1(vvP_seg1, vvV_seg1, vvBandwidth_seg1, vx, vy, vz, lambda);

			for (int ix = 0; ix < vx.size(); ix++)
			{
				vector<int>::const_iterator it = find(vvFrame_seg[iSeg].begin(), vvFrame_seg[iSeg].end(), vFrame_seg1[ix]);
				int idx1 = (int) (it - vvFrame_seg[iSeg].begin());
				vvx_seg[iSeg][idx1] = vx[ix];
				vvy_seg[iSeg][idx1] = vy[ix];
				vvz_seg[iSeg][idx1] = vz[ix];
			}

		}
		else
		{
			//cout << "window: " << window << endl;
			for (int iFrame1 = vvFrame_seg[iSeg][0]+window; iFrame1 <= vvFrame_seg[iSeg][vvFrame_seg[iSeg].size()-1]-window; iFrame1+=stride)
			{
				vector<vector<CvMat *> > vvP_seg1, vvV_seg1;
				vector<vector<double> > vvBandwidth_seg1;
				vector<int> vFrame_seg1;
				vector<double> vx, vy, vz;
				cout << vvFrame_seg[iSeg][0]+window << " " << vvFrame_seg[iSeg][vvFrame_seg[iSeg].size()-1]-window << " " <<  iFrame1 << endl;
				for (int iFrame = iFrame1-window; iFrame <= iFrame1+window; iFrame++)
				{
					vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
					int idx = (int) (it1 - vFrame_PV.begin());
					vvP_seg1.push_back(vvP[idx]);
					vvV_seg1.push_back(vvV[idx]);
					vvBandwidth_seg1.push_back(vvBandwidth[idx]);
					vFrame_seg1.push_back(iFrame);

					vector<int>::const_iterator it2 = find(vvFrame_seg[iSeg].begin(), vvFrame_seg[iSeg].end(), iFrame);
					int idx1 = (int) (it2 - vvFrame_seg[iSeg].begin());

					vx.push_back(vvx_seg[iSeg][idx1]);
					vy.push_back(vvy_seg[iSeg][idx1]);
					vz.push_back(vvz_seg[iSeg][idx1]);	
					//cout << iFrame <<" ";
				}

				cout << endl;
				OptimizePOI(vvP_seg1, vvV_seg1, vvBandwidth_seg1, vx, vy, vz, lambda);

				for (int ix = 0; ix < vx.size(); ix++)
				{
					vector<int>::const_iterator it = find(vvFrame_seg[iSeg].begin(), vvFrame_seg[iSeg].end(), vFrame_seg1[ix]);
					int idx1 = (int) (it - vvFrame_seg[iSeg].begin());
					vvx_seg[iSeg][idx1] = vx[ix];
					vvy_seg[iSeg][idx1] = vy[ix];
					vvz_seg[iSeg][idx1] = vz[ix];
				}
			}

			vector<vector<CvMat *> > vvP_seg1, vvV_seg1;
			vector<vector<double> > vvBandwidth_seg1;
			vector<int> vFrame_seg1;
			vector<double> vx, vy, vz;
			for (int iFrame = vvFrame_seg[iSeg][vvFrame_seg[iSeg].size()-1]-2*window; iFrame <= vvFrame_seg[iSeg][vvFrame_seg[iSeg].size()-1]; iFrame++)
			{
				vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
				int idx = (int) (it1 - vFrame_PV.begin());
				vvP_seg1.push_back(vvP[idx]);
				vvV_seg1.push_back(vvV[idx]);
				vvBandwidth_seg1.push_back(vvBandwidth[idx]);
				vFrame_seg1.push_back(iFrame);

				vector<int>::const_iterator it2 = find(vvFrame_seg[iSeg].begin(), vvFrame_seg[iSeg].end(), iFrame);
				int idx1 = (int) (it2 - vvFrame_seg[iSeg].begin());

				vx.push_back(vvx_seg[iSeg][idx1]);
				vy.push_back(vvy_seg[iSeg][idx1]);
				vz.push_back(vvz_seg[iSeg][idx1]);	
				//cout << iFrame <<" ";
			}

			//cout << endl;
			OptimizePOI(vvP_seg1, vvV_seg1, vvBandwidth_seg1, vx, vy, vz, lambda);

			for (int ix = 0; ix < vx.size(); ix++)
			{
				vector<int>::const_iterator it = find(vvFrame_seg[iSeg].begin(), vvFrame_seg[iSeg].end(), vFrame_seg1[ix]);
				int idx1 = (int) (it - vvFrame_seg[iSeg].begin());
				vvx_seg[iSeg][idx1] = vx[ix];
				vvy_seg[iSeg][idx1] = vy[ix];
				vvz_seg[iSeg][idx1] = vz[ix];
			}
			
		}

	}

	POI_Matches poi_matches_temp;
	for (int iSeg = 0; iSeg < vvFrame_seg.size(); iSeg++)
	{
		for (int ix = 0; ix < vvFrame_seg[iSeg].size(); ix++)
		{
			poi_matches_temp.vx.push_back(vvx_seg[iSeg][ix]);
			poi_matches_temp.vy.push_back(vvy_seg[iSeg][ix]);
			poi_matches_temp.vz.push_back(vvz_seg[iSeg][ix]);
			poi_matches_temp.vFrame.push_back(vvFrame_seg[iSeg][ix]);
		}
	}
	poi = poi_matches_temp;
}
*/
/*
void OptimizePOI(vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth,
				 vector<double> &vx, vector<double> &vy, vector<double> &vz, double lambda)
{
	AdditionalData adata;
	adata.vvP_poi = vvP;
	adata.vvV_poi = vvV;
	adata.vvBandwidth = vvBandwidth;
	adata.lambda = lambda;
	vector<double> vObj;
	for (int ix = 0; ix < vx.size(); ix++)
	{
		CvMat *x = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(x, 0, 0, vx[ix]);
		cvSetReal2D(x, 1, 0, vy[ix]);
		cvSetReal2D(x, 2, 0, vz[ix]);
		double obj = EvaulateDensityFunction(adata.vvP_poi[ix], adata.vvV_poi[ix], adata.vvBandwidth[ix], x);
		cvReleaseMat(&x);
		vObj.push_back(obj);
	}
	vector<double> measurement;
	vector<double> poi_trajectory;
	for (int iFrame = 0; iFrame < vx.size(); iFrame++)
	{
		poi_trajectory.push_back(vx[iFrame]);
		poi_trajectory.push_back(vy[iFrame]);
		poi_trajectory.push_back(vz[iFrame]);
	}
	for (int iFrame = 0; iFrame < vx.size(); iFrame++)
	{
		measurement.push_back(vObj[iFrame]+0.1);
	}
	for (int iFrame = 1; iFrame < vx.size()-1; iFrame++)
	{
		measurement.push_back(0);
		measurement.push_back(0);
		measurement.push_back(0);
	}
	double *dMeasurement = (double *) malloc(measurement.size() * sizeof(double));
	double *dPOI_trajectory = (double *) malloc(poi_trajectory.size() * sizeof(double));
	for (int i = 0; i < poi_trajectory.size(); i++)
		dPOI_trajectory[i] = poi_trajectory[i];
	for (int i = 0; i < measurement.size(); i++)
		dMeasurement[i] = measurement[i];
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(poi_trajectory.size(), measurement.size())+poi_trajectory.size()*poi_trajectory.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	cout << poi_trajectory.size() << " " << measurement.size() << endl;
	int ret ;//dlevmar_dif(ObjectiveFunction_poi_trajectory, dPOI_trajectory, dMeasurement, poi_trajectory.size(), measurement.size(),
			//				1e+3, opt, info, work, NULL, &adata);
	PrintSBAInfo(info, measurement.size());
	free(work);

	for (int iFrame = 0; iFrame < vx.size(); iFrame++)
	{
		vx[iFrame] = dPOI_trajectory[3*iFrame];
		vy[iFrame] = dPOI_trajectory[3*iFrame+1];
		vz[iFrame] = dPOI_trajectory[3*iFrame+2];			
	}	
}

void OptimizePOI1_Weight(vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, vector<double> vWeight,
	vector<int> vFrame,			 
	vector<double> &vx, vector<double> &vy, vector<double> &vz, double lambda)
{
	vector<int> startFrame;
	vector<int> endFrame;
	startFrame.push_back(vFrame[0]);
	for (int iFrame = 0; iFrame < vFrame.size()-1; iFrame++)
	{
		if (vFrame[iFrame]+1 != vFrame[iFrame+1])
		{
			endFrame.push_back(vFrame[iFrame]);
			startFrame.push_back(vFrame[iFrame+1]);
		}
	}
	endFrame.push_back(vFrame[vFrame.size()-1]);

	int time_window = 10;
	for (int iSeg = 0; iSeg < startFrame.size(); iSeg++)
	{
		for (int iTime = startFrame[iSeg]; iTime < min(iTime+time_window, endFrame[iSeg]); iTime++)
		{
			//cout << iTime << " " << min(iTime+time_window, endFrame[iSeg]) << endl;
			AdditionalData adata;
			vector<vector<CvMat *> > vvP1, vvV1;
			vector<vector<double> > vBandwidth1;
			vector<double> vX, vY, vZ;
			vector<double> measurement;
			vector<int> vIDX;
			for (int iTime1 = iTime; iTime1 < min(iTime+time_window, endFrame[iSeg]); iTime1++)
			{
				vector<int>::iterator it = find(vFrame.begin(), vFrame.end(), iTime1);
				int idx = (int) (it - vFrame.begin());
				vvP1.push_back(vvP[idx]);
				vvV1.push_back(vvV[idx]);
				vBandwidth1.push_back(vvBandwidth[idx]);
				vX.push_back(vx[idx]);
				vY.push_back(vy[idx]);
				vZ.push_back(vz[idx]);
				vIDX.push_back(idx);

				for (int ip = 0; ip < vvP[idx].size(); ip++)
				{
					measurement.push_back(0);
					measurement.push_back(0);
				}
				
				measurement.push_back(0);
				measurement.push_back(0);
				measurement.push_back(0);				
			}

			adata.vvP_poi = vvP1;
			adata.vvV_poi = vvV1;
			adata.vvBandwidth = vBandwidth1;
			adata.vWeight = vWeight;
			adata.lambda = lambda;

			double *dMeasurement = (double *) malloc(measurement.size() * sizeof(double));
			double *dPOI_trajectory = (double *) malloc(3*vX.size() * sizeof(double));
			for (int i = 0; i < vX.size(); i++)
			{
				dPOI_trajectory[3*i] = vX[i];
				dPOI_trajectory[3*i+1] = vY[i];
				dPOI_trajectory[3*i+2] = vZ[i];
			}
			for (int i = 0; i < measurement.size(); i++)
				dMeasurement[i] = measurement[i];
			double opt[5];
			opt[0] = 1e-3;
			opt[1] = 1e-9;
			opt[2] = 1e-9;
			opt[3] = 1e-9;
			opt[4] = 0;
			double info[12];
			double *work ;// (double*)malloc((LM_DIF_WORKSZ(3*vX.size(), measurement.size())+3*vX.size()*3*vX.size())*sizeof(double));
			if(!work)
				fprintf(stderr, "memory allocation request failed in main()\n");
			//cout << 3*vX.size() << " " << measurement.size() << endl;
			int ret ;//dlevmar_dif(ObjectiveFunction_poi_trajectory1_weight, dPOI_trajectory, dMeasurement, 3*vX.size(), measurement.size(),
					//				1e+3, opt, info, work, NULL, &adata);
			//PrintSBAInfo(info, measurement.size());
			free(work);

			for (int iFrame = 0; iFrame < vIDX.size(); iFrame++)
			{
				vx[vIDX[iFrame]] = dPOI_trajectory[3*iFrame];
				vy[vIDX[iFrame]] = dPOI_trajectory[3*iFrame+1];
				vz[vIDX[iFrame]] = dPOI_trajectory[3*iFrame+2];			
			}
		}
	}
}

void OptimizePOI1(vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth,
				 vector<double> &vx, vector<double> &vy, vector<double> &vz, double lambda)
{
	AdditionalData adata;
	adata.vvP_poi = vvP;
	adata.vvV_poi = vvV;
	adata.vvBandwidth = vvBandwidth;
	adata.lambda = lambda;
	vector<double> vObj;
	for (int ix = 0; ix < vx.size(); ix++)
	{
		CvMat *x = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(x, 0, 0, vx[ix]);
		cvSetReal2D(x, 1, 0, vy[ix]);
		cvSetReal2D(x, 2, 0, vz[ix]);
		double obj = EvaulateDensityFunction(adata.vvP_poi[ix], adata.vvV_poi[ix], adata.vvBandwidth[ix], x);
		cvReleaseMat(&x);
		vObj.push_back(obj);
	}
	vector<double> measurement;
	vector<double> poi_trajectory;
	for (int iFrame = 0; iFrame < vx.size(); iFrame++)
	{
		poi_trajectory.push_back(vx[iFrame]);
		poi_trajectory.push_back(vy[iFrame]);
		poi_trajectory.push_back(vz[iFrame]);
	}
	for (int iFrame = 0; iFrame < vx.size(); iFrame++)
	{
		measurement.push_back(vObj[iFrame]+0.1);
	}
	for (int iFrame = 0; iFrame < vx.size()-1; iFrame++)
	{
		measurement.push_back(0);
		measurement.push_back(0);
		measurement.push_back(0);
	}
	double *dMeasurement = (double *) malloc(measurement.size() * sizeof(double));
	double *dPOI_trajectory = (double *) malloc(poi_trajectory.size() * sizeof(double));
	for (int i = 0; i < poi_trajectory.size(); i++)
		dPOI_trajectory[i] = poi_trajectory[i];
	for (int i = 0; i < measurement.size(); i++)
		dMeasurement[i] = measurement[i];
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(poi_trajectory.size(), measurement.size())+poi_trajectory.size()*poi_trajectory.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	cout << poi_trajectory.size() << " " << measurement.size() << endl;
	int ret ;//dlevmar_dif(ObjectiveFunction_poi_trajectory1, dPOI_trajectory, dMeasurement, poi_trajectory.size(), measurement.size(),
			//				1e+3, opt, info, work, NULL, &adata);
	PrintSBAInfo(info, measurement.size());
	free(work);

	for (int iFrame = 0; iFrame < vx.size(); iFrame++)
	{
		vx[iFrame] = dPOI_trajectory[3*iFrame];
		vy[iFrame] = dPOI_trajectory[3*iFrame+1];
		vz[iFrame] = dPOI_trajectory[3*iFrame+2];			
	}	
}
*/

void POIRefinement_Seg(POI_Matches &poi, vector<int> vFrame_PV, vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, double lambda)
{
	int stride = 3;
	int window = 20;
	PrintAlgorithm("POI Refinement");	
	vector<double> vx_t, vy_t, vz_t;
	int poi_idx=0;
	vector<int> vFrame_t;
	for (int iFrame = poi.vFrame[0]; iFrame <= poi.vFrame[poi.vFrame.size()-1]; iFrame++)
	{
		vector<int>::const_iterator it = find(poi.vFrame.begin(), poi.vFrame.end(), iFrame);

		CvMat *x = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(x, 0, 0, poi.vx[poi_idx]);
		cvSetReal2D(x, 1, 0, poi.vy[poi_idx]);
		cvSetReal2D(x, 2, 0, poi.vz[poi_idx]);

		vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
		int idx = (int)(it1-vFrame_PV.begin());

		vector<double> vWeight;
		double obj = EvaulateDensityFunction(vvP[idx], vvV[idx], vvBandwidth[idx], x, vWeight);
		cvReleaseMat(&x);
		//cout << obj << " ";

		sort(vWeight.begin(), vWeight.end());
		//if (vWeight[vWeight.size()-2]/vWeight[vWeight.size()-1] > 0.01)
		//	cout << obj << " ";

		if (it!=poi.vFrame.end())
		{
			vx_t.push_back(poi.vx[poi_idx]);
			vy_t.push_back(poi.vy[poi_idx]);
			vz_t.push_back(poi.vz[poi_idx]);
			poi_idx++;
			cout << obj << endl;
			
		}
		else
		{
			vx_t.push_back(poi.vx[poi_idx]);
			vy_t.push_back(poi.vy[poi_idx]);
			vz_t.push_back(poi.vz[poi_idx]);
		}
		vFrame_t.push_back(iFrame);
		//if (vWeight[vWeight.size()-2]/vWeight[vWeight.size()-1] > 0.01)
		//	cout << endl;
	}
	cout << endl;

	if (vFrame_t.size() < window)
	{
		window = vFrame_t.size()/2-1;
		stride = 1;
	}

	for (int iFrame1 = poi.vFrame[0]+window; iFrame1 <= poi.vFrame[poi.vFrame.size()-1]-window; iFrame1 += stride)	
	{
		AdditionalData adata;
		vector<int> vFrame;		
		vector<double> vx, vy, vz;
		cout << poi.vFrame[0]+window << " " << iFrame1 << " " << poi.vFrame[poi.vFrame.size()-1]-window << endl;
		for (int iFrame = iFrame1-window; iFrame <= iFrame1+window; iFrame++)
		{
			vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
			int idx = (int) (it1 - vFrame_PV.begin());
			adata.vvP_poi.push_back(vvP[idx]);
			adata.vvV_poi.push_back(vvV[idx]);
			adata.vvBandwidth.push_back(vvBandwidth[idx]);
			vFrame.push_back(iFrame);

			vector<int>::const_iterator it2 = find(vFrame_t.begin(), vFrame_t.end(), iFrame);
			int idx2 = (int) (it2 - vFrame_t.begin());
			vx.push_back(vx_t[idx2]);
			vy.push_back(vy_t[idx2]);
			vz.push_back(vz_t[idx2]);	
		}
		adata.lambda = lambda;
		vector<double> vObj;
		for (int ix = 0; ix < vFrame.size(); ix++)
		{
			CvMat *x = cvCreateMat(3,1,CV_32FC1);
			cvSetReal2D(x, 0, 0, vx[ix]);
			cvSetReal2D(x, 1, 0, vy[ix]);
			cvSetReal2D(x, 2, 0, vz[ix]);
			double obj = EvaulateDensityFunction(adata.vvP_poi[ix], adata.vvV_poi[ix], adata.vvBandwidth[ix], x);
			cvReleaseMat(&x);
			vObj.push_back(obj);
			cout << obj << " ";
		}
		cout <<  endl;
		vector<double> measurement;
		vector<double> poi_trajectory;
		for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
		{
			poi_trajectory.push_back(vx[iFrame]);
			poi_trajectory.push_back(vy[iFrame]);
			poi_trajectory.push_back(vz[iFrame]);
		}
		for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
		{
			measurement.push_back(vObj[iFrame]+5);
		}
		for (int iFrame = 1; iFrame < vFrame.size()-1; iFrame++)
		{
			measurement.push_back(0);
			measurement.push_back(0);
			measurement.push_back(0);
			//cout << vFrame.size() << " " <<  iFrame << endl;
		}
		double *dMeasurement = (double *) malloc(measurement.size() * sizeof(double));
		double *dPOI_trajectory = (double *) malloc(poi_trajectory.size() * sizeof(double));
		for (int i = 0; i < poi_trajectory.size(); i++)
			dPOI_trajectory[i] = poi_trajectory[i];
		for (int i = 0; i < measurement.size(); i++)
			dMeasurement[i] = measurement[i];
		double opt[5];
		opt[0] = 1e-3;
		opt[1] = 1e-12;
		opt[2] = 1e-12;
		opt[3] = 1e-12;
		opt[4] = 0;
		double info[12];

		double *work ;// (double*)malloc((LM_DIF_WORKSZ(poi_trajectory.size(), measurement.size())+poi_trajectory.size()*poi_trajectory.size())*sizeof(double));
		if(!work)
			fprintf(stderr, "memory allocation request failed in main()\n");
		cout << poi_trajectory.size() << " " << measurement.size() << endl;
		int ret ;//dlevmar_dif(ObjectiveFunction_poi_trajectory, dPOI_trajectory, dMeasurement, poi_trajectory.size(), measurement.size(),
				//			  1e+4, opt, info, work, NULL, &adata);
		PrintSBAInfo(info, measurement.size());

		for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
		{
			//vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), vFrame[iFrame]);
			//int idx = (int) (it1 - vFrame_PV.begin());
			////cout << vx_t[idx] << " " <<  dPOI_trajectory[3*iFrame] << " " << vy_t[idx]<< " " <<  dPOI_trajectory[3*iFrame+1] << " " <<vz_t[idx] << " " <<  dPOI_trajectory[3*iFrame+2] << endl;
			//
			//CvMat *x = cvCreateMat(3,1,CV_32FC1);
			//cvSetReal2D(x, 0, 0, dPOI_trajectory[3*iFrame]);
			//cvSetReal2D(x, 1, 0, dPOI_trajectory[3*iFrame+1]);
			//cvSetReal2D(x, 2, 0, dPOI_trajectory[3*iFrame+2]);
			//double obj = EvaulateDensityFunction(adata.vvP_poi[iFrame], adata.vvV_poi[iFrame], adata.vvBandwidth[iFrame], x);
			//cvReleaseMat(&x);
			//cout << vObj[iFrame] << " " << obj << " ";
			
			vector<int>::const_iterator it2 = find(vFrame_t.begin(), vFrame_t.end(), vFrame[iFrame]);
			int idx2 = (int) (it2 - vFrame_t.begin());	

			//if ((iFrame > 1) && ( iFrame < vFrame.size()-1))
			//{
			//	double x1 = 2*dPOI_trajectory[3*iFrame]-dPOI_trajectory[3*(iFrame-1)]-dPOI_trajectory[3*(iFrame+1)];
			//	double x2 = 2*dPOI_trajectory[3*iFrame+1]-dPOI_trajectory[3*(iFrame-1)+1]-dPOI_trajectory[3*(iFrame+1)+1];
			//	double x3 = 2*dPOI_trajectory[3*iFrame+2]-dPOI_trajectory[3*(iFrame-1)+2]-dPOI_trajectory[3*(iFrame+1)+2];

			//	double y1 = 2*vx_t[idx2]-vx_t[idx2-1]-vx_t[idx2+1];
			//	double y2 = 2*vy_t[idx2]-vy_t[idx2-1]-vy_t[idx2+1];
			//	double y3 = 2*vz_t[idx2]-vz_t[idx2-1]-vz_t[idx2+1];

			//	cout << x1*x1+x2*x2+x3*x3 << " " <<  y1*y1+y2*y2+y3*y3 << endl;

			//}
			
			vx_t[idx2] = dPOI_trajectory[3*iFrame];
			vy_t[idx2] = dPOI_trajectory[3*iFrame+1];
			vz_t[idx2] = dPOI_trajectory[3*iFrame+2];
			
		}	
		free(work);
	}

	AdditionalData adata;
	vector<int> vFrame;		
	vector<double> vx, vy, vz;
	for (int iFrame = poi.vFrame[poi.vFrame.size()-1]-2*window; iFrame <= poi.vFrame[poi.vFrame.size()-1]; iFrame++)
	{
		vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
		int idx = (int) (it1 - vFrame_PV.begin());
		adata.vvP_poi.push_back(vvP[idx]);
		adata.vvV_poi.push_back(vvV[idx]);
		adata.vvBandwidth.push_back(vvBandwidth[idx]);
		vFrame.push_back(iFrame);

		vector<int>::const_iterator it2 = find(vFrame_t.begin(), vFrame_t.end(), iFrame);
		int idx2 = (int) (it2 - vFrame_t.begin());
		vx.push_back(vx_t[idx2]);
		vy.push_back(vy_t[idx2]);
		vz.push_back(vz_t[idx2]);	
	}
	adata.lambda = lambda;
	vector<double> vObj;
	for (int ix = 0; ix < vFrame.size(); ix++)
	{
		CvMat *x = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(x, 0, 0, vx[ix]);
		cvSetReal2D(x, 1, 0, vy[ix]);
		cvSetReal2D(x, 2, 0, vz[ix]);
		double obj = EvaulateDensityFunction(adata.vvP_poi[ix], adata.vvV_poi[ix], adata.vvBandwidth[ix], x);
		cvReleaseMat(&x);
		vObj.push_back(obj);
	}
	vector<double> measurement;
	vector<double> poi_trajectory;
	for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
	{
		poi_trajectory.push_back(vx[iFrame]);
		poi_trajectory.push_back(vy[iFrame]);
		poi_trajectory.push_back(vz[iFrame]);
	}
	for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
	{
		measurement.push_back(vObj[iFrame]+5);
	}
	for (int iFrame = 1; iFrame < vFrame.size()-1; iFrame++)
	{
		measurement.push_back(0);
		measurement.push_back(0);
		measurement.push_back(0);
		//cout << vFrame.size() << " " <<  iFrame << endl;
	}
	double *dMeasurement = (double *) malloc(measurement.size() * sizeof(double));
	double *dPOI_trajectory = (double *) malloc(poi_trajectory.size() * sizeof(double));
	for (int i = 0; i < poi_trajectory.size(); i++)
		dPOI_trajectory[i] = poi_trajectory[i];
	for (int i = 0; i < measurement.size(); i++)
		dMeasurement[i] = measurement[i];
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(poi_trajectory.size(), measurement.size())+poi_trajectory.size()*poi_trajectory.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	cout << poi_trajectory.size() << " " << measurement.size() << endl;
	int ret ;//dlevmar_dif(ObjectiveFunction_poi_trajectory, dPOI_trajectory, dMeasurement, poi_trajectory.size(), measurement.size(),
							//1e+3, opt, info, work, NULL, &adata);
	PrintSBAInfo(info, measurement.size());

	for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
	{			
		vector<int>::const_iterator it2 = find(vFrame_t.begin(), vFrame_t.end(), vFrame[iFrame]);
		int idx2 = (int) (it2 - vFrame_t.begin());	
			
		vx_t[idx2] = dPOI_trajectory[3*iFrame];
		vy_t[idx2] = dPOI_trajectory[3*iFrame+1];
		vz_t[idx2] = dPOI_trajectory[3*iFrame+2];			
	}	
	free(work);


	POI_Matches poi_matches_temp;
	poi_matches_temp.vx = vx_t;
	poi_matches_temp.vy = vy_t;
	poi_matches_temp.vz = vz_t;
	poi_matches_temp.vFrame = vFrame_t;

	for (int iFrame = 0; iFrame < vFrame_PV.size(); iFrame++)
	{
		vector<double> vWeight;
		vWeight.resize(vvP[0].size(), -1);
		poi_matches_temp.vvWeight.push_back(vWeight);
	}
	poi = poi_matches_temp;
}

void POIRefinement(POI_Matches &poi, vector<int> vFrame_PV, vector<vector<CvMat *> > vvP, vector<vector<CvMat *> > vvV, vector<vector<double> > vvBandwidth, double lambda)
{
	PrintAlgorithm("POI Refinement");
	AdditionalData adata;
	
	vector<int> vFrame;
	vector<double> vx, vy, vz;
	int poi_idx=0;
	for (int iFrame = poi.vFrame[0]; iFrame < 500; iFrame++)	
	{
		vector<int>::const_iterator it1 = find(vFrame_PV.begin(), vFrame_PV.end(), iFrame);
		int idx = (int) (it1 - vFrame_PV.begin());
		adata.vvP_poi.push_back(vvP[idx]);
		adata.vvV_poi.push_back(vvV[idx]);
		adata.vvBandwidth.push_back(vvBandwidth[idx]);
		vFrame.push_back(iFrame);
		vector<int>::const_iterator it = find(poi.vFrame.begin(), poi.vFrame.end(), iFrame);
		if (it!=poi.vFrame.end())
		{
			vx.push_back(poi.vx[poi_idx]);
			vy.push_back(poi.vy[poi_idx]);
			vz.push_back(poi.vz[poi_idx]);
			poi_idx++;
		}
		else
		{
			vx.push_back(poi.vx[poi_idx]);
			vy.push_back(poi.vy[poi_idx]);
			vz.push_back(poi.vz[poi_idx]);
		}
	}
	adata.lambda = lambda;	
	vector<double> vObj;
	for (int ix = 0; ix < vFrame.size(); ix++)
	{
		CvMat *x = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(x, 0, 0, vx[ix]);
		cvSetReal2D(x, 1, 0, vy[ix]);
		cvSetReal2D(x, 2, 0, vz[ix]);
		double obj = EvaulateDensityFunction(adata.vvP_poi[ix], adata.vvV_poi[ix], adata.vvBandwidth[ix], x);
		cvReleaseMat(&x);
		vObj.push_back(obj);
	}
	vector<double> measurement;
	vector<double> poi_trajectory;
	for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
	{
		poi_trajectory.push_back(vx[iFrame]);
		poi_trajectory.push_back(vy[iFrame]);
		poi_trajectory.push_back(vz[iFrame]);
	}
	for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
	{
		measurement.push_back(vObj[iFrame]+0.5);
	}
	for (int iFrame = 1; iFrame < vFrame.size()-1; iFrame++)
	{
		measurement.push_back(0);
		measurement.push_back(0);
		measurement.push_back(0);
		//cout << vFrame.size() << " " <<  iFrame << endl;
	}
	double *dMeasurement = (double *) malloc(measurement.size() * sizeof(double));
	double *dPOI_trajectory = (double *) malloc(poi_trajectory.size() * sizeof(double));
	for (int i = 0; i < poi_trajectory.size(); i++)
		dPOI_trajectory[i] = poi_trajectory[i];
	for (int i = 0; i < measurement.size(); i++)
		dMeasurement[i] = measurement[i];
	double opt[5];
	opt[0] = 1e-12;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(poi_trajectory.size(), measurement.size())+poi_trajectory.size()*poi_trajectory.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");

	cout << poi_trajectory.size() << " " << measurement.size() << endl;
	int ret ;//dlevmar_dif(ObjectiveFunction_poi_trajectory, dPOI_trajectory, dMeasurement, poi_trajectory.size(), measurement.size(),
						  //1e+12, opt, info, work, NULL, &adata);
	PrintSBAInfo(info, measurement.size());
	POI_Matches poi_matches_temp;
	for (int iFrame = 0; iFrame < vFrame.size(); iFrame++)
	{
		poi_matches_temp.vx.push_back(dPOI_trajectory[3*iFrame]);
		poi_matches_temp.vy.push_back(dPOI_trajectory[3*iFrame+1]);
		poi_matches_temp.vz.push_back(dPOI_trajectory[3*iFrame+2]);
		vector<double> vWeight;
		vWeight.resize(vvP[0].size(), -1);
		poi_matches_temp.vvWeight.push_back(vWeight);
		poi_matches_temp.vFrame.push_back(vFrame[iFrame]);
	}
	poi = poi_matches_temp;
}

void ObjectiveFunction_poi_trajectory(double *rt, double *hx, int m, int n, void *adata)
{
	//cout << ((AdditionalData *) adata)->vvP_poi.size() << endl;
	for (int iFrame = 0; iFrame < ((AdditionalData *) adata)->vvP_poi.size(); iFrame++)
	{
		double f = 0;
		double x1 = rt[3*iFrame];
		double x2 = rt[3*iFrame+1];
		double x3 = rt[3*iFrame+2];
		for (int ip = 0; ip < ((AdditionalData *) adata)->vvP_poi[iFrame].size(); ip++)
		{
			double v1 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 0, 0);
			double v2 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 1, 0);
			double v3 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 2, 0);

			double p1 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 0, 0);
			double p2 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 1, 0);
			double p3 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 2, 0);
			double bandwidth = ((AdditionalData *) adata)->vvBandwidth[iFrame][ip];
			if (bandwidth < 0)
				continue;

			double norm_v = sqrt((v1*v1)+(v2*v2)+(v3*v3));
			
			v1 /= norm_v;
			v2 /= norm_v;
			v3 /= norm_v;

			double xmp1 = x1-p1;
			double xmp2 = x2-p2;
			double xmp3 = x3-p3;

			double dot_product = v1*xmp1+v2*xmp2+v3*xmp3;

			double dot_ray1 = dot_product*v1;
			double dot_ray2 = dot_product*v2;
			double dot_ray3 = dot_product*v3;

			double dx1 = xmp1-dot_ray1;
			double dx2 = xmp2-dot_ray2;
			double dx3 = xmp3-dot_ray3;

			if (dot_product > 0)
			{
				double dist1 = sqrt((dx1*dx1)+(dx2*dx2)+(dx3*dx3));
				double dist2 = dot_product;
				f+= 1/bandwidth * exp(-(dist1/dist2)*(dist1/dist2)/2/bandwidth/bandwidth);
			}
		}
		f /= ((AdditionalData *) adata)->vvP_poi[iFrame].size();
		hx[iFrame] = f;
		//cout << iFrame << " " ;
	}
	//cout << hx[10] << " " << hx[11] << " " << hx[12] << endl;
	//cout << hx[3] << endl;;

	for (int iFrame=1; iFrame < ((AdditionalData *) adata)->vvP_poi.size()-1; iFrame++)
	{
		double x1 = rt[3*iFrame];	
		double x2 = rt[3*iFrame+1];
		double x3 = rt[3*iFrame+2];

		double xp1 = rt[3*(iFrame-1)];
		double xp2 = rt[3*(iFrame-1)+1];
		double xp3 = rt[3*(iFrame-1)+2];

		double xn1 = rt[3*(iFrame+1)];
		double xn2 = rt[3*(iFrame+1)+1];
		double xn3 = rt[3*(iFrame+1)+2];

		hx[((AdditionalData *) adata)->vvP_poi.size()+3*(iFrame-1)] = ((AdditionalData *) adata)->lambda*(2*x1-xp1-xn1);
		hx[((AdditionalData *) adata)->vvP_poi.size()+3*(iFrame-1)+1] = ((AdditionalData *) adata)->lambda*(2*x2-xp2-xn2);
		hx[((AdditionalData *) adata)->vvP_poi.size()+3*(iFrame-1)+2] = ((AdditionalData *) adata)->lambda*(2*x3-xp3-xn3);
	}
	//cout << endl;
}

void ObjectiveFunction_poi_trajectory1(double *rt, double *hx, int m, int n, void *adata)
{
	//cout << ((AdditionalData *) adata)->vvP_poi.size() << endl;
	for (int iFrame = 0; iFrame < ((AdditionalData *) adata)->vvP_poi.size(); iFrame++)
	{
		double f = 0;
		double x1 = rt[3*iFrame];
		double x2 = rt[3*iFrame+1];
		double x3 = rt[3*iFrame+2];
		for (int ip = 0; ip < ((AdditionalData *) adata)->vvP_poi[iFrame].size(); ip++)
		{
			double v1 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 0, 0);
			double v2 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 1, 0);
			double v3 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 2, 0);

			double p1 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 0, 0);
			double p2 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 1, 0);
			double p3 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 2, 0);
			double bandwidth = ((AdditionalData *) adata)->vvBandwidth[iFrame][ip];
			if (bandwidth < 0)
				continue;

			double norm_v = sqrt((v1*v1)+(v2*v2)+(v3*v3));
			
			v1 /= norm_v;
			v2 /= norm_v;
			v3 /= norm_v;

			double xmp1 = x1-p1;
			double xmp2 = x2-p2;
			double xmp3 = x3-p3;

			double dot_product = v1*xmp1+v2*xmp2+v3*xmp3;

			double dot_ray1 = dot_product*v1;
			double dot_ray2 = dot_product*v2;
			double dot_ray3 = dot_product*v3;

			double dx1 = xmp1-dot_ray1;
			double dx2 = xmp2-dot_ray2;
			double dx3 = xmp3-dot_ray3;

			if (dot_product > 0)
			{
				double dist1 = sqrt((dx1*dx1)+(dx2*dx2)+(dx3*dx3));
				double dist2 = dot_product;
				f+= 1/bandwidth * exp(-(dist1/dist2)*(dist1/dist2)/2/bandwidth/bandwidth);
			}
		}
		f /= ((AdditionalData *) adata)->vvP_poi[iFrame].size();
		hx[iFrame] = f;
		//cout << iFrame << " " ;
	}
	//cout << hx[10] << " " << hx[11] << " " << hx[12] << endl;
	//cout << hx[3] << endl;;

	for (int iFrame=0; iFrame < ((AdditionalData *) adata)->vvP_poi.size()-1; iFrame++)
	{
		double x1 = rt[3*iFrame];	
		double x2 = rt[3*iFrame+1];
		double x3 = rt[3*iFrame+2];

		double xn1 = rt[3*(iFrame+1)];
		double xn2 = rt[3*(iFrame+1)+1];
		double xn3 = rt[3*(iFrame+1)+2];

		hx[((AdditionalData *) adata)->vvP_poi.size()+3*(iFrame)] = ((AdditionalData *) adata)->lambda*(x1-xn1);
		hx[((AdditionalData *) adata)->vvP_poi.size()+3*(iFrame)+1] = ((AdditionalData *) adata)->lambda*(x2-xn2);
		hx[((AdditionalData *) adata)->vvP_poi.size()+3*(iFrame)+2] = ((AdditionalData *) adata)->lambda*(x3-xn3);
	}
	//cout << endl;
}

void ObjectiveFunction_poi_trajectory1_weight(double *rt, double *hx, int m, int n, void *adata)
{
	//cout << ((AdditionalData *) adata)->vvP_poi.size() << endl;
	for (int iFrame = 0; iFrame < ((AdditionalData *) adata)->vvP_poi.size(); iFrame++)
	{
		double f = 0;
		double x1 = rt[3*iFrame];
		double x2 = rt[3*iFrame+1];
		double x3 = rt[3*iFrame+2];
		for (int ip = 0; ip < ((AdditionalData *) adata)->vvP_poi[iFrame].size(); ip++)
		{
			double v1 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 0, 0);
			double v2 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 1, 0);
			double v3 = cvGetReal2D(((AdditionalData *) adata)->vvV_poi[iFrame][ip], 2, 0);

			double p1 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 0, 0);
			double p2 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 1, 0);
			double p3 = cvGetReal2D(((AdditionalData *) adata)->vvP_poi[iFrame][ip], 2, 0);
			double bandwidth = ((AdditionalData *) adata)->vvBandwidth[iFrame][ip];
			if (bandwidth < 0)
				continue;

			double norm_v = sqrt((v1*v1)+(v2*v2)+(v3*v3));
			
			v1 /= norm_v;
			v2 /= norm_v;
			v3 /= norm_v;

			double xmp1 = x1-p1;
			double xmp2 = x2-p2;
			double xmp3 = x3-p3;

			double dot_product = v1*xmp1+v2*xmp2+v3*xmp3;

			double dot_ray1 = dot_product*v1;
			double dot_ray2 = dot_product*v2;
			double dot_ray3 = dot_product*v3;

			double dx1 = xmp1-dot_ray1;
			double dx2 = xmp2-dot_ray2;
			double dx3 = xmp3-dot_ray3;

			double li = 0;
			if (((AdditionalData *) adata)->vWeight[ip] > 0.8)
				li = 1;
			else
				li = 0;

			if (dot_product > 0)
			{
				//double dist1 = sqrt((dx1*dx1)+(dx2*dx2)+(dx3*dx3));
				double dist2 = dot_product;
				//f+= 1/bandwidth * exp(-(dist1/dist2)*(dist1/dist2)/2/bandwidth/bandwidth);
				hx[2*ip+2*iFrame*((AdditionalData *) adata)->vvP_poi[iFrame].size()] = dx1/dist2*li;
				hx[2*ip+2*iFrame*((AdditionalData *) adata)->vvP_poi[iFrame].size()+1] = dx2/dist2*li;
			}
			else
			{
				hx[2*ip+2*iFrame*((AdditionalData *) adata)->vvP_poi[iFrame].size()] = 0;
				hx[2*ip+2*iFrame*((AdditionalData *) adata)->vvP_poi[iFrame].size()+1] = 0;
			}
		}
		//f /= ((AdditionalData *) adata)->vvP_poi[iFrame].size();
		//hx[iFrame] = f;
		//cout << iFrame << " " ;
	}
	//cout << hx[10] << " " << hx[11] << " " << hx[12] << endl;
	//cout << hx[3] << endl;;

	for (int iFrame=0; iFrame < ((AdditionalData *) adata)->vvP_poi.size()-1; iFrame++)
	{
		double x1 = rt[3*iFrame];	
		double x2 = rt[3*iFrame+1];
		double x3 = rt[3*iFrame+2];

		double xn1 = rt[3*(iFrame+1)];
		double xn2 = rt[3*(iFrame+1)+1];
		double xn3 = rt[3*(iFrame+1)+2];

		//cout << 2*((AdditionalData *) adata)->vvP_poi.size()*((AdditionalData *) adata)->vvP_poi[iFrame].size() << endl;
		hx[2*((AdditionalData *) adata)->vvP_poi.size()*((AdditionalData *) adata)->vvP_poi[iFrame].size()+3*(iFrame)] = ((AdditionalData *) adata)->lambda*(x1-xn1);
		hx[2*((AdditionalData *) adata)->vvP_poi.size()*((AdditionalData *) adata)->vvP_poi[iFrame].size()+3*(iFrame)+1] = ((AdditionalData *) adata)->lambda*(x2-xn2);
		hx[2*((AdditionalData *) adata)->vvP_poi.size()*((AdditionalData *) adata)->vvP_poi[iFrame].size()+3*(iFrame)+2] = ((AdditionalData *) adata)->lambda*(x3-xn3);
	}
	//cout << endl;
}


int BilinearCameraPoseEstimation_OPENCV(vector<Feature> vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat &P, CvMat &X, vector<int> &visibleStructureID)
{
	PrintAlgorithm("Bilinear Camera Pose Estimation");
	CvMat cx1, cx2, nx1, nx2;
	vector<int> visibleFeatureID;
	X = *cvCreateMat(vFeature.size(), 3, CV_32FC1);
	VisibleIntersection(vFeature, initialFrame1, initialFrame2, cx1, cx2, visibleFeatureID);
	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);

	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1,cx1.rows,CV_8UC1);
	int n = cvFindFundamentalMat(&cx1, &cx2, F, CV_FM_LMEDS , 1, 0.99, status);
	//int n = cvFindFundamentalMat(&cx1, &cx2, F, CV_FM_8POINT  , 3, 0.99, status);
	PrintMat(F, "Fundamental Matrix");
	for (int i = 0; i < cx1.rows; i++)
	{
		if (cvGetReal2D(status, 0, i) == 1)
		{
			vInlierID.push_back(visibleFeatureID[i]);
		}
	}
	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
	int temprow = 0;
	for (int i = 0; i < cx1.rows; i++)
	{
		if (cvGetReal2D(status, 0, i) == 1)
		{
			cvSetReal2D(tempCx1, temprow, 0, cvGetReal2D(&cx1, i, 0));
			cvSetReal2D(tempCx1, temprow, 1, cvGetReal2D(&cx1, i, 1));
			cvSetReal2D(tempCx2, temprow, 0, cvGetReal2D(&cx2, i, 0));
			cvSetReal2D(tempCx2, temprow, 1, cvGetReal2D(&cx2, i, 1));
			temprow++;
		}
	}
	cx1 = *cvCreateMat(tempCx1->rows, tempCx1->cols, CV_32FC1);
	cx2 = *cvCreateMat(tempCx2->rows, tempCx2->cols, CV_32FC1);
	cx1 = *cvCloneMat(tempCx1);
	cx2 = *cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *K1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *K2 = cvCreateMat(3,3,CV_32FC1);
	int camera1 = (int)((double)initialFrame1/max_nFrames);
	int camera2 = (int)((double)initialFrame2/max_nFrames);
	vector<int> ::const_iterator it1 = find(vCamera[camera1].vTakenFrame.begin(), vCamera[camera1].vTakenFrame.end(), initialFrame1%max_nFrames);
	vector<int> ::const_iterator it2 = find(vCamera[camera2].vTakenFrame.begin(), vCamera[camera2].vTakenFrame.end(), initialFrame2%max_nFrames);
	int idx1 = (int) (it1 - vCamera[camera1].vTakenFrame.begin());
	int idx2 = (int) (it2 - vCamera[camera2].vTakenFrame.begin());
	K1 = cvCloneMat(vCamera[camera1].vK[idx1]);
	K2 = cvCloneMat(vCamera[camera2].vK[idx2]);

	cvTranspose(K2, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K1, E);

	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K1, invK1);
	cvInvert(K2, invK2);
	Pxx_inhomo(invK1, &cx1, nx1);
	Pxx_inhomo(invK2, &cx2, nx2);

	GetExtrinsicParameterFromE(E, &nx1, &nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	CvMat cX;

	LinearTriangulation(&nx1, P0, &nx2, &P, cX);
	cvSetZero(&X);
	SetIndexedMatRowwise(&X, visibleFeatureID, &cX);
	cvMatMul(K2, &P, temp34);
	P = *cvCloneMat(temp34);

	visibleStructureID = visibleFeatureID;

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K1);
	cvReleaseMat(&K2);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invK2);
	cvReleaseMat(&P0);	
	return n;
}

void SetCvMatFromVectors(vector<vector<double> > x, CvMat *X)
{
	for (int i = 0; i < x.size(); i++)
	{
		for (int j = 0; j < x[i].size(); j++)
			cvSetReal2D(X, i, j, x[i][j]);
	}
}

int BilinearCameraPoseEstimation_OPENCV_mem(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X, vector<int> &visibleStructureID)
{
	PrintAlgorithm("Bilinear Camera Pose Estimation");
	vector<int> visibleFeatureID;
	vector<vector<double> > cx1_vec, cx2_vec, nx1_vec, nx2_vec;

	VisibleIntersection_mem(vFeature, initialFrame1, initialFrame2, cx1_vec, cx2_vec, visibleFeatureID);
	CvMat *cx1 = cvCreateMat(cx1_vec.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(cx2_vec.size(), 2, CV_32FC1);
	SetCvMatFromVectors(cx1_vec, cx1);
	SetCvMatFromVectors(cx2_vec, cx2);

	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1,cx1->rows,CV_8UC1);
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS , 1, 0.99, status);
	PrintMat(F, "Fundamental Matrix");
	for (int i = 0; i < cx1->rows; i++)
	{
		if (cvGetReal2D(status, 0, i) == 1)
		{
			vInlierID.push_back(visibleFeatureID[i]);
		}
	}
	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vInlierID.size(), 2, CV_32FC1);
	int temprow = 0;
	for (int i = 0; i < cx1->rows; i++)
	{
		if (cvGetReal2D(status, 0, i) == 1)
		{
			cvSetReal2D(tempCx1, temprow, 0, cvGetReal2D(cx1, i, 0));
			cvSetReal2D(tempCx1, temprow, 1, cvGetReal2D(cx1, i, 1));
			cvSetReal2D(tempCx2, temprow, 0, cvGetReal2D(cx2, i, 0));
			cvSetReal2D(tempCx2, temprow, 1, cvGetReal2D(cx2, i, 1));
			temprow++;
		}
	}
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cx1 = cvCloneMat(tempCx1);
	cx2 = cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	//CvMat *K1 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *K2 = cvCreateMat(3,3,CV_32FC1);
	int camera1 = (int)((double)initialFrame1/max_nFrames);
	int camera2 = (int)((double)initialFrame2/max_nFrames);
	vector<int> ::const_iterator it1 = find(vCamera[camera1].vTakenFrame.begin(), vCamera[camera1].vTakenFrame.end(), initialFrame1%max_nFrames);
	vector<int> ::const_iterator it2 = find(vCamera[camera2].vTakenFrame.begin(), vCamera[camera2].vTakenFrame.end(), initialFrame2%max_nFrames);
	int idx1 = (int) (it1 - vCamera[camera1].vTakenFrame.begin());
	int idx2 = (int) (it2 - vCamera[camera2].vTakenFrame.begin());
	CvMat *K1 = cvCloneMat(vCamera[camera1].vK[idx1]);
	CvMat *K2 = cvCloneMat(vCamera[camera2].vK[idx2]);

	cvTranspose(K2, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K1, E);

	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K1, invK1);
	cvInvert(K2, invK2);
	CvMat *nx1 = cvCreateMat(cx1->rows, cx1->cols, CV_32FC1);
	CvMat *nx2 = cvCreateMat(cx2->rows, cx2->cols, CV_32FC1);
	Pxx_inhomo(invK1, cx1, nx1);
	Pxx_inhomo(invK2, cx2, nx2);

	GetExtrinsicParameterFromE(E, nx1, nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	CvMat *cX = cvCreateMat(nx1->rows, 3, CV_32FC1);

	LinearTriangulation(nx1, P0, nx2, P, cX);
	cvSetZero(X);
	SetIndexedMatRowwise(X, visibleFeatureID, cX);
	cvMatMul(K2, P, temp34);
	SetSubMat(P, 0, 0, temp34);

	visibleStructureID = visibleFeatureID;

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K1);
	cvReleaseMat(&K2);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invK2);
	cvReleaseMat(&P0);	
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cvReleaseMat(&nx1);
	cvReleaseMat(&nx2);
	cvReleaseMat(&cX);
	return n;
}

int BilinearCameraPoseEstimation_OPENCV_mem_fast(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X)
{
	PrintAlgorithm("Bilinear Camera Pose Estimation");
	vector<int> visibleFeatureID;
	vector<vector<double> > cx1_vec, cx2_vec, nx1_vec, nx2_vec;

	VisibleIntersection_mem(vFeature, initialFrame1, initialFrame2, cx1_vec, cx2_vec, visibleFeatureID);
	CvMat *cx1 = cvCreateMat(cx1_vec.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(cx2_vec.size(), 2, CV_32FC1);
	SetCvMatFromVectors(cx1_vec, cx1);
	SetCvMatFromVectors(cx2_vec, cx2);

	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1,cx1->rows,CV_8UC1);
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS , 1, 0.99, status);
	PrintMat(F, "Fundamental Matrix");
	//for (int i = 0; i < cx1->rows; i++)
	//{
	//	if (cvGetReal2D(status, 0, i) == 1)
	//	{
	//		vInlierID.push_back(visibleFeatureID[i]);
	//	}
	//}

	cout << n << endl;

	vector<int> vCX_indx;
	for (int i = 0; i < cx1->rows; i++)
	{
		CvMat *xM2 = cvCreateMat(1,3,CV_32FC1);
		CvMat *xM1 = cvCreateMat(3,1,CV_32FC1);
		CvMat *s = cvCreateMat(1,1, CV_32FC1);
		cvSetReal2D(xM2, 0, 0, cvGetReal2D(cx2, i, 0));
		cvSetReal2D(xM2, 0, 1, cvGetReal2D(cx2, i, 1));
		cvSetReal2D(xM2, 0, 2, 1);
		cvSetReal2D(xM1, 0, 0, cvGetReal2D(cx1, i, 0));
		cvSetReal2D(xM1, 1, 0, cvGetReal2D(cx1, i, 1));
		cvSetReal2D(xM1, 2, 0, 1);
		cvMatMul(xM2, F, xM2);
		cvMatMul(xM2, xM1, s);			

		double l1 = cvGetReal2D(xM2, 0, 0);
		double l2 = cvGetReal2D(xM2, 0, 1);
		double l3 = cvGetReal2D(xM2, 0, 2);

		double dist = abs(cvGetReal2D(s, 0, 0))/sqrt(l1*l1+l2*l2);

		if (dist < 5)
		{
			vInlierID.push_back(visibleFeatureID[i]);
			vCX_indx.push_back(i);
		}

		cvReleaseMat(&xM2);
		cvReleaseMat(&xM1);
		cvReleaseMat(&s);
		//if (cvGetReal2D(status, 0, i) == 1)
		//{
		//	cvSetReal2D(tempCx1, temprow, 0, cvGetReal2D(cx1, i, 0));
		//	cvSetReal2D(tempCx1, temprow, 1, cvGetReal2D(cx1, i, 1));
		//	cvSetReal2D(tempCx2, temprow, 0, cvGetReal2D(cx2, i, 0));
		//	cvSetReal2D(tempCx2, temprow, 1, cvGetReal2D(cx2, i, 1));
		//	temprow++;
		//}
	}

	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierID.size(); iInlier++)
	{
		cvSetReal2D(tempCx1, iInlier, 0, cvGetReal2D(cx1, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx1, iInlier, 1, cvGetReal2D(cx1, vCX_indx[iInlier], 1));
		cvSetReal2D(tempCx2, iInlier, 0, cvGetReal2D(cx2, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx2, iInlier, 1, cvGetReal2D(cx2, vCX_indx[iInlier], 1));
	}

	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cx1 = cvCloneMat(tempCx1);
	cx2 = cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	//CvMat *K1 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *K2 = cvCreateMat(3,3,CV_32FC1);
	int camera1 = (int)((double)initialFrame1/max_nFrames);
	int camera2 = (int)((double)initialFrame2/max_nFrames);
	vector<int> ::const_iterator it1 = find(vCamera[camera1].vTakenFrame.begin(), vCamera[camera1].vTakenFrame.end(), initialFrame1%max_nFrames);
	vector<int> ::const_iterator it2 = find(vCamera[camera2].vTakenFrame.begin(), vCamera[camera2].vTakenFrame.end(), initialFrame2%max_nFrames);
	int idx1 = (int) (it1 - vCamera[camera1].vTakenFrame.begin());
	int idx2 = (int) (it2 - vCamera[camera2].vTakenFrame.begin());
	CvMat *K1 = cvCloneMat(vCamera[camera1].vK[idx1]);
	CvMat *K2 = cvCloneMat(vCamera[camera2].vK[idx2]);

	cvTranspose(K2, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K1, E);

	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K1, invK1);
	cvInvert(K2, invK2);
	CvMat *nx1 = cvCreateMat(cx1->rows, cx1->cols, CV_32FC1);
	CvMat *nx2 = cvCreateMat(cx2->rows, cx2->cols, CV_32FC1);
	Pxx_inhomo(invK1, cx1, nx1);
	Pxx_inhomo(invK2, cx2, nx2);

	GetExtrinsicParameterFromE(E, nx1, nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	
	CvMat *cX = cvCreateMat(nx1->rows, 3, CV_32FC1);

	PrintMat(P);
	cvMatMul(K1, P0, P0);
	cvMatMul(K2, P, P);

	PrintMat(P);
	//LinearTriangulation(nx1, P0, nx2, P, cX);
	LinearTriangulation(cx1, P0, cx2, P, cX);
	//PrintMat(cX);
	cvSetZero(X);
	SetIndexedMatRowwise(X, visibleFeatureID, cX);

	
	for (int i = 0; i < visibleFeatureID.size(); i++)
	{
		vFeature[visibleFeatureID[i]].isRegistered = true;
	}

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K1);
	cvReleaseMat(&K2);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invK2);
	cvReleaseMat(&P0);	
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cvReleaseMat(&nx1);
	cvReleaseMat(&nx2);
	cvReleaseMat(&cX);
	return vInlierID.size();
}

int BilinearCameraPoseEstimation_OPENCV_mem_fast_Dome(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X)
{
	PrintAlgorithm("Bilinear Camera Pose Estimation");
	vector<int> visibleFeatureID;
	vector<vector<double> > cx1_vec, cx2_vec, nx1_vec, nx2_vec;

	VisibleIntersection_mem(vFeature, initialFrame1, initialFrame2, cx1_vec, cx2_vec, visibleFeatureID);
	CvMat *cx1 = cvCreateMat(cx1_vec.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(cx2_vec.size(), 2, CV_32FC1);
	SetCvMatFromVectors(cx1_vec, cx1);
	SetCvMatFromVectors(cx2_vec, cx2);

	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1,cx1->rows,CV_8UC1);
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS , 1, 0.99, status);
	PrintMat(F, "Fundamental Matrix");
	//for (int i = 0; i < cx1->rows; i++)
	//{
	//	if (cvGetReal2D(status, 0, i) == 1)
	//	{
	//		vInlierID.push_back(visibleFeatureID[i]);
	//	}
	//}

	cout << n << endl;

	vector<int> vCX_indx;
	for (int i = 0; i < cx1->rows; i++)
	{
		CvMat *xM2 = cvCreateMat(1,3,CV_32FC1);
		CvMat *xM1 = cvCreateMat(3,1,CV_32FC1);
		CvMat *s = cvCreateMat(1,1, CV_32FC1);
		cvSetReal2D(xM2, 0, 0, cvGetReal2D(cx2, i, 0));
		cvSetReal2D(xM2, 0, 1, cvGetReal2D(cx2, i, 1));
		cvSetReal2D(xM2, 0, 2, 1);
		cvSetReal2D(xM1, 0, 0, cvGetReal2D(cx1, i, 0));
		cvSetReal2D(xM1, 1, 0, cvGetReal2D(cx1, i, 1));
		cvSetReal2D(xM1, 2, 0, 1);
		cvMatMul(xM2, F, xM2);
		cvMatMul(xM2, xM1, s);			

		double l1 = cvGetReal2D(xM2, 0, 0);
		double l2 = cvGetReal2D(xM2, 0, 1);
		double l3 = cvGetReal2D(xM2, 0, 2);

		double dist = abs(cvGetReal2D(s, 0, 0))/sqrt(l1*l1+l2*l2);

		if (dist < 5)
		{
			vInlierID.push_back(visibleFeatureID[i]);
			vCX_indx.push_back(i);
		}

		cvReleaseMat(&xM2);
		cvReleaseMat(&xM1);
		cvReleaseMat(&s);
		//if (cvGetReal2D(status, 0, i) == 1)
		//{
		//	cvSetReal2D(tempCx1, temprow, 0, cvGetReal2D(cx1, i, 0));
		//	cvSetReal2D(tempCx1, temprow, 1, cvGetReal2D(cx1, i, 1));
		//	cvSetReal2D(tempCx2, temprow, 0, cvGetReal2D(cx2, i, 0));
		//	cvSetReal2D(tempCx2, temprow, 1, cvGetReal2D(cx2, i, 1));
		//	temprow++;
		//}
	}

	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierID.size(); iInlier++)
	{
		cvSetReal2D(tempCx1, iInlier, 0, cvGetReal2D(cx1, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx1, iInlier, 1, cvGetReal2D(cx1, vCX_indx[iInlier], 1));
		cvSetReal2D(tempCx2, iInlier, 0, cvGetReal2D(cx2, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx2, iInlier, 1, cvGetReal2D(cx2, vCX_indx[iInlier], 1));
	}

	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cx1 = cvCloneMat(tempCx1);
	cx2 = cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	//CvMat *K1 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *K2 = cvCreateMat(3,3,CV_32FC1);
	int camera1 = (int)((double)initialFrame1/max_nFrames);
	int camera2 = (int)((double)initialFrame2/max_nFrames);
	vector<int> ::const_iterator it1 = find(vCamera[camera1].vTakenFrame.begin(), vCamera[camera1].vTakenFrame.end(), initialFrame1%max_nFrames);
	vector<int> ::const_iterator it2 = find(vCamera[camera2].vTakenFrame.begin(), vCamera[camera2].vTakenFrame.end(), initialFrame2%max_nFrames);
	int idx1 = (int) (it1 - vCamera[camera1].vTakenFrame.begin());
	int idx2 = (int) (it2 - vCamera[camera2].vTakenFrame.begin());
	CvMat *K1 = cvCloneMat(vCamera[camera1].vK[idx1]);
	CvMat *K2 = cvCloneMat(vCamera[camera2].vK[idx2]);

	cvTranspose(K2, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K1, E);

	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K1, invK1);
	cvInvert(K2, invK2);
	CvMat *nx1 = cvCreateMat(cx1->rows, cx1->cols, CV_32FC1);
	CvMat *nx2 = cvCreateMat(cx2->rows, cx2->cols, CV_32FC1);
	Pxx_inhomo(invK1, cx1, nx1);
	Pxx_inhomo(invK2, cx2, nx2);

	GetExtrinsicParameterFromE(E, nx1, nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	
	CvMat *cX = cvCreateMat(nx1->rows, 3, CV_32FC1);

	PrintMat(P);
	cvMatMul(K1, P0, P0);
	cvMatMul(K2, P, P);

	PrintMat(P);
	//LinearTriangulation(nx1, P0, nx2, P, cX);
	LinearTriangulation(cx1, P0, cx2, P, cX);
	//PrintMat(cX);
	cvSetZero(X);
	SetIndexedMatRowwise(X, visibleFeatureID, cX);

	
	for (int i = 0; i < visibleFeatureID.size(); i++)
	{
		vFeature[visibleFeatureID[i]].isRegistered = true;
		vFeature[visibleFeatureID[i]].vVisibleFrame.push_back(initialFrame1);
		vFeature[visibleFeatureID[i]].vVisibleFrame.push_back(initialFrame2);

		vector<int>::iterator it_i1 = find(vFeature[visibleFeatureID[i]].vFrame.begin(), 
										vFeature[visibleFeatureID[i]].vFrame.end(),
										initialFrame1);
		vector<int>::iterator it_i2 = find(vFeature[visibleFeatureID[i]].vFrame.begin(), 
										vFeature[visibleFeatureID[i]].vFrame.end(),
										initialFrame2);
		int i1 = (int) (it_i1 - vFeature[visibleFeatureID[i]].vFrame.begin());
		int i2 = (int) (it_i2 - vFeature[visibleFeatureID[i]].vFrame.begin());

		vFeature[visibleFeatureID[i]].vVisible_x.push_back(vFeature[visibleFeatureID[i]].vx[i1]);
		vFeature[visibleFeatureID[i]].vVisible_y.push_back(vFeature[visibleFeatureID[i]].vy[i1]);

		vFeature[visibleFeatureID[i]].vVisible_dis_x.push_back(vFeature[visibleFeatureID[i]].vx_dis[i1]);
		vFeature[visibleFeatureID[i]].vVisible_dis_y.push_back(vFeature[visibleFeatureID[i]].vy_dis[i1]);

		vFeature[visibleFeatureID[i]].vVisible_x.push_back(vFeature[visibleFeatureID[i]].vx[i2]);
		vFeature[visibleFeatureID[i]].vVisible_y.push_back(vFeature[visibleFeatureID[i]].vy[i2]);

		vFeature[visibleFeatureID[i]].vVisible_dis_x.push_back(vFeature[visibleFeatureID[i]].vx_dis[i2]);
		vFeature[visibleFeatureID[i]].vVisible_dis_y.push_back(vFeature[visibleFeatureID[i]].vy_dis[i2]);

	}

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K1);
	cvReleaseMat(&K2);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invK2);
	cvReleaseMat(&P0);	
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cvReleaseMat(&nx1);
	cvReleaseMat(&nx2);
	cvReleaseMat(&cX);
	return vInlierID.size();
}

int BilinearCameraPoseEstimation_OPENCV_mem_fast_AD(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, vector<Camera> vCamera, CvMat *P, CvMat *X, vector<vector<int> > &vvPointIndex)
{
	PrintAlgorithm("Bilinear Camera Pose Estimation");
	vector<int> visibleFeatureID;
	vector<vector<double> > cx1_vec, cx2_vec, nx1_vec, nx2_vec;

	VisibleIntersection_mem(vFeature, initialFrame1, initialFrame2, cx1_vec, cx2_vec, visibleFeatureID);
	CvMat *cx1 = cvCreateMat(cx1_vec.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(cx2_vec.size(), 2, CV_32FC1);
	SetCvMatFromVectors(cx1_vec, cx1);
	SetCvMatFromVectors(cx2_vec, cx2);

	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1,cx1->rows,CV_8UC1);
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS , 1, 0.99, status);
	PrintMat(F, "Fundamental Matrix");
	//for (int i = 0; i < cx1->rows; i++)
	//{
	//	if (cvGetReal2D(status, 0, i) == 1)
	//	{
	//		vInlierID.push_back(visibleFeatureID[i]);
	//	}
	//}

	cout << n << endl;

	vector<int> vCX_indx;
	for (int i = 0; i < cx1->rows; i++)
	{
		CvMat *xM2 = cvCreateMat(1,3,CV_32FC1);
		CvMat *xM1 = cvCreateMat(3,1,CV_32FC1);
		CvMat *s = cvCreateMat(1,1, CV_32FC1);
		cvSetReal2D(xM2, 0, 0, cvGetReal2D(cx2, i, 0));
		cvSetReal2D(xM2, 0, 1, cvGetReal2D(cx2, i, 1));
		cvSetReal2D(xM2, 0, 2, 1);
		cvSetReal2D(xM1, 0, 0, cvGetReal2D(cx1, i, 0));
		cvSetReal2D(xM1, 1, 0, cvGetReal2D(cx1, i, 1));
		cvSetReal2D(xM1, 2, 0, 1);
		cvMatMul(xM2, F, xM2);
		cvMatMul(xM2, xM1, s);			

		double l1 = cvGetReal2D(xM2, 0, 0);
		double l2 = cvGetReal2D(xM2, 0, 1);
		double l3 = cvGetReal2D(xM2, 0, 2);

		double dist = abs(cvGetReal2D(s, 0, 0))/sqrt(l1*l1+l2*l2);

		if (dist < 5)
		{
			vInlierID.push_back(visibleFeatureID[i]);
			vCX_indx.push_back(i);
		}

		cvReleaseMat(&xM2);
		cvReleaseMat(&xM1);
		cvReleaseMat(&s);
		//if (cvGetReal2D(status, 0, i) == 1)
		//{
		//	cvSetReal2D(tempCx1, temprow, 0, cvGetReal2D(cx1, i, 0));
		//	cvSetReal2D(tempCx1, temprow, 1, cvGetReal2D(cx1, i, 1));
		//	cvSetReal2D(tempCx2, temprow, 0, cvGetReal2D(cx2, i, 0));
		//	cvSetReal2D(tempCx2, temprow, 1, cvGetReal2D(cx2, i, 1));
		//	temprow++;
		//}
	}

	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierID.size(); iInlier++)
	{
		cvSetReal2D(tempCx1, iInlier, 0, cvGetReal2D(cx1, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx1, iInlier, 1, cvGetReal2D(cx1, vCX_indx[iInlier], 1));
		cvSetReal2D(tempCx2, iInlier, 0, cvGetReal2D(cx2, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx2, iInlier, 1, cvGetReal2D(cx2, vCX_indx[iInlier], 1));
	}

	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cx1 = cvCloneMat(tempCx1);
	cx2 = cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	//CvMat *K1 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *K2 = cvCreateMat(3,3,CV_32FC1);
	int camera1 = (int)((double)initialFrame1/max_nFrames);
	int camera2 = (int)((double)initialFrame2/max_nFrames);
	vector<int> ::const_iterator it1 = find(vCamera[camera1].vTakenFrame.begin(), vCamera[camera1].vTakenFrame.end(), initialFrame1%max_nFrames);
	vector<int> ::const_iterator it2 = find(vCamera[camera2].vTakenFrame.begin(), vCamera[camera2].vTakenFrame.end(), initialFrame2%max_nFrames);
	int idx1 = (int) (it1 - vCamera[camera1].vTakenFrame.begin());
	int idx2 = (int) (it2 - vCamera[camera2].vTakenFrame.begin());
	CvMat *K1 = cvCloneMat(vCamera[camera1].vK[idx1]);
	CvMat *K2 = cvCloneMat(vCamera[camera2].vK[idx2]);

	cvTranspose(K2, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K1, E);

	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K1, invK1);
	cvInvert(K2, invK2);
	CvMat *nx1 = cvCreateMat(cx1->rows, cx1->cols, CV_32FC1);
	CvMat *nx2 = cvCreateMat(cx2->rows, cx2->cols, CV_32FC1);
	Pxx_inhomo(invK1, cx1, nx1);
	Pxx_inhomo(invK2, cx2, nx2);

	GetExtrinsicParameterFromE(E, nx1, nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	
	CvMat *cX = cvCreateMat(nx1->rows, 3, CV_32FC1);

	PrintMat(P);
	cvMatMul(K1, P0, P0);
	cvMatMul(K2, P, P);

	PrintMat(P);
	//LinearTriangulation(nx1, P0, nx2, P, cX);
	LinearTriangulation(cx1, P0, cx2, P, cX);
	//PrintMat(cX);
	cvSetZero(X);
	SetIndexedMatRowwise(X, visibleFeatureID, cX);
	vvPointIndex.push_back(visibleFeatureID);
	vvPointIndex.push_back(visibleFeatureID);
	
	for (int i = 0; i < visibleFeatureID.size(); i++)
	{
		vFeature[visibleFeatureID[i]].isRegistered = true;
	}

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K1);
	cvReleaseMat(&K2);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invK2);
	cvReleaseMat(&P0);	
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cvReleaseMat(&nx1);
	cvReleaseMat(&nx2);
	cvReleaseMat(&cX);
	return vInlierID.size();
}

int BilinearCameraPoseEstimation_OPENCV_OrientationRefinement(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, int max_nFrames, 
															  vector<Camera> vCamera, CvMat *M, CvMat *m, vector<int> &vVisibleID)
{
	//PrintAlgorithm("Bilinear Camera Pose Estimation");
	vector<int> visibleFeatureID;
	vector<vector<double> > cx1_vec, cx2_vec, nx1_vec, nx2_vec;

	VisibleIntersection_mem(vFeature, initialFrame1, initialFrame2, cx1_vec, cx2_vec, visibleFeatureID);
	if (visibleFeatureID.size() < 40)
		return 0;
	CvMat *cx1 = cvCreateMat(cx1_vec.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(cx2_vec.size(), 2, CV_32FC1);
	SetCvMatFromVectors(cx1_vec, cx1);
	SetCvMatFromVectors(cx2_vec, cx2);

	if (visibleFeatureID.size() < 8)
	{
		return visibleFeatureID.size();
	}
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1,cx1->rows,CV_8UC1);
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS , 1, 0.99, status);

	//cout << n << endl;

	vector<int> vCX_indx;
	for (int i = 0; i < cx1->rows; i++)
	{
		CvMat *xM2 = cvCreateMat(1,3,CV_32FC1);
		CvMat *xM1 = cvCreateMat(3,1,CV_32FC1);
		CvMat *s = cvCreateMat(1,1, CV_32FC1);
		cvSetReal2D(xM2, 0, 0, cvGetReal2D(cx2, i, 0));
		cvSetReal2D(xM2, 0, 1, cvGetReal2D(cx2, i, 1));
		cvSetReal2D(xM2, 0, 2, 1);
		cvSetReal2D(xM1, 0, 0, cvGetReal2D(cx1, i, 0));
		cvSetReal2D(xM1, 1, 0, cvGetReal2D(cx1, i, 1));
		cvSetReal2D(xM1, 2, 0, 1);
		cvMatMul(xM2, F, xM2);
		cvMatMul(xM2, xM1, s);			

		double l1 = cvGetReal2D(xM2, 0, 0);
		double l2 = cvGetReal2D(xM2, 0, 1);
		double l3 = cvGetReal2D(xM2, 0, 2);

		double dist = abs(cvGetReal2D(s, 0, 0))/sqrt(l1*l1+l2*l2);

		if (dist < 5)
		{
			vInlierID.push_back(visibleFeatureID[i]);
			vCX_indx.push_back(i);
		}

		cvReleaseMat(&xM2);
		cvReleaseMat(&xM1);
		cvReleaseMat(&s);
	}

	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierID.size(); iInlier++)
	{
		cvSetReal2D(tempCx1, iInlier, 0, cvGetReal2D(cx1, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx1, iInlier, 1, cvGetReal2D(cx1, vCX_indx[iInlier], 1));
		cvSetReal2D(tempCx2, iInlier, 0, cvGetReal2D(cx2, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx2, iInlier, 1, cvGetReal2D(cx2, vCX_indx[iInlier], 1));
	}

	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cx1 = cvCloneMat(tempCx1);
	cx2 = cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	//CvMat *K1 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *K2 = cvCreateMat(3,3,CV_32FC1);
	int camera1 = (int)((double)initialFrame1/max_nFrames);
	int camera2 = (int)((double)initialFrame2/max_nFrames);
	vector<int> ::const_iterator it1 = find(vCamera[camera1].vTakenFrame.begin(), vCamera[camera1].vTakenFrame.end(), initialFrame1%max_nFrames);
	vector<int> ::const_iterator it2 = find(vCamera[camera2].vTakenFrame.begin(), vCamera[camera2].vTakenFrame.end(), initialFrame2%max_nFrames);
	int idx1 = (int) (it1 - vCamera[camera1].vTakenFrame.begin());
	int idx2 = (int) (it2 - vCamera[camera2].vTakenFrame.begin());
	CvMat *K1 = cvCloneMat(vCamera[camera1].vK[idx1]);
	CvMat *K2 = cvCloneMat(vCamera[camera2].vK[idx2]);

	cvTranspose(K2, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K1, E);

	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K1, invK1);
	cvInvert(K2, invK2);
	CvMat *nx1 = cvCreateMat(cx1->rows, cx1->cols, CV_32FC1);
	CvMat *nx2 = cvCreateMat(cx2->rows, cx2->cols, CV_32FC1);
	Pxx_inhomo(invK1, cx1, nx1);
	Pxx_inhomo(invK2, cx2, nx2);

	CvMat *P = cvCreateMat(3,4,CV_32FC1);
	GetExtrinsicParameterFromE(E, nx1, nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cvSetReal2D(M, i, j, cvGetReal2D(P, i, j));
		}
	}

	for (int i = 0; i < 3; i++)
	{
		cvSetReal2D(m, i, 0, cvGetReal2D(P, i, 3));
	}
	
	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K1);
	cvReleaseMat(&K2);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invK2);
	cvReleaseMat(&P0);	
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cvReleaseMat(&nx1);
	cvReleaseMat(&nx2);
	cvReleaseMat(&P);
	vVisibleID = vInlierID;
	return vInlierID.size();
}

void BilinearCameraPoseEstimation(vector<Feature> vFeature, int initialFrame1, int initialFrame2, double ransacThreshold, int ransacMaxIter, int max_nFrames, vector<Camera> vCamera, CvMat &P, CvMat &X, vector<int> &visibleStructureID)
{
	PrintAlgorithm("Bilinear Camera Pose Estimation");
	CvMat cx1, cx2, nx1, nx2;
	vector<int> visibleFeatureID;
	X = *cvCreateMat(vFeature.size(), 3, CV_32FC1);
	VisibleIntersection(vFeature, initialFrame1, initialFrame2, cx1, cx2, visibleFeatureID);
	cout << "Visible intersection between 2 and 3: " << visibleFeatureID.size() << endl;
	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	EightPointAlgorithm(&cx1, &cx2, F);
	ScalarMul(F, 1/cvGetReal2D(F, 2,2), F);
	PrintMat(F, "Fundamental Matrix");





















	//Classifier classifier;
	//vector<int> visibleID;
	//classifier.SetRansacParam(ransacThreshold, ransacMaxIter);
	//classifier.SetCorrespondance(&cx1, &cx2, visibleFeatureID);
	//classifier.Classify();
	//vector<int> vInlierID, vOutlierID;
	//classifier.GetClassificationResultByFeatureID(vInlierID, vOutlierID);
	//visibleFeatureID = vInlierID;
	//F = cvCloneMat(classifier.F);
	//double F33 = cvGetReal2D(F, 2,2);
	//ScalarMul(F, 1/F33, F);
	//PrintMat(F, "Fundamental Matrix");
	//cx1 = *cvCreateMat(classifier.inlier1->rows, classifier.inlier1->cols, CV_32FC1);
	//cx2 = *cvCreateMat(classifier.inlier2->rows, classifier.inlier2->cols, CV_32FC1);
	//cx1 = *cvCloneMat(classifier.inlier1);
	//cx2 = *cvCloneMat(classifier.inlier2);

	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *K1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *K2 = cvCreateMat(3,3,CV_32FC1);
	int camera1 = (int)((double)initialFrame1/max_nFrames);
	int camera2 = (int)((double)initialFrame2/max_nFrames);
	vector<int> ::const_iterator it1 = find(vCamera[camera1].vTakenFrame.begin(), vCamera[camera1].vTakenFrame.end(), initialFrame1%max_nFrames);
	vector<int> ::const_iterator it2 = find(vCamera[camera2].vTakenFrame.begin(), vCamera[camera2].vTakenFrame.end(), initialFrame2%max_nFrames);
	int idx1 = (int) (it1 - vCamera[camera1].vTakenFrame.begin());
	int idx2 = (int) (it2 - vCamera[camera2].vTakenFrame.begin());
	K1 = cvCloneMat(vCamera[camera1].vK[idx1]);
	K2 = cvCloneMat(vCamera[camera2].vK[idx1]);

	cvTranspose(K2, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K1, E);

	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K1, invK1);
	cvInvert(K2, invK2);
	Pxx_inhomo(invK1, &cx1, nx1);
	Pxx_inhomo(invK2, &cx2, nx2);

	GetExtrinsicParameterFromE(E, &nx1, &nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	CvMat cX;

	LinearTriangulation(&nx1, P0, &nx2, &P, cX);
	cvSetZero(&X);
	SetIndexedMatRowwise(&X, visibleFeatureID, &cX);
	cvMatMul(K2, &P, temp34);
	P = *cvCloneMat(temp34);

	visibleStructureID = visibleFeatureID;

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K1);
	cvReleaseMat(&K2);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invK2);
	cvReleaseMat(&P0);	
}

void EightPointAlgorithm(CvMat *x1_8, CvMat *x2_8, CvMat *F_8)
{
	CvMat *A = cvCreateMat(x1_8->rows, 9, CV_32FC1);
	CvMat *U = cvCreateMat(x1_8->rows, x1_8->rows, CV_32FC1);
	CvMat *D = cvCreateMat(x1_8->rows, 9, CV_32FC1);
	CvMat *V = cvCreateMat(9, 9, CV_32FC1);
	for (int iIdx = 0; iIdx < x1_8->rows; iIdx++)
	{
		double x11 = cvGetReal2D(x1_8, iIdx, 0);
		double x12 = cvGetReal2D(x1_8, iIdx, 1);
		double x21 = cvGetReal2D(x2_8, iIdx, 0);
		double x22 = cvGetReal2D(x2_8, iIdx, 1);
		cvSetReal2D(A, iIdx, 0, x21*x11);
		cvSetReal2D(A, iIdx, 1, x21*x12);
		cvSetReal2D(A, iIdx, 2, x21);
		cvSetReal2D(A, iIdx, 3, x22*x11);
		cvSetReal2D(A, iIdx, 4, x22*x12);
		cvSetReal2D(A, iIdx, 5, x22);
		cvSetReal2D(A, iIdx, 6, x11);
		cvSetReal2D(A, iIdx, 7, x12);
		cvSetReal2D(A, iIdx, 8, 1);
	}

	cvSVD(A, D, U, V, 0);
	cvSetReal2D(F_8, 0, 0, cvGetReal2D(V, 0, 8));	cvSetReal2D(F_8, 0, 1, cvGetReal2D(V, 1, 8));	cvSetReal2D(F_8, 0, 2, cvGetReal2D(V, 2, 8));
	cvSetReal2D(F_8, 1, 0, cvGetReal2D(V, 3, 8));	cvSetReal2D(F_8, 1, 1, cvGetReal2D(V, 4, 8));	cvSetReal2D(F_8, 1, 2, cvGetReal2D(V, 5, 8));
	cvSetReal2D(F_8, 2, 0, cvGetReal2D(V, 6, 8));	cvSetReal2D(F_8, 2, 1, cvGetReal2D(V, 7, 8));	cvSetReal2D(F_8, 2, 2, cvGetReal2D(V, 8, 8));

	CvMat *UD, *Vt;
	U = cvCreateMat(3, 3, CV_32FC1);
	D = cvCreateMat(3, 3, CV_32FC1);
	V = cvCreateMat(3, 3, CV_32FC1);
	UD = cvCreateMat(U->rows, D->cols, CV_32FC1);
	Vt = cvCreateMat(V->cols, V->rows, CV_32FC1);

	cvSVD(F_8, D, U, V, 0);
	cvSetReal2D(D, 2, 2, 0);

	cvMatMul(U, D, UD);
	cvTranspose(V, Vt);
	cvMatMul(UD, Vt, F_8);

	cvReleaseMat(&UD);
	cvReleaseMat(&Vt);
	cvReleaseMat(&A);
	cvReleaseMat(&U);
	cvReleaseMat(&D);
	cvReleaseMat(&V);
}




void VisibleIntersection(vector<Feature> vFeature, int frame1, int frame2, CvMat &cx1, CvMat &cx2, vector<int> &visibleFeatureID)
{
	vector<double> x1, y1, x2, y2;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != vFeature[iFeature].vFrame.end()))
		{
			int idx = int(it1-vFeature[iFeature].vFrame.begin());
			x1.push_back(vFeature[iFeature].vx[idx]);
			y1.push_back(vFeature[iFeature].vy[idx]);
			idx = int(it2-vFeature[iFeature].vFrame.begin());
			x2.push_back(vFeature[iFeature].vx[idx]);
			y2.push_back(vFeature[iFeature].vy[idx]);
			visibleFeatureID.push_back(vFeature[iFeature].id);
		}
	}

	cout << "# intersection: " << visibleFeatureID.size() << endl;
	cx1 = *cvCreateMat(x1.size(), 2, CV_32FC1);
	cx2 = *cvCreateMat(x1.size(), 2, CV_32FC1);
	for (int i = 0; i < x1.size(); i++)
	{
		cvSetReal2D(&cx1, i, 0, x1[i]);		cvSetReal2D(&cx1, i, 1, y1[i]);
		cvSetReal2D(&cx2, i, 0, x2[i]);		cvSetReal2D(&cx2, i, 1, y2[i]);
	}
}

void VisibleIntersection_mem(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleFeatureID)
{
	vector<double> x1, y1, x2, y2;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != vFeature[iFeature].vFrame.end()))
		{
			int idx = int(it1-vFeature[iFeature].vFrame.begin());
			x1.push_back(vFeature[iFeature].vx[idx]);
			y1.push_back(vFeature[iFeature].vy[idx]);
			idx = int(it2-vFeature[iFeature].vFrame.begin());
			x2.push_back(vFeature[iFeature].vx[idx]);
			y2.push_back(vFeature[iFeature].vy[idx]);
			visibleFeatureID.push_back(vFeature[iFeature].id);
		}
	}
	//cout << "# intersection: " << visibleFeatureID.size() << endl;
	for (int i = 0; i < x1.size(); i++)
	{
		vector<double> x1_vec, x2_vec;
		x1_vec.push_back(x1[i]);
		x1_vec.push_back(y1[i]);

		x2_vec.push_back(x2[i]);
		x2_vec.push_back(y2[i]);

		cx1.push_back(x1_vec);
		cx2.push_back(x2_vec);
	}
}

void VisibleIntersection_Simple(vector<Feature> vFeature, int frame1, int frame2, vector<int> &visibleFeatureID)
{
	vector<double> x1, y1, x2, y2;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != vFeature[iFeature].vFrame.end()))
		{
			visibleFeatureID.push_back(vFeature[iFeature].id);
		}
	}
}

int VisibleIntersection23(vector<Feature> vFeature, int frame1, CvMat *X, vector<int> visibleStructureID, CvMat &cx, CvMat &cX, vector<int> &visibleID)
{
	vector<double> x_, y_, X_, Y_, Z_;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		vector<int>::iterator it2 = find(visibleStructureID.begin(),visibleStructureID.end(),vFeature[iFeature].id);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != visibleStructureID.end()))
		{
			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
			int idx2 = int(it2-visibleStructureID.begin());
			x_.push_back(vFeature[iFeature].vx[idx1]);
			y_.push_back(vFeature[iFeature].vy[idx1]);
			X_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 0));
			Y_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 1));
			Z_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 2));
			visibleID.push_back(vFeature[iFeature].id);
		}
	}

	if (x_.size() < 1)
		return 0;
	cx = *cvCreateMat(x_.size(), 2, CV_32FC1);
	cX = *cvCreateMat(x_.size(), 3, CV_32FC1);
	for (int i = 0; i < x_.size(); i++)
	{
		cvSetReal2D(&cx, i, 0, x_[i]);		cvSetReal2D(&cx, i, 1, y_[i]);
		cvSetReal2D(&cX, i, 0, X_[i]);		cvSetReal2D(&cX, i, 1, Y_[i]);		cvSetReal2D(&cX, i, 2, Z_[i]); 
	}
	return 1;
}

int VisibleIntersection23_mem(vector<Feature> vFeature, int frame1, CvMat *X, vector<int> visibleStructureID, vector<vector<double> > &cx, vector<vector<double> > &cX, vector<int> &visibleID)
{
	vector<double> x_, y_, X_, Y_, Z_;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		vector<int>::iterator it2 = find(visibleStructureID.begin(),visibleStructureID.end(),vFeature[iFeature].id);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != visibleStructureID.end()))
		{
			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
			int idx2 = int(it2-visibleStructureID.begin());
			x_.push_back(vFeature[iFeature].vx[idx1]);
			y_.push_back(vFeature[iFeature].vy[idx1]);
			X_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 0));
			Y_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 1));
			Z_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 2));
			visibleID.push_back(vFeature[iFeature].id);
		}
	}

	if (x_.size() < 1)
		return 0;
	for (int i = 0; i < x_.size(); i++)
	{
		vector<double> cx_vec, cX_vec;
		cx_vec.push_back(x_[i]);
		cx_vec.push_back(y_[i]);

		cX_vec.push_back(X_[i]);
		cX_vec.push_back(Y_[i]);
		cX_vec.push_back(Z_[i]);

		cx.push_back(cx_vec);
		cX.push_back(cX_vec);
	}
	return visibleID.size();
}

int VisibleIntersection23_mem_fast(vector<Feature> &vFeature, int frame1, CvMat *X, vector<vector<double> > &cx, vector<vector<double> > &cX, vector<int> &visibleID)
{
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<double> cx_vec, cX_vec;
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		if ((it1 != vFeature[iFeature].vFrame.end()) && (vFeature[iFeature].isRegistered))
		{
			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());

			cx_vec.push_back(vFeature[iFeature].vx[idx1]);
			cx_vec.push_back(vFeature[iFeature].vy[idx1]);

			cX_vec.push_back(cvGetReal2D(X, vFeature[iFeature].id, 0));
			cX_vec.push_back(cvGetReal2D(X, vFeature[iFeature].id, 1));
			cX_vec.push_back(cvGetReal2D(X, vFeature[iFeature].id, 2));

			cx.push_back(cx_vec);
			cX.push_back(cX_vec);

			visibleID.push_back(vFeature[iFeature].id);
		}
	}

	if (cx.size() < 1)
		return 0;
	return visibleID.size();
}

int VisibleIntersection23_Simple(vector<Feature> &vFeature, int frame1, vector<int> visibleStructureID, vector<int> &visibleID)
{
	vector<double> x_, y_, X_, Y_, Z_;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		if (it1 == vFeature[iFeature].vFrame.end())
			continue;

		vector<int>::iterator it2 = find(visibleStructureID.begin(),visibleStructureID.end(),vFeature[iFeature].id);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != visibleStructureID.end()))
		{
			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
			int idx2 = int(it2-visibleStructureID.begin());
			visibleID.push_back(vFeature[iFeature].id);
		}
	}
	x_.clear();
	y_.clear();
	X_.clear();
	Y_.clear();
	Z_.clear();
	return visibleID.size();
}

int VisibleIntersection23_Simple_fast(vector<Feature> &vFeature, int frame1)
{
	int count = 0;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		if (it1 == vFeature[iFeature].vFrame.end())
			continue;
		else if (vFeature[iFeature].isRegistered)
			count++;
	}
	return count;
}

//int VisibleIntersection23(vector<Feature> vFeature, int frame1, CvMat *X, vector<int> visibleStructureID, CvMat *x, vector<int> &visibleID)
//{
//	vector<double> x_, y_, X_, Y_, Z_;
//	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
//	{
//		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
//		vector<int>::iterator it2 = find(visibleStructureID.begin(),visibleStructureID.end(),vFeature[iFeature].id);
//
//		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != visibleStructureID.end()))
//		{
//			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
//			int idx2 = int(it2-visibleStructureID.begin());
//			x_.push_back(vFeature[iFeature].vx[idx1]);
//			y_.push_back(vFeature[iFeature].vy[idx1]);
//			X_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 0));
//			Y_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 1));
//			Z_.push_back(cvGetReal2D(X, vFeature[iFeature].id, 2));
//			visibleID.push_back(vFeature[iFeature].id);
//		}
//	}
//
//	if (x_.size() < 1)
//		return 0;
//	cx = *cvCreateMat(x_.size(), 2, CV_32FC1);
//	cX = *cvCreateMat(x_.size(), 3, CV_32FC1);
//	for (int i = 0; i < x_.size(); i++)
//	{
//		cvSetReal2D(&cx, i, 0, x_[i]);		cvSetReal2D(&cx, i, 1, y_[i]);
//		cvSetReal2D(&cX, i, 0, X_[i]);		cvSetReal2D(&cX, i, 1, Y_[i]);		cvSetReal2D(&cX, i, 2, Z_[i]); 
//	}
//	return 1;
//}

int VisibleIntersectionXOR3(vector<Feature> vFeature, int frame1, int frame2, vector<int> visibleStructureID, CvMat &cx1, CvMat &cx2, vector<int> &visibleID)
{
	vector<double> x1_, y1_, x2_, y2_;
	visibleID.clear();
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);
		vector<int>::iterator it3 = find(visibleStructureID.begin(),visibleStructureID.end(),vFeature[iFeature].id);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != vFeature[iFeature].vFrame.end()) && (it3 == visibleStructureID.end()))
		{
			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
			int idx2 = int(it2-vFeature[iFeature].vFrame.begin());
			x1_.push_back(vFeature[iFeature].vx[idx1]);
			y1_.push_back(vFeature[iFeature].vy[idx1]);
			x2_.push_back(vFeature[iFeature].vx[idx2]);
			y2_.push_back(vFeature[iFeature].vy[idx2]);
			visibleID.push_back(vFeature[iFeature].id);
		}
	}

	if (x1_.size() == 0)
	{
		visibleID.clear();
		return 0;
	}

	cx1 = *cvCreateMat(x1_.size(), 2, CV_32FC1);
	cx2 = *cvCreateMat(x1_.size(), 2, CV_32FC1);
	for (int i = 0; i < x1_.size(); i++)
	{
		cvSetReal2D(&cx1, i, 0, x1_[i]);		cvSetReal2D(&cx1, i, 1, y1_[i]);
		cvSetReal2D(&cx2, i, 0, x2_[i]);		cvSetReal2D(&cx2, i, 1, y2_[i]);
	}
	return 1;
}

int VisibleIntersectionXOR3_mem(vector<Feature> &vFeature, int frame1, int frame2, vector<int> visibleStructureID, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID)
{
	vector<double> x1_, y1_, x2_, y2_;
	visibleID.clear();
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);
		vector<int>::iterator it3 = find(visibleStructureID.begin(),visibleStructureID.end(),vFeature[iFeature].id);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != vFeature[iFeature].vFrame.end()) && (it3 == visibleStructureID.end()))
		{
			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
			int idx2 = int(it2-vFeature[iFeature].vFrame.begin());
			x1_.push_back(vFeature[iFeature].vx[idx1]);
			y1_.push_back(vFeature[iFeature].vy[idx1]);
			x2_.push_back(vFeature[iFeature].vx[idx2]);
			y2_.push_back(vFeature[iFeature].vy[idx2]);
			visibleID.push_back(vFeature[iFeature].id);
		}
	}

	if (x1_.size() == 0)
	{
		visibleID.clear();
		return 0;
	}
	for (int i = 0; i < x1_.size(); i++)
	{
		vector<double> cx1_vec, cx2_vec;
		cx1_vec.push_back(x1_[i]);
		cx1_vec.push_back(y1_[i]);

		cx2_vec.push_back(x2_[i]);
		cx2_vec.push_back(y2_[i]);

		cx1.push_back(cx1_vec);
		cx2.push_back(cx2_vec);
	}
	return cx1.size();
}

int VisibleIntersectionXOR3_mem_fast(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID)
{
	visibleID.clear();
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
			continue;
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		if (it1 == vFeature[iFeature].vFrame.end())
			continue;
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);
		if (it2 == vFeature[iFeature].vFrame.end())
			continue;

		vector<double> cx1_vec, cx2_vec;

		int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
		int idx2 = int(it2-vFeature[iFeature].vFrame.begin());

		cx1_vec.push_back(vFeature[iFeature].vx[idx1]);
		cx1_vec.push_back(vFeature[iFeature].vy[idx1]);

		cx2_vec.push_back(vFeature[iFeature].vx[idx2]);
		cx2_vec.push_back(vFeature[iFeature].vy[idx2]);

		cx1.push_back(cx1_vec);
		cx2.push_back(cx2_vec);
		visibleID.push_back(vFeature[iFeature].id);
	}

	if (visibleID.size() == 0)
	{
		return 0;
	}

	return cx1.size();
}

int VisibleIntersectionXOR3_mem_fast(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID,
		double omega1, double omega2, double px1, double px2, double py1, double py2)
{
	visibleID.clear();
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
			continue;
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		if (it1 == vFeature[iFeature].vFrame.end())
			continue;
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);
		if (it2 == vFeature[iFeature].vFrame.end())
			continue;

		vector<double> cx1_vec, cx2_vec;

		int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
		int idx2 = int(it2-vFeature[iFeature].vFrame.begin());

		double x1 = vFeature[iFeature].vx_dis[idx1];
		double y1 = vFeature[iFeature].vy_dis[idx1];

		double x = x1-px1;
		double y = y1-py1;
		double r_d = sqrt(x*x+y*y);
		double r_u = tan(r_d*omega1)/2/tan(omega1/2); 
		double x_u = r_u/r_d*x;
		double y_u = r_u/r_d*y;
		x1 = x_u+px1;
		y1 = y_u+py1;

		double x2 = vFeature[iFeature].vx_dis[idx2];
		double y2 = vFeature[iFeature].vy_dis[idx2];

		x = x2-px2;
		y = y2-py2;
		r_d = sqrt(x*x+y*y);
		r_u = tan(r_d*omega2)/2/tan(omega2/2); 
		x_u = r_u/r_d*x;
		y_u = r_u/r_d*y;
		x2 = x_u+px2;
		y2 = y_u+py2;

		cx1_vec.push_back(x1);
		cx1_vec.push_back(y1);

		cx2_vec.push_back(x2);
		cx2_vec.push_back(y2);

		cx1.push_back(cx1_vec);
		cx2.push_back(cx2_vec);
		visibleID.push_back(vFeature[iFeature].id);
	}

	if (visibleID.size() == 0)
	{
		return 0;
	}

	return cx1.size();
}

void Undistort_iPhone(double fx, double fy, double px, double py, double k1, double u, double v, double &u1, double &v1)
{
	double nx = (u-px)/fx;
	double ny = (v-py)/fy;
	const double t2 = ny*ny; 
    const double t3 = t2*t2*t2; 
    const double t4 = nx*nx; 
    const double t7 = k1*(t2+t4); 

	const double t9 = t3/(t7*t7*4.0); 
    const double t11 = t3/(t7*t7*t7*27.0); 
    const std::complex<double> t12 = t9+t11; 
    const std::complex<double> t13 = sqrt(t12); 
    const double t14 = t2/t7; 
    const double t15 = t14*ny*0.5; 
    const std::complex<double> t16 = t13+t15; 
    const std::complex<double> t17 = pow(t16,1.0/3.0); 
    const std::complex<double> t18 = (t17+t14/(t17*3.0))*std::complex<double>(0.0,sqrt(3.0)); 
    const std::complex<double> t19 = -0.5*(t17+t18)+t14/(t17*6.0); 
		
	double ux = t19.real()*nx/ny;
	double uy = t19.real();

	u1 = ux*fx+px;
	v1 = uy*fy+py;
}

int VisibleIntersectionXOR3_mem_fast_iPhone(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID,
		CvMat *K1, CvMat *K2, double k11, double k12)
{
	visibleID.clear();
	double fx1, fy1, fx2, fy2, px1, py1, px2, py2;
	fx1 = cvGetReal2D(K1, 0, 0);
	fy1 = cvGetReal2D(K1, 1, 1);
	px1 = cvGetReal2D(K1, 0, 2);
	py1 = cvGetReal2D(K1, 1, 2);
	fx2 = cvGetReal2D(K2, 0, 0);
	fy2 = cvGetReal2D(K2, 1, 1);
	px2 = cvGetReal2D(K2, 0, 2);
	py2 = cvGetReal2D(K2, 1, 2);
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
			continue;
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
		if (it1 == vFeature[iFeature].vFrame.end())
			continue;
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);
		if (it2 == vFeature[iFeature].vFrame.end())
			continue;

		vector<double> cx1_vec, cx2_vec;

		int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
		int idx2 = int(it2-vFeature[iFeature].vFrame.begin());

		double x1 = vFeature[iFeature].vx_dis[idx1];
		double y1 = vFeature[iFeature].vy_dis[idx1];

		double x_u1, y_u1;
		Undistort_iPhone(fx1, fy1, px1, py1, k11, x1, y1, x_u1, y_u1);

		double x2 = vFeature[iFeature].vx_dis[idx2];
		double y2 = vFeature[iFeature].vy_dis[idx2];

		double x_u2, y_u2;
		Undistort_iPhone(fx2, fy2, px2, py2, k12, x2, y2, x_u2, y_u2);

		cx1_vec.push_back(x_u1);
		cx1_vec.push_back(y_u1);

		cx2_vec.push_back(x_u2);
		cx2_vec.push_back(y_u2);

		cx1.push_back(cx1_vec);
		cx2.push_back(cx2_vec);
		visibleID.push_back(vFeature[iFeature].id);
	}

	if (visibleID.size() == 0)
	{
		return 0;
	}

	return cx1.size();
}


//int VisibleIntersectionXOR3(vector<Feature> vFeature, int frame1, int frame2, vector<int> visibleStructureID, vector<int> &visibleID)
//{
//	vector<double> x1_, y1_, x2_, y2_;
//	visibleID.clear();
//	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
//	{
//		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame1);
//		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),frame2);
//		vector<int>::iterator it3 = find(visibleStructureID.begin(),visibleStructureID.end(),vFeature[iFeature].id);
//
//		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != vFeature[iFeature].vFrame.end()) && (it3 == visibleStructureID.end()))
//		{
//			int idx1 = int(it1-vFeature[iFeature].vFrame.begin());
//			int idx2 = int(it2-vFeature[iFeature].vFrame.begin());
//			x1_.push_back(vFeature[iFeature].vx[idx1]);
//			y1_.push_back(vFeature[iFeature].vy[idx1]);
//			x2_.push_back(vFeature[iFeature].vx[idx2]);
//			y2_.push_back(vFeature[iFeature].vy[idx2]);
//			visibleID.push_back(vFeature[iFeature].id);
//		}
//	}
//
//	if (x1_.size() == 0)
//	{
//		visibleID.clear();
//		return 0;
//	}
//
//	cx1 = *cvCreateMat(x1_.size(), 2, CV_32FC1);
//	cx2 = *cvCreateMat(x1_.size(), 2, CV_32FC1);
//	for (int i = 0; i < x1_.size(); i++)
//	{
//		cvSetReal2D(&cx1, i, 0, x1_[i]);		cvSetReal2D(&cx1, i, 1, y1_[i]);
//		cvSetReal2D(&cx2, i, 0, x2_[i]);		cvSetReal2D(&cx2, i, 1, y2_[i]);
//	}
//	return 1;
//}

//int ExcludeOutliers(CvMat *cx1, CvMat *cx2, double ransacThreshold, double ransacMaxIter, vector<int> visibleID, CvMat &ex1, CvMat &ex2, vector<int> &eVisibleID)
//{
//	Classifier classifier;
//	classifier.SetRansacParam(ransacThreshold, ransacMaxIter);
//	classifier.SetCorrespondance(cx1, cx2, visibleID);
//	classifier.Classify();
//	vector<int> vInlierID, vOutlierID;
//	classifier.GetClassificationResultByFeatureID(vInlierID, vOutlierID);
//
//	if (vInlierID.size() > 0)
//	{
//		ex1 = *cvCreateMat(classifier.inlier1->rows, classifier.inlier1->cols, CV_32FC1);
//		ex2 = *cvCreateMat(classifier.inlier1->rows, classifier.inlier1->cols, CV_32FC1);
//		eVisibleID = vInlierID;
//		ex1 = *cvCloneMat(classifier.inlier1);
//		ex2 = *cvCloneMat(classifier.inlier2);
//		return 1;
//	}
//	else
//	{
//		return 0;
//	}
//}

int ExcludeOutliers(CvMat *cx1, CvMat *P1, CvMat *cx2, CvMat *P2, CvMat *K, double threshold, vector<int> visibleID, CvMat &ex1, CvMat &ex2, vector<int> &eVisibleID)
{
	// Find epipole
	CvMat *e_homo = cvCreateMat(3,1,CV_32FC1);
	CvMat *C = cvCreateMat(3,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);

	GetCameraParameter(P1, K, R, C);
	CvMat *C_homo = cvCreateMat(4,1,CV_32FC1);
	Inhomo2HomoVec(C, C_homo);
	cvMatMul(P2, C_homo, e_homo);

	double enorm = NormL2(e_homo);
	ScalarMul(e_homo, 1/enorm, e_homo);
	CvMat *pinvP1 = cvCreateMat(4,3,CV_32FC1);
	cvInvert(P1, pinvP1, CV_SVD);
	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	cvMatMul(P2, pinvP1, temp33);
	CvMat *skewE = cvCreateMat(3,3,CV_32FC1);

	Vec2Skew(e_homo, skewE);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	CvMat *FF = cvCreateMat(3,3,CV_32FC1);
	cvMatMul(skewE, temp33, F);
	double Fnorm = NormL2(F);
	ScalarMul(F, 1/Fnorm, F);
	cvTranspose(F, temp33);
	cvMatMul(temp33, F, FF);

	CvMat *D1=cvCreateMat(cx1->rows,1,CV_32FC1), *D2=cvCreateMat(cx1->rows,1,CV_32FC1), *D=cvCreateMat(cx1->rows,1,CV_32FC1);
	xPy_inhomo(cx2, cx1, F, D1);
	xPx_inhomo(cx1, FF, D2);
	for (int iIdx = 0; iIdx < D2->rows; iIdx++)
	{
		cvSetReal2D(D2, iIdx, 0, sqrt(cvGetReal2D(D2, iIdx, 0)));
	}
	cvDiv(D1, D2, D);
	cvMul(D, D, D);

	eVisibleID.clear();
	for (int iIdx = 0; iIdx < cx1->rows; iIdx++)
	{
		if (abs(cvGetReal2D(D, iIdx, 0)) < threshold)
		{
			eVisibleID.push_back(visibleID[iIdx]);	
		}
	}

	if (eVisibleID.size() > 0)
	{
		ex1 = *cvCreateMat(eVisibleID.size(), 2, CV_32FC1);
		ex2 = *cvCreateMat(eVisibleID.size(), 2, CV_32FC1);
		int k = 0;
		for (int iIdx = 0; iIdx < cx1->rows; iIdx++)
		{
			if (abs(cvGetReal2D(D, iIdx, 0)) < threshold)
			{
				cvSetReal2D(&ex1, k, 0, cvGetReal2D(cx1, iIdx, 0));
				cvSetReal2D(&ex1, k, 1, cvGetReal2D(cx1, iIdx, 1));
				cvSetReal2D(&ex2, k, 0, cvGetReal2D(cx2, iIdx, 0));
				cvSetReal2D(&ex2, k, 1, cvGetReal2D(cx2, iIdx, 1));
				k++;
			}
		}
	}

	cvReleaseMat(&e_homo);
	cvReleaseMat(&C);
	cvReleaseMat(&R);
	cvReleaseMat(&C_homo);
	cvReleaseMat(&pinvP1);
	cvReleaseMat(&temp33);
	cvReleaseMat(&skewE);
	cvReleaseMat(&F);
	cvReleaseMat(&FF);
	cvReleaseMat(&D1);
	cvReleaseMat(&D2);
	cvReleaseMat(&D);

	if (eVisibleID.size() >0)
		return 1;
	else 
		return 0;

}

int ExcludeOutliers_mem(CvMat *cx1, CvMat *P1, CvMat *cx2, CvMat *P2, CvMat *K, double threshold, vector<int> visibleID, vector<vector<double> > &ex1, vector<vector<double> > &ex2, vector<int> &eVisibleID)
{
	// Find epipole
	CvMat *e_homo = cvCreateMat(3,1,CV_32FC1);
	CvMat *C = cvCreateMat(3,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);

	GetCameraParameter(P1, K, R, C);
	CvMat *C_homo = cvCreateMat(4,1,CV_32FC1);
	Inhomo2HomoVec(C, C_homo);
	cvMatMul(P2, C_homo, e_homo);

	double enorm = NormL2(e_homo);
	ScalarMul(e_homo, 1/enorm, e_homo);
	CvMat *pinvP1 = cvCreateMat(4,3,CV_32FC1);
	cvInvert(P1, pinvP1, CV_SVD);
	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	cvMatMul(P2, pinvP1, temp33);
	CvMat *skewE = cvCreateMat(3,3,CV_32FC1);

	Vec2Skew(e_homo, skewE);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	CvMat *FF = cvCreateMat(3,3,CV_32FC1);
	cvMatMul(skewE, temp33, F);
	double Fnorm = NormL2(F);
	ScalarMul(F, 1/Fnorm, F);
	cvTranspose(F, temp33);
	cvMatMul(temp33, F, FF);

	CvMat *D1=cvCreateMat(cx1->rows,1,CV_32FC1), *D2=cvCreateMat(cx1->rows,1,CV_32FC1), *D=cvCreateMat(cx1->rows,1,CV_32FC1);
	xPy_inhomo(cx2, cx1, F, D1);
	xPx_inhomo(cx1, FF, D2);
	for (int iIdx = 0; iIdx < D2->rows; iIdx++)
	{
		cvSetReal2D(D2, iIdx, 0, sqrt(cvGetReal2D(D2, iIdx, 0)));
	}
	cvDiv(D1, D2, D);
	cvMul(D, D, D);

	eVisibleID.clear();
	for (int iIdx = 0; iIdx < cx1->rows; iIdx++)
	{
		if (abs(cvGetReal2D(D, iIdx, 0)) < threshold)
		{
			eVisibleID.push_back(visibleID[iIdx]);	
		}
	}

	if (eVisibleID.size() > 0)
	{
		//ex1 = *cvCreateMat(eVisibleID.size(), 2, CV_32FC1);
		//ex2 = *cvCreateMat(eVisibleID.size(), 2, CV_32FC1);
		int k = 0;
		for (int iIdx = 0; iIdx < cx1->rows; iIdx++)
		{
			if (abs(cvGetReal2D(D, iIdx, 0)) < threshold)
			{
				vector<double> ex1_vec, ex2_vec;
				ex1_vec.push_back(cvGetReal2D(cx1, iIdx, 0));
				ex1_vec.push_back(cvGetReal2D(cx1, iIdx, 1));

				ex2_vec.push_back(cvGetReal2D(cx2, iIdx, 0));
				ex2_vec.push_back(cvGetReal2D(cx2, iIdx, 1));

				ex1.push_back(ex1_vec);
				ex2.push_back(ex2_vec);
				k++;
			}
		}
	}

	cvReleaseMat(&e_homo);
	cvReleaseMat(&C);
	cvReleaseMat(&R);
	cvReleaseMat(&C_homo);
	cvReleaseMat(&pinvP1);
	cvReleaseMat(&temp33);
	cvReleaseMat(&skewE);
	cvReleaseMat(&F);
	cvReleaseMat(&FF);
	cvReleaseMat(&D1);
	cvReleaseMat(&D2);
	cvReleaseMat(&D);

	if (eVisibleID.size() >0)
		return 1;
	else 
		return 0;

}

int ExcludeOutliers_mem_fast(CvMat *cx1, CvMat *P1, CvMat *cx2, CvMat *P2, CvMat *K, double threshold, vector<int> visibleID, vector<vector<double> > &ex1, vector<vector<double> > &ex2, vector<int> &eVisibleID)
{
	// Find epipole
	CvMat *e_homo = cvCreateMat(3,1,CV_32FC1);
	CvMat *C = cvCreateMat(3,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);

	GetCameraParameter(P1, K, R, C);
	CvMat *C_homo = cvCreateMat(4,1,CV_32FC1);
	Inhomo2HomoVec(C, C_homo);
	cvMatMul(P2, C_homo, e_homo);

	double enorm = NormL2(e_homo);
	ScalarMul(e_homo, 1/enorm, e_homo);
	CvMat *pinvP1 = cvCreateMat(4,3,CV_32FC1);
	cvInvert(P1, pinvP1, CV_SVD);
	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	cvMatMul(P2, pinvP1, temp33);
	CvMat *skewE = cvCreateMat(3,3,CV_32FC1);

	Vec2Skew(e_homo, skewE);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	CvMat *FF = cvCreateMat(3,3,CV_32FC1);
	cvMatMul(skewE, temp33, F);
	double Fnorm = NormL2(F);
	ScalarMul(F, 1/Fnorm, F);
	cvTranspose(F, temp33);
	cvMatMul(temp33, F, FF);

	//CvMat *D1=cvCreateMat(cx1->rows,1,CV_32FC1), *D2=cvCreateMat(cx1->rows,1,CV_32FC1), *D=cvCreateMat(cx1->rows,1,CV_32FC1);
	//xPy_inhomo(cx2, cx1, F, D1);
	//xPx_inhomo(cx1, FF, D2);
	//for (int iIdx = 0; iIdx < D2->rows; iIdx++)
	//{
	//	cvSetReal2D(D2, iIdx, 0, sqrt(cvGetReal2D(D2, iIdx, 0)));
	//}
	//cvDiv(D1, D2, D);
	//cvMul(D, D, D);

	eVisibleID.clear();
	for (int iIdx = 0; iIdx < cx1->rows; iIdx++)
	{
		CvMat *xM2 = cvCreateMat(1, 3, CV_32FC1);
		CvMat *xM1 = cvCreateMat(3, 1, CV_32FC1);
		CvMat *s = cvCreateMat(1, 1, CV_32FC1);

		cvSetReal2D(xM2, 0, 0, cvGetReal2D(cx2, iIdx, 0));
		cvSetReal2D(xM2, 0, 1, cvGetReal2D(cx2, iIdx, 1));
		cvSetReal2D(xM2, 0, 2, 1);
		cvSetReal2D(xM1, 0, 0, cvGetReal2D(cx1, iIdx, 0));
		cvSetReal2D(xM1, 1, 0, cvGetReal2D(cx2, iIdx, 1));
		cvSetReal2D(xM1, 2, 0, 1);
		cvMatMul(xM2, F, xM2);
		cvMatMul(xM2, xM1, s);			

		double l1 = cvGetReal2D(xM2, 0, 0);
		double l2 = cvGetReal2D(xM2, 0, 1);
		double l3 = cvGetReal2D(xM2, 0, 2);

		double dist = abs(cvGetReal2D(s, 0, 0))/sqrt(l1*l1+l2*l2);
		cout << dist << endl;

		if (abs(dist) < threshold)
		{
			vector<double> ex1_vec, ex2_vec;
			ex1_vec.push_back(cvGetReal2D(cx1, iIdx, 0));
			ex1_vec.push_back(cvGetReal2D(cx1, iIdx, 1));

			ex2_vec.push_back(cvGetReal2D(cx2, iIdx, 0));
			ex2_vec.push_back(cvGetReal2D(cx2, iIdx, 1));

			ex1.push_back(ex1_vec);
			ex2.push_back(ex2_vec);
			eVisibleID.push_back(visibleID[iIdx]);	
		}

		cvReleaseMat(&xM2);
		cvReleaseMat(&xM1);
		cvReleaseMat(&s);
	}

	cvReleaseMat(&e_homo);
	cvReleaseMat(&C);
	cvReleaseMat(&R);
	cvReleaseMat(&C_homo);
	cvReleaseMat(&pinvP1);
	cvReleaseMat(&temp33);
	cvReleaseMat(&skewE);
	cvReleaseMat(&F);
	cvReleaseMat(&FF);
	//cvReleaseMat(&D1);
	//cvReleaseMat(&D2);
	//cvReleaseMat(&D);

	if (eVisibleID.size() > 0)
		return 1;
	else 
		return 0;

}



void NonlinearTriangulation(CvMat *x1, CvMat *x2, CvMat *F, CvMat &xhat1, CvMat &xhat2)
{
	for (int ix = 0; ix < x1->rows; ix++)
	{
		CvMat *T1 = cvCreateMat(3,3,CV_32FC1);
		CvMat *T2 = cvCreateMat(3,3,CV_32FC1);
		cvSetZero(T1);
		cvSetReal2D(T1, 0, 0, 1.0);
		cvSetReal2D(T1, 1, 1, 1.0);
		cvSetReal2D(T1, 2, 2, 1.0);

		cvSetReal2D(T1, 0, 2, -cvGetReal2D(x1, ix, 0));
		cvSetReal2D(T1, 1, 2, -cvGetReal2D(x1, ix, 1));
		
		cvSetZero(T2);
		cvSetReal2D(T2, 0, 0, 1.0);
		cvSetReal2D(T2, 1, 1, 1.0);
		cvSetReal2D(T2, 2, 2, 1.0);
		cvSetReal2D(T2, 0, 2, -cvGetReal2D(x2, ix, 0));
		cvSetReal2D(T2, 1, 2, -cvGetReal2D(x2, ix, 1));
		CvMat *nF = cvCreateMat(3,3,CV_32FC1);
		nF = cvCloneMat(F);
		CvMat *FinvT1 = cvCreateMat(3,3,CV_32FC1);
		CvMat *invT1 = cvCreateMat(3,3,CV_32FC1);
		CvMat *invT2 = cvCreateMat(3,3,CV_32FC1);
		CvMat *invT2t = cvCreateMat(3,3,CV_32FC1);

		cvInvert(T1, invT1);	cvInvert(T2, invT2);	cvTranspose(invT2, invT2t);
		cvMatMul(nF, invT1, FinvT1);
		cvMatMul(invT2t, FinvT1, nF);
		CvMat *U = cvCreateMat(3,3,CV_32FC1);
		CvMat *D = cvCreateMat(3,3,CV_32FC1);
		CvMat *Vt = cvCreateMat(3,3,CV_32FC1);
		cvSVD(nF, D, U, Vt, CV_SVD_V_T);		cvSetReal2D(D, 2, 2, 0);
		CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);	CvMat *temp33_1 = cvCreateMat(3,3,CV_32FC1);
		cvMatMul(U, D, temp33);
		cvMatMul(temp33, Vt, nF);
	
		CvMat *e1 = cvCreateMat(3,1,CV_32FC1);
		CvMat *e2 = cvCreateMat(3,1,CV_32FC1);
		double e11 = cvGetReal2D(Vt,2,0);	double e12 = cvGetReal2D(Vt,2,1);	double e13 = cvGetReal2D(Vt,2,2);
		CvMat *nFt = cvCreateMat(3,3,CV_32FC1);
		cvTranspose(nF, nFt);
		cvSVD(nFt, D, U, Vt, CV_SVD_V_T);
		double norm_e1 = sqrt(e11*e11+e12*e12);
		double e21 = cvGetReal2D(Vt,2,0);	double e22 = cvGetReal2D(Vt,2,1);	double e23 = cvGetReal2D(Vt,2,2);
		double norm_e2 = sqrt(e21*e21+e22*e22);
		cvSetReal2D(e1,0,0,e11/norm_e1);	cvSetReal2D(e1,1,0,e12/norm_e1);	cvSetReal2D(e1,2,0,e13/norm_e1);	
		cvSetReal2D(e2,0,0,e21/norm_e2);	cvSetReal2D(e2,1,0,e22/norm_e2);	cvSetReal2D(e2,2,0,e23/norm_e2);	

		CvMat *R1 = cvCreateMat(3,3,CV_32FC1);
		CvMat *R2 = cvCreateMat(3,3,CV_32FC1);
		cvSetIdentity(R1);		cvSetIdentity(R2);
		cvSetReal2D(R1, 0, 0, cvGetReal2D(e1, 0, 0));	cvSetReal2D(R1, 0, 1, cvGetReal2D(e1, 1, 0));	
		cvSetReal2D(R1, 1, 0, -cvGetReal2D(e1, 1, 0));	cvSetReal2D(R1, 1, 1, cvGetReal2D(e1, 0, 0));		
		cvSetReal2D(R2, 0, 0, cvGetReal2D(e2, 0, 0));	cvSetReal2D(R2, 0, 1, cvGetReal2D(e2, 1, 0));	
		cvSetReal2D(R2, 1, 0, -cvGetReal2D(e2, 1, 0));	cvSetReal2D(R2, 1, 1, cvGetReal2D(e2, 0, 0));	

		cvMatMul(R2, nF, temp33);
		cvTranspose(R1, temp33_1);
		cvMatMul(temp33, temp33_1, nF);
		double f1 = cvGetReal2D(e1, 2, 0);
		double f2 = cvGetReal2D(e2, 2, 0);
		double a = cvGetReal2D(nF, 1, 1);
		double b = cvGetReal2D(nF, 1, 2);
		double c = cvGetReal2D(nF, 2, 1);
		double d = cvGetReal2D(nF, 2, 2);
		double g1 = -(a*d-b*c)*f1*f1*f1*f1*a*c;
		double g2 = (a*a+f2*f2*c*c)*(a*a+f2*f2*c*c)-(a*d-b*c)*f1*f1*f1*f1*b*c-(a*d-b*c)*f1*f1*f1*f1*a*d;
		double g3 = (2*(2*b*a+2*f2*f2*d*c)*(a*a+f2*f2*c*c)-2*(a*d-b*c)*f1*f1*a*c-(a*d-b*c)*f1*f1*f1*f1*b*d);
		double g4 = (-2*(a*d-b*c)*f1*f1*b*c-2*(a*d-b*c)*f1*f1*a*d+2*(b*b+f2*f2*d*d)*(a*a+f2*f2*c*c)+(2*b*a+2*f2*f2*d*c)*(2*b*a+2*f2*f2*d*c));
		double g5 = (-(a*d-b*c)*a*c-2*(a*d-b*c)*f1*f1*b*d+2*(b*b+f2*f2*d*d)*(2*b*a+2*f2*f2*d*c));
		double g6 = ((b*b+f2*f2*d*d)*(b*b+f2*f2*d*d)-(a*d-b*c)*b*c-(a*d-b*c)*a*d);
		double g7 = -(a*d-b*c)*b*d;

		CvMat *G = cvCreateMat(7,1,CV_32FC1);
		CvMat *root = cvCreateMat(6,1, CV_32FC2);
		cvSetReal2D(G, 0, 0, g1);
		cvSetReal2D(G, 1, 0, g2);
		cvSetReal2D(G, 2, 0, g3);
		cvSetReal2D(G, 3, 0, g4);
		cvSetReal2D(G, 4, 0, g5);
		cvSetReal2D(G, 5, 0, g6);
		cvSetReal2D(G, 6, 0, g7);

		cvSolvePoly(G, root, 1e+3, 10);
		for (int i = 0; i < 6; i++)
		{
			CvScalar r = cvGet2D(root, i, 0);
			cout << r.val[0] << " " << r.val[1]<< endl;
		}


	}

	//	e1 = null(F);   e1 = e1/sqrt(e1(1)^2+e1(2)^2);
	//e2 = null(F');  e2 = e2/sqrt(e2(1)^2+e2(2)^2);
	//	R1 = [e1(1) e1(2) 0; -e1(2) e1(1) 0; 0 0 1];
	//R2 = [e2(1) e2(2) 0; -e2(2) e2(1) 0; 0 0 1];
	//F = R2*F*R1';
	//	f1 = e1(3); f2 = e2(3);
	//a = F(2,2); b = F(2,3); c = F(3,2); d = F(3,3);
	//g = [-(a*d-b*c)*f1^4*a*c,...
	//	(a^2+f2^2*c^2)^2-(a*d-b*c)*f1^4*b*c-(a*d-b*c)*f1^4*a*d,...
	//	(2*(2*b*a+2*f2^2*d*c)*(a^2+f2^2*c^2)-2*(a*d-b*c)*f1^2*a*c-(a*d-b*c)*f1^4*b*d),...
	//	(-2*(a*d-b*c)*f1^2*b*c-2*(a*d-b*c)*f1^2*a*d+2*(b^2+f2^2*d^2)*(a^2+f2^2*c^2)+(2*b*a+2*f2^2*d*c)^2),...
	//	(-(a*d-b*c)*a*c-2*(a*d-b*c)*f1^2*b*d+2*(b^2+f2^2*d^2)*(2*b*a+2*f2^2*d*c)),...
	//	((b^2+f2^2*d^2)^2-(a*d-b*c)*b*c-(a*d-b*c)*a*d),...
	//	-(a*d-b*c)*b*d];
	//t = roots(g);
	//t = real(t);
	//s = t.^2./(1+f1^2*t.^2) + (c*t+d).^2./((a*t+b).^2 + f2^2*(c*t+d).^2);
	//s(end+1) = 1/f1^2+c^2/(a^2+f2^2*c^2);
	//[mins,minidx] = min(s);
	//if minidx <= length(t)
	//	t = t(minidx);
	//else
	//	t = 1e+6;
	//end
	//	l1 = [t*f1, 1, -t];
	//l2 = F*[0; t; 1];
	//xhat1_t = [-l1(1)*l1(3); -l1(2)*l1(3); l1(1)^2+l1(2)^2];
	//xhat2_t = [-l2(1)*l2(3); -l2(2)*l2(3); l2(1)^2+l2(2)^2];
	//xhat1_t = inv(T1)*R1'*xhat1_t;
	//	xhat2_t = inv(T2)*R2'*xhat2_t;
	//	xhat1_t = xhat1_t/xhat1_t(3);
	//xhat2_t = xhat2_t/xhat2_t(3);
	//xhat1(i,:) = xhat1_t;
	//xhat2(i,:) = xhat2_t;
	//end

}

void GetExtrinsicParameterFromE(CvMat *E, CvMat *x1, CvMat *x2, CvMat &P)
{
	CvMat *W = cvCreateMat(3, 3, CV_32FC1);
	CvMat *U = cvCreateMat(3, 3, CV_32FC1);
	CvMat *D = cvCreateMat(3, 3, CV_32FC1);
	CvMat *Vt = cvCreateMat(3, 3, CV_32FC1);
	CvMat *Wt = cvCreateMat(3, 3, CV_32FC1);
	cvSVD(E, D, U, Vt, CV_SVD_V_T);

	//cvSetReal2D(D, 1,1,cvGetReal2D(D,0,0));
	//cvMatMul(U, D, E);
	//cvMatMul(E, Vt, E);
	//cvSVD(E, D, U, Vt, CV_SVD_V_T);

	cvSetReal2D(W, 0, 0, 0);	cvSetReal2D(W, 0, 1, -1);	cvSetReal2D(W, 0, 2, 0);
	cvSetReal2D(W, 1, 0, 1);	cvSetReal2D(W, 1, 1, 0);	cvSetReal2D(W, 1, 2, 0);
	cvSetReal2D(W, 2, 0, 0);	cvSetReal2D(W, 2, 1, 0);	cvSetReal2D(W, 2, 2, 1);
	cvTranspose(W, Wt);

	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);
	P = *cvCreateMat(3, 4, CV_32FC1);

	CvMat *P1 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P2 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P3 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P4 = cvCreateMat(3, 4, CV_32FC1);

	CvMat *R1 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *R2 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *t1 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *t2 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);

	cvMatMul(U, W, temp33);
	cvMatMul(temp33, Vt, R1);
	cvMatMul(U, Wt, temp33);
	cvMatMul(temp33, Vt, R2);
	
	cvSetReal2D(t1, 0, 0, cvGetReal2D(U,0,2));
	cvSetReal2D(t1, 1, 0, cvGetReal2D(U,1,2));
	cvSetReal2D(t1, 2, 0, cvGetReal2D(U,2,2));
	ScalarMul(t1, -1, t2);

	SetSubMat(P1, 0, 0, R1);
	SetSubMat(P1, 0, 3, t1);
	SetSubMat(P2, 0, 0, R1);
	SetSubMat(P2, 0, 3, t2);
	SetSubMat(P3, 0, 0, R2);
	SetSubMat(P3, 0, 3, t1);
	SetSubMat(P4, 0, 0, R2);
	SetSubMat(P4, 0, 3, t2);
	if (cvDet(R1) < 0)
	{
		ScalarMul(P1, -1, P1);		
		ScalarMul(P2, -1, P2);		
	}

	if (cvDet(R2) < 0)
	{
		ScalarMul(P3, -1, P3);		
		ScalarMul(P4, -1, P4);		
	}
	CvMat X1;
	LinearTriangulation(x1, P0, x2, P1, X1);
	CvMat X2;
	LinearTriangulation(x1, P0, x2, P2, X2);
	CvMat X3;
	LinearTriangulation(x1, P0, x2, P3, X3);
	CvMat X4;
	LinearTriangulation(x1, P0, x2, P4, X4);

	int x1neg = 0, x2neg = 0, x3neg = 0, x4neg = 0;
	CvMat *H1 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH1 = cvCreateMat(4, 4, CV_32FC1);		CvMat HX1;
	cvSetIdentity(H1);
	SetSubMat(H1, 0, 0, P1);
	cvInvert(H1, invH1);
	Pxx_inhomo(H1, &X1, HX1);

	CvMat *H2 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH2 = cvCreateMat(4, 4, CV_32FC1);		CvMat HX2;
	cvSetIdentity(H2);
	SetSubMat(H2, 0, 0, P2);
	cvInvert(H2, invH2);
	Pxx_inhomo(H2, &X2, HX2);
	CvMat *H3 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH3 = cvCreateMat(4, 4, CV_32FC1);		CvMat HX3;
	cvSetIdentity(H3);
	SetSubMat(H3, 0, 0, P3);
	cvInvert(H3, invH3);
	Pxx_inhomo(H3, &X3, HX3);
	CvMat *H4 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH4 = cvCreateMat(4, 4, CV_32FC1);		CvMat HX4;
	cvSetIdentity(H4);
	SetSubMat(H4, 0, 0, P4);
	cvInvert(H4, invH4);
	Pxx_inhomo(H4, &X4, HX4);

	for (int ix = 0; ix < x1->rows; ix++)
	{
		if ((cvGetReal2D(&X1, ix, 2)<0) || (cvGetReal2D(&HX1, ix, 2)<0))
			x1neg++;
		if ((cvGetReal2D(&X2, ix, 2)<0) || (cvGetReal2D(&HX2, ix, 2)<0))
			x2neg++;
		if ((cvGetReal2D(&X3, ix, 2)<0) || (cvGetReal2D(&HX3, ix, 2)<0))
			x3neg++;
		if ((cvGetReal2D(&X4, ix, 2)<0) || (cvGetReal2D(&HX4, ix, 2)<0))
			x4neg++;
	}

	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	if ((x1neg <= x2neg) && (x1neg <= x3neg) && (x1neg <= x4neg))
		P = *cvCloneMat(P1);
	else if ((x2neg <= x1neg) && (x2neg <= x3neg) && (x2neg <= x4neg))
		P = *cvCloneMat(P2);
	else if ((x3neg <= x1neg) && (x3neg <= x2neg) && (x3neg <= x4neg))
		P = *cvCloneMat(P3);
	else
		P = *cvCloneMat(P4);

	cvReleaseMat(&W);
	cvReleaseMat(&U);
	cvReleaseMat(&D);
	cvReleaseMat(&Vt);
	cvReleaseMat(&Wt);
	cvReleaseMat(&P0);
	cvReleaseMat(&P1);
	cvReleaseMat(&P2);
	cvReleaseMat(&P3);
	cvReleaseMat(&P4);
	cvReleaseMat(&R1);
	cvReleaseMat(&R2);
	cvReleaseMat(&t1);
	cvReleaseMat(&t2);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&H1);
	cvReleaseMat(&invH1);
	cvReleaseMat(&H2);
	cvReleaseMat(&invH2);
	cvReleaseMat(&H3);
	cvReleaseMat(&invH3);
	cvReleaseMat(&H4);
	cvReleaseMat(&invH4);
}

void GetExtrinsicParameterFromE(CvMat *E, CvMat *x1, CvMat *x2, CvMat *P)
{
	CvMat *W = cvCreateMat(3, 3, CV_32FC1);
	CvMat *U = cvCreateMat(3, 3, CV_32FC1);
	CvMat *D = cvCreateMat(3, 3, CV_32FC1);
	CvMat *Vt = cvCreateMat(3, 3, CV_32FC1);
	CvMat *Wt = cvCreateMat(3, 3, CV_32FC1);
	cvSVD(E, D, U, Vt, CV_SVD_V_T);

	//cvSetReal2D(D, 1,1,cvGetReal2D(D,0,0));
	//cvMatMul(U, D, E);
	//cvMatMul(E, Vt, E);
	//cvSVD(E, D, U, Vt, CV_SVD_V_T);

	cvSetReal2D(W, 0, 0, 0);	cvSetReal2D(W, 0, 1, -1);	cvSetReal2D(W, 0, 2, 0);
	cvSetReal2D(W, 1, 0, 1);	cvSetReal2D(W, 1, 1, 0);	cvSetReal2D(W, 1, 2, 0);
	cvSetReal2D(W, 2, 0, 0);	cvSetReal2D(W, 2, 1, 0);	cvSetReal2D(W, 2, 2, 1);
	cvTranspose(W, Wt);

	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);

	CvMat *P1 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P2 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P3 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P4 = cvCreateMat(3, 4, CV_32FC1);

	CvMat *R1 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *R2 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *t1 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *t2 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);

	cvMatMul(U, W, temp33);
	cvMatMul(temp33, Vt, R1);
	cvMatMul(U, Wt, temp33);
	cvMatMul(temp33, Vt, R2);

	cvSetReal2D(t1, 0, 0, cvGetReal2D(U,0,2));
	cvSetReal2D(t1, 1, 0, cvGetReal2D(U,1,2));
	cvSetReal2D(t1, 2, 0, cvGetReal2D(U,2,2));
	ScalarMul(t1, -1, t2);

	SetSubMat(P1, 0, 0, R1);
	SetSubMat(P1, 0, 3, t1);
	SetSubMat(P2, 0, 0, R1);
	SetSubMat(P2, 0, 3, t2);
	SetSubMat(P3, 0, 0, R2);
	SetSubMat(P3, 0, 3, t1);
	SetSubMat(P4, 0, 0, R2);
	SetSubMat(P4, 0, 3, t2);
	if (cvDet(R1) < 0)
	{
		ScalarMul(P1, -1, P1);		
		ScalarMul(P2, -1, P2);		
	}

	if (cvDet(R2) < 0)
	{
		ScalarMul(P3, -1, P3);		
		ScalarMul(P4, -1, P4);		
	}
	CvMat *X1 = cvCreateMat(x1->rows, 3, CV_32FC1);
	LinearTriangulation(x1, P0, x2, P1, X1);
	CvMat *X2 = cvCreateMat(x1->rows, 3, CV_32FC1);;
	LinearTriangulation(x1, P0, x2, P2, X2);
	CvMat *X3 = cvCreateMat(x1->rows, 3, CV_32FC1);;
	LinearTriangulation(x1, P0, x2, P3, X3);
	CvMat *X4 = cvCreateMat(x1->rows, 3, CV_32FC1);;
	LinearTriangulation(x1, P0, x2, P4, X4);

	int x1neg = 0, x2neg = 0, x3neg = 0, x4neg = 0;
	CvMat *H1 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH1 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX1 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H1);
	SetSubMat(H1, 0, 0, P1);
	cvInvert(H1, invH1);
	Pxx_inhomo(H1, X1, HX1);

	CvMat *H2 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH2 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX2 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H2);
	SetSubMat(H2, 0, 0, P2);
	cvInvert(H2, invH2);
	Pxx_inhomo(H2, X2, HX2);
	CvMat *H3 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH3 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX3 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H3);
	SetSubMat(H3, 0, 0, P3);
	cvInvert(H3, invH3);
	Pxx_inhomo(H3, X3, HX3);
	CvMat *H4 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH4 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX4 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H4);
	SetSubMat(H4, 0, 0, P4);
	cvInvert(H4, invH4);
	Pxx_inhomo(H4, X4, HX4);

	for (int ix = 0; ix < x1->rows; ix++)
	{
		if ((cvGetReal2D(X1, ix, 2)<0) || (cvGetReal2D(HX1, ix, 2)<0))
			x1neg++;
		if ((cvGetReal2D(X2, ix, 2)<0) || (cvGetReal2D(HX2, ix, 2)<0))
			x2neg++;
		if ((cvGetReal2D(X3, ix, 2)<0) || (cvGetReal2D(HX3, ix, 2)<0))
			x3neg++;
		if ((cvGetReal2D(X4, ix, 2)<0) || (cvGetReal2D(HX4, ix, 2)<0))
			x4neg++;
	}

	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	if ((x1neg <= x2neg) && (x1neg <= x3neg) && (x1neg <= x4neg))
		SetSubMat(P, 0, 0, P1);
	else if ((x2neg <= x1neg) && (x2neg <= x3neg) && (x2neg <= x4neg))
		SetSubMat(P, 0, 0, P2);
	else if ((x3neg <= x1neg) && (x3neg <= x2neg) && (x3neg <= x4neg))
		SetSubMat(P, 0, 0, P3);
	else
		SetSubMat(P, 0, 0, P4);

	//cout << x1neg << " " << x2neg << " " << " " << x3neg << " " << x4neg << endl;
	cvReleaseMat(&W);
	cvReleaseMat(&U);
	cvReleaseMat(&D);
	cvReleaseMat(&Vt);
	cvReleaseMat(&Wt);
	cvReleaseMat(&P0);
	cvReleaseMat(&P1);
	cvReleaseMat(&P2);
	cvReleaseMat(&P3);
	cvReleaseMat(&P4);
	cvReleaseMat(&R1);
	cvReleaseMat(&R2);
	cvReleaseMat(&t1);
	cvReleaseMat(&t2);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&H1);
	cvReleaseMat(&invH1);
	cvReleaseMat(&H2);
	cvReleaseMat(&invH2);
	cvReleaseMat(&H3);
	cvReleaseMat(&invH3);
	cvReleaseMat(&H4);
	cvReleaseMat(&invH4);
	cvReleaseMat(&X1);
	cvReleaseMat(&X2);
	cvReleaseMat(&X3);
	cvReleaseMat(&X4);

	cvReleaseMat(&HX1);
	cvReleaseMat(&HX2);
	cvReleaseMat(&HX3);
	cvReleaseMat(&HX4);
}

void LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, CvMat &X)
{
	X = *cvCreateMat(x1->rows, 3, CV_32FC1);
	cvSetZero(&X);
	for (int ix = 0; ix < x1->rows; ix++)
	{
		CvMat *A = cvCreateMat(4, 4, CV_32FC1);
		CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A4 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_3 = cvCreateMat(1, 4, CV_32FC1);

		CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);

		GetSubMatRowwise(P1, 0, 0, P1_1);
		GetSubMatRowwise(P1, 1, 1, P1_2);
		GetSubMatRowwise(P1, 2, 2, P1_3);
		GetSubMatRowwise(P2, 0, 0, P2_1);
		GetSubMatRowwise(P2, 1, 1, P2_2);
		GetSubMatRowwise(P2, 2, 2, P2_3);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 0), temp14_1);
		cvSub(temp14_1, P1_1, A1);
		
		ScalarMul(P1_3, cvGetReal2D(x1, ix, 1), temp14_1);
		cvSub(temp14_1, P1_2, A2);

		ScalarMul(P2_3, cvGetReal2D(x2, ix, 0), temp14_1);
		cvSub(temp14_1, P2_1, A3);
		ScalarMul(P2_3, cvGetReal2D(x2, ix, 1), temp14_1);
		cvSub(temp14_1, P2_2, A4);
		SetSubMat(A, 0, 0, A1);
		SetSubMat(A, 1, 0, A2);
		SetSubMat(A, 2, 0, A3);
		SetSubMat(A, 3, 0, A4);

		CvMat x;
		LS_homogeneous(A, x);
		double v = cvGetReal2D(&x, 3, 0);
		cvSetReal2D(&X, ix, 0, cvGetReal2D(&x, 0, 0)/v);
		cvSetReal2D(&X, ix, 1, cvGetReal2D(&x, 1, 0)/v);
		cvSetReal2D(&X, ix, 2, cvGetReal2D(&x, 2, 0)/v);

		cvReleaseMat(&A);
		cvReleaseMat(&A1);
		cvReleaseMat(&A2);
		cvReleaseMat(&A3);
		cvReleaseMat(&A4);
		cvReleaseMat(&P1_1);
		cvReleaseMat(&P1_2);
		cvReleaseMat(&P1_3);
		cvReleaseMat(&P2_1);
		cvReleaseMat(&P2_2);
		cvReleaseMat(&P2_3);
		cvReleaseMat(&temp14_1);
	}
}

void LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, CvMat *X)
{
	for (int ix = 0; ix < x1->rows; ix++)
	{
		CvMat *A = cvCreateMat(4, 4, CV_32FC1);
		CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A4 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_3 = cvCreateMat(1, 4, CV_32FC1);

		CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);

		GetSubMatRowwise(P1, 0, 0, P1_1);
		GetSubMatRowwise(P1, 1, 1, P1_2);
		GetSubMatRowwise(P1, 2, 2, P1_3);
		GetSubMatRowwise(P2, 0, 0, P2_1);
		GetSubMatRowwise(P2, 1, 1, P2_2);
		GetSubMatRowwise(P2, 2, 2, P2_3);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 0), temp14_1);
		cvSub(temp14_1, P1_1, A1);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 1), temp14_1);
		cvSub(temp14_1, P1_2, A2);

		ScalarMul(P2_3, cvGetReal2D(x2, ix, 0), temp14_1);
		cvSub(temp14_1, P2_1, A3);
		ScalarMul(P2_3, cvGetReal2D(x2, ix, 1), temp14_1);
		cvSub(temp14_1, P2_2, A4);
		SetSubMat(A, 0, 0, A1);
		SetSubMat(A, 1, 0, A2);
		SetSubMat(A, 2, 0, A3);
		SetSubMat(A, 3, 0, A4);

		CvMat *x = cvCreateMat(A->cols, 1, CV_32FC1);
		LS_homogeneous(A, x);
		double v = cvGetReal2D(x, 3, 0);
		cvSetReal2D(X, ix, 0, cvGetReal2D(x, 0, 0)/v);
		cvSetReal2D(X, ix, 1, cvGetReal2D(x, 1, 0)/v);
		cvSetReal2D(X, ix, 2, cvGetReal2D(x, 2, 0)/v);

		cvReleaseMat(&A);
		cvReleaseMat(&A1);
		cvReleaseMat(&A2);
		cvReleaseMat(&A3);
		cvReleaseMat(&A4);
		cvReleaseMat(&P1_1);
		cvReleaseMat(&P1_2);
		cvReleaseMat(&P1_3);
		cvReleaseMat(&P2_1);
		cvReleaseMat(&P2_2);
		cvReleaseMat(&P2_3);
		cvReleaseMat(&temp14_1);
		cvReleaseMat(&x);
	}
}


int LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> featureID, CvMat &X, vector<int> &filteredFeatureID)
{
	//X = *cvCreateMat(x1->rows, 3, CV_32FC1);
	//cvSetZero(&X);
	vector<double> X1, X2, X3;
	filteredFeatureID.clear();
	for (int ix = 0; ix < x1->rows; ix++)
	{
		CvMat *A = cvCreateMat(4, 4, CV_32FC1);
		CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A4 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_3 = cvCreateMat(1, 4, CV_32FC1);

		CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);

		GetSubMatRowwise(P1, 0, 0, P1_1);
		GetSubMatRowwise(P1, 1, 1, P1_2);
		GetSubMatRowwise(P1, 2, 2, P1_3);
		GetSubMatRowwise(P2, 0, 0, P2_1);
		GetSubMatRowwise(P2, 1, 1, P2_2);
		GetSubMatRowwise(P2, 2, 2, P2_3);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 0), temp14_1);
		cvSub(temp14_1, P1_1, A1);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 1), temp14_1);
		cvSub(temp14_1, P1_2, A2);

		ScalarMul(P2_3, cvGetReal2D(x2, ix, 0), temp14_1);
		cvSub(temp14_1, P2_1, A3);
		ScalarMul(P2_3, cvGetReal2D(x2, ix, 1), temp14_1);
		cvSub(temp14_1, P2_2, A4);
		SetSubMat(A, 0, 0, A1);
		SetSubMat(A, 1, 0, A2);
		SetSubMat(A, 2, 0, A3);
		SetSubMat(A, 3, 0, A4);

		CvMat x;
		LS_homogeneous(A, x);
		
		double v = cvGetReal2D(&x, 3, 0);
		//cout << v  << " " << abs(v)<< endl;
		if (abs(v) < POINT_AT_INFINITY_ZERO)
			continue;
		X1.push_back(cvGetReal2D(&x, 0, 0)/v);
		X2.push_back(cvGetReal2D(&x, 1, 0)/v);
		X3.push_back(cvGetReal2D(&x, 2, 0)/v);
		filteredFeatureID.push_back(featureID[ix]);

		cvReleaseMat(&A);
		cvReleaseMat(&A1);
		cvReleaseMat(&A2);
		cvReleaseMat(&A3);
		cvReleaseMat(&A4);
		cvReleaseMat(&P1_1);
		cvReleaseMat(&P1_2);
		cvReleaseMat(&P1_3);
		cvReleaseMat(&P2_1);
		cvReleaseMat(&P2_2);
		cvReleaseMat(&P2_3);
		cvReleaseMat(&temp14_1);
	}

	if (X1.size() == 0)
		return 0;

	X = *cvCreateMat(X1.size(), 3, CV_32FC1);
	for (int i = 0; i < X1.size(); i++)
	{
		cvSetReal2D(&X, i, 0, X1[i]);
		cvSetReal2D(&X, i, 1, X2[i]);
		cvSetReal2D(&X, i, 2, X3[i]);
	}
	return 1;
}

int LinearTriangulation_mem(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> featureID, vector<vector<double> > &X, vector<int> &filteredFeatureID)
{
	vector<double> X1, X2, X3;
	filteredFeatureID.clear();
	for (int ix = 0; ix < x1->rows; ix++)
	{
		CvMat *A = cvCreateMat(4, 4, CV_32FC1);
		CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A4 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_3 = cvCreateMat(1, 4, CV_32FC1);

		CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);

		GetSubMatRowwise(P1, 0, 0, P1_1);
		GetSubMatRowwise(P1, 1, 1, P1_2);
		GetSubMatRowwise(P1, 2, 2, P1_3);
		GetSubMatRowwise(P2, 0, 0, P2_1);
		GetSubMatRowwise(P2, 1, 1, P2_2);
		GetSubMatRowwise(P2, 2, 2, P2_3);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 0), temp14_1);
		cvSub(temp14_1, P1_1, A1);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 1), temp14_1);
		cvSub(temp14_1, P1_2, A2);

		ScalarMul(P2_3, cvGetReal2D(x2, ix, 0), temp14_1);
		cvSub(temp14_1, P2_1, A3);
		ScalarMul(P2_3, cvGetReal2D(x2, ix, 1), temp14_1);
		cvSub(temp14_1, P2_2, A4);
		SetSubMat(A, 0, 0, A1);
		SetSubMat(A, 1, 0, A2);
		SetSubMat(A, 2, 0, A3);
		SetSubMat(A, 3, 0, A4);

		CvMat *x = cvCreateMat(A->cols, 1, CV_32FC1);
		LS_homogeneous(A, x);

		double v = cvGetReal2D(x, 3, 0);
		if (abs(v) < POINT_AT_INFINITY_ZERO)
		{
			cvReleaseMat(&A);
			cvReleaseMat(&A1);
			cvReleaseMat(&A2);
			cvReleaseMat(&A3);
			cvReleaseMat(&A4);
			cvReleaseMat(&P1_1);
			cvReleaseMat(&P1_2);
			cvReleaseMat(&P1_3);
			cvReleaseMat(&P2_1);
			cvReleaseMat(&P2_2);
			cvReleaseMat(&P2_3);
			cvReleaseMat(&temp14_1);
			cvReleaseMat(&x);
			continue;
		}
		X1.push_back(cvGetReal2D(x, 0, 0)/v);
		X2.push_back(cvGetReal2D(x, 1, 0)/v);
		X3.push_back(cvGetReal2D(x, 2, 0)/v);
		filteredFeatureID.push_back(featureID[ix]);

		cvReleaseMat(&A);
		cvReleaseMat(&A1);
		cvReleaseMat(&A2);
		cvReleaseMat(&A3);
		cvReleaseMat(&A4);
		cvReleaseMat(&P1_1);
		cvReleaseMat(&P1_2);
		cvReleaseMat(&P1_3);
		cvReleaseMat(&P2_1);
		cvReleaseMat(&P2_2);
		cvReleaseMat(&P2_3);
		cvReleaseMat(&temp14_1);
		cvReleaseMat(&x);
	}

	if (X1.size() == 0)
		return 0;

	for (int i = 0; i < X1.size(); i++)
	{
		vector<double> X_vec;
		X_vec.push_back(X1[i]);
		X_vec.push_back(X2[i]);
		X_vec.push_back(X3[i]);
		X.push_back(X_vec);
	}
	return X.size();
}

int LinearTriangulation_mem_fast(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> &featureID, vector<vector<double> > &X, vector<int> &filteredFeatureID)
{
	filteredFeatureID.clear();
	CvMat *A = cvCreateMat(4, 4, CV_32FC1);
	CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *A3 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *A4 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P1_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P1_2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P1_3 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P2_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P2_2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P2_3 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *x = cvCreateMat(A->cols, 1, CV_32FC1);

	for (int ix = 0; ix < x1->rows; ix++)
	{
		GetSubMatRowwise(P1, 0, 0, P1_1);
		GetSubMatRowwise(P1, 1, 1, P1_2);
		GetSubMatRowwise(P1, 2, 2, P1_3);
		GetSubMatRowwise(P2, 0, 0, P2_1);
		GetSubMatRowwise(P2, 1, 1, P2_2);
		GetSubMatRowwise(P2, 2, 2, P2_3);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 0), temp14_1);
		cvSub(temp14_1, P1_1, A1);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 1), temp14_1);
		cvSub(temp14_1, P1_2, A2);

		ScalarMul(P2_3, cvGetReal2D(x2, ix, 0), temp14_1);
		cvSub(temp14_1, P2_1, A3);
		ScalarMul(P2_3, cvGetReal2D(x2, ix, 1), temp14_1);
		cvSub(temp14_1, P2_2, A4);
		SetSubMat(A, 0, 0, A1);
		SetSubMat(A, 1, 0, A2);
		SetSubMat(A, 2, 0, A3);
		SetSubMat(A, 3, 0, A4);

		LS_homogeneous(A, x);

		double v = cvGetReal2D(x, 3, 0);
		if (abs(v) < POINT_AT_INFINITY_ZERO)
		{
			continue;
		}

		vector<double> X_vec;
		X_vec.push_back(cvGetReal2D(x, 0, 0)/v);
		X_vec.push_back(cvGetReal2D(x, 1, 0)/v);
		X_vec.push_back(cvGetReal2D(x, 2, 0)/v);
		X.push_back(X_vec);
		filteredFeatureID.push_back(featureID[ix]);
	}

	cvReleaseMat(&A);
	cvReleaseMat(&A1);
	cvReleaseMat(&A2);
	cvReleaseMat(&A3);
	cvReleaseMat(&A4);
	cvReleaseMat(&P1_1);
	cvReleaseMat(&P1_2);
	cvReleaseMat(&P1_3);
	cvReleaseMat(&P2_1);
	cvReleaseMat(&P2_2);
	cvReleaseMat(&P2_3);
	cvReleaseMat(&temp14_1);
	cvReleaseMat(&x);

	if (filteredFeatureID.size() == 0)
		return 0;

	return X.size();
}

bool LinearTriangulation(vector<CvMat *> vP, vector<double> vx, vector<double> vy, double &X, double &Y, double &Z)
{
	if (vP.size() < 2)
		return false;

	CvMat *A = cvCreateMat(2*vP.size(), 4, CV_32FC1);
	CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P_2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P_3 = cvCreateMat(1, 4, CV_32FC1);

	CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *x = cvCreateMat(A->cols, 1, CV_32FC1);
	
	for (int iP = 0; iP < vP.size(); iP++)
	{
		GetSubMatRowwise(vP[iP], 0, 0, P_1);
		GetSubMatRowwise(vP[iP], 1, 1, P_2);
		GetSubMatRowwise(vP[iP], 2, 2, P_3);

		ScalarMul(P_3, vx[iP], temp14_1);
		cvSub(temp14_1, P_1, A1);

		ScalarMul(P_3, vy[iP], temp14_1);
		cvSub(temp14_1, P_2, A2);

		SetSubMat(A, 2*iP, 0, A1);
		SetSubMat(A, 2*iP+1, 0, A2);
	}

	LS_homogeneous(A, x);

	CvMat *Ax = cvCreateMat(A->rows, 1, CV_32FC1);
	cvMatMul(A, x, Ax);
	//PrintMat(Ax);

	double v = cvGetReal2D(x, 3, 0);
	if (abs(v) < POINT_AT_INFINITY_ZERO)
	{
		return false;
	}

	X = cvGetReal2D(x, 0, 0)/v;
	Y = cvGetReal2D(x, 1, 0)/v;
	Z = cvGetReal2D(x, 2, 0)/v;

	cvReleaseMat(&A);
	cvReleaseMat(&A1);
	cvReleaseMat(&A2);
	cvReleaseMat(&P_1);
	cvReleaseMat(&P_2);
	cvReleaseMat(&P_3);
	cvReleaseMat(&temp14_1);
	cvReleaseMat(&x);

	return true;
}


int DLT_ExtrinsicCameraParamEstimation(CvMat *X, CvMat *x, CvMat *K, CvMat *P)
{
	if (X->rows < 6)
		return 0;

	CvMat *Xtilde = cvCreateMat(X->rows, 3, CV_32FC1);
	CvMat *xtilde = cvCreateMat(x->rows, 2, CV_32FC1);
	CvMat *U = cvCreateMat(4,4, CV_32FC1);
	CvMat *T = cvCreateMat(3,3, CV_32FC1);

	Normalization3D(X, Xtilde, U);
	Normalization(x, xtilde, T);


	CvMat *A = cvCreateMat(X->rows*2,12,CV_32FC1);
	for (int iX = 0; iX < X->rows; iX++)
	{
		CvMat *A1 = cvCreateMat(1, 12, CV_32FC1);
		CvMat *A2 = cvCreateMat(1, 12, CV_32FC1);
		cvSetZero(A1);	cvSetZero(A2);
		CvMat *A1_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A1_3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2_3 = cvCreateMat(1, 4, CV_32FC1);
		// A1_2
		cvSetReal2D(A1_2, 0, 0, -cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A1_2, 0, 1, -cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A1_2, 0, 2, -cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A1_2, 0, 3, -1);
		
		// A1_3
		cvSetReal2D(A1_3, 0, 0, cvGetReal2D(xtilde, iX, 1)*cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A1_3, 0, 1, cvGetReal2D(xtilde, iX, 1)*cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A1_3, 0, 2, cvGetReal2D(xtilde, iX, 1)*cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A1_3, 0, 3, cvGetReal2D(xtilde, iX, 1));

		// A2_1
		cvSetReal2D(A2_1, 0, 0, cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A2_1, 0, 1, cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A2_1, 0, 2, cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A2_1, 0, 3, 1);
		// A1_3
		cvSetReal2D(A2_3, 0, 0, -cvGetReal2D(xtilde, iX, 0)*cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A2_3, 0, 1, -cvGetReal2D(xtilde, iX, 0)*cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A2_3, 0, 2, -cvGetReal2D(xtilde, iX, 0)*cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A2_3, 0, 3, -cvGetReal2D(xtilde, iX, 0));
		SetSubMat(A1, 0, 4, A1_2);	SetSubMat(A1, 0, 8, A1_3);
		SetSubMat(A2, 0, 0, A2_1);	SetSubMat(A2, 0, 8, A2_3);
		
		SetSubMat(A, 2*iX, 0, A1);
		SetSubMat(A, 2*iX+1, 0, A2);

		cvReleaseMat(&A1);
		cvReleaseMat(&A2);
		cvReleaseMat(&A1_2);
		cvReleaseMat(&A1_3);
		cvReleaseMat(&A2_1);
		cvReleaseMat(&A2_3);
	}
	CvMat *p = cvCreateMat(A->cols, 1, CV_32FC1);
	LS_homogeneous(A, p);

	cvSetReal2D(P, 0, 0, cvGetReal2D(p, 0, 0));	cvSetReal2D(P, 0, 1, cvGetReal2D(p, 1, 0));	cvSetReal2D(P, 0, 2, cvGetReal2D(p, 2, 0));	cvSetReal2D(P, 0, 3, cvGetReal2D(p, 3, 0));
	cvSetReal2D(P, 1, 0, cvGetReal2D(p, 4, 0));	cvSetReal2D(P, 1, 1, cvGetReal2D(p, 5, 0));	cvSetReal2D(P, 1, 2, cvGetReal2D(p, 6, 0));	cvSetReal2D(P, 1, 3, cvGetReal2D(p, 7, 0));
	cvSetReal2D(P, 2, 0, cvGetReal2D(p, 8, 0));	cvSetReal2D(P, 2, 1, cvGetReal2D(p, 9, 0));	cvSetReal2D(P, 2, 2, cvGetReal2D(p, 10, 0));	cvSetReal2D(P, 2, 3, cvGetReal2D(p, 11, 0));

	CvMat *temp34 = cvCreateMat(3,4,CV_32FC1);
	CvMat *invT = cvCreateMat(3,3,CV_32FC1);
	cvInvert(T, invT);
	cvMatMul(invT, P, temp34);
	cvMatMul(temp34, U, P);

	CvMat *P_ = cvCreateMat(3,4, CV_32FC1);
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K, invK);
	cvMatMul(invK, P, P_);
	CvMat *P1 = cvCreateMat(3,1,CV_32FC1);
	GetSubMatColwise(P_, 0, 0, P1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	GetSubMatColwise(P_, 0, 2, R);
	double norm = NormL2(P1);
	double determinant = cvDet(R);
	double sign = 1;
	if (determinant < 0)
		sign = -1;
	ScalarMul(P_,sign/norm, P);
	cvMatMul(K, P, P);

	cvReleaseMat(&temp34);
	cvReleaseMat(&invT);
	cvReleaseMat(&P_);
	cvReleaseMat(&invK);
	cvReleaseMat(&P1);
	cvReleaseMat(&R);
	cvReleaseMat(&Xtilde);
	cvReleaseMat(&xtilde);
	cvReleaseMat(&U);
	cvReleaseMat(&T);
	cvReleaseMat(&A);
	cvReleaseMat(&p);
	return 1;
}

int EPNP_ExtrinsicCameraParamEstimation(CvMat *X, CvMat *x, CvMat *K, CvMat *P)
{
	epnp PnP;

	PnP.set_internal_parameters(cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2), cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1));
	PnP.set_maximum_number_of_correspondences(X->rows);
	PnP.reset_correspondences();
	for(int i = 0; i < X->rows; i++) {
		PnP.add_correspondence(cvGetReal2D(X, i, 0), cvGetReal2D(X, i, 1), cvGetReal2D(X, i, 2), cvGetReal2D(x, i, 0), cvGetReal2D(x, i, 1));
	}

	double R_est[3][3], t_est[3];
	double err2 = PnP.compute_pose(R_est, t_est);

	cvSetReal2D(P, 0, 3, t_est[0]);
	cvSetReal2D(P, 1, 3, t_est[1]);
	cvSetReal2D(P, 2, 3, t_est[2]);

	cvSetReal2D(P, 0, 0, R_est[0][0]);		cvSetReal2D(P, 0, 1, R_est[0][1]);		cvSetReal2D(P, 0, 2, R_est[0][2]);
	cvSetReal2D(P, 1, 0, R_est[1][0]);		cvSetReal2D(P, 1, 1, R_est[1][1]);		cvSetReal2D(P, 1, 2, R_est[1][2]);
	cvSetReal2D(P, 2, 0, R_est[2][0]);		cvSetReal2D(P, 2, 1, R_est[2][1]);		cvSetReal2D(P, 2, 2, R_est[2][2]);
	cvMatMul(K, P, P);

	return 1;
}




int DLT_ExtrinsicCameraParamEstimation_KRT(CvMat *X, CvMat *x, CvMat *K, CvMat *P)
{
	if (X->rows < 6)
		return 0;

	CvMat *Xtilde = cvCreateMat(X->rows, 3, CV_32FC1);
	CvMat *xtilde = cvCreateMat(x->rows, 2, CV_32FC1);
	CvMat *U = cvCreateMat(4,4, CV_32FC1);
	CvMat *T = cvCreateMat(3,3, CV_32FC1);

	Normalization3D(X, Xtilde, U);
	Normalization(x, xtilde, T);


	CvMat *A = cvCreateMat(X->rows*2,12,CV_32FC1);
	for (int iX = 0; iX < X->rows; iX++)
	{
		CvMat *A1 = cvCreateMat(1, 12, CV_32FC1);
		CvMat *A2 = cvCreateMat(1, 12, CV_32FC1);
		cvSetZero(A1);	cvSetZero(A2);
		CvMat *A1_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A1_3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2_3 = cvCreateMat(1, 4, CV_32FC1);
		// A1_2
		cvSetReal2D(A1_2, 0, 0, -cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A1_2, 0, 1, -cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A1_2, 0, 2, -cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A1_2, 0, 3, -1);

		// A1_3
		cvSetReal2D(A1_3, 0, 0, cvGetReal2D(xtilde, iX, 1)*cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A1_3, 0, 1, cvGetReal2D(xtilde, iX, 1)*cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A1_3, 0, 2, cvGetReal2D(xtilde, iX, 1)*cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A1_3, 0, 3, cvGetReal2D(xtilde, iX, 1));

		// A2_1
		cvSetReal2D(A2_1, 0, 0, cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A2_1, 0, 1, cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A2_1, 0, 2, cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A2_1, 0, 3, 1);
		// A1_3
		cvSetReal2D(A2_3, 0, 0, -cvGetReal2D(xtilde, iX, 0)*cvGetReal2D(Xtilde, iX, 0));	
		cvSetReal2D(A2_3, 0, 1, -cvGetReal2D(xtilde, iX, 0)*cvGetReal2D(Xtilde, iX, 1));
		cvSetReal2D(A2_3, 0, 2, -cvGetReal2D(xtilde, iX, 0)*cvGetReal2D(Xtilde, iX, 2));
		cvSetReal2D(A2_3, 0, 3, -cvGetReal2D(xtilde, iX, 0));
		SetSubMat(A1, 0, 4, A1_2);	SetSubMat(A1, 0, 8, A1_3);
		SetSubMat(A2, 0, 0, A2_1);	SetSubMat(A2, 0, 8, A2_3);

		SetSubMat(A, 2*iX, 0, A1);
		SetSubMat(A, 2*iX+1, 0, A2);

		cvReleaseMat(&A1);
		cvReleaseMat(&A2);
		cvReleaseMat(&A1_2);
		cvReleaseMat(&A1_3);
		cvReleaseMat(&A2_1);
		cvReleaseMat(&A2_3);
	}
	CvMat *p = cvCreateMat(A->cols, 1, CV_32FC1);
	LS_homogeneous(A, p);

	cvSetReal2D(P, 0, 0, cvGetReal2D(p, 0, 0));	cvSetReal2D(P, 0, 1, cvGetReal2D(p, 1, 0));	cvSetReal2D(P, 0, 2, cvGetReal2D(p, 2, 0));	cvSetReal2D(P, 0, 3, cvGetReal2D(p, 3, 0));
	cvSetReal2D(P, 1, 0, cvGetReal2D(p, 4, 0));	cvSetReal2D(P, 1, 1, cvGetReal2D(p, 5, 0));	cvSetReal2D(P, 1, 2, cvGetReal2D(p, 6, 0));	cvSetReal2D(P, 1, 3, cvGetReal2D(p, 7, 0));
	cvSetReal2D(P, 2, 0, cvGetReal2D(p, 8, 0));	cvSetReal2D(P, 2, 1, cvGetReal2D(p, 9, 0));	cvSetReal2D(P, 2, 2, cvGetReal2D(p, 10, 0));	cvSetReal2D(P, 2, 3, cvGetReal2D(p, 11, 0));

	//for (int i = 0; i < X->rows; i++)
	//{
	//	CvMat *tX = cvCreateMat(4,1,CV_32FC1);
	//	cvSetReal2D(tX, 0, 0, cvGetReal2D(Xtilde, i, 0));
	//	cvSetReal2D(tX, 1, 0, cvGetReal2D(Xtilde, i, 1));
	//	cvSetReal2D(tX, 2, 0, cvGetReal2D(Xtilde, i, 2));
	//	cvSetReal2D(tX, 3, 0, 1);
	//	CvMat *tx = cvCreateMat(3,1, CV_32FC1);
	//	cvMatMul(P, tX, tx);
	//	ScalarMul(tx, 1/cvGetReal2D(tx, 2, 0), tx);
	//	PrintMat(tx, "tx");
	//	CvMat *ttx = cvCreateMat(2,1,CV_32FC1);
	//	cvSetReal2D(ttx, 0, 0, cvGetReal2D(xtilde, i, 0));
	//	cvSetReal2D(ttx, 1, 0, cvGetReal2D(xtilde, i, 1));
	//	PrintMat(ttx, "ttx");
	//}

	CvMat *temp34 = cvCreateMat(3,4,CV_32FC1);
	CvMat *invT = cvCreateMat(3,3,CV_32FC1);
	cvInvert(T, invT);
	cvMatMul(invT, P, temp34);
	cvMatMul(temp34, U, P);

	ScalarMul(P, 1/cvGetReal2D(P, 2, 3), P);
	
	CvMat *R = cvCreateMat(3,3, CV_32FC1);
	CvMat *C = cvCreateMat(4,1, CV_32FC1);
	//ScalarMul(P, -1, P);
	cvDecomposeProjectionMatrix(P, K, R, C);
	//cout << "det " << cvDet(R) << endl;
	//PrintMat(C, "C");

	//CvMat *P_ = cvCreateMat(3,4, CV_32FC1);
	//CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	//cvInvert(K, invK);
	//cvMatMul(invK, P, P_);
	//CvMat *P1 = cvCreateMat(3,1,CV_32FC1);
	//GetSubMatColwise(P_, 0, 0, P1);
	//CvMat *R = cvCreateMat(3,3,CV_32FC1);
	//GetSubMatColwise(P_, 0, 2, R);
	//double norm = NormL2(P1);
	//double determinant = cvDet(R);
	//double sign = 1;
	//if (determinant < 0)
	//	sign = -1;
	//ScalarMul(P_,sign/norm, P);
	//cvMatMul(K, P, P);

	cvReleaseMat(&temp34);
	cvReleaseMat(&invT);
	//cvReleaseMat(&P_);
	//cvReleaseMat(&invK);
	//cvReleaseMat(&P1);
	cvReleaseMat(&R);
	cvReleaseMat(&Xtilde);
	cvReleaseMat(&xtilde);
	cvReleaseMat(&U);
	cvReleaseMat(&T);
	cvReleaseMat(&A);
	cvReleaseMat(&p);
	cvReleaseMat(&C);
	return 1;
}

int DLT_ExtrinsicCameraParamEstimationWRansac(CvMat *X, CvMat *x, CvMat *K, CvMat &P, double ransacThreshold, int ransacMaxIter)
{
	int min_set = 5;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vInlierIndex, vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;
	
	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	int nIter = 0;

	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{
		nIter++;
		if (nIter > 1e+4)
			return 0;
		int *randIdx = (int *) malloc(min_set * sizeof(int));
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
		CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
		CvMat *randP = cvCreateMat(3,4,CV_32FC1);
		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		free(randIdx);

		DLT_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		CvMat *H = cvCreateMat(4, 4, CV_32FC1); CvMat *HX = cvCreateMat(randX->rows, randX->cols, CV_32FC1);
		cvSetIdentity(H);
		SetSubMat(H, 0, 0, randP);
		Pxx_inhomo(H, randX, HX);

		bool isFront = true;
		for (int i = 0; i < min_set; i++)
		{
			if (cvGetReal2D(HX, i, 2) < 0)
				isFront = false;
		}
		cvReleaseMat(&H);
		cvReleaseMat(&HX);

		if (!isFront)
		{
			cvReleaseMat(&randx);
			cvReleaseMat(&randX);
			cvReleaseMat(&randP);
			iRansacIter--;
			continue;
		}

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
			CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0)/cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0)/cvGetReal2D(reproj, 2, 0);

			//if ((ip == randIdx[0]) || (ip == randIdx[1]) || (ip == randIdx[2]) || (ip == randIdx[3]))
			//	cout << cvGetReal2D(x, ip, 0) << " " << u << " " << cvGetReal2D(x, ip, 1) << " " << v << endl;
			double dist = (u-cvGetReal2D(x, ip, 0))*(u-cvGetReal2D(x, ip, 0))+(v-cvGetReal2D(x, ip, 1))*(v-cvGetReal2D(x, ip, 1));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}
			cvReleaseMat(&reproj);
			cvReleaseMat(&homo_X);
		}


		// Distance function
		//CvMat *x_ = cvCreateMat(3, X->rows, CV_32FC1);
		//CvMat *e = cvCreateMat(3, X->rows, CV_32FC1);
		//cvMatMul(randP, X_homoT, x_);
		//NormalizingByRow(x_, 2);
		//cvSub(x_homoT, x_, e);
		//for (int ie = 0; ie < e->cols; ie++)
		//{
		//	CvMat *ei = cvCreateMat(3,1,CV_32FC1);
		//	CvMat *xi = cvCreateMat(3,1, CV_32FC1);
		//	GetSubMatColwise(x_homoT, ie, ie, xi);
		//	GetSubMatColwise(e, ie, ie, ei);
		//	double norm = NormL2(ei);
		//	double denorm = NormL2(xi);
		//	double d = norm;
		//	if (d < ransacThreshold)
		//		vInlier.push_back(ie);
		//	else
		//		vOutlier.push_back(ie);
		//	cvReleaseMat(&ei);
		//	cvReleaseMat(&xi);
		//}

		//if (vInlier.size() > maxInlier)
		//{
		//	maxInlier = vInlier.size();
		//	P = *cvCloneMat(randP);
		//	vInlierIndex = vInlier;
		//	vOutlierIndex = vOutlier;
		//}
		//cvReleaseMat(&x_);
		//cvReleaseMat(&e);
		cvReleaseMat(&randx);
		cvReleaseMat(&randX);
		cvReleaseMat(&randP);
	}

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&X_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	if (vInlierIndex.size() < min_set)
		return 0;
	cout << "Number of features to do DLT camera pose estimation: " << vInlierIndex.size() << endl;
	return 1;
}

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP(CvMat *X, CvMat *x, CvMat *K, CvMat &P, double ransacThreshold, int ransacMaxIter)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vInlierIndex, vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{
		int *randIdx = (int *) malloc(min_set * sizeof(int));
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
		CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
		CvMat *randP = cvCreateMat(3,4,CV_32FC1);
		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		

		//DLT_ExtrinsicCameraParamEstimation(randX, randx, K, randP);
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
			CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0)/cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0)/cvGetReal2D(reproj, 2, 0);

			//if ((ip == randIdx[0]) || (ip == randIdx[1]) || (ip == randIdx[2]) || (ip == randIdx[3]))
			//	cout << cvGetReal2D(x, ip, 0) << " " << u << " " << cvGetReal2D(x, ip, 1) << " " << v << endl;
			double dist = (u-cvGetReal2D(x, ip, 0))*(u-cvGetReal2D(x, ip, 0))+(v-cvGetReal2D(x, ip, 1))*(v-cvGetReal2D(x, ip, 1));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}
			cvReleaseMat(&reproj);
			cvReleaseMat(&homo_X);
		}
		free(randIdx);

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			P = *cvCloneMat(randP);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}
		cvReleaseMat(&randx);
		cvReleaseMat(&randX);
		cvReleaseMat(&randP);
	}

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&X_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	if (vInlierIndex.size() < 20)
		return 0;
	cout << "Number of features to do DLT camera pose estimation: " << vInlierIndex.size() << endl;
	return 1;
}

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_Dome(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlier_dome)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vInlierIndex, vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
	CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
	CvMat *randP = cvCreateMat(3,4,CV_32FC1);
	int *randIdx = (int *) malloc(min_set * sizeof(int));

	CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
	CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);
	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{		
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0)/cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0)/cvGetReal2D(reproj, 2, 0);
			double dist = sqrt((u-cvGetReal2D(x, ip, 0))*(u-cvGetReal2D(x, ip, 0))+(v-cvGetReal2D(x, ip, 1))*(v-cvGetReal2D(x, ip, 1)));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}

		}
		

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			SetSubMat(P, 0, 0, randP);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}

		if (vInlier.size() > X->rows * 0.8)
		{
			break;
		}
	}
	CvMat *Xin = cvCreateMat(vInlierIndex.size(), 3, CV_32FC1);
	CvMat *xin = cvCreateMat(vInlierIndex.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierIndex.size(); iInlier++)
	{
		cvSetReal2D(Xin, iInlier, 0, cvGetReal2D(X, vInlierIndex[iInlier], 0));
		cvSetReal2D(Xin, iInlier, 1, cvGetReal2D(X, vInlierIndex[iInlier], 1));
		cvSetReal2D(Xin, iInlier, 2, cvGetReal2D(X, vInlierIndex[iInlier], 2));

		cvSetReal2D(xin, iInlier, 0, cvGetReal2D(x, vInlierIndex[iInlier], 0));
		cvSetReal2D(xin, iInlier, 1, cvGetReal2D(x, vInlierIndex[iInlier], 1));
	}
	EPNP_ExtrinsicCameraParamEstimation(Xin, xin, K, P);

	cvReleaseMat(&Xin);
	cvReleaseMat(&xin);
	cvReleaseMat(&reproj);
	cvReleaseMat(&homo_X);
	free(randIdx);
	cvReleaseMat(&randx);
	cvReleaseMat(&randX);
	cvReleaseMat(&randP);

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&x_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	if (vInlierIndex.size() < 40)
		return 0;
	cout << "Number of features ePnP: " << vInlierIndex.size() << endl;
	vInlier_dome = vInlierIndex;
	return vInlierIndex.size();
}

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vInlierIndex, vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
	CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
	CvMat *randP = cvCreateMat(3,4,CV_32FC1);
	int *randIdx = (int *) malloc(min_set * sizeof(int));

	CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
	CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);
	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{		
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0)/cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0)/cvGetReal2D(reproj, 2, 0);
			double dist = sqrt((u-cvGetReal2D(x, ip, 0))*(u-cvGetReal2D(x, ip, 0))+(v-cvGetReal2D(x, ip, 1))*(v-cvGetReal2D(x, ip, 1)));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}

		}
		

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			SetSubMat(P, 0, 0, randP);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}

		if (vInlier.size() > X->rows * 0.8)
		{
			break;
		}
	}
	CvMat *Xin = cvCreateMat(vInlierIndex.size(), 3, CV_32FC1);
	CvMat *xin = cvCreateMat(vInlierIndex.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierIndex.size(); iInlier++)
	{
		cvSetReal2D(Xin, iInlier, 0, cvGetReal2D(X, vInlierIndex[iInlier], 0));
		cvSetReal2D(Xin, iInlier, 1, cvGetReal2D(X, vInlierIndex[iInlier], 1));
		cvSetReal2D(Xin, iInlier, 2, cvGetReal2D(X, vInlierIndex[iInlier], 2));

		cvSetReal2D(xin, iInlier, 0, cvGetReal2D(x, vInlierIndex[iInlier], 0));
		cvSetReal2D(xin, iInlier, 1, cvGetReal2D(x, vInlierIndex[iInlier], 1));
	}
	//EPNP_ExtrinsicCameraParamEstimation(Xin, xin, K, P);

	cvReleaseMat(&Xin);
	cvReleaseMat(&xin);
	cvReleaseMat(&reproj);
	cvReleaseMat(&homo_X);
	free(randIdx);
	cvReleaseMat(&randx);
	cvReleaseMat(&randX);
	cvReleaseMat(&randP);

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&x_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	if (vInlierIndex.size() < 30)
		return 0;
	cout << "Number of features ePnP: " << vInlierIndex.size() << endl;
	return vInlierIndex.size();
}

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_AD(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlier1)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vInlierIndex, vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
	CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
	CvMat *randP = cvCreateMat(3,4,CV_32FC1);
	int *randIdx = (int *) malloc(min_set * sizeof(int));

	CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
	CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);
	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{		
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0)/cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0)/cvGetReal2D(reproj, 2, 0);
			double dist = sqrt((u-cvGetReal2D(x, ip, 0))*(u-cvGetReal2D(x, ip, 0))+(v-cvGetReal2D(x, ip, 1))*(v-cvGetReal2D(x, ip, 1)));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}

		}
		

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			SetSubMat(P, 0, 0, randP);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}

		if (vInlier.size() > X->rows * 0.8)
		{
			break;
		}
	}
	CvMat *Xin = cvCreateMat(vInlierIndex.size(), 3, CV_32FC1);
	CvMat *xin = cvCreateMat(vInlierIndex.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierIndex.size(); iInlier++)
	{
		cvSetReal2D(Xin, iInlier, 0, cvGetReal2D(X, vInlierIndex[iInlier], 0));
		cvSetReal2D(Xin, iInlier, 1, cvGetReal2D(X, vInlierIndex[iInlier], 1));
		cvSetReal2D(Xin, iInlier, 2, cvGetReal2D(X, vInlierIndex[iInlier], 2));

		cvSetReal2D(xin, iInlier, 0, cvGetReal2D(x, vInlierIndex[iInlier], 0));
		cvSetReal2D(xin, iInlier, 1, cvGetReal2D(x, vInlierIndex[iInlier], 1));
	}
	EPNP_ExtrinsicCameraParamEstimation(Xin, xin, K, P);

	cvReleaseMat(&Xin);
	cvReleaseMat(&xin);
	cvReleaseMat(&reproj);
	cvReleaseMat(&homo_X);
	free(randIdx);
	cvReleaseMat(&randx);
	cvReleaseMat(&randX);
	cvReleaseMat(&randP);

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&x_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	vInlier1 = vInlierIndex;
	if (vInlierIndex.size() < 40)
		return 0;
	cout << "Number of features ePnP: " << vInlierIndex.size() << endl;
	return vInlierIndex.size();
}

int PnP_Opencv(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double th, int nIter, vector<int> &vInlier)
{
	cv::Mat cameraMatrix(3,3,CV_64F);
	cameraMatrix.at<double>(0, 0) = cvGetReal2D(K, 0, 0);
	cameraMatrix.at<double>(0, 1) = cvGetReal2D(K, 0, 1);
	cameraMatrix.at<double>(0, 2) = cvGetReal2D(K, 0, 2);

	cameraMatrix.at<double>(1, 0) = cvGetReal2D(K, 1, 0);
	cameraMatrix.at<double>(1, 1) = cvGetReal2D(K, 1, 1);
	cameraMatrix.at<double>(1, 2) = cvGetReal2D(K, 1, 2);

	cameraMatrix.at<double>(2, 0) = cvGetReal2D(K, 2, 0);
	cameraMatrix.at<double>(2, 1) = cvGetReal2D(K, 2, 1);
	cameraMatrix.at<double>(2, 2) = cvGetReal2D(K, 2, 2);
	//cout << "daf1" << endl;


	vector<cv::Point3f> objectPoints;
	vector<cv::Point2f> imagePoints;
	for (int iPoint = 0; iPoint < X->rows; iPoint++)
	{
		cv::Point3f X1;
		cv::Point2f x1;
		X1.x = cvGetReal2D(X, iPoint, 0);
		X1.y = cvGetReal2D(X, iPoint, 1);
		X1.z = cvGetReal2D(X, iPoint, 2);

		x1.x = cvGetReal2D(x, iPoint, 0);
		x1.y = cvGetReal2D(x, iPoint, 1);
		objectPoints.push_back(X1);
		imagePoints.push_back(x1);
		//cout << iPoint << endl;
	}
	//cout << "daf1" << endl;
	cv::Mat  rvec(3,1,CV_64F), tvec(3,1,CV_64F);
	cv::Mat diff(4,1,CV_64F);
	cv::Mat inl;
	diff.at<double>(0,0) = 0;
	diff.at<double>(1,0) = 0;
	diff.at<double>(2,0) = 0;
	diff.at<double>(3,0) = 0;

	//cout << "daf1" << endl;

	//timeval t1, t2;
	//gettimeofday(&t1, NULL);
	cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, diff, rvec, tvec, false, nIter, th, 100);//, inl, CV_EPNP);
	//gettimeofday(&t2, NULL);
	//cout << "daf" << endl;
	cv::Mat R_mat;
	cv::Rodrigues(rvec, R_mat);

	//cout << "pmp : " << measureTime1(t1,t2) << endl;

	cvSetReal2D(P, 0, 0, R_mat.at<double>(0, 0));
	cvSetReal2D(P, 0, 1, R_mat.at<double>(0, 1));
	cvSetReal2D(P, 0, 2, R_mat.at<double>(0, 2));

	cvSetReal2D(P, 1, 0, R_mat.at<double>(1, 0));
	cvSetReal2D(P, 1, 1, R_mat.at<double>(1, 1));
	cvSetReal2D(P, 1, 2, R_mat.at<double>(1, 2));

	cvSetReal2D(P, 2, 0, R_mat.at<double>(2, 0));
	cvSetReal2D(P, 2, 1, R_mat.at<double>(2, 1));
	cvSetReal2D(P, 2, 2, R_mat.at<double>(2, 2));

	cvSetReal2D(P, 0, 3, tvec.at<double>(0));
	cvSetReal2D(P, 1, 3, tvec.at<double>(1));
	cvSetReal2D(P, 2, 3, tvec.at<double>(2));

	cvMatMul(K, P, P);

	CvMat *X_ = cvCreateMat(4,1,CV_32FC1);
	CvMat *x_ = cvCreateMat(3,1,CV_32FC1);
	for (int i = 0; i < X->rows; i++)
	{
		cvSetReal2D(X_, 0, 0, cvGetReal2D(X, i, 0));
		cvSetReal2D(X_, 1, 0, cvGetReal2D(X, i, 1));
		cvSetReal2D(X_, 2, 0, cvGetReal2D(X, i, 2));
		cvSetReal2D(X_, 3, 0, 1);
		cvMatMul(P, X_, x_);
		double u = cvGetReal2D(x_, 0, 0)/cvGetReal2D(x_, 2, 0);
		double v = cvGetReal2D(x_, 1, 0)/cvGetReal2D(x_, 2, 0);
		
		double dist = sqrt((u-cvGetReal2D(x, i, 0))*(u-cvGetReal2D(x, i, 0))+(v-cvGetReal2D(x, i, 1))*(v-cvGetReal2D(x, i, 1)));
		if (dist < th)
			vInlier.push_back(i);
		
	}
}

int PnP_Opencv1(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double th, int nIter, vector<int> &vInlier)
{
	cv::Mat cameraMatrix(3,3,CV_64F);
	cameraMatrix.at<double>(0, 0) = cvGetReal2D(K, 0, 0);
	cameraMatrix.at<double>(0, 1) = cvGetReal2D(K, 0, 1);
	cameraMatrix.at<double>(0, 2) = cvGetReal2D(K, 0, 2);

	cameraMatrix.at<double>(1, 0) = cvGetReal2D(K, 1, 0);
	cameraMatrix.at<double>(1, 1) = cvGetReal2D(K, 1, 1);
	cameraMatrix.at<double>(1, 2) = cvGetReal2D(K, 1, 2);

	cameraMatrix.at<double>(2, 0) = cvGetReal2D(K, 2, 0);
	cameraMatrix.at<double>(2, 1) = cvGetReal2D(K, 2, 1);
	cameraMatrix.at<double>(2, 2) = cvGetReal2D(K, 2, 2);
	//cout << "daf1" << endl;


	vector<cv::Point3f> objectPoints;
	vector<cv::Point2f> imagePoints;
	for (int iPoint = 0; iPoint < X->rows; iPoint++)
	{
		cv::Point3f X1;
		cv::Point2f x1;
		X1.x = cvGetReal2D(X, iPoint, 0);
		X1.y = cvGetReal2D(X, iPoint, 1);
		X1.z = cvGetReal2D(X, iPoint, 2);

		x1.x = cvGetReal2D(x, iPoint, 0);
		x1.y = cvGetReal2D(x, iPoint, 1);
		objectPoints.push_back(X1);
		imagePoints.push_back(x1);
		//cout << iPoint << endl;
	}
	//cout << "daf1" << endl;
	cv::Mat  rvec(3,1,CV_64F), tvec(3,1,CV_64F);
	cv::Mat diff(4,1,CV_64F);
	cv::Mat inl;
	diff.at<double>(0,0) = 0;
	diff.at<double>(1,0) = 0;
	diff.at<double>(2,0) = 0;
	diff.at<double>(3,0) = 0;

	//cout << "daf1" << endl;
	cv::solvePnP(objectPoints, imagePoints, cameraMatrix, diff, rvec, tvec, false, CV_P3P);//, inl, CV_EPNP);
	//cout << "daf" << endl;
	cv::Mat R_mat;
	cv::Rodrigues(rvec, R_mat);

	cvSetReal2D(P, 0, 0, R_mat.at<double>(0, 0));
	cvSetReal2D(P, 0, 1, R_mat.at<double>(0, 1));
	cvSetReal2D(P, 0, 2, R_mat.at<double>(0, 2));

	cvSetReal2D(P, 1, 0, R_mat.at<double>(1, 0));
	cvSetReal2D(P, 1, 1, R_mat.at<double>(1, 1));
	cvSetReal2D(P, 1, 2, R_mat.at<double>(1, 2));

	cvSetReal2D(P, 2, 0, R_mat.at<double>(2, 0));
	cvSetReal2D(P, 2, 1, R_mat.at<double>(2, 1));
	cvSetReal2D(P, 2, 2, R_mat.at<double>(2, 2));

	cvSetReal2D(P, 0, 3, tvec.at<double>(0));
	cvSetReal2D(P, 1, 3, tvec.at<double>(1));
	cvSetReal2D(P, 2, 3, tvec.at<double>(2));

	cvMatMul(K, P, P);

	CvMat *X_ = cvCreateMat(4,1,CV_32FC1);
	CvMat *x_ = cvCreateMat(3,1,CV_32FC1);
	for (int i = 0; i < X->rows; i++)
	{
		cvSetReal2D(X_, 0, 0, cvGetReal2D(X, i, 0));
		cvSetReal2D(X_, 1, 0, cvGetReal2D(X, i, 1));
		cvSetReal2D(X_, 2, 0, cvGetReal2D(X, i, 2));
		cvSetReal2D(X_, 3, 0, 1);
		cvMatMul(P, X_, x_);
		double u = cvGetReal2D(x_, 0, 0)/cvGetReal2D(x_, 2, 0);
		double v = cvGetReal2D(x_, 1, 0)/cvGetReal2D(x_, 2, 0);
		
		double dist = sqrt((u-cvGetReal2D(x, i, 0))*(u-cvGetReal2D(x, i, 0))+(v-cvGetReal2D(x, i, 1))*(v-cvGetReal2D(x, i, 1)));
		if (dist < th)
			vInlier.push_back(i);
		
	}
}

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierIndex)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
	CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
	CvMat *randP = cvCreateMat(3,4,CV_32FC1);
	int *randIdx = (int *) malloc(min_set * sizeof(int));

	CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
	CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);
	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{		
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0)/cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0)/cvGetReal2D(reproj, 2, 0);
			double dist = sqrt((u-cvGetReal2D(x, ip, 0))*(u-cvGetReal2D(x, ip, 0))+(v-cvGetReal2D(x, ip, 1))*(v-cvGetReal2D(x, ip, 1)));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}

		}
		

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			SetSubMat(P, 0, 0, randP);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}

		//if (vInlier.size() > X->rows * 0.8)
		//{
		//	break;
		//}
	}

	CvMat *Xin = cvCreateMat(vInlierIndex.size(), 3, CV_32FC1);
	CvMat *xin = cvCreateMat(vInlierIndex.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierIndex.size(); iInlier++)
	{
		cvSetReal2D(Xin, iInlier, 0, cvGetReal2D(X, vInlierIndex[iInlier], 0));
		cvSetReal2D(Xin, iInlier, 1, cvGetReal2D(X, vInlierIndex[iInlier], 1));
		cvSetReal2D(Xin, iInlier, 2, cvGetReal2D(X, vInlierIndex[iInlier], 2));

		cvSetReal2D(xin, iInlier, 0, cvGetReal2D(x, vInlierIndex[iInlier], 0));
		cvSetReal2D(xin, iInlier, 1, cvGetReal2D(x, vInlierIndex[iInlier], 1));
	}
	//EPNP_ExtrinsicCameraParamEstimation(Xin, xin, K, P);

	cvReleaseMat(&Xin);
	cvReleaseMat(&xin);
	cvReleaseMat(&reproj);
	cvReleaseMat(&homo_X);
	free(randIdx);
	cvReleaseMat(&randx);
	cvReleaseMat(&randX);
	cvReleaseMat(&randP);

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&x_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	//if (vInlierIndex.size() < 20)
	//	return 0;
	//cout << "Number of features to do ePNP camera pose estimation: " << vInlierIndex.size() << endl;
	return vInlierIndex.size();
}

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs_global(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierIndex)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
	CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
	CvMat *randP = cvCreateMat(3,4,CV_32FC1);
	int *randIdx = (int *) malloc(min_set * sizeof(int));

	CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
	CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);
	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{		
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0)/cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0)/cvGetReal2D(reproj, 2, 0);
			double dist = sqrt((u-cvGetReal2D(x, ip, 0))*(u-cvGetReal2D(x, ip, 0))+(v-cvGetReal2D(x, ip, 1))*(v-cvGetReal2D(x, ip, 1)));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}

		}
		

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			SetSubMat(P, 0, 0, randP);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}

		//if (vInlier.size() > X->rows * 0.8)
		//{
		//	break;
		//}
	}

	CvMat *Xin = cvCreateMat(vInlierIndex.size(), 3, CV_32FC1);
	CvMat *xin = cvCreateMat(vInlierIndex.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierIndex.size(); iInlier++)
	{
		cvSetReal2D(Xin, iInlier, 0, cvGetReal2D(X, vInlierIndex[iInlier], 0));
		cvSetReal2D(Xin, iInlier, 1, cvGetReal2D(X, vInlierIndex[iInlier], 1));
		cvSetReal2D(Xin, iInlier, 2, cvGetReal2D(X, vInlierIndex[iInlier], 2));

		cvSetReal2D(xin, iInlier, 0, cvGetReal2D(x, vInlierIndex[iInlier], 0));
		cvSetReal2D(xin, iInlier, 1, cvGetReal2D(x, vInlierIndex[iInlier], 1));
	}
	EPNP_ExtrinsicCameraParamEstimation(Xin, xin, K, P);

	cvReleaseMat(&Xin);
	cvReleaseMat(&xin);
	cvReleaseMat(&reproj);
	cvReleaseMat(&homo_X);
	free(randIdx);
	cvReleaseMat(&randx);
	cvReleaseMat(&randX);
	cvReleaseMat(&randP);

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&x_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	//if (vInlierIndex.size() < 20)
	//	return 0;
	//cout << "Number of features to do ePNP camera pose estimation: " << vInlierIndex.size() << endl;
	return vInlierIndex.size();
}


int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_face(CvMat *X, CvMat *x, CvMat *K, CvMat &P)
{
	CvMat *P_ = cvCreateMat(3,4,CV_32FC1);
	EPNP_ExtrinsicCameraParamEstimation(X, x, K, P_);
	P = *cvCloneMat(P_);
	cvReleaseMat(&P_);
	return 1;
}


int DLT_ExtrinsicCameraParamEstimationWRansac_KRT(CvMat *X, CvMat *x, CvMat *K, CvMat &P, double ransacThreshold, int ransacMaxIter)
{
	if (X->rows < 6)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vInlierIndex, vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{
		int *randIdx = (int *) malloc(6 * sizeof(int));
		for (int iIdx = 0; iIdx < 6; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		CvMat *randx = cvCreateMat(6, 2, CV_32FC1);
		CvMat *randX = cvCreateMat(6, 3, CV_32FC1);
		CvMat *randP = cvCreateMat(3,4,CV_32FC1);
		for (int iIdx = 0; iIdx < 6; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		free(randIdx);

		CvMat *K_ = cvCreateMat(3,3,CV_32FC1);
		DLT_ExtrinsicCameraParamEstimation_KRT(randX, randx, K_, randP);
		double k33 = cvGetReal2D(K_, 2, 2);
		ScalarMul(K_, 1/k33, K_);

		CvMat *H = cvCreateMat(4, 4, CV_32FC1); CvMat *HX = cvCreateMat(randX->rows, randX->cols, CV_32FC1);
		cvSetIdentity(H);
		SetSubMat(H, 0, 0, randP);
		Pxx_inhomo(H, randX, HX);

		bool isFront = true;
		for (int i = 0; i < 6; i++)
		{
			if (cvGetReal2D(HX, i, 2) < 0)
				isFront = false;
		}
		cvReleaseMat(&H);
		cvReleaseMat(&HX);

		if ((!isFront) || (cvGetReal2D(K, 0,0) < 0))
		{
			cvReleaseMat(&randx);
			cvReleaseMat(&randX);
			cvReleaseMat(&randP);
			iRansacIter--;
			continue;
		}

		vInlier.clear();
		vOutlier.clear();
		// Distance function
		CvMat *x_ = cvCreateMat(3, X->rows, CV_32FC1);
		CvMat *e = cvCreateMat(3, X->rows, CV_32FC1);
		cvMatMul(randP, X_homoT, x_);
		NormalizingByRow(x_, 2);
		cvSub(x_homoT, x_, e);
		for (int ie = 0; ie < e->cols; ie++)
		{
			CvMat *ei = cvCreateMat(3,1,CV_32FC1);
			CvMat *xi = cvCreateMat(3,1, CV_32FC1);
			GetSubMatColwise(x_homoT, ie, ie, xi);
			GetSubMatColwise(e, ie, ie, ei);
			double norm = NormL2(ei);
			double denorm = NormL2(xi);
			double d = norm;
			if (d < ransacThreshold)
				vInlier.push_back(ie);
			else
				vOutlier.push_back(ie);
			cvReleaseMat(&ei);
			cvReleaseMat(&xi);
		}

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			P = *cvCloneMat(randP);
			K = cvCloneMat(K_);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}

		if (vInlier.size() > 0.8*X->rows)
			break;
		cvReleaseMat(&x_);
		cvReleaseMat(&e);
		cvReleaseMat(&randx);
		cvReleaseMat(&randX);
		cvReleaseMat(&randP);
		cvReleaseMat(&K_);
	}

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&X_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	if (vInlierIndex.size() < 10)
		return 0;
	cout << "Number of features to do DLT camera pose estimation: " << vInlierIndex.size() << endl;
	return 1;
}

int DLT_ExtrinsicCameraParamEstimation_KRT(CvMat *X, CvMat *x, CvMat *K, CvMat &P)
{
	if (X->rows < 6)
		return 0;

	CvMat *P_ = cvCreateMat(3,4,CV_32FC1);
	DLT_ExtrinsicCameraParamEstimation_KRT(X, x, K, P_);
	P = *cvCloneMat(P_);
	return 1;
}


/*
void SparseBundleAdjustment_MOT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, CvMat *K, vector<int> visibleStructureID)
{
	//PrintAlgorithm("Sparse bundle adjustment motion only");
	//vector<double> cameraParameter, feature2DParameter;
	//vector<char> vMask;
	//double *dCovFeatures = 0;
	//AdditionalData adata;// focal_x focal_y princ_x princ_y
	//double intrinsic[4];
	//intrinsic[0] = cvGetReal2D(K, 0, 0);
	//intrinsic[1] = cvGetReal2D(K, 1, 1);
	//intrinsic[2] = cvGetReal2D(K, 0, 2);
	//intrinsic[3] = cvGetReal2D(K, 1, 2);
	//adata.intrinsic = intrinsic;

	//GetParameterForSBA(vFeature, vUsedFrame, cP, X, K, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	//double nCameraParam = 7;
	//int nFeatures = vFeature.size(); 
	//int nFrames = vUsedFrame.size(); 
	//char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	//double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	//double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	//
	//for (int i = 0; i < cameraParameter.size(); i++)
	//	dCameraParameter[i] = cameraParameter[i];
	//for (int i = 0; i < vMask.size(); i++)
	//	dVMask[i] = vMask[i];
	//for (int i = 0; i < feature2DParameter.size(); i++)
	//	dFeature2DParameter[i] = feature2DParameter[i];

	//adata.XYZ = &(dCameraParameter[7*vUsedFrame.size()]);
	//double opt[5];
	//opt[0] = 1e-3;
	//opt[1] = 1e-12;
	//opt[2] = 1e-12;
	//opt[3] = 1e-12;
	//opt[4] = 0;
	//double info[12];
	//sba_mot_levmar(visibleStructureID.size(), vUsedFrame.size(), 1, dVMask,  dCameraParameter, 7, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOT, NULL, &adata,
	//				1e+3, 0, opt, info);
	//PrintSBAInfo(info);
	//RetrieveParameterFromSBA(dCameraParameter, K, cP, X, visibleStructureID);
	//free(dVMask);
	//free(dFeature2DParameter);
	//free(dCameraParameter);
}

void SparseBundleAdjustment_MOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, CvMat *K, vector<int> visibleStructureID)
{
	//PrintAlgorithm("Sparse bundle adjustment motion and structure");
	//vector<double> cameraParameter, feature2DParameter;
	//vector<char> vMask;
	//double *dCovFeatures = 0;
	//AdditionalData adata;// focal_x focal_y princ_x princ_y
	//double intrinsic[4];
	//intrinsic[0] = cvGetReal2D(K, 0, 0);
	//intrinsic[1] = cvGetReal2D(K, 1, 1);
	//intrinsic[2] = cvGetReal2D(K, 0, 2);
	//intrinsic[3] = cvGetReal2D(K, 1, 2);
	//adata.intrinsic = intrinsic;

	//GetParameterForSBA(vFeature, vUsedFrame, cP, X, K, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	//int NZ = 0;
	//for (int i = 0; i < vMask.size(); i++)
	//{
	//	if (vMask[i])
	//		NZ++;
	//}
	//double nCameraParam = 7;
	//int nFeatures = vFeature.size(); 
	//int nFrames = vUsedFrame.size(); 
	//char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	//double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	//double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	//for (int i = 0; i < cameraParameter.size(); i++)
	//	dCameraParameter[i] = cameraParameter[i];
	//for (int i = 0; i < vMask.size(); i++)
	//	dVMask[i] = vMask[i];
	//for (int i = 0; i < feature2DParameter.size(); i++)
	//	dFeature2DParameter[i] = feature2DParameter[i];

	//double opt[5];
	//opt[0] = 1e-3;
	//opt[1] = 1e-5;//1e-12;
	//opt[2] = 1e-5;//1e-12;
	//opt[3] = 1e-5;//1e-12;
	//opt[4] = 0;
	//double info[12];
	//sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR, NULL, &adata,
	//	1e+3, 0, opt, info);
	//PrintSBAInfo(info);
	//RetrieveParameterFromSBA(dCameraParameter, K, cP, X, visibleStructureID);

	//free(dVMask);
	//free(dFeature2DParameter);
	//free(dCameraParameter);
}

void SparseBundleAdjustment_MOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		double *intrinsic = (double *) malloc(4 * sizeof(double));
		intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
		intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
		intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
		intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
		adata.vIntrinsic.push_back(intrinsic);
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 7;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR, NULL, &adata,
		1e+3, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	cout << "kk" << endl;
	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	cout << "kk" << endl;
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		free(adata.vIntrinsic[i]);
	}
	cout << "kk" << endl;
}

void SparseBundleAdjustment_MOTSTR_mem(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		double *intrinsic = (double *) malloc(4 * sizeof(double));
		intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
		intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
		intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
		intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
		adata.vIntrinsic.push_back(intrinsic);
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 7;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	PrintMat(cP[1]);
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast, NULL, &adata,
		1e+3, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	PrintMat(cP[1]);
	cout << "----------------" << endl;
	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		free(adata.vIntrinsic[i]);
	}
	adata.vIntrinsic.clear();
}

void SparseBundleAdjustment_MOTSTR_mem_fast(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<int> visibleStructureID;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			visibleStructureID.push_back(vFeature[iFeature].id);
		}
	}

	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	for (int iFrame = 0; iFrame < vCamera[0].vTakenFrame.size(); iFrame++)
	{
		double *intrinsic = (double *) malloc(4 * sizeof(double));
		intrinsic[0] = cvGetReal2D(vCamera[0].vK[iFrame], 0, 0);
		intrinsic[1] = cvGetReal2D(vCamera[0].vK[iFrame], 1, 1);
		intrinsic[2] = cvGetReal2D(vCamera[0].vK[iFrame], 0, 2);
		intrinsic[3] = cvGetReal2D(vCamera[0].vK[iFrame], 1, 2);
		adata.vIntrinsic.push_back(intrinsic);
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 7;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 2, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast, NULL, &adata,
		1e+2, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		free(adata.vIntrinsic[i]);
	}
	adata.vIntrinsic.clear();
}

void SparseBundleAdjustment_MOTSTR_mem_fast_Dome(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<int> visibleStructureID;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			visibleStructureID.push_back(vFeature[iFeature].id);
		}
	}

	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	for (int iFrame = 0; iFrame < vCamera[0].vTakenFrame.size(); iFrame++)
	{
		double *intrinsic = (double *) malloc(4 * sizeof(double));
		intrinsic[0] = cvGetReal2D(vCamera[0].vK[iFrame], 0, 0);
		intrinsic[1] = cvGetReal2D(vCamera[0].vK[iFrame], 1, 1);
		intrinsic[2] = cvGetReal2D(vCamera[0].vK[iFrame], 0, 2);
		intrinsic[3] = cvGetReal2D(vCamera[0].vK[iFrame], 1, 2);
		adata.vIntrinsic.push_back(intrinsic);
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA_Dome(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 11;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-15;
	opt[2] = 1e-15;
	opt[3] = 1e-15;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast_Dome, NULL, &adata,
		1e+2, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem_Dome(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		free(adata.vIntrinsic[i]);
	}
	adata.vIntrinsic.clear();
}


void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, double omega)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<int> visibleStructureID;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			visibleStructureID.push_back(vFeature[iFeature].id);
		}
	}

	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		double *intrinsic = (double *) malloc(6 * sizeof(double));
		intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
		intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
		intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
		intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
		intrinsic[4] = omega;
		intrinsic[5] = 2*tan(omega/2);

		adata.vIntrinsic.push_back(intrinsic);
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA_Distortion(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 7;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast_Distortion, NULL, &adata,
		1e+2, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		free(adata.vIntrinsic[i]);
	}
	adata.vIntrinsic.clear();
}

void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera,
	double omega, double princ_x1, double princ_y1)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<int> visibleStructureID;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			visibleStructureID.push_back(vFeature[iFeature].id);
		}
	}

	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		double *intrinsic = (double *) malloc(8 * sizeof(double));
		intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
		intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
		intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
		intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
		intrinsic[4] = omega;
		intrinsic[5] = 2*tan(omega/2);
		intrinsic[6] = princ_x1;
		intrinsic[7] = princ_y1;
		adata.vIntrinsic.push_back(intrinsic);
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	//cout << "OK" << endl;
	//for (int i = 0; i < cP.size(); i++)
	//{
	//	PrintMat(cP[i]);
	//}
	//for (int i = 0; i < visibleStructureID.size(); i++)
	//{
	//	cout << cvGetReal2D(X, visibleStructureID[i], 0) << " " <<cvGetReal2D(X, visibleStructureID[i], 1) << " " <<cvGetReal2D(X, visibleStructureID[i], 2) << " " << endl; 
	//}
	
	GetParameterForSBA_Distortion(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	cout << "OK" << endl;
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	
	double nCameraParam = 7;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	cout << "OK" << endl;
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast_Distortion_ObstacleDetection, NULL, &adata,
		1e+2, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		free(adata.vIntrinsic[i]);
	}
	adata.vIntrinsic.clear();
}

void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera,
	vector<double> &vOmega, vector<double> &vprinc_x1, vector<double> &vprinc_y1, vector<CvMat *> &vK)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<int> visibleStructureID;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			visibleStructureID.push_back(vFeature[iFeature].id);
		}
	}

	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	//for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	//{
	//	double *intrinsic = (double *) malloc(8 * sizeof(double));
	//	intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
	//	intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
	//	intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
	//	intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
	//	intrinsic[4] = 0;
	//	intrinsic[5] = 0;
	//	intrinsic[6] = 0;
	//	intrinsic[7] = 0;
	//	adata.vIntrinsic.push_back(intrinsic);
	//}

	adata.vOmega = vOmega;
	adata.vpx1 = vprinc_x1;
	adata.vpy1 = vprinc_y1;
	for (int i = 0; i < vK.size(); i++)
	{
		adata.vK.push_back(cvCloneMat(vK[i]));
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	//cout << "OK" << endl;
	//for (int i = 0; i < cP.size(); i++)
	//{
	//	PrintMat(cP[i]);
	//}
	//for (int i = 0; i < visibleStructureID.size(); i++)
	//{
	//	cout << cvGetReal2D(X, visibleStructureID[i], 0) << " " <<cvGetReal2D(X, visibleStructureID[i], 1) << " " <<cvGetReal2D(X, visibleStructureID[i], 2) << " " << endl; 
	//}
	
	GetParameterForSBA_Distortion_Each(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask, vK,
		vOmega, vprinc_x1, vprinc_y1);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	
	double nCameraParam = 14;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 0, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast_Distortion_ObstacleDetection1, NULL, &adata,
		5e+2, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem_Each(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames, vOmega, vprinc_x1, vprinc_y1, vK);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		//free(adata.vIntrinsic[i]);
		cvReleaseMat(&adata.vK[i]);
	}
	adata.vK.clear();

	//adata.vIntrinsic.clear();
}

void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_iPhone(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera,
	vector<double> &vk1, vector<CvMat *> &vK)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<int> visibleStructureID;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			visibleStructureID.push_back(vFeature[iFeature].id);
		}
	}

	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	//for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	//{
	//	double *intrinsic = (double *) malloc(8 * sizeof(double));
	//	intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
	//	intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
	//	intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
	//	intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
	//	intrinsic[4] = 0;
	//	intrinsic[5] = 0;
	//	intrinsic[6] = 0;
	//	intrinsic[7] = 0;
	//	adata.vIntrinsic.push_back(intrinsic);
	//}

	//adata.vk1 = vk1;
	for (int i = 0; i < vK.size(); i++)
	{
		adata.vK.push_back(cvCloneMat(vK[i]));
	}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	//cout << "OK" << endl;
	//for (int i = 0; i < cP.size(); i++)
	//{
	//	PrintMat(cP[i]);
	//}
	//for (int i = 0; i < visibleStructureID.size(); i++)
	//{
	//	cout << cvGetReal2D(X, visibleStructureID[i], 0) << " " <<cvGetReal2D(X, visibleStructureID[i], 1) << " " <<cvGetReal2D(X, visibleStructureID[i], 2) << " " << endl; 
	//}
	
	GetParameterForSBA_Distortion_Each_iPhone(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask, vK,
		vk1);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	
	double nCameraParam = 12;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-10;
	opt[1] = 1e-20;
	opt[2] = 1e-20;
	opt[3] = 1e-20;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 0, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast_Distortion_iPhone, NULL, &adata,
		1e+2, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem_Each_iPhone(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames, vk1, vK);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		//free(adata.vIntrinsic[i]);
		cvReleaseMat(&adata.vK[i]);
	}
	adata.vK.clear();

	//adata.vIntrinsic.clear();
}

void SparseBundleAdjustment_MOTSTR_mem_fast_Distortion_GoPro(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<CvMat *> vK, vector<double> vOmega, vector<double> vpx, vector<double> vpy)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<int> visibleStructureID;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			visibleStructureID.push_back(vFeature[iFeature].id);
		}
	}

	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	for (int iFrame = 0; iFrame < vK.size(); iFrame++)
	{
		double *intrinsic = (double *) malloc(8 * sizeof(double));
		intrinsic[0] = cvGetReal2D(vK[iFrame], 0, 0);
		intrinsic[1] = cvGetReal2D(vK[iFrame], 1, 1);
		intrinsic[2] = cvGetReal2D(vK[iFrame], 0, 2);
		intrinsic[3] = cvGetReal2D(vK[iFrame], 1, 2);
		intrinsic[4] = vOmega[iFrame];
		intrinsic[5] = 2*tan(vOmega[iFrame]/2);
		intrinsic[6] = vpx[iFrame];
		intrinsic[7] = vpy[iFrame];
		adata.vIntrinsic.push_back(intrinsic);
	}
	cout << "OK" << endl;
	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	//cout << "OK" << endl;
	//for (int i = 0; i < cP.size(); i++)
	//{
	//	PrintMat(cP[i]);
	//}
	//for (int i = 0; i < visibleStructureID.size(); i++)
	//{
	//	cout << cvGetReal2D(X, visibleStructureID[i], 0) << " " <<cvGetReal2D(X, visibleStructureID[i], 1) << " " <<cvGetReal2D(X, visibleStructureID[i], 2) << " " << endl; 
	//}
	
	GetParameterForSBA_Distortion(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask, vK);
	cout << "OK" << endl;
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	
	double nCameraParam = 7;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	cout << "OK" << endl;
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_MOTSTR_fast_Distortion_GoPro, NULL, &adata,
		1e+2, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_mem(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
	free(dCovFeatures);
	for (int i = 0; i < adata.vIntrinsic.size(); i++)
	{
		free(adata.vIntrinsic[i]);
	}
	adata.vIntrinsic.clear();
}



void SparseBundleAdjustment_KMOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	//for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	//{
	//	double *intrinsic = (double *) malloc(4 * sizeof(double));
	//	intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
	//	intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
	//	intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
	//	intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
	//	adata.vIntrinsic.push_back(intrinsic);
	//}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA_KRT(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 7+4;
	int nFeatures = visibleStructureID.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
	{
		dVMask[i] = vMask[i];
	}
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_KMOTSTR, NULL, &adata,
		1e+3, 0, opt, info);
	PrintSBAInfo(info, feature2DParameter.size()/2);
	RetrieveParameterFromSBA_KRT(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
}

void SparseBundleAdjustment_KMOTSTR_x(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	//for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	//{
	//	double *intrinsic = (double *) malloc(4 * sizeof(double));
	//	intrinsic[0] = cvGetReal2D(vCamera[iCamera].K, 0, 0);
	//	intrinsic[1] = cvGetReal2D(vCamera[iCamera].K, 1, 1);
	//	intrinsic[2] = cvGetReal2D(vCamera[iCamera].K, 0, 2);
	//	intrinsic[3] = cvGetReal2D(vCamera[iCamera].K, 1, 2);
	//	adata.vIntrinsic.push_back(intrinsic);
	//}

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA_KRT(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 7+4;
	int nFeatures = visibleStructureID.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
	{
		dVMask[i] = vMask[i];
	}
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_KMOTSTR, NULL, &adata,
		1e+3, 0, opt, info);
	PrintSBAInfo(info);
	RetrieveParameterFromSBA_KRT(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
}


void SparseBundleAdjustment_KDMOTSTR(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID)
{
	PrintAlgorithm("Sparse bundle adjustment motion and structure");
	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA_KDRT(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);
	int NZ = 0;
	for (int i = 0; i < vMask.size(); i++)
	{
		if (vMask[i])
			NZ++;
	}
	double nCameraParam = 7+4+2;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(visibleStructureID.size(), 0, vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_KDMOTSTR, NULL, &adata,
		1e+3, 0, opt, info);
	PrintSBAInfo(info);
	RetrieveParameterFromSBA_KDRT(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
}

void PatchTriangulationRefinement(CvMat *K, vector<CvMat *> vC, vector<CvMat *> vR, 
								  vector<double> vx11, vector<double> vy11, 
								  vector<double> vx12, vector<double> vy12,
								  vector<double> vx21, vector<double> vy21,
								  vector<double> vx22, vector<double> vy22,
								  double X, double Y, double Z, 
								  CvMat *X11, CvMat *X12, CvMat *X21, CvMat *X22, CvMat *pi)
{
	PrintAlgorithm("Patch Triangulation Refinement");
	vector<CvMat *> vP;
	for (int i = 0; i < vC.size(); i++)
	{
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		ScalarMul(vC[i], -1, t); 
		cvSetIdentity(P);
		SetSubMat(P, 0, 3, t);
		cvMatMul(vR[i], P, P);
		cvMatMul(K, P, P);
		vP.push_back(P);
	}
	AdditionalData adata;
	adata.vP = vP;
	adata.X = X;	adata.Y = Y;	adata.Z = Z;

	vector<double> XParameter, feature2DParameter;
	XParameter.push_back(cvGetReal2D(pi, 0, 0));
	XParameter.push_back(cvGetReal2D(pi, 1, 0));
	XParameter.push_back(cvGetReal2D(pi, 2, 0));
	XParameter.push_back(cvGetReal2D(X11, 0, 0));
	XParameter.push_back(cvGetReal2D(X11, 1, 0));
	XParameter.push_back(cvGetReal2D(X12, 0, 0));
	XParameter.push_back(cvGetReal2D(X12, 1, 0));
	XParameter.push_back(cvGetReal2D(X21, 0, 0));
	XParameter.push_back(cvGetReal2D(X21, 1, 0));
	XParameter.push_back(cvGetReal2D(X22, 0, 0));
	XParameter.push_back(cvGetReal2D(X22, 1, 0));
	
	for (int iFeature = 0; iFeature < vx11.size(); iFeature++)
	{
		feature2DParameter.push_back(vx11[iFeature]);
		feature2DParameter.push_back(vy11[iFeature]);
		feature2DParameter.push_back(vx12[iFeature]);
		feature2DParameter.push_back(vy12[iFeature]);
		feature2DParameter.push_back(vx21[iFeature]);
		feature2DParameter.push_back(vy21[iFeature]);
		feature2DParameter.push_back(vx22[iFeature]);
		feature2DParameter.push_back(vy22[iFeature]);
	}

	//for (int i = 0; i < feature2DParameter.size(); i++)
	//{
	//	cout << feature2DParameter[i] << " ";
	//}
	//cout << endl;


	int nFeatures = vx11.size(); 
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dXParameter = (double *) malloc(XParameter.size() * sizeof(double));
	
	for (int i = 0; i < XParameter.size(); i++)
		dXParameter[i] = XParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(XParameter.size(), feature2DParameter.size())+XParameter.size()*XParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");

	int ret ;//dlevmar_dif(Projection3Donto2D_Patch, dXParameter, dFeature2DParameter, XParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);

	double pi1 = dXParameter[0];
	double pi2 = dXParameter[1];
	double pi3 = dXParameter[2];
	double pi4 = -pi1*X-pi2*Y-pi3*Z;

	cvSetReal2D(pi, 0, 0, dXParameter[0]);
	cvSetReal2D(pi, 1, 0, dXParameter[1]);
	cvSetReal2D(pi, 2, 0, dXParameter[2]);
	cvSetReal2D(pi, 3, 0, pi4);

	cvSetReal2D(X11, 0, 0, dXParameter[3]);
	cvSetReal2D(X11, 1, 0, dXParameter[4]);
	double Z11 = (-pi4-pi1*cvGetReal2D(X11, 0, 0)-pi2*cvGetReal2D(X11, 1, 0))/pi3;
	cvSetReal2D(X11, 2, 0, Z11);

	cvSetReal2D(X12, 0, 0, dXParameter[5]);
	cvSetReal2D(X12, 1, 0, dXParameter[6]);
	double Z12 = (-pi4-pi1*cvGetReal2D(X12, 0, 0)-pi2*cvGetReal2D(X12, 1, 0))/pi3;
	cvSetReal2D(X12, 2, 0, Z12);

	cvSetReal2D(X21, 0, 0, dXParameter[7]);
	cvSetReal2D(X21, 1, 0, dXParameter[8]);
	double Z21 = (-pi4-pi1*cvGetReal2D(X21, 0, 0)-pi2*cvGetReal2D(X21, 1, 0))/pi3;
	cvSetReal2D(X21, 2, 0, Z21);

	cvSetReal2D(X22, 0, 0, dXParameter[9]);
	cvSetReal2D(X22, 1, 0, dXParameter[10]);
	double Z22 = (-pi4-pi1*cvGetReal2D(X22, 0, 0)-pi2*cvGetReal2D(X22, 1, 0))/pi3;
	cvSetReal2D(X22, 2, 0, Z22);

	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dXParameter);
	free(work);

	for (int i = 0; i < vP.size(); i++)
	{
		cvReleaseMat(&vP[i]);
	}
}

void TriangulationRefinement(CvMat *K, double omega, vector<CvMat *> vP, vector<double> vx, vector<double> vy, double &X, double &Y, double &Z)
{
	PrintAlgorithm("Point Triangulation Refinement");
	AdditionalData adata;
	double intrinsic[5] = {cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1), cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2), omega};
	adata.vIntrinsic.push_back(intrinsic);
	adata.vP = vP;

	vector<double> XParameter, feature2DParameter;
	XParameter.push_back(X);
	XParameter.push_back(Y);
	XParameter.push_back(Z);
	
	for (int iFeature = 0; iFeature < vx.size(); iFeature++)
	{
		feature2DParameter.push_back(vx[iFeature]);
		feature2DParameter.push_back(vy[iFeature]);
	}

	int nFeatures = vx.size(); 
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dXParameter = (double *) malloc(XParameter.size() * sizeof(double));
	
	for (int i = 0; i < XParameter.size(); i++)
		dXParameter[i] = XParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(XParameter.size(), feature2DParameter.size())+XParameter.size()*XParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");

	int ret ;//dlevmar_dif(Projection3Donto2D_STR_fast_SO, dXParameter, dFeature2DParameter, XParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);

	X = dXParameter[0];
	Y = dXParameter[1];
	Z = dXParameter[2];
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dXParameter);
	free(work);

	//for (int i = 0; i < adata.vP.size(); i++)
	//{
	//	cvReleaseMat(&adata.vP[i]);
	//}
}

void FaceRefinement(vector<CvMat *> &vX1, vector<CvMat *> vP1, vector<CvMat *> vx1, vector<int> vIdx1)
{
	cout << "Face Refinement" << endl;

	int windowsize = 5;
	int stepsize = 1;
	
	
	for (int iw = 0; iw < vX1.size(); iw+=stepsize)
	{
		vector<CvMat *> vX, vP, vx;
		vector<int> vIdx;
		vector<int> vRefer;
		for (int ii = iw; ii < MIN(vX1.size(), iw+windowsize); ii++)
		{
			vX.push_back(vX1[ii]);
			vRefer.push_back(ii);
			for (int id = 0; id < vIdx1.size(); id++)
			{
				if (vIdx1[id] == ii)
				{
					vP.push_back(vP1[id]);
					vx.push_back(vx1[id]);
					vIdx.push_back(vIdx1[id]-iw);					
				}
			}
		}

		AdditionalData adata;
		adata.vP = vP;
		adata.vx = vx;
		adata.vIdx = vIdx;
		adata.nFrames = vX.size();

		vector<double> XParameter, feature2DParameter;

		for (int iX = 0; iX < vX.size(); iX++)
		{
			XParameter.push_back(cvGetReal2D(vX[iX], 0, 0));
			XParameter.push_back(cvGetReal2D(vX[iX], 1, 0));
			XParameter.push_back(cvGetReal2D(vX[iX], 2, 0));
		}
	
		for (int iX = 0; iX < vX.size()-1; iX++)
		{
			feature2DParameter.push_back(0);
			feature2DParameter.push_back(0);
			feature2DParameter.push_back(0);
		}

		for (int iFeature = 0; iFeature < vx.size(); iFeature++)
		{
			feature2DParameter.push_back(cvGetReal2D(vx[iFeature], 0, 0));
			feature2DParameter.push_back(cvGetReal2D(vx[iFeature], 1, 0));
		}

		//for (int i = 0; i < feature2DParameter.size(); i++)
		//{
		//	cout << feature2DParameter[i] << " ";
		//}
		//cout <<  endl;


		//int nFeatures = vx.size(); 
		double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
		double *dXParameter = (double *) malloc(XParameter.size() * sizeof(double));
	
		for (int i = 0; i < XParameter.size(); i++)
			dXParameter[i] = XParameter[i];
		for (int i = 0; i < feature2DParameter.size(); i++)
			dFeature2DParameter[i] = feature2DParameter[i];
		double opt[5];
		opt[0] = 1e-3;
		opt[1] = 1e-12;
		opt[2] = 1e-12;
		opt[3] = 1e-12;
		opt[4] = 0;
		double info[12];

		adata.measurements = feature2DParameter;

		double *work ;// (double*)malloc((LM_DIF_WORKSZ(XParameter.size(), feature2DParameter.size())+XParameter.size()*XParameter.size())*sizeof(double));
		if(!work)
			fprintf(stderr, "memory allocation request failed in main()\n");

		int ret ;//dlevmar_dif(Projection3Donto2D_FaceRefinement, dXParameter, dFeature2DParameter, XParameter.size(), feature2DParameter.size(),
							  //1e+3, opt, info, work, NULL, &adata);

		for (int iX = 0; iX < XParameter.size()/3; iX++)
		{
			cvSetReal2D(vX1[vRefer[iX]], 0, 0, dXParameter[3*iX]);
			cvSetReal2D(vX1[vRefer[iX]], 1, 0, dXParameter[3*iX+1]);
			cvSetReal2D(vX1[vRefer[iX]], 2, 0, dXParameter[3*iX+2]);

		}
		//PrintSBAInfo(info, feature2DParameter.size());
		free(dFeature2DParameter);
		free(dXParameter);
		free(work);
	}
}

void TriangulationRefinement_NoDistortion(vector<vector<CvMat *> > vvP, vector<vector<double> > vvx, vector<vector<double> > vvy, vector<double> &vX, vector<double> &vY, vector<double> &vZ)
{
	//PrintAlgorithm("Point Triangulation Refinement");
	AdditionalData adata;
	adata.vvP = vvP;

	vector<double> XParameter, feature2DParameter;

	for (int iX = 0; iX < vX.size(); iX++)
	{
		XParameter.push_back(vX[iX]);
		XParameter.push_back(vY[iX]);
		XParameter.push_back(vZ[iX]);
	}
	
	for (int iFeature = 0; iFeature < vvx.size(); iFeature++)
	{
		for (int iFeature1 = 0; iFeature1 < vvx[iFeature].size(); iFeature1++)
		{
			feature2DParameter.push_back(vvx[iFeature][iFeature1]);
			feature2DParameter.push_back(vvy[iFeature][iFeature1]);
		}		
	}

	//int nFeatures = vx.size(); 
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dXParameter = (double *) malloc(XParameter.size() * sizeof(double));
	
	for (int i = 0; i < XParameter.size(); i++)
		dXParameter[i] = XParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(XParameter.size(), feature2DParameter.size())+XParameter.size()*XParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");

	int ret ;//dlevmar_dif(Projection3Donto2D_STR_fast_SO_NoDistortion_All, dXParameter, dFeature2DParameter, XParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);
	vX.clear(); vY.clear(); vZ.clear();
	for (int iX = 0; iX < XParameter.size()/3; iX++)
	{
		vX.push_back(dXParameter[3*iX]);
		vY.push_back(dXParameter[3*iX+1]);
		vZ.push_back(dXParameter[3*iX+2]);
	}
	//PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dXParameter);
	free(work);

	//for (int i = 0; i < adata.vP.size(); i++)
	//{
	//	cvReleaseMat(&adata.vP[i]);
	//}
}

void TriangulationRefinement_NoDistortion(CvMat *K, vector<CvMat *> vP, vector<double> vx, vector<double> vy, double &X, double &Y, double &Z)
{
	PrintAlgorithm("Point Triangulation Refinement");
	AdditionalData adata;
	double intrinsic[4] = {cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1), cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2)};
	adata.vIntrinsic.push_back(intrinsic);
	adata.vP = vP;

	vector<double> XParameter, feature2DParameter;
	XParameter.push_back(X);
	XParameter.push_back(Y);
	XParameter.push_back(Z);
	
	for (int iFeature = 0; iFeature < vx.size(); iFeature++)
	{
		feature2DParameter.push_back(vx[iFeature]);
		feature2DParameter.push_back(vy[iFeature]);
	}

	int nFeatures = vx.size(); 
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dXParameter = (double *) malloc(XParameter.size() * sizeof(double));
	
	for (int i = 0; i < XParameter.size(); i++)
		dXParameter[i] = XParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(XParameter.size(), feature2DParameter.size())+XParameter.size()*XParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");

	int ret ;//dlevmar_dif(Projection3Donto2D_STR_fast_SO_NoDistortion, dXParameter, dFeature2DParameter, XParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);

	X = dXParameter[0];
	Y = dXParameter[1];
	Z = dXParameter[2];
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dXParameter);
	free(work);

	//for (int i = 0; i < adata.vP.size(); i++)
	//{
	//	cvReleaseMat(&adata.vP[i]);
	//}
}
*/
//void POI_TriangulationRefinement_NoDistortion(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vWeight, double &X, double &Y, double &Z)
//{
//	PrintAlgorithm("Point Triangulation Refinement");
//
//	AdditionalData adata;
//	adata.vP = vP;
//	adata.vV = vV;
//	adata.vWeight = vWeight;
//
//	vector<double> XParameter, feature2DParameter;
//	XParameter.push_back(X);
//	XParameter.push_back(Y);
//	XParameter.push_back(Z);
//	
//	for (int iFeature = 0; iFeature < vP.size(); iFeature++)
//	{
//		feature2DParameter.push_back(0);
//		feature2DParameter.push_back(0);
//	}
//
//	int nFeatures = vP.size(); 
//	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
//	double *dXParameter = (double *) malloc(XParameter.size() * sizeof(double));
//	
//	for (int i = 0; i < XParameter.size(); i++)
//		dXParameter[i] = XParameter[i];
//	for (int i = 0; i < feature2DParameter.size(); i++)
//		dFeature2DParameter[i] = feature2DParameter[i];
//	double opt[5];
//	opt[0] = 1e-3;
//	opt[1] = 1e-12;
//	opt[2] = 1e-12;
//	opt[3] = 1e-12;
//	opt[4] = 0;
//	double info[12];
//
//	double *work ;// (double*)malloc((LM_DIF_WORKSZ(XParameter.size(), feature2DParameter.size())+XParameter.size()*XParameter.size())*sizeof(double));
//	if(!work)
//		fprintf(stderr, "memory allocation request failed in main()\n");
//
//	int ret ;//dlevmar_dif(Projection3Donto2D_STR_Ray_ICCV, dXParameter, dFeature2DParameter, XParameter.size(), feature2DParameter.size(),
//						  1e+3, opt, info, work, NULL, &adata);
//
//	X = dXParameter[0];
//	Y = dXParameter[1];
//	Z = dXParameter[2];
//	PrintSBAInfo(info, feature2DParameter.size());
//	free(dFeature2DParameter);
//	free(dXParameter);
//	free(work);
//
//	//for (int i = 0; i < adata.vP.size(); i++)
//	//{
//	//	cvReleaseMat(&adata.vP[i]);
//	//}
//}



void CfMRefinement(Camera &virtualCamera, vector<Camera> &vCamera, vector<CvMat *> &vT)
{
	PrintAlgorithm("CfM Refinement");
	AdditionalData adata;
	adata.stride = 2;

	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		vector<vector<Correspondence2D3D> > vvCorr;
		for (int iFrame = 0; iFrame <= virtualCamera.vTakenFrame[virtualCamera.vTakenFrame.size()-1]; iFrame++)
		{
			vector<Correspondence2D3D> vCorr;
			for (int iPoint = 0; iPoint < vCamera[iCamera].vvCorr[iFrame].size(); iPoint++)
				vCorr.push_back(vCamera[iCamera].vvCorr[iFrame][iPoint]);
			vvCorr.push_back(vCorr);
		}
		adata.vvvCorr.push_back(vvCorr);
	}

	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		//vector<int> frameidx;
		//for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		//{
		//	vector<int>::const_iterator it = find(virtualCamera.vTakenFrame.begin(), virtualCamera.vTakenFrame.end(), vCamera[iCamera].vTakenFrame[iFrame]);
		//	if (it == virtualCamera.vTakenFrame.end())
		//		continue;
		//	frameidx.push_back((int) (it - virtualCamera.vTakenFrame.begin()));
		//}
		//adata.vvCameraIndex_CfM.push_back(frameidx);
		double intrinsic[4] = {cvGetReal2D(vCamera[iCamera].K, 0, 0), cvGetReal2D(vCamera[iCamera].K, 1, 1), cvGetReal2D(vCamera[iCamera].K, 0, 2), cvGetReal2D(vCamera[iCamera].K, 1, 2)};
		adata.vIntrinsic.push_back(intrinsic);
	}
	adata.nFrames = virtualCamera.vTakenFrame.size();

	vector<double> cameraParameter, feature2DParameter;
	for (int iFrame = 0; iFrame < virtualCamera.vTakenFrame.size(); iFrame++)
	{
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		Rotation2Quaternion(virtualCamera.vR[iFrame], q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));
		cameraParameter.push_back(cvGetReal2D(virtualCamera.vC[iFrame], 0, 0));
		cameraParameter.push_back(cvGetReal2D(virtualCamera.vC[iFrame], 1, 0));
		cameraParameter.push_back(cvGetReal2D(virtualCamera.vC[iFrame], 2, 0));
		cvReleaseMat(&q);
	}

	for (int iT = 0; iT < vT.size(); iT++)
	{
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		GetSubMat(vT[iT], 0, 2, 0, 2, R);
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));
		cameraParameter.push_back(cvGetReal2D(vT[iT], 0, 3));
		cameraParameter.push_back(cvGetReal2D(vT[iT], 1, 3));
		cameraParameter.push_back(cvGetReal2D(vT[iT], 2, 3));

		cvReleaseMat(&q);
		cvReleaseMat(&R);
	}
	
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < virtualCamera.vTakenFrame.size(); iFrame++)
		{
			for (int iPoint = 0; iPoint < vCamera[iCamera].vvCorr[iFrame].size(); iPoint += adata.stride)
			{
				feature2DParameter.push_back(vCamera[iCamera].vvCorr[iFrame][iPoint].u);
				feature2DParameter.push_back(vCamera[iCamera].vvCorr[iFrame][iPoint].v);
			}
		}
	}

	adata.vMeasurement = feature2DParameter;

	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-20;
	opt[2] = 1e-20;
	opt[3] = 1e-20;
	opt[4] = 1e-5;
	double info[12];
	cout << "Aa" << endl;
	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	int ret ;//dlevmar_dif(Projection3Donto2D_CfM, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);
	cout << "done" << endl;
	for (int iFrame = 0; iFrame < virtualCamera.vTakenFrame.size(); iFrame++)
	{
		cvReleaseMat(&virtualCamera.vC[iFrame]);
		cvReleaseMat(&virtualCamera.vR[iFrame]);
	}
	virtualCamera.vC.clear(); 
	virtualCamera.vR.clear();

	for (int iFrame = 0; iFrame < virtualCamera.vTakenFrame.size(); iFrame++)
	{
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[iFrame*7+0]);
		cvSetReal2D(q, 1, 0, dCameraParameter[iFrame*7+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[iFrame*7+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[iFrame*7+3]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		Quaternion2Rotation(q, R);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(C, 0, 0, dCameraParameter[iFrame*7+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[iFrame*7+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[iFrame*7+6]);
		virtualCamera.vR.push_back(R);
		virtualCamera.vC.push_back(C);
		cvReleaseMat(&q);
	}

	for (int iT = 0; iT < vT.size(); iT++)
	{
		cvReleaseMat(&vT[iT]);
	}
	vT.clear();
	cout << vCamera.size() << endl;
	for (int iCamera=0; iCamera < vCamera.size(); iCamera++)
	{
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[7*virtualCamera.vTakenFrame.size()+iCamera*7+0]);
		cvSetReal2D(q, 1, 0, dCameraParameter[7*virtualCamera.vTakenFrame.size()+iCamera*7+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[7*virtualCamera.vTakenFrame.size()+iCamera*7+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[7*virtualCamera.vTakenFrame.size()+iCamera*7+3]);
		Quaternion2Rotation(q, R);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(t, 0, 0, dCameraParameter[7*virtualCamera.vTakenFrame.size()+iCamera*7+4]);
		cvSetReal2D(t, 1, 0, dCameraParameter[7*virtualCamera.vTakenFrame.size()+iCamera*7+5]);
		cvSetReal2D(t, 2, 0, dCameraParameter[7*virtualCamera.vTakenFrame.size()+iCamera*7+6]);

		CvMat *T = cvCreateMat(4,4,CV_32FC1);
		cvSetIdentity(T);
		SetSubMat(T, 0, 0, R);
		SetSubMat(T, 0, 3, t);

		cvReleaseMat(&q);
		cvReleaseMat(&R);
		cvReleaseMat(&t);
		vT.push_back(T);
		cout << iCamera << endl;
		PrintMat(T);
	}
	cout << vCamera.size() << " " << vT.size() << endl;
	
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		vCamera[iCamera].vTakenFrame.clear();
		for (int iFrame = 0; iFrame < vCamera[iCamera].vC.size(); iFrame++)
		{
			cvReleaseMat(&vCamera[iCamera].vC[iFrame]);
			cvReleaseMat(&vCamera[iCamera].vR[iFrame]);
		}
		vCamera[iCamera].vC.clear();
		vCamera[iCamera].vR.clear();
	}
	
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < virtualCamera.vTakenFrame.size(); iFrame++)
		{
			CvMat *Tw1 = cvCreateMat(4,4,CV_32FC1);
			cvSetIdentity(Tw1);
			CvMat *Rt = cvCreateMat(3,3,CV_32FC1);
			cvTranspose(virtualCamera.vR[iFrame], Rt);
			SetSubMat(Tw1, 0, 0, Rt);
			SetSubMat(Tw1, 0, 3, virtualCamera.vC[iFrame]);
			CvMat *Twi = cvCreateMat(4,4,CV_32FC1);
			CvMat *T1i = cvCreateMat(4,4,CV_32FC1);
			cvInvert(vT[iCamera], T1i);			
			cvMatMul(Tw1, T1i, Twi);

			GetSubMat(Twi, 0, 2, 0, 2, Rt);
			cvTranspose(Rt, Rt);
			vCamera[iCamera].vR.push_back(Rt);
			CvMat *C = cvCreateMat(3,1,CV_32FC1);
			GetSubMat(Twi, 0, 2, 3, 3, C);
			vCamera[iCamera].vC.push_back(C);
			vCamera[iCamera].vTakenFrame.push_back(virtualCamera.vTakenFrame[iFrame]);

			//CvMat *P = cvCreateMat(3,4,CV_32FC1);
			//cvSetIdentity(P);
			//CvMat *t = cvCreateMat(3,1,CV_32FC1);
			//ScalarMul(C, -1, t);
			//SetSubMat(P, 0, 3, t);
			//cvMatMul(Rt, P, P);
			//cvMatMul(vCamera[iCamera].K, P, P);
			//for (int ix = 0; ix < vCamera[iCamera].vvCorr[iFrame].size(); ix++)
			//{
			//	CvMat *X = cvCreateMat(4,1,CV_32FC1);
			//	cvSetReal2D(X, 0, 0, vCamera[iCamera].vvCorr[iFrame][ix].x);
			//	cvSetReal2D(X, 1, 0, vCamera[iCamera].vvCorr[iFrame][ix].y);
			//	cvSetReal2D(X, 2, 0, vCamera[iCamera].vvCorr[iFrame][ix].z);
			//	cvSetReal2D(X, 3, 0, 1);

			//	CvMat *x = cvCreateMat(3,1,CV_32FC1);
			//	cvMatMul(P, X, x);
			//	cout << cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0) << " " << cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0) << " ";
			//	cout << vCamera[iCamera].vvCorr[iFrame][ix].u << " " << vCamera[iCamera].vvCorr[iFrame][ix].v << endl;
			//}
			

			cvReleaseMat(&Tw1);
			cvReleaseMat(&Twi);
			cvReleaseMat(&T1i);
			
		}
	}
	//cout << vCamera.size() << endl;
	
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dCameraParameter);
}

void CfMRefinement_corner(vector<CvMat *> &vH0i, vector<Camera> vCamera, vector<double> &vReproj_x, vector<double> &vReproj_y)
{
	PrintAlgorithm("CfM Refinement");
	AdditionalData adata;
	adata.lambda = 1e-1;

	vector<double> cameraParameter, feature2DParameter;

	for (int  i = 0; i <vH0i.size(); i++)
	{
		PrintMat(vH0i[i]);
	}

	for (int i = 0; i < vH0i.size(); i++)
	{
		cameraParameter.push_back(cvGetReal2D(vH0i[i], 0, 0));
		cameraParameter.push_back(cvGetReal2D(vH0i[i], 0, 1));
		cameraParameter.push_back(cvGetReal2D(vH0i[i], 0, 2));

		cameraParameter.push_back(cvGetReal2D(vH0i[i], 1, 0));
		cameraParameter.push_back(cvGetReal2D(vH0i[i], 1, 1));
		cameraParameter.push_back(cvGetReal2D(vH0i[i], 1, 2));

		cameraParameter.push_back(cvGetReal2D(vH0i[i], 2, 0));
		cameraParameter.push_back(cvGetReal2D(vH0i[i], 2, 1));
	}
	
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		adata.vvcp.push_back(vCamera[iCamera].vCornerPoint);
		for (int iPoint = 0; iPoint < vCamera[iCamera].vCornerPoint[0].vx.size(); iPoint++)
		{
			feature2DParameter.push_back(0);
			feature2DParameter.push_back(0);
		}
	}


	for (int icp = 0; icp < adata.vvcp.size()-1; icp++)
	{
		for (int j = 0; j < adata.vvcp[icp][1].vx.size(); j++)
		{
			double x1 = adata.vvcp[icp][1].vx[j];
			double y1 = adata.vvcp[icp][1].vy[j];

			double x2 = adata.vvcp[icp+1][0].vx[j];
			double y2 = adata.vvcp[icp+1][0].vy[j];

			double x1n = (cvGetReal2D(vH0i[icp],0,0)*x1+cvGetReal2D(vH0i[icp],0,1)*y1+cvGetReal2D(vH0i[icp],0,2))/(cvGetReal2D(vH0i[icp],2,0)*x1+cvGetReal2D(vH0i[icp],2,1)*y1+cvGetReal2D(vH0i[icp],2,2));
			double y1n = (cvGetReal2D(vH0i[icp],1,0)*x1+cvGetReal2D(vH0i[icp],1,1)*y1+cvGetReal2D(vH0i[icp],1,2))/(cvGetReal2D(vH0i[icp],2,0)*x1+cvGetReal2D(vH0i[icp],2,1)*y1+cvGetReal2D(vH0i[icp],2,2));

			double x2n = (cvGetReal2D(vH0i[icp+1],0,0)*x2+cvGetReal2D(vH0i[icp+1],0,1)*y2+cvGetReal2D(vH0i[icp+1],0,2))/(cvGetReal2D(vH0i[icp+1],2,0)*x2+cvGetReal2D(vH0i[icp+1],2,1)*y2+cvGetReal2D(vH0i[icp+1],2,2));
			double y2n = (cvGetReal2D(vH0i[icp+1],1,0)*x2+cvGetReal2D(vH0i[icp+1],1,1)*y2+cvGetReal2D(vH0i[icp+1],1,2))/(cvGetReal2D(vH0i[icp+1],2,0)*x2+cvGetReal2D(vH0i[icp+1],2,1)*y2+cvGetReal2D(vH0i[icp+1],2,2));


			cout << x1n << " " << x2n << " " << y1n << " " << y2n << endl;

			feature2DParameter.push_back((1-adata.lambda)*(x1n+x2n)/2);
			feature2DParameter.push_back((1-adata.lambda)*(y1n+y2n)/2);
		}
	}

	for (int j = 0; j < adata.vvcp[0][1].vx.size(); j++)
	{
		double x1 = adata.vvcp[adata.vvcp.size()-1][1].vx[j];
		double y1 = adata.vvcp[adata.vvcp.size()-1][1].vy[j];

		double x2 = adata.vvcp[0][0].vx[j];
		double y2 = adata.vvcp[0][0].vy[j];

		double x1n = (cvGetReal2D(vH0i[adata.vvcp.size()-1],0,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],0,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],0,2))/(cvGetReal2D(vH0i[adata.vvcp.size()-1],2,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,2));
		double y1n = (cvGetReal2D(vH0i[adata.vvcp.size()-1],1,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],1,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],1,2))/(cvGetReal2D(vH0i[adata.vvcp.size()-1],2,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,2));

		double x2n = (cvGetReal2D(vH0i[0],0,0)*x2+cvGetReal2D(vH0i[0],0,1)*y2+cvGetReal2D(vH0i[0],0,2))/(cvGetReal2D(vH0i[0],2,0)*x2+cvGetReal2D(vH0i[0],2,1)*y2+cvGetReal2D(vH0i[0],2,2));
		double y2n = (cvGetReal2D(vH0i[0],1,0)*x2+cvGetReal2D(vH0i[0],1,1)*y2+cvGetReal2D(vH0i[0],1,2))/(cvGetReal2D(vH0i[0],2,0)*x2+cvGetReal2D(vH0i[0],2,1)*y2+cvGetReal2D(vH0i[0],2,2));


		cout << x1n << " " << x2n << " " << y1n << " " << y2n << endl;

		feature2DParameter.push_back((1-adata.lambda)*(x1n+x2n)/2);
		feature2DParameter.push_back((1-adata.lambda)*(y1n+y2n)/2);
	}


	//for (int iCamera = 0; iCamera < cameraParameter.size(); iCamera++)
	//{
	//	feature2DParameter.push_back((1-adata.lambda)*cameraParameter[iCamera]);
	//}

	adata.vMeasurement = feature2DParameter;

	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-20;
	opt[2] = 1e-20;
	opt[3] = 1e-20;
	opt[4] = 1e-5;
	double info[12];
	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	int ret ;//dlevmar_dif(Projection3Donto2D_CfM_corner, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);
	cout << "done" << endl;

	for (int iH = 0; iH < vH0i.size(); iH++)
	{
		cvSetReal2D(vH0i[iH], 0, 0, dCameraParameter[(iH)*8]);
		cvSetReal2D(vH0i[iH], 0, 1, dCameraParameter[(iH)*8+1]);
		cvSetReal2D(vH0i[iH], 0, 2, dCameraParameter[(iH)*8+2]);

		cvSetReal2D(vH0i[iH], 1, 0, dCameraParameter[(iH)*8+3]);
		cvSetReal2D(vH0i[iH], 1, 1, dCameraParameter[(iH)*8+4]);
		cvSetReal2D(vH0i[iH], 1, 2, dCameraParameter[(iH)*8+5]);

		cvSetReal2D(vH0i[iH], 2, 0, dCameraParameter[(iH)*8+6]);
		cvSetReal2D(vH0i[iH], 2, 1, dCameraParameter[(iH)*8+7]);
	}

	for (int  i = 0; i <vH0i.size(); i++)
	{
		PrintMat(vH0i[i]);
	}

	for (int icp = 0; icp < adata.vvcp.size()-1; icp++)
	{
		for (int j = 0; j < adata.vvcp[icp][1].vx.size(); j++)
		{
			double x1 = adata.vvcp[icp][1].vx[j];
			double y1 = adata.vvcp[icp][1].vy[j];

			double x2 = adata.vvcp[icp+1][0].vx[j];
			double y2 = adata.vvcp[icp+1][0].vy[j];

			double x1n = (cvGetReal2D(vH0i[icp],0,0)*x1+cvGetReal2D(vH0i[icp],0,1)*y1+cvGetReal2D(vH0i[icp],0,2))/(cvGetReal2D(vH0i[icp],2,0)*x1+cvGetReal2D(vH0i[icp],2,1)*y1+cvGetReal2D(vH0i[icp],2,2));
			double y1n = (cvGetReal2D(vH0i[icp],1,0)*x1+cvGetReal2D(vH0i[icp],1,1)*y1+cvGetReal2D(vH0i[icp],1,2))/(cvGetReal2D(vH0i[icp],2,0)*x1+cvGetReal2D(vH0i[icp],2,1)*y1+cvGetReal2D(vH0i[icp],2,2));

			double x2n = (cvGetReal2D(vH0i[icp+1],0,0)*x2+cvGetReal2D(vH0i[icp+1],0,1)*y2+cvGetReal2D(vH0i[icp+1],0,2))/(cvGetReal2D(vH0i[icp+1],2,0)*x2+cvGetReal2D(vH0i[icp+1],2,1)*y2+cvGetReal2D(vH0i[icp+1],2,2));
			double y2n = (cvGetReal2D(vH0i[icp+1],1,0)*x2+cvGetReal2D(vH0i[icp+1],1,1)*y2+cvGetReal2D(vH0i[icp+1],1,2))/(cvGetReal2D(vH0i[icp+1],2,0)*x2+cvGetReal2D(vH0i[icp+1],2,1)*y2+cvGetReal2D(vH0i[icp+1],2,2));


			cout << x1n << " " << x2n << " " << y1n << " " << y2n << endl;

			vReproj_x.push_back(x1n);
			vReproj_x.push_back(x2n);
			vReproj_y.push_back(y1n);
			vReproj_y.push_back(y2n);
		}
	}

	for (int j = 0; j < adata.vvcp[0][1].vx.size(); j++)
	{
		double x1 = adata.vvcp[adata.vvcp.size()-1][1].vx[j];
		double y1 = adata.vvcp[adata.vvcp.size()-1][1].vy[j];

		double x2 = adata.vvcp[0][0].vx[j];
		double y2 = adata.vvcp[0][0].vy[j];

		double x1n = (cvGetReal2D(vH0i[adata.vvcp.size()-1],0,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],0,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],0,2))/(cvGetReal2D(vH0i[adata.vvcp.size()-1],2,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,2));
		double y1n = (cvGetReal2D(vH0i[adata.vvcp.size()-1],1,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],1,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],1,2))/(cvGetReal2D(vH0i[adata.vvcp.size()-1],2,0)*x1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,1)*y1+cvGetReal2D(vH0i[adata.vvcp.size()-1],2,2));

		double x2n = (cvGetReal2D(vH0i[0],0,0)*x2+cvGetReal2D(vH0i[0],0,1)*y2+cvGetReal2D(vH0i[0],0,2))/(cvGetReal2D(vH0i[0],2,0)*x2+cvGetReal2D(vH0i[0],2,1)*y2+cvGetReal2D(vH0i[0],2,2));
		double y2n = (cvGetReal2D(vH0i[0],1,0)*x2+cvGetReal2D(vH0i[0],1,1)*y2+cvGetReal2D(vH0i[0],1,2))/(cvGetReal2D(vH0i[0],2,0)*x2+cvGetReal2D(vH0i[0],2,1)*y2+cvGetReal2D(vH0i[0],2,2));


		cout << x1n << " " << x2n << " " << y1n << " " << y2n << endl;

		vReproj_x.push_back(x1n);
		vReproj_x.push_back(x2n);
		vReproj_y.push_back(y1n);
		vReproj_y.push_back(y2n);
	}

	
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dCameraParameter);
}

void CfMRefinement_ECCM_Each(CvMat *H0i, vector<double> vx1, vector<double> vy1, vector<double> vx2, vector<double> vy2)
{
	PrintAlgorithm("CfM Refinement");
	AdditionalData adata;

	vector<double> cameraParameter, feature2DParameter;

	cameraParameter.push_back(cvGetReal2D(H0i, 0, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 0, 1)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 0, 2)/cvGetReal2D(H0i, 2, 2));

	cameraParameter.push_back(cvGetReal2D(H0i, 1, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 1, 1)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 1, 2)/cvGetReal2D(H0i, 2, 2));

	cameraParameter.push_back(cvGetReal2D(H0i, 2, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 2, 1)/cvGetReal2D(H0i, 2, 2));

	adata.lambda = 5e+5;
	for (int ix = 0; ix < vx1.size(); ix++)
	{
		feature2DParameter.push_back(vx2[ix]/adata.lambda);
		feature2DParameter.push_back(vy2[ix]/adata.lambda);
	}
	
	//if (vx1.size() > 4)
	//	adata.lambda = 80;

	feature2DParameter.push_back(1);
	feature2DParameter.push_back(0);
	feature2DParameter.push_back(0);

	feature2DParameter.push_back(0);
	feature2DParameter.push_back(1);
	feature2DParameter.push_back(0);

	feature2DParameter.push_back(0);
	feature2DParameter.push_back(0);

	adata.vx1d = vx1;
	adata.vy1d = vy1;

	for (int ix = 0; ix < vx1.size(); ix++)
	{
		adata.vMeasurement.push_back(vx2[ix]);
		adata.vMeasurement.push_back(vy2[ix]);
	}	

	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-10;
	opt[2] = 1e-15;
	opt[3] = 1e-15;
	opt[4] = 1e-5;
	double info[12];
	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	int ret ;//dlevmar_dif(Projection3Donto2D_CfM_ECCM, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);
	cout << "done" << endl;

	cvSetReal2D(H0i, 0, 0, dCameraParameter[0]);
	cvSetReal2D(H0i, 0, 1, dCameraParameter[1]);
	cvSetReal2D(H0i, 0, 2, dCameraParameter[2]);

	cvSetReal2D(H0i, 1, 0, dCameraParameter[3]);
	cvSetReal2D(H0i, 1, 1, dCameraParameter[4]);
	cvSetReal2D(H0i, 1, 2, dCameraParameter[5]);

	cvSetReal2D(H0i, 2, 0, dCameraParameter[6]);
	cvSetReal2D(H0i, 2, 1, dCameraParameter[7]);
	cvSetReal2D(H0i, 2, 2, 1);
	
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dCameraParameter);
}

void CfMRefinement_ECCM_Each(CvMat *H0i, vector<double> vx1, vector<double> vy1, vector<double> vx2, vector<double> vy2, vector<double> vx3, vector<double> vy3)
{
	PrintAlgorithm("CfM Refinement");
	AdditionalData adata;

	vector<double> cameraParameter, feature2DParameter;

	cameraParameter.push_back(cvGetReal2D(H0i, 0, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 0, 1)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 0, 2)/cvGetReal2D(H0i, 2, 2));

	cameraParameter.push_back(cvGetReal2D(H0i, 1, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 1, 1)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 1, 2)/cvGetReal2D(H0i, 2, 2));

	cameraParameter.push_back(cvGetReal2D(H0i, 2, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 2, 1)/cvGetReal2D(H0i, 2, 2));

	adata.lambda = 10;
	for (int ix = 0; ix < vx1.size(); ix++)
	{
		feature2DParameter.push_back(vx2[ix]/adata.lambda);
		feature2DParameter.push_back(vy2[ix]/adata.lambda);
	}

	for (int ix = 0; ix < vx3.size(); ix++)
	{
		feature2DParameter.push_back(vx3[ix]);
		feature2DParameter.push_back(vy3[ix]);
	}
	
	//if (vx1.size() > 4)
	//	adata.lambda = 80;

	adata.vx1d = vx1;
	adata.vy1d = vy1;

	adata.vx3d = vx3;
	adata.vy3d = vy3;

	for (int ix = 0; ix < vx1.size(); ix++)
	{
		adata.vMeasurement.push_back(vx2[ix]);
		adata.vMeasurement.push_back(vy2[ix]);
	}	

	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-10;
	opt[2] = 1e-15;
	opt[3] = 1e-15;
	opt[4] = 1e-5;
	double info[12];
	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	int ret ;//dlevmar_dif(Projection3Donto2D_CfM_ECCM1, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);
	cout << "done" << endl;

	cvSetReal2D(H0i, 0, 0, dCameraParameter[0]);
	cvSetReal2D(H0i, 0, 1, dCameraParameter[1]);
	cvSetReal2D(H0i, 0, 2, dCameraParameter[2]);

	cvSetReal2D(H0i, 1, 0, dCameraParameter[3]);
	cvSetReal2D(H0i, 1, 1, dCameraParameter[4]);
	cvSetReal2D(H0i, 1, 2, dCameraParameter[5]);

	cvSetReal2D(H0i, 2, 0, dCameraParameter[6]);
	cvSetReal2D(H0i, 2, 1, dCameraParameter[7]);
	cvSetReal2D(H0i, 2, 2, 1);
	
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dCameraParameter);
}

void CfMRefinement_ECCM_Final(vector<CvMat *> vH_inv_r, vector<vector<double> > vVx1, vector<vector<double> > vVy1, vector<vector<double> > vVx2, vector<vector<double> > vVy2)
{
	PrintAlgorithm("CfM Refinement");
	AdditionalData adata;
	adata.vVx1 = vVx1;
	adata.vVy1 = vVy1;
	adata.vVx2 = vVx2;
	adata.vVy2 = vVy2;
	vector<double> cameraParameter, feature2DParameter;
	for (int i = 0; i < vH_inv_r.size(); i++)
	{
		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 0, 0)/cvGetReal2D(vH_inv_r[i], 2, 2));
		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 0, 1)/cvGetReal2D(vH_inv_r[i], 2, 2));
		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 0, 2)/cvGetReal2D(vH_inv_r[i], 2, 2));

		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 1, 0)/cvGetReal2D(vH_inv_r[i], 2, 2));
		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 1, 1)/cvGetReal2D(vH_inv_r[i], 2, 2));
		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 1, 2)/cvGetReal2D(vH_inv_r[i], 2, 2));

		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 2, 0)/cvGetReal2D(vH_inv_r[i], 2, 2));
		cameraParameter.push_back(cvGetReal2D(vH_inv_r[i], 2, 1)/cvGetReal2D(vH_inv_r[i], 2, 2));
	}

	feature2DParameter.push_back(vVx2[0][0]);
	feature2DParameter.push_back(vVy2[0][0]);
	feature2DParameter.push_back(vVx2[0][1]);
	feature2DParameter.push_back(vVy2[0][1]);
	feature2DParameter.push_back(vVx2[0][2]);
	feature2DParameter.push_back(vVy2[0][2]);
	feature2DParameter.push_back(vVx2[0][3]);
	feature2DParameter.push_back(vVy2[0][3]);

	for (int i = 0; i < 4; i++)
	{
		feature2DParameter.push_back(0);
		feature2DParameter.push_back(0);
		feature2DParameter.push_back(0);
		feature2DParameter.push_back(0);
		feature2DParameter.push_back(0);
		feature2DParameter.push_back(0);
		feature2DParameter.push_back(0);
		feature2DParameter.push_back(0);
	}

	feature2DParameter.push_back(vVx1[5][0]);
	feature2DParameter.push_back(vVy1[5][0]);
	feature2DParameter.push_back(vVx1[5][1]);
	feature2DParameter.push_back(vVy1[5][1]);
	feature2DParameter.push_back(vVx1[5][2]);
	feature2DParameter.push_back(vVy1[5][2]);
	feature2DParameter.push_back(vVx1[5][3]);
	feature2DParameter.push_back(vVy1[5][3]);

	for (int ix = 0; ix < feature2DParameter.size(); ix++)
	{
		adata.vMeasurement.push_back(feature2DParameter[ix]);
	}	

	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-3;
	opt[2] = 1e-5;
	opt[3] = 1e-5;
	opt[4] = 1e-5;
	double info[12];
	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	int ret ;//dlevmar_dif(Projection3Donto2D_CfM_ECCM_Final, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);
	cout << "done" << endl;

	for (int i = 0; i < vH_inv_r.size(); i++)
	{
		cvSetReal2D(vH_inv_r[i], 0, 0, dCameraParameter[0+8*i]);
		cvSetReal2D(vH_inv_r[i], 0, 1, dCameraParameter[1+8*i]);
		cvSetReal2D(vH_inv_r[i], 0, 2, dCameraParameter[2+8*i]);

		cvSetReal2D(vH_inv_r[i], 1, 0, dCameraParameter[3+8*i]);
		cvSetReal2D(vH_inv_r[i], 1, 1, dCameraParameter[4+8*i]);
		cvSetReal2D(vH_inv_r[i], 1, 2, dCameraParameter[5+8*i]);

		cvSetReal2D(vH_inv_r[i], 2, 0, dCameraParameter[6+8*i]);
		cvSetReal2D(vH_inv_r[i], 2, 1, dCameraParameter[7+8*i]);
		cvSetReal2D(vH_inv_r[i], 2, 2, 1);
	}
		
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dCameraParameter);
}



void CfMRefinement_ECCM_Each(CvMat *H0i, vector<double> vx1, vector<double> vy1, vector<double> vx2, vector<double> vy2, double lambda)
{
	PrintAlgorithm("CfM Refinement");
	AdditionalData adata;

	vector<double> cameraParameter, feature2DParameter;

	cameraParameter.push_back(cvGetReal2D(H0i, 0, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 0, 1)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 0, 2)/cvGetReal2D(H0i, 2, 2));

	cameraParameter.push_back(cvGetReal2D(H0i, 1, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 1, 1)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 1, 2)/cvGetReal2D(H0i, 2, 2));

	cameraParameter.push_back(cvGetReal2D(H0i, 2, 0)/cvGetReal2D(H0i, 2, 2));
	cameraParameter.push_back(cvGetReal2D(H0i, 2, 1)/cvGetReal2D(H0i, 2, 2));

	adata.lambda = lambda;
	for (int ix = 0; ix < vx1.size(); ix++)
	{
		feature2DParameter.push_back(vx2[ix]/adata.lambda);
		feature2DParameter.push_back(vy2[ix]/adata.lambda);
	}
	
	//if (vx1.size() > 4)
	//	adata.lambda = 80;

	feature2DParameter.push_back(1);
	feature2DParameter.push_back(0);
	feature2DParameter.push_back(0);

	feature2DParameter.push_back(0);
	feature2DParameter.push_back(1);
	feature2DParameter.push_back(0);

	feature2DParameter.push_back(0);
	feature2DParameter.push_back(0);

	adata.vx1d = vx1;
	adata.vy1d = vy1;

	for (int ix = 0; ix < vx1.size(); ix++)
	{
		adata.vMeasurement.push_back(vx2[ix]);
		adata.vMeasurement.push_back(vy2[ix]);
	}	

	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-14;
	opt[2] = 1e-14;
	opt[3] = 1e-14;
	opt[4] = 1e-5;
	double info[12];
	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");
	int ret ;//dlevmar_dif(Projection3Donto2D_CfM_ECCM, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);
	cout << "done" << endl;

	cvSetReal2D(H0i, 0, 0, dCameraParameter[0]);
	cvSetReal2D(H0i, 0, 1, dCameraParameter[1]);
	cvSetReal2D(H0i, 0, 2, dCameraParameter[2]);

	cvSetReal2D(H0i, 1, 0, dCameraParameter[3]);
	cvSetReal2D(H0i, 1, 1, dCameraParameter[4]);
	cvSetReal2D(H0i, 1, 2, dCameraParameter[5]);

	cvSetReal2D(H0i, 2, 0, dCameraParameter[6]);
	cvSetReal2D(H0i, 2, 1, dCameraParameter[7]);
	cvSetReal2D(H0i, 2, 2, 1);
	
	PrintSBAInfo(info, feature2DParameter.size());
	free(dFeature2DParameter);
	free(dCameraParameter);
}
/*
void AbsoluteCameraPoseRefinement(CvMat *X, CvMat *x, CvMat *P, CvMat *K)
{
	PrintAlgorithm("Absolute Camera Pose Refinement");
	vector<double> cameraParameter, feature2DParameter;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	double intrinsic[4] = {cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1), cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2)};
	adata.vIntrinsic.push_back(intrinsic);

	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	CvMat *t = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR = cvCreateMat(3,3,CV_32FC1);
	GetSubMatColwise(P, 0, 2, R);
	GetSubMatColwise(P, 3, 3, t);
	cvInvert(K, invK);
	cvMatMul(invK, R, R);
	cvInvert(R, invR);
	cvMatMul(invK, t, t);
	cvMatMul(invR, t, t);
	ScalarMul(t, -1, t);
	Rotation2Quaternion(R, q);
	cameraParameter.push_back(cvGetReal2D(q, 0, 0));
	cameraParameter.push_back(cvGetReal2D(q, 1, 0));
	cameraParameter.push_back(cvGetReal2D(q, 2, 0));
	cameraParameter.push_back(cvGetReal2D(q, 3, 0));
	cameraParameter.push_back(cvGetReal2D(t, 0, 0));
	cameraParameter.push_back(cvGetReal2D(t, 1, 0));
	cameraParameter.push_back(cvGetReal2D(t, 2, 0));
	
	for (int iFeature = 0; iFeature < X->rows; iFeature++)
	{
		feature2DParameter.push_back(cvGetReal2D(x, iFeature, 0));
		feature2DParameter.push_back(cvGetReal2D(x, iFeature, 1));
	}

	int nCameraParam = 7;
	int nFeatures = X->rows; 
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	double *dXParameter = (double *) malloc(X->rows * 3 * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];
	for (int i = 0; i < X->rows; i++)
	{
		dXParameter[3*i] = cvGetReal2D(X, i, 0);
		dXParameter[3*i+1] = cvGetReal2D(X, i, 1);
		dXParameter[3*i+2] = cvGetReal2D(X, i, 2);
	}

	adata.XYZ = dXParameter;
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	

	double *work = (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");

	int ret = dlevmar_dif(Projection3Donto2D_MOT_fast, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  1e+3, opt, info, work, NULL, &adata);

	cvSetIdentity(P);
	cvSetReal2D(P, 0, 3, -dCameraParameter[4]);
	cvSetReal2D(P, 1, 3, -dCameraParameter[5]);
	cvSetReal2D(P, 2, 3, -dCameraParameter[6]);

	cvSetReal2D(q, 0, 0, dCameraParameter[0]);
	cvSetReal2D(q, 1, 0, dCameraParameter[1]);
	cvSetReal2D(q, 2, 0, dCameraParameter[2]);
	cvSetReal2D(q, 3, 0, dCameraParameter[3]);

	Quaternion2Rotation(q, R);
	cvMatMul(K, R, R);
	cvMatMul(R, P, P);

	//cout << ret << endl;
	PrintSBAInfo(info, X->rows);
	free(dFeature2DParameter);
	free(dCameraParameter);

	cvReleaseMat(&R);
	cvReleaseMat(&t);
	cvReleaseMat(&q);
	cvReleaseMat(&invK);
	cvReleaseMat(&invR);
}
*/
void AbsoluteCameraPoseRefinement_Dome(CvMat *X, CvMat *x, CvMat *P, CvMat *K)
{
	PrintAlgorithm("Absolute Camera Pose Refinement");
	vector<double> cameraParameter, feature2DParameter;
	AdditionalData adata;// focal_x focal_y princ_x princ_y
	double intrinsic[4] = {cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1), cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2)};
	adata.vIntrinsic.push_back(intrinsic);

	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	CvMat *t = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR = cvCreateMat(3,3,CV_32FC1);
	GetSubMatColwise(P, 0, 2, R);
	GetSubMatColwise(P, 3, 3, t);
	cvInvert(K, invK);
	cvMatMul(invK, R, R);
	cvInvert(R, invR);
	cvMatMul(invK, t, t);
	cvMatMul(invR, t, t);
	ScalarMul(t, -1, t);
	Rotation2Quaternion(R, q);
	cameraParameter.push_back(cvGetReal2D(q, 0, 0));
	cameraParameter.push_back(cvGetReal2D(q, 1, 0));
	cameraParameter.push_back(cvGetReal2D(q, 2, 0));
	cameraParameter.push_back(cvGetReal2D(q, 3, 0));
	cameraParameter.push_back(cvGetReal2D(t, 0, 0));
	cameraParameter.push_back(cvGetReal2D(t, 1, 0));
	cameraParameter.push_back(cvGetReal2D(t, 2, 0));

	cameraParameter.push_back(cvGetReal2D(K, 0, 0));
	cameraParameter.push_back(cvGetReal2D(K, 1, 1));
	cameraParameter.push_back(cvGetReal2D(K, 0, 2));
	cameraParameter.push_back(cvGetReal2D(K, 1, 2));
	
	for (int iFeature = 0; iFeature < X->rows; iFeature++)
	{
		feature2DParameter.push_back(cvGetReal2D(x, iFeature, 0));
		feature2DParameter.push_back(cvGetReal2D(x, iFeature, 1));
	}

	int nCameraParam = 11;
	int nFeatures = X->rows; 
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	double *dXParameter = (double *) malloc(X->rows * 3 * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];
	for (int i = 0; i < X->rows; i++)
	{
		dXParameter[3*i] = cvGetReal2D(X, i, 0);
		dXParameter[3*i+1] = cvGetReal2D(X, i, 1);
		dXParameter[3*i+2] = cvGetReal2D(X, i, 2);
	}

	adata.XYZ = dXParameter;
	double opt[5];
	opt[0] = 1e-5;
	opt[1] = 1e-15;
	opt[2] = 1e-15;
	opt[3] = 1e-15;
	opt[4] = 0;
	double info[12];

	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	if(!work)
		fprintf(stderr, "memory allocation request failed in main()\n");

	int ret ;//dlevmar_dif(Projection3Donto2D_MOT_fast, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
						  //1e+3, opt, info, work, NULL, &adata);

	cvSetIdentity(P);
	cvSetReal2D(P, 0, 3, -dCameraParameter[4]);
	cvSetReal2D(P, 1, 3, -dCameraParameter[5]);
	cvSetReal2D(P, 2, 3, -dCameraParameter[6]);

	cvSetReal2D(q, 0, 0, dCameraParameter[0]);
	cvSetReal2D(q, 1, 0, dCameraParameter[1]);
	cvSetReal2D(q, 2, 0, dCameraParameter[2]);
	cvSetReal2D(q, 3, 0, dCameraParameter[3]);

	cvSetReal2D(K, 0, 0, dCameraParameter[7]);
	cvSetReal2D(K, 1, 1, dCameraParameter[8]);
	cvSetReal2D(K, 0, 2, dCameraParameter[9]);
	cvSetReal2D(K, 1, 2, dCameraParameter[10]);

	Quaternion2Rotation(q, R);
	cvMatMul(K, R, R);
	cvMatMul(R, P, P);

	//cout << ret << endl;
	PrintSBAInfo(info, X->rows);
	free(dFeature2DParameter);
	free(dCameraParameter);

	cvReleaseMat(&R);
	cvReleaseMat(&t);
	cvReleaseMat(&q);
	cvReleaseMat(&invK);
	cvReleaseMat(&invR);
}

void Projection3Donto2D_Patch(double *rt, double *hx, int m, int n, void *adata)
{
	// Set intrinsic parameter
	double X = ((AdditionalData *) adata)->X;
	double Y = ((AdditionalData *) adata)->Y;
	double Z = ((AdditionalData *) adata)->Z;

	double pi1 = rt[0];
	double pi2 = rt[1];
	double pi3 = rt[2];
	double pi4 = -pi1*X - pi2*Y - pi3*Z;

	double X11 = rt[3];
	double Y11 = rt[4];
	double Z11 = (-pi4-pi1*X11-pi2*Y11)/pi3;

	double X12 = rt[5];
	double Y12 = rt[6];
	double Z12 = (-pi4-pi1*X12-pi2*Y12)/pi3;

	double X21 = rt[7];
	double Y21 = rt[8];
	double Z21 = (-pi4-pi1*X21-pi2*Y21)/pi3;

	double X22 = rt[9];
	double Y22 = rt[10];
	double Z22 = (-pi4-pi1*X22-pi2*Y22)/pi3;
	for (int iP = 0; iP < ((AdditionalData *) adata)->vP.size(); iP++)
	{
		double P11 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 0);
		double P12 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 1);
		double P13 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 2);
		double P14 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 3);

		double P21 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 0);
		double P22 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 1);
		double P23 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 2);
		double P24 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 3);

		double P31 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 0);
		double P32 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 1);
		double P33 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 2);
		double P34 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 3);

		double x11 = P11*X11+P12*Y11+P13*Z11+P14;
		double y11 = P21*X11+P22*Y11+P23*Z11+P24;
		double z11 = P31*X11+P32*Y11+P33*Z11+P34;

		double x12 = P11*X12+P12*Y12+P13*Z12+P14;
		double y12 = P21*X12+P22*Y12+P23*Z12+P24;
		double z12 = P31*X12+P32*Y12+P33*Z12+P34;

		double x21 = P11*X21+P12*Y21+P13*Z21+P14;
		double y21 = P21*X21+P22*Y21+P23*Z21+P24;
		double z21 = P31*X21+P32*Y21+P33*Z21+P34;

		double x22 = P11*X22+P12*Y22+P13*Z22+P14;
		double y22 = P21*X22+P22*Y22+P23*Z22+P24;
		double z22 = P31*X22+P32*Y22+P33*Z22+P34;

		hx[8*iP] = x11/z11;
		hx[8*iP+1] = y11/z11;

		hx[8*iP+2] = x12/z12;
		hx[8*iP+3] = y12/z12;

		hx[8*iP+4] = x21/z21;
		hx[8*iP+5] = y21/z21;

		hx[8*iP+6] = x22/z22;
		hx[8*iP+7] = y22/z22;

		//cout << x11/z11 << " " << y11/z11 << " ";
		//cout << x12/z12 << " " << y12/z12 << " ";
		//cout << x21/z21 << " " << y21/z21 << " ";
		//cout << x22/z22 << " " << y22/z22 << endl;
	}

}


void Projection3Donto2D_STR_fast_SO(double *rt, double *hx, int m, int n, void *adata)
{
	// Set intrinsic parameter
	double K11 = ((((AdditionalData *) adata)->vIntrinsic)[0])[0];
	double K12 = 0;
	double K13 = ((((AdditionalData *) adata)->vIntrinsic)[0])[2];
	double K21 = 0;
	double K22 = ((((AdditionalData *) adata)->vIntrinsic)[0])[1];
	double K23 = ((((AdditionalData *) adata)->vIntrinsic)[0])[3];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	double omega = ((((AdditionalData *) adata)->vIntrinsic)[0])[4];

	double tan_omega_half_2 = 2*tan(omega/2); 

	double X = rt[0];
	double Y = rt[1];
	double Z = rt[2];
	for (int iP = 0; iP < ((AdditionalData *) adata)->vP.size(); iP++)
	{
		double P11 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 0);
		double P12 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 1);
		double P13 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 2);
		double P14 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 3);

		double P21 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 0);
		double P22 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 1);
		double P23 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 2);
		double P24 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 3);

		double P31 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 0);
		double P32 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 1);
		double P33 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 2);
		double P34 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 3);

		double x = P11*X+P12*Y+P13*Z+P14;
		double y = P21*X+P22*Y+P23*Z+P24;
		double z = P31*X+P32*Y+P33*Z+P34;

		x /= z;
		y /= z;

		double u_n = x/K11 - K13/K11;
		double v_n = y/K22 - K23/K22;

		double r_u = sqrt(u_n*u_n+v_n*v_n);
		double r_d = 1/omega*atan(r_u*tan_omega_half_2);

		double u_d_n = r_d/r_u * u_n;
		double v_d_n = r_d/r_u * v_n;

		double u_d = u_d_n*K11 + K13;
		double v_d = v_d_n*K22 + K23;

		hx[2*iP] = u_d;
		hx[2*iP+1] = v_d;
	}

}

void Projection3Donto2D_STR_fast_SO_NoDistortion(double *rt, double *hx, int m, int n, void *adata)
{
	// Set intrinsic parameter
	double K11 = ((((AdditionalData *) adata)->vIntrinsic)[0])[0];
	double K12 = 0;
	double K13 = ((((AdditionalData *) adata)->vIntrinsic)[0])[2];
	double K21 = 0;
	double K22 = ((((AdditionalData *) adata)->vIntrinsic)[0])[1];
	double K23 = ((((AdditionalData *) adata)->vIntrinsic)[0])[3];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	//double omega = ((((AdditionalData *) adata)->vIntrinsic)[0])[4];

	//double tan_omega_half_2 = 2*tan(omega/2); 

	double X = rt[0];
	double Y = rt[1];
	double Z = rt[2];
	for (int iP = 0; iP < ((AdditionalData *) adata)->vP.size(); iP++)
	{
		double P11 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 0);
		double P12 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 1);
		double P13 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 2);
		double P14 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 3);

		double P21 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 0);
		double P22 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 1);
		double P23 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 2);
		double P24 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 3);

		double P31 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 0);
		double P32 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 1);
		double P33 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 2);
		double P34 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 3);

		double x = P11*X+P12*Y+P13*Z+P14;
		double y = P21*X+P22*Y+P23*Z+P24;
		double z = P31*X+P32*Y+P33*Z+P34;

		x /= z;
		y /= z;

		//double u_n = x/K11 - K13/K11;
		//double v_n = y/K22 - K23/K22;

		//double r_u = sqrt(u_n*u_n+v_n*v_n);
		//double r_d = 1/omega*atan(r_u*tan_omega_half_2);

		//double u_d_n = r_d/r_u * u_n;
		//double v_d_n = r_d/r_u * v_n;

		//double u_d = u_d_n*K11 + K13;
		//double v_d = v_d_n*K22 + K23;

		hx[2*iP] = x;
		hx[2*iP+1] = y;
	}

}

void Projection3Donto2D_STR_Ray_ICCV(double *rt, double *hx, int m, int n, void *adata)
{

	//double omega = ((((AdditionalData *) adata)->vIntrinsic)[0])[4];

	//double tan_omega_half_2 = 2*tan(omega/2); 

	double X = rt[0];
	double Y = rt[1];
	double Z = rt[2];
	for (int iP = 0; iP < ((AdditionalData *) adata)->vP.size(); iP++)
	{
		double P1 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 0, 0);
		double P2 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 1, 0);
		double P3 = cvGetReal2D(((AdditionalData *) adata)->vP[iP], 2, 0);

		double V1 = cvGetReal2D(((AdditionalData *) adata)->vV[iP], 0, 0);
		double V2 = cvGetReal2D(((AdditionalData *) adata)->vV[iP], 1, 0);
		double V3 = cvGetReal2D(((AdditionalData *) adata)->vV[iP], 2, 0);

		double bandwidth = ((AdditionalData *) adata)->vBandwidth[iP];

		if (bandwidth < 0)
			continue;
		double norm_v = sqrt(V1*V1+V2*V2+V3*V3);
		V1 /= norm_v;	V2 /= norm_v;	V3 /= norm_v;
		double xmp1 = X-P1;
		double xmp2 = X-P2;
		double xmp3 = X-P3;
		double dot = V1*xmp1+V2*xmp2+V3*xmp3;
		
		double dot_ray1 = V1*dot;
		double dot_ray2 = V2*dot;
		double dot_ray3 = V3*dot;

		double dx1 = xmp1-dot_ray1;
		double dx2 = xmp2-dot_ray2;
		double dx3 = xmp3-dot_ray3;

		double x,y;

		if (dot > 0)
		{
			double dist1 = sqrt(dx1*dx1+dx2*dx2+dx3*dx3);
			double dist2 = dot;

			x = dx1/dot*((AdditionalData *) adata)->vWeight[iP];
			y = dx2/dot*((AdditionalData *) adata)->vWeight[iP];
		}
		else
		{
			x = 100;
			y - 100;
		}

		hx[2*iP] = x;
		hx[2*iP+1] = y;
	}

}

void Projection3Donto2D_FaceRefinement(double *rt, double *hx, int m, int n, void *adata)
{
	double lambda = 5;
	for (int i = 0; i < ((AdditionalData *) adata)->nFrames-1; i++)
	{
		hx[3*i] = lambda*(rt[3*i]-rt[3*(i+1)]);
		hx[3*i+1] = lambda*(rt[3*i+1]-rt[3*(i+1)+1]);
		hx[3*i+2] = lambda*(rt[3*i+2]-rt[3*(i+1)+2]);

		//cout << hx[3*i] << " " << ((AdditionalData *) adata)->measurements[3*i] << endl;
	}

	int k = 0;

	for (int ivP = 0; ivP < ((AdditionalData *) adata)->vP.size(); ivP++)
	{
		int idx = ((AdditionalData *) adata)->vIdx[ivP];
		double X = rt[3*idx];
		double Y = rt[3*idx+1];
		double Z = rt[3*idx+2];
		

		double P11 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 0, 0);
		double P12 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 0, 1);
		double P13 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 0, 2);
		double P14 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 0, 3);

		double P21 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 1, 0);
		double P22 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 1, 1);
		double P23 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 1, 2);
		double P24 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 1, 3);

		double P31 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 2, 0);
		double P32 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 2, 1);
		double P33 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 2, 2);
		double P34 = cvGetReal2D(((AdditionalData *) adata)->vP[ivP], 2, 3);

		double x = P11*X+P12*Y+P13*Z+P14;
		double y = P21*X+P22*Y+P23*Z+P24;
		double z = P31*X+P32*Y+P33*Z+P34;

		x /= z;
		y /= z;

		hx[(((AdditionalData *) adata)->nFrames-1)*3+2*k] = x;
		hx[(((AdditionalData *) adata)->nFrames-1)*3+2*k+1] = y;
		

		//cout << idx << " " << x << " " << ((AdditionalData *) adata)->measurements[(((AdditionalData *) adata)->nFrames-1)*3+2*k] << " ";
		//cout << y << " " << ((AdditionalData *) adata)->measurements[(((AdditionalData *) adata)->nFrames-1)*3+2*k+1] << endl;

		k++;
	}
}

void Projection3Donto2D_STR_fast_SO_NoDistortion_All(double *rt, double *hx, int m, int n, void *adata)
{
	int k = 0;
	for (int ivP = 0; ivP < ((AdditionalData *) adata)->vvP.size(); ivP++)
	{
		double X = rt[0];
		double Y = rt[1];
		double Z = rt[2];

		for (int iP = 0; iP < ((AdditionalData *) adata)->vvP[ivP].size(); iP++)
		{
			double P11 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 0, 0);
			double P12 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 0, 1);
			double P13 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 0, 2);
			double P14 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 0, 3);

			double P21 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 1, 0);
			double P22 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 1, 1);
			double P23 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 1, 2);
			double P24 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 1, 3);

			double P31 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 2, 0);
			double P32 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 2, 1);
			double P33 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 2, 2);
			double P34 = cvGetReal2D(((AdditionalData *) adata)->vvP[ivP][iP], 2, 3);

			double x = P11*X+P12*Y+P13*Z+P14;
			double y = P21*X+P22*Y+P23*Z+P24;
			double z = P31*X+P32*Y+P33*Z+P34;

			x /= z;
			y /= z;

			hx[2*k] = x;
			hx[2*k+1] = y;
			k++;
		}

	}
}

void Projection3Donto2D_CfM_corner(double *rt, double *hx, int m, int n, void *adata)
{
	//vector<vector<int> > vvFrameIdx = ((AdditionalData *) adata)->vvCameraIndex_CfM;
	//vector<vector<vector<Correspondence2D3D>>> vvvCorr = ((AdditionalData *) adata)->vvvCorr;
	//vector<double> vMeasurement = ((AdditionalData *) adata)->vMeasurement;

	vector<vector<CornerPoint> > vvcp = ((AdditionalData *) adata)->vvcp;
	double lambda = ((AdditionalData *) adata)->lambda;

	vector<double> vH11, vH12, vH13;
	vector<double> vH21, vH22, vH23;
	vector<double> vH31, vH32, vH33;

	//vH11.push_back(1);	vH12.push_back(0);	vH13.push_back(0);
	//vH21.push_back(0);	vH22.push_back(1);	vH23.push_back(0);
	//vH31.push_back(0);	vH32.push_back(0);	vH33.push_back(1);

	for (int icp = 0; icp < vvcp.size(); icp++)
	{
		vH11.push_back(rt[icp*8]);
		vH12.push_back(rt[icp*8+1]);
		vH13.push_back(rt[icp*8+2]);

		vH21.push_back(rt[icp*8+3]);
		vH22.push_back(rt[icp*8+4]);
		vH23.push_back(rt[icp*8+5]);

		vH31.push_back(rt[icp*8+6]);
		vH32.push_back(rt[icp*8+7]);
		vH33.push_back(1);
	}

	int i = 0;
	for (int icp = 0; icp < vvcp.size()-1; icp++)
	{
		for (int j = 0; j < vvcp[icp][1].vx.size(); j++)
		{
			double x1 = vvcp[icp][1].vx[j];
			double y1 = vvcp[icp][1].vy[j];

			double x2 = vvcp[icp+1][0].vx[j];
			double y2 = vvcp[icp+1][0].vy[j];

			double x1n = (vH11[icp]*x1+vH12[icp]*y1+vH13[icp])/(vH31[icp]*x1+vH32[icp]*y1+vH33[icp]);
			double y1n = (vH21[icp]*x1+vH22[icp]*y1+vH23[icp])/(vH31[icp]*x1+vH32[icp]*y1+vH33[icp]);
			
			double x2n = (vH11[icp+1]*x2+vH12[icp+1]*y2+vH13[icp+1])/(vH31[icp+1]*x2+vH32[icp+1]*y2+vH33[icp+1]);
			double y2n = (vH21[icp+1]*x2+vH22[icp+1]*y2+vH23[icp+1])/(vH31[icp+1]*x2+vH32[icp+1]*y2+vH33[icp+1]);

			hx[2*i] = lambda*(x1n-x2n);
			hx[2*i+1] = lambda*(y1n-y2n);
			i++;
		}
	}

	for (int j = 0; j < vvcp[0][1].vx.size(); j++)
	{
		double x1 = vvcp[vvcp.size()-1][1].vx[j];
		double y1 = vvcp[vvcp.size()-1][1].vy[j];

		double x2 = vvcp[0][0].vx[j];
		double y2 = vvcp[0][0].vy[j];

		double x1n = (vH11[vvcp.size()-1]*x1+vH12[vvcp.size()-1]*y1+vH13[vvcp.size()-1])/(vH31[vvcp.size()-1]*x1+vH32[vvcp.size()-1]*y1+vH33[vvcp.size()-1]);
		double y1n = (vH21[vvcp.size()-1]*x1+vH22[vvcp.size()-1]*y1+vH23[vvcp.size()-1])/(vH31[vvcp.size()-1]*x1+vH32[vvcp.size()-1]*y1+vH33[vvcp.size()-1]);

		double x2n = (vH11[0]*x2+vH12[0]*y2+vH13[0])/(vH31[0]*x2+vH32[0]*y2+vH33[0]);
		double y2n = (vH21[0]*x2+vH22[0]*y2+vH23[0])/(vH31[0]*x2+vH32[0]*y2+vH33[0]);

		hx[2*i] = lambda*(x1n-x2n);
		hx[2*i+1] = lambda*(y1n-y2n);
		i++;
	}


	for (int icp = 0; icp < vvcp.size()-1; icp++)
	{
		for (int j = 0; j < vvcp[icp][1].vx.size(); j++)
		{
			double x1 = vvcp[icp][1].vx[j];
			double y1 = vvcp[icp][1].vy[j];

			double x2 = vvcp[icp+1][0].vx[j];
			double y2 = vvcp[icp+1][0].vy[j];

			double x1n = (vH11[icp]*x1+vH12[icp]*y1+vH13[icp])/(vH31[icp]*x1+vH32[icp]*y1+vH33[icp]);
			double y1n = (vH21[icp]*x1+vH22[icp]*y1+vH23[icp])/(vH31[icp]*x1+vH32[icp]*y1+vH33[icp]);
			
			double x2n = (vH11[icp+1]*x2+vH12[icp+1]*y2+vH13[icp+1])/(vH31[icp+1]*x2+vH32[icp+1]*y2+vH33[icp+1]);
			double y2n = (vH21[icp+1]*x2+vH22[icp+1]*y2+vH23[icp+1])/(vH31[icp+1]*x2+vH32[icp+1]*y2+vH33[icp+1]);

			hx[2*i] = (1-lambda)*(x1n+x2n)/2;
			hx[2*i+1] = (1-lambda)*(y1n+y2n)/2;
			i++;
		}
	}

	for (int j = 0; j < vvcp[0][1].vx.size(); j++)
	{
		double x1 = vvcp[vvcp.size()-1][1].vx[j];
		double y1 = vvcp[vvcp.size()-1][1].vy[j];

		double x2 = vvcp[0][0].vx[j];
		double y2 = vvcp[0][0].vy[j];

		double x1n = (vH11[vvcp.size()-1]*x1+vH12[vvcp.size()-1]*y1+vH13[vvcp.size()-1])/(vH31[vvcp.size()-1]*x1+vH32[vvcp.size()-1]*y1+vH33[vvcp.size()-1]);
		double y1n = (vH21[vvcp.size()-1]*x1+vH22[vvcp.size()-1]*y1+vH23[vvcp.size()-1])/(vH31[vvcp.size()-1]*x1+vH32[vvcp.size()-1]*y1+vH33[vvcp.size()-1]);

		double x2n = (vH11[0]*x2+vH12[0]*y2+vH13[0])/(vH31[0]*x2+vH32[0]*y2+vH33[0]);
		double y2n = (vH21[0]*x2+vH22[0]*y2+vH23[0])/(vH31[0]*x2+vH32[0]*y2+vH33[0]);

		hx[2*i] = (1-lambda)*(x1n+x2n)/2;
		hx[2*i+1] = (1-lambda)*(y1n+y2n)/2;
		i++;
	}

	//for (int iH = 0; iH < vH11.size(); iH++)
	//{
	//	hx[2*i+8*(iH)] = (1-lambda)*vH11[iH];
	//	hx[2*i+8*(iH)+1] = (1-lambda)*vH12[iH];
	//	hx[2*i+8*(iH)+2] = (1-lambda)*vH13[iH];

	//	hx[2*i+8*(iH)+3] = (1-lambda)*vH21[iH];
	//	hx[2*i+8*(iH)+4] = (1-lambda)*vH22[iH];
	//	hx[2*i+8*(iH)+5] = (1-lambda)*vH23[iH];

	//	hx[2*i+8*(iH)+6] = (1-lambda)*vH31[iH];
	//	hx[2*i+8*(iH)+7] = (1-lambda)*vH32[iH];
	//}

	double sum = 0;

	for (int k = 0; k < i; k++)
	{
		sum += sqrt(hx[2*k]*hx[2*k]+hx[2*k+1]*hx[2*k+1]);
	}
	//cout << sum << endl;

	//for (int k = 0; k < ((AdditionalData *) adata)->vMeasurement.size(); k++)
	//{
	//	cout << k << hx[k] << " " << ((AdditionalData *) adata)->vMeasurement[k] << endl;
	//}
	//cout << ((AdditionalData *) adata)->vMeasurement.size() << endl;
}

void Projection3Donto2D_CfM_ECCM(double *rt, double *hx, int m, int n, void *adata)
{
	//vector<vector<int> > vvFrameIdx = ((AdditionalData *) adata)->vvCameraIndex_CfM;
	//vector<vector<vector<Correspondence2D3D>>> vvvCorr = ((AdditionalData *) adata)->vvvCorr;
	vector<double> vMeasurement = ((AdditionalData *) adata)->vMeasurement;

	vector<double> vx1 = ((AdditionalData *) adata)->vx1d;
	vector<double> vy1 = ((AdditionalData *) adata)->vy1d;
	double lambda = ((AdditionalData *) adata)->lambda;

	double H11 = rt[0];	double H12 = rt[1];	double H13 = rt[2];
	double H21 = rt[3];	double H22 = rt[4];	double H23 = rt[5];
	double H31 = rt[6];	double H32 = rt[7];	double H33 = 1;

	int i = 0;
	for (int ix = 0; ix < vx1.size(); ix++)
	{
		double x1 = vx1[ix];
		double y1 = vy1[ix];

		double x1n = (H11*x1+H12*y1+H13)/(H31*x1+H32*y1+H33);
		double y1n = (H21*x1+H22*y1+H23)/(H31*x1+H32*y1+H33);
			
		hx[2*ix] = x1n/lambda;
		hx[2*ix+1] = y1n/lambda;

		//hx[2*ix] = vMeasurement[ix*2];
		//hx[2*ix+1] = vMeasurement[ix*2+1];

		//cout << x1n << " " << y1n << " " << vMeasurement[ix*2] << " " << vMeasurement[ix*2+1] << endl;
	}
	hx[2*vx1.size()] = H11/H33;
	hx[2*vx1.size()+1] = H12/H33;
	hx[2*vx1.size()+2] = H13/H33;

	hx[2*vx1.size()+3] = H21/H33;
	hx[2*vx1.size()+4] = H22/H33;
	hx[2*vx1.size()+5] = H23/H33;

	hx[2*vx1.size()+6] = H31/H33;
	hx[2*vx1.size()+7] = H32/H33;

}

void Projection3Donto2D_CfM_ECCM1(double *rt, double *hx, int m, int n, void *adata)
{
	//vector<vector<int> > vvFrameIdx = ((AdditionalData *) adata)->vvCameraIndex_CfM;
	//vector<vector<vector<Correspondence2D3D>>> vvvCorr = ((AdditionalData *) adata)->vvvCorr;
	vector<double> vMeasurement = ((AdditionalData *) adata)->vMeasurement;

	vector<double> vx1 = ((AdditionalData *) adata)->vx1d;
	vector<double> vy1 = ((AdditionalData *) adata)->vy1d;

	vector<double> vx3 = ((AdditionalData *) adata)->vx3d;
	vector<double> vy3 = ((AdditionalData *) adata)->vy3d;

	double lambda = ((AdditionalData *) adata)->lambda;

	double H11 = rt[0];	double H12 = rt[1];	double H13 = rt[2];
	double H21 = rt[3];	double H22 = rt[4];	double H23 = rt[5];
	double H31 = rt[6];	double H32 = rt[7];	double H33 = 1;

	int i = 0;
	for (int ix = 0; ix < vx1.size(); ix++)
	{
		double x1 = vx1[ix];
		double y1 = vy1[ix];

		double x1n = (H11*x1+H12*y1+H13)/(H31*x1+H32*y1+H33);
		double y1n = (H21*x1+H22*y1+H23)/(H31*x1+H32*y1+H33);
			
		hx[2*ix] = x1n/lambda;
		hx[2*ix+1] = y1n/lambda;

		//hx[2*ix] = vMeasurement[ix*2];
		//hx[2*ix+1] = vMeasurement[ix*2+1];

		//cout << x1n << " " << y1n << " " << vMeasurement[ix*2] << " " << vMeasurement[ix*2+1] << endl;
	}

	for (int ix = 0; ix < vx3.size(); ix++)
	{
		double x1 = vx3[ix];
		double y1 = vy3[ix];

		double x1n = (H11*x1+H12*y1+H13)/(H31*x1+H32*y1+H33);
		double y1n = (H21*x1+H22*y1+H23)/(H31*x1+H32*y1+H33);
			
		hx[vx1.size()*2+2*ix] = x1n;
		hx[vx1.size()*2+2*ix+1] = y1n;
	}

	//hx[2*vx1.size()] = H11/H33;
	//hx[2*vx1.size()+1] = H12/H33;
	//hx[2*vx1.size()+2] = H13/H33;

	//hx[2*vx1.size()+3] = H21/H33;
	//hx[2*vx1.size()+4] = H22/H33;
	//hx[2*vx1.size()+5] = H23/H33;

	//hx[2*vx1.size()+6] = H31/H33;
	//hx[2*vx1.size()+7] = H32/H33;

}

void Projection3Donto2D_CfM_ECCM_Final(double *rt, double *hx, int m, int n, void *adata)
{
	vector<double> vMeasurement = ((AdditionalData *) adata)->vMeasurement;
	vector<double> vH11, vH12, vH13;
	vector<double> vH21, vH22, vH23;
	vector<double> vH31, vH32, vH33;
	vector<vector<double> > vVx1, vVy1, vVx2, vVy2;
	vVx1 = ((AdditionalData *) adata)->vVx1;
	vVy1 = ((AdditionalData *) adata)->vVy1;
	vVx2 = ((AdditionalData *) adata)->vVx2;
	vVy2 = ((AdditionalData *) adata)->vVy2;

	for (int i = 0; i < 5; i++)
	{
		vH11.push_back(rt[8*i+0]);	vH12.push_back(rt[8*i+1]);	vH13.push_back(rt[8*i+2]);
		vH21.push_back(rt[8*i+3]);	vH22.push_back(rt[8*i+4]);	vH23.push_back(rt[8*i+5]);
		vH31.push_back(rt[8*i+6]);	vH32.push_back(rt[8*i+7]);	vH33.push_back(1);
	}

	int k = 0;
	for (int i = 0; i < 4; i++)
	{
		double u = vH11[0]*vVx1[0][i]+vH12[0]*vVy1[0][i]+vH13[0];
		double v = vH21[0]*vVx1[0][i]+vH22[0]*vVy1[0][i]+vH23[0];
		double w = vH31[0]*vVx1[0][i]+vH32[0]*vVy1[0][i]+vH33[0];
		hx[2*k+0] = u/w;
		hx[2*k+1] = v/w;
		k++;
	}

	for (int j = 0; j < 4; j++)
	{
		for (int i = 0; i < 4; i++)
		{
			double u1 = vH11[j]*vVx2[j+1][i]+vH12[j]*vVy2[j+1][i]+vH13[j];
			double v1 = vH21[j]*vVx2[j+1][i]+vH22[j]*vVy2[j+1][i]+vH23[j];
			double w1 = vH31[j]*vVx2[j+1][i]+vH32[j]*vVy2[j+1][i]+vH33[j];

			double u2 = vH11[j+1]*vVx1[j+1][i]+vH12[j+1]*vVy1[j+1][i]+vH13[j+1];
			double v2 = vH21[j+1]*vVx1[j+1][i]+vH22[j+1]*vVy1[j+1][i]+vH23[j+1];
			double w2 = vH31[j+1]*vVx1[j+1][i]+vH32[j+1]*vVy1[j+1][i]+vH33[j+1];

			hx[2*k+0] = u1/w1-u2/w2;
			hx[2*k+1] = v1/w1-v2/w2;
			k++;
		}
	}

	for (int i = 0; i < 4; i++)
	{
		double u = vH11[4]*vVx2[5][i]+vH12[4]*vVy2[5][i]+vH13[4];
		double v = vH21[4]*vVx2[5][i]+vH22[4]*vVy2[5][i]+vH23[4];
		double w = vH31[4]*vVx2[5][i]+vH32[4]*vVy2[5][i]+vH33[4];
		hx[2*k+0] = u/w;
		hx[2*k+1] = v/w;
		k++;
	}

	//for (int i = 0; i < vMeasurement.size(); i++)
	//{
	//	cout << vMeasurement[i] << " " << hx[i] << " / ";
	//}
}



void Projection3Donto2D_CfM(double *rt, double *hx, int m, int n, void *adata)
{
	//vector<vector<int> > vvFrameIdx = ((AdditionalData *) adata)->vvCameraIndex_CfM;
	//vector<vector<vector<Correspondence2D3D>>> vvvCorr = ((AdditionalData *) adata)->vvvCorr;
	vector<double> vMeasurement = ((AdditionalData *) adata)->vMeasurement;

	int i = 0;
	for (int iCamera = 0; iCamera < ((AdditionalData *) adata)->vvvCorr.size(); iCamera++)
	{
		double K11 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[0];
		double K12 = 0;
		double K13 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[2];
		double K21 = 0;
		double K22 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[1];
		double K23 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[3];
		double K31 = 0;
		double K32 = 0;
		double K33 = 1;

		for (int iFrame = 0; iFrame < ((AdditionalData *) adata)->vvvCorr[iCamera].size(); iFrame++)
		{
			//double error = 0;
			for (int iPoint = 0; iPoint < ((AdditionalData *) adata)->vvvCorr[iCamera][iFrame].size(); iPoint += ((AdditionalData *) adata)->stride)
			{
				double X1 = ((AdditionalData *) adata)->vvvCorr[iCamera][iFrame][iPoint].x;
				double X2 = ((AdditionalData *) adata)->vvvCorr[iCamera][iFrame][iPoint].y;
				double X3 = ((AdditionalData *) adata)->vvvCorr[iCamera][iFrame][iPoint].z;

				// virtual camera
				//cout << iCamera << " " << iFrame << " " << vvFrameIdx[iCamera][iFrame] << endl;
				double qw = rt[iFrame*7+0];
				double qx = rt[iFrame*7+1];
				double qy = rt[iFrame*7+2];
				double qz = rt[iFrame*7+3];

				double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
				qw /= q_norm;
				qx /= q_norm;
				qy /= q_norm;
				qz /= q_norm;
	
				double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
				double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
				double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

				double c1 = rt[iFrame*7+4];
				double c2 = rt[iFrame*7+5];
				double c3 = rt[iFrame*7+6];

				// transformation
				double qw_t = rt[((AdditionalData *) adata)->nFrames*7+iCamera*7+0];
				double qx_t = rt[((AdditionalData *) adata)->nFrames*7+iCamera*7+1];
				double qy_t = rt[((AdditionalData *) adata)->nFrames*7+iCamera*7+2];
				double qz_t = rt[((AdditionalData *) adata)->nFrames*7+iCamera*7+3];

				double q_norm_t = sqrt(qw_t*qw_t + qx_t*qx_t + qy_t*qy_t + qz_t*qz_t);
				qw_t /= q_norm_t;
				qx_t /= q_norm_t;
				qy_t /= q_norm_t;
				qz_t /= q_norm_t;
	
				double R11_t = 1.0-2*qy_t*qy_t-2*qz_t*qz_t; 		double R12_t = 2*qx_t*qy_t-2*qz_t*qw_t;		double R13_t = 2*qx_t*qz_t+2*qy_t*qw_t;
				double R21_t = 2*qx_t*qy_t+2*qz_t*qw_t;				double R22_t = 1.0-2*qx_t*qx_t-2*qz_t*qz_t;	double R23_t = 2*qz_t*qy_t-2*qx_t*qw_t;
				double R31_t = 2*qx_t*qz_t-2*qy_t*qw_t;				double R32_t = 2*qy_t*qz_t+2*qx_t*qw_t;		double R33_t = 1.0-2*qx_t*qx_t-2*qy_t*qy_t;

				double c1_t = rt[((AdditionalData *) adata)->nFrames*7+iCamera*7+4];
				double c2_t = rt[((AdditionalData *) adata)->nFrames*7+iCamera*7+5];
				double c3_t = rt[((AdditionalData *) adata)->nFrames*7+iCamera*7+6];

				double RX1 = R11*X1+R12*X2+R13*X3;
				double RX2 = R21*X1+R22*X2+R23*X3;
				double RX3 = R31*X1+R32*X2+R33*X3;

				double RtRX1 = R11_t*RX1+R12_t*RX2+R13_t*RX3;
				double RtRX2 = R21_t*RX1+R22_t*RX2+R23_t*RX3;
				double RtRX3 = R31_t*RX1+R32_t*RX2+R33_t*RX3;

				double KRtRX1 = K11*RtRX1+K12*RtRX2+K13*RtRX3;
				double KRtRX2 = K21*RtRX1+K22*RtRX2+K23*RtRX3;
				double KRtRX3 = K31*RtRX1+K32*RtRX2+K33*RtRX3;

				double RC1 = R11*c1+R12*c2+R13*c3;
				double RC2 = R21*c1+R22*c2+R23*c3;
				double RC3 = R31*c1+R32*c2+R33*c3;

				double RtRC1 = R11_t*RC1+R12_t*RC2+R13_t*RC3;
				double RtRC2 = R21_t*RC1+R22_t*RC2+R23_t*RC3;
				double RtRC3 = R31_t*RC1+R32_t*RC2+R33_t*RC3;

				RtRC1 = c1_t-RtRC1;
				RtRC2 = c2_t-RtRC2;
				RtRC3 = c3_t-RtRC3;

				double KRtRC1 = K11*RtRC1+K12*RtRC2+K13*RtRC3;
				double KRtRC2 = K21*RtRC1+K22*RtRC2+K23*RtRC3;
				double KRtRC3 = K31*RtRC1+K32*RtRC2+K33*RtRC3;

				//double R_tTc1 = R11_t*c1_t+R21_t*c2_t+R31_t*c3_t;
				//double R_tTc2 = R12_t*c1_t+R22_t*c2_t+R32_t*c3_t;
				//double R_tTc3 = R13_t*c1_t+R23_t*c2_t+R33_t*c3_t;

				//double RtTRTc1 = R11*R_tTc1+R21*R_tTc2+R31*R_tTc3;
				//double RtTRTc2 = R12*R_tTc1+R22*R_tTc2+R32*R_tTc3;
				//double RtTRTc3 = R13*R_tTc1+R23*R_tTc2+R33*R_tTc3;

				//RtTRTc1 = RtTRTc1-c1;
				//RtTRTc2 = RtTRTc2-c2;
				//RtTRTc3 = RtTRTc3-c3;

				//double RRtTRTc1 = R11*RtTRTc1+R12*RtTRTc2+R13*RtTRTc3;
				//double RRtTRTc2 = R21*RtTRTc1+R22*RtTRTc2+R23*RtTRTc3;
				//double RRtTRTc3 = R31*RtTRTc1+R32*RtTRTc2+R33*RtTRTc3;

				//double RtRRtTRTc1 = R11_t*RRtTRTc1+R12_t*RRtTRTc2+R13_t*RRtTRTc3;
				//double RtRRtTRTc2 = R21_t*RRtTRTc1+R22_t*RRtTRTc2+R23_t*RRtTRTc3;
				//double RtRRtTRTc3 = R31_t*RRtTRTc1+R32_t*RRtTRTc2+R33_t*RRtTRTc3;

				//double KRtTRTc1 = K11*RRtTRTc1+K12*RRtTRTc2+K13*RRtTRTc3;
				//double KRtTRTc2 = K21*RRtTRTc1+K22*RRtTRTc2+K23*RRtTRTc3;
				//double KRtTRTc3 = K31*RRtTRTc1+K32*RRtTRTc2+K33*RRtTRTc3;

				double proj1 = KRtRX1+KRtRC1;
				double proj2 = KRtRX2+KRtRC2;
				double proj3 = KRtRX3+KRtRC3;

				hx[2*i] = proj1/proj3;
				hx[2*i+1] = proj2/proj3;

				//error += sqrt((hx[2*i]-vMeasurement[2*i])*(hx[2*i]-vMeasurement[2*i])+(hx[2*i+1]-vMeasurement[2*i+1])*(hx[2*i+1]-vMeasurement[2*i+1]));
				//cout << 2*i+1  << " " << ((AdditionalData *) adata)->nFrames*7+iCamera*7+6 << " " << vvFrameIdx[iCamera][iFrame]*7+6 << endl;
				//cout << hx[2*i] << " " << hx[2*i+1] << " " << vMeasurement[2*i] << " " << vMeasurement[2*i+1] << endl;;
				//cout << i << endl;
				i++;
			}
			//cout << 2*i+1  << " " << ((AdditionalData *) adata)->nFrames*7+iCamera*7+6 << " " << vvFrameIdx[iCamera][iFrame]*7+6 << endl;
			//cout << error << endl;
		}
	}

	//for (int k = 0; k < i; k++)
	//{
	//	cout << k << " " << hx[2*k] << " " << hx[2*k+1] << endl; 
	//}
	//cout << ((AdditionalData *) adata)->vMeasurement.size() << endl;
}


void Projection3Donto2D_MOT_fast(double *rt, double *hx, int m, int n, void *adata)
{
	// Set intrinsic parameter
	double K11 = ((((AdditionalData *) adata)->vIntrinsic)[0])[0];
	double K12 = 0;
	double K13 = ((((AdditionalData *) adata)->vIntrinsic)[0])[2];
	double K21 = 0;
	double K22 = ((((AdditionalData *) adata)->vIntrinsic)[0])[1];
	double K23 = ((((AdditionalData *) adata)->vIntrinsic)[0])[3];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;
	
	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];
	
	for (int iPoint = 0; iPoint < n/2; iPoint++)
	{
		double X1 = ((AdditionalData *) adata)->XYZ[3*iPoint];
		double X2 = ((AdditionalData *) adata)->XYZ[3*iPoint+1];
		double X3 = ((AdditionalData *) adata)->XYZ[3*iPoint+2];

		// Building projection 
		double RX1 = R11*X1+R12*X2+R13*X3;
		double RX2 = R21*X1+R22*X2+R23*X3;
		double RX3 = R31*X1+R32*X2+R33*X3;

		double KRX1 = K11*RX1+K12*RX2+K13*RX3;
		double KRX2 = K21*RX1+K22*RX2+K23*RX3;
		double KRX3 = K31*RX1+K32*RX2+K33*RX3;

		double RC1 = R11*C1+R12*C2+R13*C3;
		double RC2 = R21*C1+R22*C2+R23*C3;
		double RC3 = R31*C1+R32*C2+R33*C3;

		double KRC1 = K11*RC1+K12*RC2+K13*RC3;
		double KRC2 = K21*RC1+K22*RC2+K23*RC3;
		double KRC3 = K31*RC1+K32*RC2+K33*RC3;

		double proj1 = KRX1-KRC1;
		double proj2 = KRX2-KRC2;
		double proj3 = KRX3-KRC3;

		hx[2*iPoint] = proj1/proj3;
		hx[2*iPoint+1] = proj2/proj3;
	}
}

void Projection3Donto2D_MOT_fast_Dome(double *rt, double *hx, int m, int n, void *adata)
{
	// Set intrinsic parameter
	double K11 = rt[7];
	double K12 = 0;
	double K13 = rt[9];
	double K21 = 0;
	double K22 = rt[8];
	double K23 = rt[10];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;
	
	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];
	
	for (int iPoint = 0; iPoint < n/2; iPoint++)
	{
		double X1 = ((AdditionalData *) adata)->XYZ[3*iPoint];
		double X2 = ((AdditionalData *) adata)->XYZ[3*iPoint+1];
		double X3 = ((AdditionalData *) adata)->XYZ[3*iPoint+2];

		// Building projection 
		double RX1 = R11*X1+R12*X2+R13*X3;
		double RX2 = R21*X1+R22*X2+R23*X3;
		double RX3 = R31*X1+R32*X2+R33*X3;

		double KRX1 = K11*RX1+K12*RX2+K13*RX3;
		double KRX2 = K21*RX1+K22*RX2+K23*RX3;
		double KRX3 = K31*RX1+K32*RX2+K33*RX3;

		double RC1 = R11*C1+R12*C2+R13*C3;
		double RC2 = R21*C1+R22*C2+R23*C3;
		double RC3 = R31*C1+R32*C2+R33*C3;

		double KRC1 = K11*RC1+K12*RC2+K13*RC3;
		double KRC2 = K21*RC1+K22*RC2+K23*RC3;
		double KRC3 = K31*RC1+K32*RC2+K33*RC3;

		double proj1 = KRX1-KRC1;
		double proj2 = KRX2-KRC2;
		double proj3 = KRX3-KRC3;

		hx[2*iPoint] = proj1/proj3;
		hx[2*iPoint+1] = proj2/proj3;
	}
}


void ObjectiveOrientationRefinement(double *rt, double *hx, int m, int n, void *adata)
{
	int nFrames = (((AdditionalData *) adata)->nFrames);

	vector<double> vR11, vR12, vR13, vR21, vR22, vR23, vR31, vR32, vR33;
	vector<double> vm1, vm2, vm3;

	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		// Set orientation
		double qw = rt[7*iFrame+0];
		double qx = rt[7*iFrame+1];
		double qy = rt[7*iFrame+2];
		double qz = rt[7*iFrame+3];
		double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
		qw /= q_norm;
		qx /= q_norm;
		qy /= q_norm;
		qz /= q_norm;

		double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
		double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
		double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;
		vR11.push_back(R11);		vR12.push_back(R12);		vR13.push_back(R13);
		vR21.push_back(R21);		vR22.push_back(R22);		vR23.push_back(R23);
		vR31.push_back(R31);		vR32.push_back(R32);		vR33.push_back(R33);

		vm1.push_back(rt[7*iFrame+4]);
		vm2.push_back(rt[7*iFrame+5]);
		vm3.push_back(rt[7*iFrame+6]);

		//cout << R11 << " " << R12 << " " << R13 << endl;
		//cout << R21 << " " << R22 << " " << R23 << endl;
		//cout << R31 << " " << R32 << " " << R33 << endl;
 	}

	//vector<CvMat *> vP = (((AdditionalData *) adata)->vP);

	double R11 = 1;		double R12 = 0;		double R13 = 0;
	double R21 = 0;		double R22 = 1;		double R23 = 0;
	double R31 = 0;		double R32 = 0;		double R33 = 1;
	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		double R11_t, R12_t, R13_t, R21_t, R22_t, R23_t, R31_t, R32_t, R33_t; 
		R11_t = vR11[iFrame]*R11 + vR12[iFrame]*R21 + vR13[iFrame]*R31;
		R12_t = vR11[iFrame]*R12 + vR12[iFrame]*R22 + vR13[iFrame]*R32;
		R13_t = vR11[iFrame]*R13 + vR12[iFrame]*R23 + vR13[iFrame]*R33;

		R21_t = vR21[iFrame]*R11 + vR22[iFrame]*R21 + vR23[iFrame]*R31;
		R22_t = vR21[iFrame]*R12 + vR22[iFrame]*R22 + vR23[iFrame]*R32;
		R23_t = vR21[iFrame]*R13 + vR22[iFrame]*R23 + vR23[iFrame]*R33;

		R31_t = vR31[iFrame]*R11 + vR32[iFrame]*R21 + vR33[iFrame]*R31;
		R32_t = vR31[iFrame]*R12 + vR32[iFrame]*R22 + vR33[iFrame]*R32;
		R33_t = vR31[iFrame]*R13 + vR32[iFrame]*R23 + vR33[iFrame]*R33;

		R11 = R11_t;		R12 = R12_t;		R13 = R13_t;
		R21 = R21_t;		R22 = R22_t;		R23 = R23_t;
		R31 = R31_t;		R32 = R32_t;		R33 = R33_t;
	}
	
	double qw = sqrt(abs(1.0+R11+R22+R33))/2;
	double qx, qy, qz;
	if (qw > QW_ZERO)
	{
		qx = (R32-R23)/4/qw;
		qy = (R13-R31)/4/qw;
		qz = (R21-R12)/4/qw;
	}
	else
	{
		double d = sqrt((R12*R12*R13*R13+R12*R12*R23*R23+R13*R13*R23*R23));
		qx = R12*R13/d;
		qy = R12*R23/d;
		qz = R13*R23/d;
	}

	double norm_q = sqrt(qx*qx+qy*qy+qz*qz+qw*qw);
	hx[0] = qw/norm_q;
	hx[1] = qx/norm_q;
	hx[2] = qy/norm_q;
	hx[3] = qz/norm_q;

	//cout << qw/norm_q << " " << qx/norm_q << " " << qy/norm_q << " " << qz/norm_q << endl;
	vector<CvMat *> *vx1 = (((AdditionalData *) adata)->vx1);
	vector<CvMat *> *vx2 = (((AdditionalData *) adata)->vx2);

	int idx = 4;
	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		double E11 = -vm3[iFrame]*vR21[iFrame] + vm2[iFrame]*vR31[iFrame];
		double E12 = -vm3[iFrame]*vR22[iFrame] + vm2[iFrame]*vR32[iFrame];
		double E13 = -vm3[iFrame]*vR23[iFrame] + vm2[iFrame]*vR33[iFrame];

		double E21 = vm3[iFrame]*vR11[iFrame] - vm1[iFrame]*vR31[iFrame];
		double E22 = vm3[iFrame]*vR12[iFrame] - vm1[iFrame]*vR32[iFrame];
		double E23 = vm3[iFrame]*vR13[iFrame] - vm1[iFrame]*vR33[iFrame];

		double E31 = -vm2[iFrame]*vR11[iFrame] + vm1[iFrame]*vR21[iFrame];
		double E32 = -vm2[iFrame]*vR12[iFrame] + vm1[iFrame]*vR22[iFrame];
		double E33 = -vm2[iFrame]*vR13[iFrame] + vm1[iFrame]*vR23[iFrame];

		double out = 0;
		if ((*vx1)[iFrame]->rows == 1)
			continue;
		for (int ix = 0; ix < (*vx1)[iFrame]->rows; ix++)
		{
			double x1 = cvGetReal2D((*vx1)[iFrame], ix, 0);
			double x2 = cvGetReal2D((*vx1)[iFrame], ix, 1);
			double x3 = 1;

			double xp1 = cvGetReal2D((*vx2)[iFrame], ix, 0);
			double xp2 = cvGetReal2D((*vx2)[iFrame], ix, 1);
			double xp3 = 1;

			double Ex1 = E11*x1 + E12*x2 + E13*x3;
			double Ex2 = E21*x1 + E22*x2 + E23*x3;
			double Ex3 = E31*x1 + E32*x2 + E33*x3;

			double Etxp1 = E11*xp1 + E21*xp2 + E31*xp3;
			double Etxp2 = E12*xp1 + E22*xp2 + E32*xp3;
			double Etxp3 = E13*xp1 + E23*xp2 + E33*xp3;

			double xpEx = xp1*Ex1 + xp2*Ex2 + xp3*Ex3;

			double dist = xpEx*xpEx * (1/(Ex1*Ex1+Ex2*Ex2)+1/(Etxp1*Etxp1+Etxp2*Etxp2));

			out += sqrt(dist)/(*vx1)[iFrame]->rows;
			//hx[idx] = dist;
			//idx++;
		}	
		for (int j = 0; j < 10; j++)
		{
			idx++;
			hx[idx] = 0.1*out;
		}
	}
}

void ObjectiveOrientationRefinement1(double *rt, double *hx, int m, int n, void *adata)
{
	// Set orientation
	double qw = (((AdditionalData *) adata)->qw);
	double qx = (((AdditionalData *) adata)->qx);
	double qy = (((AdditionalData *) adata)->qy);
	double qz = (((AdditionalData *) adata)->qz);
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;

	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	double m1 = rt[0];
	double m2 = rt[1];
	double m3 = rt[2];

	double norm_m = sqrt(m1*m1+m2*m2+m3*m3);
	m1 /= norm_m;
	m2 /= norm_m;
	m3 /= norm_m;

	vector<double> *vx1 = (((AdditionalData *) adata)->vx1_a);
	vector<double> *vy1 = (((AdditionalData *) adata)->vy1_a);

	vector<double> *vx2 = (((AdditionalData *) adata)->vx2_a);
	vector<double> *vy2 = (((AdditionalData *) adata)->vy2_a);

	double E11 = -m3*R21 + m2*R31;
	double E12 = -m3*R22 + m2*R32;
	double E13 = -m3*R23 + m2*R33;

	double E21 = m3*R11 - m1*R31;
	double E22 = m3*R12 - m1*R32;
	double E23 = m3*R13 - m1*R33;

	double E31 = -m2*R11 + m1*R21;
	double E32 = -m2*R12 + m1*R22;
	double E33 = -m2*R13 + m1*R23;

	for (int ix = 0; ix < (*vx1).size(); ix++)
	{
		double x1 = (*vx1)[ix];
		double x2 = (*vy1)[ix];
		double x3 = 1;

		double xp1 = (*vx2)[ix];
		double xp2 = (*vy2)[ix];
		double xp3 = 1;

		double Ex1 = E11*x1 + E12*x2 + E13*x3;
		double Ex2 = E21*x1 + E22*x2 + E23*x3;
		double Ex3 = E31*x1 + E32*x2 + E33*x3;

		double Etxp1 = E11*xp1 + E21*xp2 + E31*xp3;
		double Etxp2 = E12*xp1 + E22*xp2 + E32*xp3;
		double Etxp3 = E13*xp1 + E23*xp2 + E33*xp3;

		double xpEx = xp1*Ex1 + xp2*Ex2 + xp3*Ex3;

		double dist = xpEx*xpEx * (1/(Ex1*Ex1+Ex2*Ex2)+1/(Etxp1*Etxp1+Etxp2*Etxp2));

		hx[ix] = 100*dist;
	}	
}

void CameraCenterInterpolationWithDegree(CvMat *R_1, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat *> vM, vector<CvMat *> vm,
													 vector<int> vFrame1_r, vector<int> vFrame2_r, vector<CvMat *> vM_r, vector<CvMat *> vm_r, 
										 vector<CvMat *> vC_c, vector<CvMat *> vR_c, vector<int> vFrame_c, vector<CvMat *> &vC, vector<CvMat *> &vR, double weight)
{
	// Frame normalization
	int first = vFrame1[0];
	for (int iFrame = 0; iFrame < vFrame1.size(); iFrame++)
	{
		vFrame1[iFrame] = vFrame1[iFrame] - first;
		vFrame2[iFrame] = vFrame2[iFrame] - first;
		double m1 = cvGetReal2D(vm[iFrame], 0, 0);
		double m2 = cvGetReal2D(vm[iFrame], 1, 0);
		double m3 = cvGetReal2D(vm[iFrame], 2, 0);
		double norm_m = sqrt(m1*m1+m2*m2+m3*m3);
		ScalarMul(vm[iFrame], 1/norm_m, vm[iFrame]);
	}

	for (int iFrame = 0; iFrame < vFrame1_r.size(); iFrame++)
	{
		vFrame1_r[iFrame] = vFrame1_r[iFrame] - first;
		vFrame2_r[iFrame] = vFrame2_r[iFrame] - first;
		double m1 = cvGetReal2D(vm_r[iFrame], 0, 0);
		double m2 = cvGetReal2D(vm_r[iFrame], 1, 0);
		double m3 = cvGetReal2D(vm_r[iFrame], 2, 0);
		double norm_m = sqrt(m1*m1+m2*m2+m3*m3);
		ScalarMul(vm_r[iFrame], 1/norm_m, vm_r[iFrame]);
	}

	for (int iFrame = 0; iFrame < vFrame_c.size(); iFrame++)
	{
		vFrame_c[iFrame] = vFrame_c[iFrame] - first;
	}
	int nFrames = vFrame_c[vFrame_c.size()-1] - vFrame_c[0]+1;
	vR.resize(nFrames);
	vector<bool> vIsR(nFrames, false);
	for (int ic = 0; ic < vFrame_c.size(); ic++)
	{
		vR[vFrame_c[ic]] = cvCloneMat(vR_c[ic]);
		vIsR[vFrame_c[ic]] = true;
	}
	for (int iFrame = 0; iFrame < vFrame1.size(); iFrame++)
	{
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		int j = 0;
		if (!vIsR[vFrame1[iFrame]-j])
		{
			while (!vIsR[vFrame1[iFrame]-j])
			{
				j++;
			}
			for (int i = j-1; i >= 0; i--)
			{
				vR[vFrame1[iFrame]-i] = cvCloneMat(vR[vFrame1[iFrame]-j]);
			}
		}

		cvMatMul(vM[iFrame], vR[vFrame1[iFrame]], R);
		vR[vFrame2[iFrame]] = cvCloneMat(R);
		vIsR[vFrame2[iFrame]] = true;
		cvReleaseMat(&R);
	}

	for (int ic = 0; ic < vFrame_c.size(); ic++)
	{
		vR[vFrame_c[ic]] = cvCloneMat(vR_c[ic]);
	}
	
	int nBasis = floor((double)nFrames/3); 
	CvMat *theta_all = cvCreateMat(nFrames, nFrames, CV_32FC1);
	GetIDCTMappingMatrix(theta_all, nFrames);
	CvMat *theta_i = cvCreateMat(1, nBasis, CV_32FC1);
	CvMat *Theta_i = cvCreateMat(3, 3*nBasis, CV_32FC1);

	CvMat *A = cvCreateMat(3*vM.size(), 3*nBasis, CV_32FC1);
	CvMat *b = cvCreateMat(3*vM.size(), 1, CV_32FC1);
	cvSetZero(b);

	for (int im = 0; im < vM.size(); im++)
	{
		for (int ith = 0; ith < nBasis; ith++)
		{
			cvSetReal2D(theta_i, 0, ith, cvGetReal2D(theta_all, vFrame1[im], ith)-cvGetReal2D(theta_all, vFrame2[im], ith));
		}
		SetSubMat(Theta_i, 0, 0, theta_i);
		SetSubMat(Theta_i, 1, nBasis, theta_i);
		SetSubMat(Theta_i, 2, 2*nBasis, theta_i);

		CvMat *skewm = cvCreateMat(3,3,CV_32FC1);
		Vec2Skew(vm[im], skewm);
		CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
		cvMatMul(skewm, vR[vFrame2[im]], temp33);
		CvMat *temp3nBasis = cvCreateMat(3, 3*nBasis, CV_32FC1);
		cvMatMul(temp33, Theta_i, temp3nBasis);
		SetSubMat(A, 3*im, 0, temp3nBasis);		
		cvReleaseMat(&skewm);
		cvReleaseMat(&temp33);
		cvReleaseMat(&temp3nBasis);
	}

	CvMat *A_r = cvCreateMat(3*vM_r.size(), 3*nBasis, CV_32FC1);
	CvMat *b_r = cvCreateMat(3*vM_r.size(), 1, CV_32FC1);
	cvSetZero(b_r);

	for (int im = 0; im < vM_r.size(); im++)
	{
		for (int ith = 0; ith < nBasis; ith++)
		{
			cvSetReal2D(theta_i, 0, ith, cvGetReal2D(theta_all, vFrame1_r[im], ith)-cvGetReal2D(theta_all, vFrame2_r[im], ith));
		}
		SetSubMat(Theta_i, 0, 0, theta_i);
		SetSubMat(Theta_i, 1, nBasis, theta_i);
		SetSubMat(Theta_i, 2, 2*nBasis, theta_i);

		CvMat *skewm = cvCreateMat(3,3,CV_32FC1);
		Vec2Skew(vm_r[im], skewm);
		CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
		cvMatMul(skewm, vR[vFrame2_r[im]], temp33);
		CvMat *temp3nBasis = cvCreateMat(3, 3*nBasis, CV_32FC1);
		cvMatMul(temp33, Theta_i, temp3nBasis);
		SetSubMat(A_r, 3*im, 0, temp3nBasis);	
		cvReleaseMat(&skewm);
		cvReleaseMat(&temp33);
		cvReleaseMat(&temp3nBasis);
	}

	CvMat *A_c = cvCreateMat(3*vFrame_c.size(), 3*nBasis, CV_32FC1);
	CvMat *b_c = cvCreateMat(3*vFrame_c.size(), 1, CV_32FC1);

	for (int ic = 0; ic < vFrame_c.size(); ic++)
	{
		for (int ith = 0; ith < nBasis; ith++)
		{
			cvSetReal2D(theta_i, 0, ith, cvGetReal2D(theta_all, vFrame_c[ic], ith));
		}
		SetSubMat(Theta_i, 0, 0, theta_i);
		SetSubMat(Theta_i, 1, nBasis, theta_i);
		SetSubMat(Theta_i, 2, 2*nBasis, theta_i);
		SetSubMat(A_c, 3*ic, 0, Theta_i);
		SetSubMat(b_c, 3*ic, 0, vC_c[ic]);
	}
	ScalarMul(A_c, weight, A_c);
	ScalarMul(b_c, weight, b_c);

	CvMat *At = cvCreateMat(A->rows+A_r->rows+A_c->rows, A->cols, CV_32FC1);
	CvMat *bt = cvCreateMat(A->rows+A_r->rows+A_c->rows, 1, CV_32FC1);	

	SetSubMat(At, 0, 0, A);
	SetSubMat(bt, 0, 0, b);

	SetSubMat(At, A->rows, 0, A_r);
	SetSubMat(bt, A->rows, 0, b_r);

	SetSubMat(At, A->rows+A_r->rows, 0, A_c);
	SetSubMat(bt, A->rows+A_r->rows, 0, b_c);

	CvMat *beta = cvCreateMat(3*nBasis, 1, CV_32FC1);
	cvSolve(At, bt, beta);

	//cvSolve(A_c, b_c, beta);

	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		for (int ith = 0; ith < nBasis; ith++)
		{
			cvSetReal2D(theta_i, 0, ith, cvGetReal2D(theta_all, iFrame, ith));
		}
		SetSubMat(Theta_i, 0, 0, theta_i);
		SetSubMat(Theta_i, 1, nBasis, theta_i);
		SetSubMat(Theta_i, 2, 2*nBasis, theta_i);

		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvMatMul(Theta_i, beta, C);
		vC.push_back(C);
	}
	cvReleaseMat(&beta);
	cvReleaseMat(&At);
	cvReleaseMat(&bt);
	cvReleaseMat(&A);
	cvReleaseMat(&b);
	cvReleaseMat(&A_r);
	cvReleaseMat(&b_r);
	cvReleaseMat(&A_c);
	cvReleaseMat(&b_c);
	cvReleaseMat(&theta_all);
	cvReleaseMat(&theta_i);
	cvReleaseMat(&Theta_i);
}
/*
void SparseBundleAdjustment_KDMOT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> &cP, CvMat *X, vector<Camera> vCamera, vector<int> visibleStructureID)
{
	PrintAlgorithm("Sparse bundle adjustment motion only");
	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	adata.vUsedFrame = vUsedFrame;

	GetParameterForSBA_KDRT(vFeature, vUsedFrame, cP, X, vCamera, max_nFrames, visibleStructureID, cameraParameter, feature2DParameter, vMask);

	int nCameraParam = 7+4+2;
	int nFeatures = vFeature.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
	
	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	adata.XYZ = &(dCameraParameter[nCameraParam*vUsedFrame.size()]);
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-12;
	opt[2] = 1e-12;
	opt[3] = 1e-12;
	opt[4] = 0;
	double info[12];
	sba_mot_levmar(visibleStructureID.size(), vUsedFrame.size(), 1, dVMask,  dCameraParameter, nCameraParam, dFeature2DParameter, dCovFeatures, 2, Projection3Donto2D_KDMOT, NULL, &adata,
					1e+3, 0, opt, info);
	PrintSBAInfo(info);
	RetrieveParameterFromSBA_KDRT(dCameraParameter, vCamera, cP, X, visibleStructureID, vUsedFrame, max_nFrames);
	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
}


void SparseBundleAdjustment_TEMPORAL(vector<Feature> vFeature, vector<Theta> &vTheta, vector<Camera> &vCamera)
{
	PrintAlgorithm("Sparse bundle adjustment - Temporal adjustment");
	vector<double> cameraParameter, feature2DParameter;
	vector<char> vMask;
	double *dCovFeatures = 0;
	AdditionalData adata;// focal_x focal_y princ_x princ_y

	int max_nFrames = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
				max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
		}
	}
	max_nFrames++;
	adata.max_nFrames = max_nFrames;
	vector<int> vUsedFrame;
	vector<CvMat *> vP;
	vector<double> vdP;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			vUsedFrame.push_back(iCamera*max_nFrames+vCamera[iCamera].vTakenFrame[iFrame]);
			vP.push_back(vCamera[iCamera].vP[iFrame]);
			for (int iP = 0; iP < 3; iP++)
				for (int jP = 0; jP < 4; jP++)
					vdP.push_back(cvGetReal2D(vCamera[iCamera].vP[iFrame], iP, jP));
		}
	}
	adata.vUsedFrame = vUsedFrame;
	adata.vP = vP;
	adata.vdP = vdP;
	adata.nBase = vTheta[0].thetaX.size();
	adata.vTheta = vTheta;

	GetParameterForSBA_TEMPORAL(vFeature, vTheta, vCamera, max_nFrames, cameraParameter, feature2DParameter, vMask);

	int nCameraParam = 1;
	int nFeatures = vTheta.size(); 
	int nFrames = vUsedFrame.size(); 
	char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	for (int i = 0; i < cameraParameter.size(); i++)
		dCameraParameter[i] = cameraParameter[i];
	for (int i = 0; i < vMask.size(); i++)
		dVMask[i] = vMask[i];
	for (int i = 0; i < feature2DParameter.size(); i++)
		dFeature2DParameter[i] = feature2DParameter[i];

	adata.XYZ = &(dCameraParameter[nCameraParam*vUsedFrame.size()]);
	double opt[5];
	opt[0] = 1e-3;
	opt[1] = 1e-5;
	opt[2] = 1e-5;
	opt[3] = 1e-5;
	opt[4] = 0;
	double info[12];
	sba_motstr_levmar(vTheta.size(), 0, vUsedFrame.size(), 0, dVMask,  dCameraParameter, nCameraParam, 3*adata.nBase, dFeature2DParameter, dCovFeatures, 4, ProjectionThetaonto2D_TEMPORAL, NULL, &adata,
		1e+3, 0, opt, info);
	PrintSBAInfo(info);
	RetrieveParameterFromSBA_TEMPORAL(dCameraParameter, vCamera, vTheta);

	free(dVMask);
	free(dFeature2DParameter);
	free(dCameraParameter);
}

void SparseBundleAdjustment_TEMPORAL_LEVMAR(vector<Feature> vFeature, vector<Theta> &vTheta, vector<Camera> &vCamera)
{
	//for (int i = 0; i < 106; i++)
	//{
	//	vFeature.pop_back();
	//	vTheta.pop_back();
	//}
	//PrintAlgorithm("Sparse bundle adjustment - Temporal adjustment, Levenburg-Marquedt");
	//vector<double> cameraParameter, feature2DParameter;
	//vector<char> vMask;
	//double *dCovFeatures = 0;
	//AdditionalData adata;// focal_x focal_y princ_x princ_y

	//int max_nFrames = 0;
	//for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	//{
	//	for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
	//	{
	//		if (vCamera[iCamera].vTakenFrame[iFrame] > max_nFrames)
	//			max_nFrames = vCamera[iCamera].vTakenFrame[iFrame];
	//	}
	//}
	//max_nFrames++;
	//adata.max_nFrames = max_nFrames;
	//vector<int> vUsedFrame;
	//vector<CvMat *> vP;
	//vector<double> vdP;
	//for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	//{
	//	for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
	//	{
	//		vUsedFrame.push_back(iCamera*max_nFrames+vCamera[iCamera].vTakenFrame[iFrame]);
	//		vP.push_back(vCamera[iCamera].vP[iFrame]);
	//		for (int iP = 0; iP < 3; iP++)
	//			for (int jP = 0; jP < 4; jP++)
	//				vdP.push_back(cvGetReal2D(vCamera[iCamera].vP[iFrame], iP, jP));
	//	}
	//}
	//adata.vUsedFrame = vUsedFrame;
	////adata.vP = vP;
	//adata.vdP = vdP;
	//adata.nBase = vTheta[0].thetaX.size();
	//adata.nFeatures = vTheta.size();
	//adata.vTheta = vTheta;

	//GetParameterForSBA_TEMPORAL_LEVMAR(vFeature, vTheta, vCamera, max_nFrames, cameraParameter, feature2DParameter);

	//int nCameraParam = 1;
	//int nFeatures = vTheta.size(); 
	//int nFrames = vUsedFrame.size(); 
	////char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	//double *dFeature2DParameter = (double *) malloc((feature2DParameter.size()+nFeatures) * sizeof(double));
	//double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	//for (int i = 0; i < cameraParameter.size(); i++)
	//	dCameraParameter[i] = cameraParameter[i];
	//for (int i = 0; i < feature2DParameter.size(); i++)
	//	dFeature2DParameter[i] = feature2DParameter[i]/1e-6;
	//for (int i = 0; i < nFeatures; i++)
	//	dFeature2DParameter[feature2DParameter.size()+i] = 0.0;

	//adata.isStatic = false;

	////adata.XYZ = &(dCameraParameter[nCameraParam*vUsedFrame.size()]);
	//adata.measurements = feature2DParameter;
	//double opt[5];
	//opt[0] = 1e-1;
	//opt[1] = 1e-20;
	//opt[2] = 1e-15;
	//opt[3] = 1e-15;
	//opt[4] = 0;

	//double info[12];

	//double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size()+nFeatures)+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	//if(!work)
	//	fprintf(stderr, "memory allocation request failed in main()\n");

	//int ret ;//dlevmar_dif(ProjectionThetaonto2D_TEMPORAL_LEVMAR, dCameraParameter, dFeature2DParameter, cameraParameter.size(), feature2DParameter.size()+nFeatures,
	//	1e+3, opt, info, work, NULL, &adata);
	//cout << ret << endl;
	//PrintSBAInfo(info);
	//RetrieveParameterFromSBA_TEMPORAL(dCameraParameter, vCamera, vTheta);

	//free(dFeature2DParameter);
	//free(dCameraParameter);
}
*/



void GlobalBundleAdjustment(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames, int nBase, int nFeatures_static)
{
	//PrintAlgorithm("Global bundle adjustment motion and structure");
	//vector<double> cameraParameter, feature2DParameter;
	//vector<char> vMask;
	//double *dCovFeatures = 0;
	//AdditionalData adata;// focal_x focal_y princ_x princ_y
	//double intrinsic[4];
	//intrinsic[0] = cvGetReal2D(K, 0, 0);
	//intrinsic[1] = cvGetReal2D(K, 1, 1);
	//intrinsic[2] = cvGetReal2D(K, 0, 2);
	//intrinsic[3] = cvGetReal2D(K, 1, 2);
	//adata.intrinsic = intrinsic;
	//adata.nCameraParameters = 7;
	//adata.nImagePointPrameraters = 2;
	//adata.nBase = nBase;
	//adata.nFrames = nFrames;
	//adata.nFeature_static = nFeatures_static;
	//bool *isStatic = (bool *) malloc(vFeature.size() * sizeof(bool));
	//for (int iStatic = 0; iStatic < vFeature.size(); iStatic++)
	//{
	//	if (iStatic < nFeatures_static)
	//		isStatic[iStatic] = true;
	//	else
	//		isStatic[iStatic] = false;
	//}
	//adata.isStatic = isStatic;
	//GetParameterForGBA(vFeature, vCamera, vTheta, K, nFrames, cameraParameter, feature2DParameter, vMask);

	//int NZ = 0;
	//for (int i = 0; i < vMask.size(); i++)
	//{
	//	if (vMask[i])
	//		NZ++;
	//}
	//double nCameraParam = 7;
	//int nFeatures = vFeature.size();  
	//char *dVMask = (char *) malloc(vMask.size() * sizeof(char));
	//double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	//double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	//for (int i = 0; i < cameraParameter.size(); i++)
	//	dCameraParameter[i] = cameraParameter[i];
	//for (int i = 0; i < vMask.size(); i++)
	//	dVMask[i] = vMask[i];
	//for (int i = 0; i < feature2DParameter.size(); i++)
	//	dFeature2DParameter[i] = feature2DParameter[i];

	//double opt[5];
	//opt[0] = 1e-3;
	//opt[1] = 1e-5;//1e-12;
	//opt[2] = 1e-5;//1e-12;
	//opt[3] = 1e-5;//1e-12;
	//opt[4] = 0;
	//double info[12];
	//sba_motstr_levmar_x(vFeature.size(), 0, nFrames*vCamera.size(), 1, dVMask,  dCameraParameter, nCameraParam, 3*nBase, dFeature2DParameter, dCovFeatures, 2, ProjectionThetaonto2D_MOTSTR_x, NULL, &adata,
	//	1e+3, 0, opt, info);
	//PrintSBAInfo(info);
	//RetrieveParameterFromGBA(dCameraParameter, K, vCamera, vTheta, nFrames);

	//free(isStatic);
	//free(dVMask);
	//free(dFeature2DParameter);
	//free(dCameraParameter);
}

void GlobalBundleAdjustment_LEVMAR(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames, int nBase, int nFeatures_static)
{
	//PrintAlgorithm("Global bundle adjustment motion and structure");
	//vector<double> cameraParameter, feature2DParameter;
	//double *dCovFeatures = 0;
	//AdditionalData adata;// focal_x focal_y princ_x princ_y
	//double intrinsic[4];
	//intrinsic[0] = cvGetReal2D(K, 0, 0);
	//intrinsic[1] = cvGetReal2D(K, 1, 1);
	//intrinsic[2] = cvGetReal2D(K, 0, 2);
	//intrinsic[3] = cvGetReal2D(K, 1, 2);
	//adata.intrinsic = intrinsic;
	//adata.nCameraParameters = 7;
	//adata.nImagePointPrameraters = 2;
	//adata.nBase = nBase;
	//adata.nFrames = nFrames*vCamera.size();
	//adata.nFeature_static = nFeatures_static;
	//bool *isStatic = (bool *) malloc(vFeature.size() * sizeof(bool));
	//for (int iStatic = 0; iStatic < vFeature.size(); iStatic++)
	//{
	//	if (iStatic < nFeatures_static)
	//		isStatic[iStatic] = true;
	//	else
	//		isStatic[iStatic] = false;
	//}
	//adata.isStatic = isStatic;
	//adata.nNZ = (double)feature2DParameter.size()/2;
	//CvMat visibilityMask;
	//GetParameterForGBA(vFeature, vCamera, vTheta, K, nFrames, cameraParameter, feature2DParameter, visibilityMask);
	//CvMat *vMask = cvCreateMat(visibilityMask.rows, visibilityMask.cols, CV_32FC1);
	//vMask = cvCloneMat(&visibilityMask);
	//adata.visibilityMask = vMask;

	//double nCameraParam = 7;
	//int nFeatures = vFeature.size();  
	//double *dFeature2DParameter = (double *) malloc(feature2DParameter.size() * sizeof(double));
	//double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	//double opt[5];
	//opt[0] = 1e-3;
	//opt[1] = 1e-5;//1e-12;
	//opt[2] = 1e-5;//1e-12;
	//opt[3] = 1e-5;//1e-12;
	//opt[4] = 0;
	//double info[12];
	//double *work=(double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), feature2DParameter.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	//if(!work)
	//	fprintf(stderr, "memory allocation request failed in main()\n");

	//feature2DParameter.clear();
	//cameraParameter.clear();
	//for (int i = 0; i < 10000;i++)
	//	feature2DParameter.push_back(0);
	//for (int i = 0; i < 10000-1; i++)
	//	cameraParameter.push_back(0);

	//for (int i = 0; i < cameraParameter.size(); i++)
	//	dCameraParameter[i] = cameraParameter[i];
	//for (int i = 0; i < feature2DParameter.size(); i++)
	//	dFeature2DParameter[i] = feature2DParameter[i];
	//slevmar_dif(ProjectionThetaonto2D_MOTSTR_LEVMAR, (float*)dCameraParameter, (float*)dFeature2DParameter, cameraParameter.size(), feature2DParameter.size(),
	//			1e+3, (float*)opt, (float*)info, (float*)work, NULL, &adata);

	//PrintSBAInfo(info);
	//RetrieveParameterFromGBA(dCameraParameter, K, vCamera, vTheta, nFrames);

	//cvReleaseMat(&vMask);
	//free(isStatic);
	//free(dFeature2DParameter);
	//free(dCameraParameter);
}

void ProjectionThetaonto2D_MOTSTR_LEVMAR(double *p, double *hx, int m, int n, void *adata_)
{
	//for (int i = 0; i <n; i++)
	//	hx[i] = 1;
	//return;
	//AdditionalData *adata = (AdditionalData *)adata_;
	//bool *isStatic = adata->isStatic;
	//int nFrames = adata->nFrames;
	//int nCameraParameters = adata->nCameraParameters;
	//int nImagePointParameters = adata->nImagePointPrameraters;
	//int nBase = adata->nBase;
	//int nFeature_static = adata->nFeature_static;
	//double *XYZ = p + nCameraParameters*nFrames;
	//double *rt;
	//double *xij;
	//
	//int last = 0;
	//for (int iFeature = 0; iFeature < ((AdditionalData*)adata_)->visibilityMask->rows; iFeature++)
	//{
	//	for (int iFrame = 0; iFrame < ((AdditionalData*)adata_)->visibilityMask->cols; iFrame++)
	//	{
	//		if (cvGetReal2D(((AdditionalData*)adata_)->visibilityMask, iFeature, iFrame))
	//		{
	//			if (isStatic[iFeature])
	//			{
	//				double x, y;
	//				//xij = hx + nImagePointParameters*last;
	//				rt = p + iFrame*nCameraParameters;
	//				XYZ = p + nFrames*nCameraParameters + iFeature*3;
	//				ProjectionThetaonto2D_MOTSTR_LEVMAR(iFrame, iFeature, rt, XYZ, x, y, adata);
	//				hx[nImagePointParameters*last] = 0;
	//				hx[nImagePointParameters*last+1] = 0;
	//				last++;
	//			}
	//			else
	//			{
	//				double x, y;
	//				//xij = hx + nImagePointParameters*last;
	//				rt = p + iFrame*nCameraParameters;
	//				XYZ = p + nFrames*nCameraParameters + nFeature_static*3 + (iFeature-nFeature_static)*3*nBase;
	//				ProjectionThetaonto2D_MOTSTR_LEVMAR(iFrame, iFeature, rt, XYZ, x, y, adata);
	//				hx[nImagePointParameters*last] = 0;
	//				hx[nImagePointParameters*last+1] = 0;
	//				last++;
	//			}				
	//		}
	//	}
	//}
}

void ProjectionThetaonto2D_TEMPORAL_LEVMAR(double *p, double *hx, int m, int n, void *adata)
{
	if (!((AdditionalData *)adata)->isStatic)
	{
		((AdditionalData *)adata)->isStatic = true;
		((AdditionalData *)adata)->ptr = p;
	}
	p = ((AdditionalData *)adata)->ptr;
	for (int ihx = 0; ihx < n; ihx++)
		hx[ihx] = 0.0;
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	int nFeatures = ((AdditionalData *) adata)->nFeatures;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int nBase = ((AdditionalData*) adata)->nBase;
	vector<Theta> vTheta = ((AdditionalData *) adata)->vTheta;
	vector<double> measurements = ((AdditionalData *) adata)->measurements;

	for (int iFeature = 0; iFeature < nFeatures; iFeature++)
	{
		double *xyz = p+vUsedFrame.size()+nBase*3*iFeature;
		for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
		{
			double *rt = p+iFrame;

			int iCamera = (int)((double)vUsedFrame[iFrame]/max_nFrames);
			int cFrame = vUsedFrame[iFrame] % max_nFrames;
			CvMat *P = cvCreateMat(3,4, CV_32FC1);
			for (int iP = 0; iP < 3; iP++)
				for (int jP = 0; jP < 4; jP++)
					cvSetReal2D(P, iP, jP, ((AdditionalData *) adata)->vdP[12*iFrame+iP*4+jP]);
			double dt = rt[0];
			double t = (double)cFrame + dt;
			CvMat *B = cvCreateMat(1, nBase, CV_32FC1);
			cvSetReal2D(B, 0, 0, sqrt(1.0/(double)max_nFrames));
			for (int iB = 1; iB < nBase; iB++)
				cvSetReal2D(B, 0, iB, sqrt(2.0/(double)max_nFrames)*cos((2*t-1)*(iB)*PI/2.0/(double)max_nFrames));

			CvMat *thetaX = cvCreateMat(nBase,1, CV_32FC1);
			CvMat *thetaY = cvCreateMat(nBase,1, CV_32FC1);
			CvMat *thetaZ = cvCreateMat(nBase,1, CV_32FC1);
			for (int iTheta = 0; iTheta < nBase; iTheta++)
				cvSetReal2D(thetaX, iTheta, 0, xyz[iTheta]);
			for (int iTheta = 0; iTheta < nBase; iTheta++)
				cvSetReal2D(thetaY, iTheta, 0, xyz[nBase+iTheta]);
			for (int iTheta = 0; iTheta < nBase; iTheta++)
				cvSetReal2D(thetaZ, iTheta, 0, xyz[2*nBase+iTheta]);

			CvMat *X3 = cvCreateMat(1,1,CV_32FC1);
			CvMat *Y3 = cvCreateMat(1,1,CV_32FC1);
			CvMat *Z3 = cvCreateMat(1,1,CV_32FC1);
			cvMatMul(B, thetaX, X3);
			cvMatMul(B, thetaY, Y3);
			cvMatMul(B, thetaZ, Z3);
			CvMat *X = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(X, 0, 0, cvGetReal2D(X3,0,0));
			cvSetReal2D(X, 1, 0, cvGetReal2D(Y3,0,0));
			cvSetReal2D(X, 2, 0, cvGetReal2D(Z3,0,0));
			cvSetReal2D(X, 3, 0, 1);

			CvMat *x = cvCreateMat(3,1,CV_32FC1);
			cvMatMul(P, X, x);

			double pm_x = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
			double pm_y = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);
			double m_x = measurements[2*(iFeature*vUsedFrame.size()+iFrame)];
			double m_y = measurements[2*(iFeature*vUsedFrame.size()+iFrame)+1];
			//hx[2*(iFeature*vUsedFrame.size()+iFrame)] = (pm_x-m_x)*(pm_x-m_x);
			//hx[2*(iFeature*vUsedFrame.size()+iFrame)+1] = (pm_y-m_y)*(pm_y-m_y);

			hx[2*(iFeature*vUsedFrame.size()+iFrame)] = pm_x/1e-6;
			hx[2*(iFeature*vUsedFrame.size()+iFrame)+1] = pm_y/1e-6;

			cvReleaseMat(&X);
			cvReleaseMat(&x);
			cvReleaseMat(&P);
			cvReleaseMat(&B);
			cvReleaseMat(&thetaX);
			cvReleaseMat(&thetaY);
			cvReleaseMat(&thetaZ);
			cvReleaseMat(&X3);
			cvReleaseMat(&Y3);
			cvReleaseMat(&Z3);
		}
		int error = 0;
		for (int iBase = 0; iBase < nBase; iBase++)
		{
			error += (xyz[iBase]-vTheta[iFeature].thetaX[iBase])*(xyz[iBase]-vTheta[iFeature].thetaX[iBase]);
		}
		for (int iBase = 0; iBase < nBase; iBase++)
		{
			error += (xyz[nBase+iBase]-vTheta[iFeature].thetaY[iBase])*(xyz[nBase+iBase]-vTheta[iFeature].thetaY[iBase]);
		}
		for (int iBase = 0; iBase < nBase; iBase++)
		{
			error += (xyz[2*nBase+iBase]-vTheta[iFeature].thetaZ[iBase])*(xyz[2*nBase+iBase]-vTheta[iFeature].thetaZ[iBase]);
		}

		hx[2*(nFeatures*vUsedFrame.size())+iFeature] = 0.0;//error/10;
	}
	//for (int i = 0; i < n; i++)
	//	cout << hx[i] << " ";
	//cout << endl;
}

void ProjectionThetaonto2D_MOTSTR_LEVMAR(float *p, float *hx, int m, int n, void *adata_)
{
	//for (int i = 0; i <n; i++)
	//	hx[i] = 1;
	//return;
	//AdditionalData *adata = (AdditionalData *)adata_;
	//bool *isStatic = adata->isStatic;
	//int nFrames = adata->nFrames;
	//int nCameraParameters = adata->nCameraParameters;
	//int nImagePointParameters = adata->nImagePointPrameraters;
	//int nBase = adata->nBase;
	//int nFeature_static = adata->nFeature_static;
	//float *XYZ = p + nCameraParameters*nFrames;
	//float *rt;
	//float *xij;

	//int last = 0;
	////for (int iFeature = 0; iFeature < ((AdditionalData*)adata_)->visibilityMask->rows; iFeature++)
	////{
	////	for (int iFrame = 0; iFrame < ((AdditionalData*)adata_)->visibilityMask->cols; iFrame++)
	////	{
	////		if (cvGetReal2D(((AdditionalData*)adata_)->visibilityMask, iFeature, iFrame))
	////		{
	////			if (isStatic[iFeature])
	////			{
	////				float x, y;
	////				//xij = hx + nImagePointParameters*last;
	////				rt = p + iFrame*nCameraParameters;
	////				XYZ = p + nFrames*nCameraParameters + iFeature*3;
	////				ProjectionThetaonto2D_MOTSTR_LEVMAR(iFrame, iFeature, (double*)rt, (double*)XYZ, (double)x, (double)y, adata);
	////				hx[nImagePointParameters*last] = 0;
	////				hx[nImagePointParameters*last+1] = 0;
	////				last++;
	////			}
	////			else
	////			{
	////				float x, y;
	////				//xij = hx + nImagePointParameters*last;
	////				rt = p + iFrame*nCameraParameters;
	////				XYZ = p + nFrames*nCameraParameters + nFeature_static*3 + (iFeature-nFeature_static)*3*nBase;
	////				ProjectionThetaonto2D_MOTSTR_LEVMAR(iFrame, iFeature, (double*)rt, (double*)XYZ, (double)x, (double)y, adata);
	////				hx[nImagePointParameters*last] = 0;
	////				hx[nImagePointParameters*last+1] = 0;
	////				last++;
	////			}				
	////		}
	////	}
	////}
}


void ProjectionThetaonto2D_MOTSTR_x(double *p, struct sba_crsm *idxij, int *rcidxs, int *rcsubs, double *hx, void *adata)
{
	//int i, j;
	//int cnp, pnp, mnp;
	//double *pa, *pb, *pqr, *pt, *ppt, *pmeas, *Kparms, *pr0, lrot[4], trot[4];
	////int n;
	//int m, nnz;
	//AdditionalData *gl;

	//gl= (AdditionalData *)adata;
	//cnp=gl->nCameraParameters; 
	//mnp=gl->nImagePointPrameraters;

	////n=idxij->nr;
	//m=idxij->nc;
	//pa=p; // Pointer for camera
	//pb=p+m*cnp; // Point for xyz

	//for(j=0; j<m; ++j)
	//{
	//	/* j-th camera parameters */
	//	double *rt = pa + j*cnp;
	//	double *xyz;
	//	nnz=sba_crsm_col_elmidxs(idxij, j, rcidxs, rcsubs); /* find nonzero hx_ij, i=0...n-1 */

	//	for(i=0; i<nnz; ++i)
	//	{
	//		if (gl->isStatic[i])
	//		{
	//			xyz = pb + rcsubs[i]*3;
	//		}
	//		else
	//		{
	//			xyz = pb + gl->nFeature_static*3 + (rcsubs[i]-gl->nFeature_static)*3*gl->nBase;
	//		}
	//		double *xij = hx + idxij->val[rcidxs[i]]*mnp; // set pmeas to point to hx_ij

	//		ProjectionThetaonto2D_MOTSTR(j, i, rt, xyz, xij, adata);
	//		//calcImgProjFullR(Kparms, trot, pt, ppt, pmeas); // evaluate Q in pmeas
	//		//calcImgProj(Kparms, pr0, pqr, pt, ppt, pmeas); // evaluate Q in pmeas
	//	}
	//}
}

void PrintSBAInfo(double *info)
{
	cout << "SBA result -------" << endl;
	cout << "Mean squared reprojection error: " << info[0] << " ==> " << info[1] << endl;
	cout << "Total number of iteration: " << info[5] << endl;
	cout << "Reason for terminating: ";
	if (info[6] == 1)
		cout << "Stopped by small ||J^T e||" <<endl;
	else if (info[6] == 2)
		cout << "Stopped by small ||delta||" << endl;
	else if (info[6] == 3)
		cout << "Stopped by maximum iteration" << endl;
	else if (info[6] == 4)
		cout << "Stopped by small relative reduction in ||e||" << endl;
	else if (info[6] == 5)
		cout << "Stopped by small ||e||" << endl;
	else if (info[6] == 6)
		cout << "Stopped due to excessive failed attempts to increase damping for getting a positive definite normal equations" << endl;
	else if (info[6] == 7)
		cout << "Stopped due to infinite values in the coordinates of the set of predicted projections x" << endl;
	cout << "Total number of projection function evaluation: " << info[7] <<endl;
	cout << "Total number of times that normal equations were solved: " << info[9] << endl;
 
}

void PrintSBAInfo(double *info, int nVisiblePoints)
{
	cout << "SBA result -------" << endl;
	cout << "Mean squared reprojection error: " << info[0]/(double)nVisiblePoints << " ==> " << info[1]/(double)nVisiblePoints << endl;
	cout << "Total number of iteration: " << info[5] << endl;
	cout << "Reason for terminating: ";
	if (info[6] == 1)
		cout << "Stopped by small ||J^T e||" <<endl;
	else if (info[6] == 2)
		cout << "Stopped by small ||delta||" << endl;
	else if (info[6] == 3)
		cout << "Stopped by maximum iteration" << endl;
	else if (info[6] == 4)
		cout << "Stopped by small relative reduction in ||e||" << endl;
	else if (info[6] == 5)
		cout << "Stopped by small ||e||" << endl;
	else if (info[6] == 6)
		cout << "Stopped due to excessive failed attempts to increase damping for getting a positive definite normal equations" << endl;
	else if (info[6] == 7)
		cout << "Stopped due to infinite values in the coordinates of the set of predicted projections x" << endl;
	cout << "Total number of projection function evaluation: " << info[7] <<endl;
	cout << "Total number of times that normal equations were solved: " << info[9] << endl;

}

void RetrieveParameterFromSBA(double *dCameraParameter, CvMat *K, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID)
{	
	int nFrames = cP.size();
	cP.clear();
	CvMat *P;
	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[7*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[7*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[7*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[7*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[7*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[7*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[7*iFrame+6]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P_ = cvCreateMat(3,4,CV_32FC1);
		P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		CreateCameraMatrix(R, C, K, P_);
		P = cvCloneMat(P_);

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
		cvReleaseMat(&P_);
	}
	P = cvCreateMat(3,4,CV_32FC1);
	cvReleaseMat(&P);

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[7*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[7*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[7*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
}

void RetrieveParameterFromSBA_mem(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames)
{	
	int nFrames = cP.size();
	for (int i = 0; i < cP.size(); i++)	
		cvReleaseMat(&(cP[i]));
	cP.clear();

	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		int frame = vUsedFrame[iFrame]%max_nFrames;
		int cam = (int) vUsedFrame[iFrame]/max_nFrames;

		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[7*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[7*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[7*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[7*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[7*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[7*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[7*iFrame+6]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		CreateCameraMatrix(R, C, vCamera[cam].vK[frame], P);

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
	}

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[7*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[7*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[7*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
	cvReleaseMat(&X_);
}

void RetrieveParameterFromSBA_mem_Each(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames
	, vector<double> &vOmega, vector<double> &vpx, vector<double> &vpy, vector<CvMat *> &vK)
{	
	int nFrames = cP.size();
	for (int i = 0; i < cP.size(); i++)	
		cvReleaseMat(&(cP[i]));
	cP.clear();

	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		int frame = vUsedFrame[iFrame]%max_nFrames;
		int cam = (int) vUsedFrame[iFrame]/max_nFrames;

		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[14*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[14*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[14*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[14*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[14*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[14*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[14*iFrame+6]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		cvSetReal2D(vK[iFrame], 0, 0, dCameraParameter[14*iFrame+7]);
		cvSetReal2D(vK[iFrame], 1, 1, dCameraParameter[14*iFrame+8]);
		cvSetReal2D(vK[iFrame], 0, 2, dCameraParameter[14*iFrame+9]);
		cvSetReal2D(vK[iFrame], 1, 2, dCameraParameter[14*iFrame+10]);
		CreateCameraMatrix(R, C, vK[iFrame], P);

		cvSetReal2D(vCamera[cam].vK[frame], 0, 0, dCameraParameter[14*iFrame+7]);
		cvSetReal2D(vCamera[cam].vK[frame], 1, 1, dCameraParameter[14*iFrame+8]);
		cvSetReal2D(vCamera[cam].vK[frame], 0, 2, dCameraParameter[14*iFrame+9]);
		cvSetReal2D(vCamera[cam].vK[frame], 1, 2, dCameraParameter[14*iFrame+10]);
		//CreateCameraMatrix(R, C, vCamera[cam].vK[frame], P);
		vOmega[iFrame] = dCameraParameter[14*iFrame+11];
		vpx[iFrame] = dCameraParameter[14*iFrame+12];
		vpy[iFrame] = dCameraParameter[14*iFrame+13];

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
	}

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[14*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[14*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[14*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
	cvReleaseMat(&X_);
}

void RetrieveParameterFromSBA_mem_Each_iPhone(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames
	, vector<double> &vk1, vector<CvMat *> &vK)
{	
	int nFrames = cP.size();
	for (int i = 0; i < cP.size(); i++)	
		cvReleaseMat(&(cP[i]));
	cP.clear();

	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		int frame = vUsedFrame[iFrame]%max_nFrames;
		int cam = (int) vUsedFrame[iFrame]/max_nFrames;

		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[12*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[12*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[12*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[12*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[12*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[12*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[12*iFrame+6]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		cvSetReal2D(vK[iFrame], 0, 0, dCameraParameter[12*iFrame+7]);
		cvSetReal2D(vK[iFrame], 1, 1, dCameraParameter[12*iFrame+8]);
		cvSetReal2D(vK[iFrame], 0, 2, dCameraParameter[12*iFrame+9]);
		cvSetReal2D(vK[iFrame], 1, 2, dCameraParameter[12*iFrame+10]);
		CreateCameraMatrix(R, C, vK[iFrame], P);

		cvSetReal2D(vCamera[cam].vK[frame], 0, 0, dCameraParameter[12*iFrame+7]);
		cvSetReal2D(vCamera[cam].vK[frame], 1, 1, dCameraParameter[12*iFrame+8]);
		cvSetReal2D(vCamera[cam].vK[frame], 0, 2, dCameraParameter[12*iFrame+9]);
		cvSetReal2D(vCamera[cam].vK[frame], 1, 2, dCameraParameter[12*iFrame+10]);
		//CreateCameraMatrix(R, C, vCamera[cam].vK[frame], P);
		vk1[iFrame] = dCameraParameter[12*iFrame+11];

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
	}

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[12*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[12*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[12*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
	cvReleaseMat(&X_);
}

void RetrieveParameterFromSBA_mem_Dome(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames)
{	
	int nFrames = cP.size();
	for (int i = 0; i < cP.size(); i++)	
		cvReleaseMat(&(cP[i]));
	cP.clear();

	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		int frame = vUsedFrame[iFrame]%max_nFrames;
		int cam = (int) vUsedFrame[iFrame]/max_nFrames;

		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[11*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[11*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[11*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[11*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[11*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[11*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[11*iFrame+6]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);

		cvSetReal2D(vCamera[cam].vK[frame], 0, 0, dCameraParameter[11*iFrame+7]);
		cvSetReal2D(vCamera[cam].vK[frame], 1, 1, dCameraParameter[11*iFrame+8]);
		cvSetReal2D(vCamera[cam].vK[frame], 0, 2, dCameraParameter[11*iFrame+9]);
		cvSetReal2D(vCamera[cam].vK[frame], 1, 2, dCameraParameter[11*iFrame+10]);


		CreateCameraMatrix(R, C, vCamera[cam].vK[frame], P);


		//cout << dCameraParameter[11*iFrame+7] << endl;



		//vCamera[cam].vk1[frame] = dCameraParameter[13*iFrame+11];
		//vCamera[cam].vk2[frame] = dCameraParameter[13*iFrame+12];

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
	}

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[11*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[11*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[11*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
	cvReleaseMat(&X_);
}

void RetrieveParameterFromSBA(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames)
{	
	int nFrames = cP.size();
	cP.clear();

	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		int frame = vUsedFrame[iFrame]%max_nFrames;
		int cam = (int) vUsedFrame[iFrame]/max_nFrames;

		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[7*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[7*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[7*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[7*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[7*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[7*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[7*iFrame+6]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		CreateCameraMatrix(R, C, vCamera[cam].K, P);

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
	}

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[7*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[7*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[7*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
	cvReleaseMat(&X_);
}

void RetrieveParameterFromSBA_KRT(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames)
{	
	int nFrames = cP.size();
	cP.clear();
	CvMat *P;
	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		int frame = vUsedFrame[iFrame]%max_nFrames;
		int cam = (int) vUsedFrame[iFrame]/max_nFrames;

		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[11*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[11*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[11*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[11*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[11*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[11*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[11*iFrame+6]);

		vector<int>::const_iterator it = find(vCamera[cam].vTakenFrame.begin(), vCamera[cam].vTakenFrame.end(), frame);
		if (it == vCamera[cam].vTakenFrame.end())
			return;
		int iTakenFrame = (int) (it - vCamera[cam].vTakenFrame.begin());
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 0, 0, dCameraParameter[11*iFrame+7]);
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 1, 1, dCameraParameter[11*iFrame+8]);
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 0, 2, dCameraParameter[11*iFrame+9]);
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 1, 2, dCameraParameter[11*iFrame+10]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P_ = cvCreateMat(3,4,CV_32FC1);
		P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		CreateCameraMatrix(R, C, vCamera[cam].vK[iTakenFrame], P_);
		P = cvCloneMat(P_);

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
		cvReleaseMat(&P_);
	}
	P = cvCreateMat(3,4,CV_32FC1);
	cvReleaseMat(&P);

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[11*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[11*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[11*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
	cvReleaseMat(&X_);
}


void RetrieveParameterFromSBA_TEMPORAL(double *dCameraParameter, vector<Camera> &vCamera, vector<Theta> &vTheta)
{	
	int cFrame = 0;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			vCamera[iCamera].vTakenInstant.push_back(vCamera[iCamera].vTakenFrame[iFrame]+dCameraParameter[cFrame]);
			cout << vCamera[iCamera].vTakenFrame[iFrame]+dCameraParameter[cFrame] << " ";
			cFrame++;
		}
	}

	for (int iTheta = 0; iTheta < vTheta.size(); iTheta++)
	{
		Theta theta;
		theta = vTheta[iTheta];
		theta.thetaX.clear();
		theta.thetaY.clear();
		theta.thetaZ.clear();
		for (int i = 0; i < vTheta[0].thetaX.size(); i++)	
		{
			theta.thetaX.push_back(dCameraParameter[cFrame]);
			cFrame++;
		}
		for (int i = 0; i < vTheta[0].thetaX.size(); i++)	
		{
			theta.thetaY.push_back(dCameraParameter[cFrame]);
			cFrame++;
		}
		for (int i = 0; i < vTheta[0].thetaX.size(); i++)	
		{
			theta.thetaZ.push_back(dCameraParameter[cFrame]);
			cFrame++;
		}

		vTheta[iTheta] = theta;
	}
}

void RetrieveParameterFromSBA_KDRT(double *dCameraParameter, vector<Camera> vCamera, vector<CvMat *> &cP, CvMat *X, vector<int> visibleStructureID, vector<int> vUsedFrame, int max_nFrames)
{	
	int nFrames = cP.size();
	cP.clear();
	CvMat *P;
	for (int iFrame = 0; iFrame < nFrames; iFrame++)
	{
		int frame = vUsedFrame[iFrame]%max_nFrames;
		int cam = (int) vUsedFrame[iFrame]/max_nFrames;

		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[13*iFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[13*iFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[13*iFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[13*iFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[13*iFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[13*iFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[13*iFrame+6]);

		vector<int>::const_iterator it = find(vCamera[cam].vTakenFrame.begin(), vCamera[cam].vTakenFrame.end(), frame);
		if (it == vCamera[cam].vTakenFrame.end())
			return;
		int iTakenFrame = (int) (it - vCamera[cam].vTakenFrame.begin());
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 0, 0, dCameraParameter[13*iFrame+7]);
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 1, 1, dCameraParameter[13*iFrame+8]);
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 0, 2, dCameraParameter[13*iFrame+9]);
		cvSetReal2D(vCamera[cam].vK[iTakenFrame], 1, 2, dCameraParameter[13*iFrame+10]);
		vCamera[cam].vk1[iTakenFrame] = dCameraParameter[13*iFrame+11];
		vCamera[cam].vk2[iTakenFrame] = dCameraParameter[13*iFrame+12];
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P_ = cvCreateMat(3,4,CV_32FC1);
		P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		CreateCameraMatrix(R, C, vCamera[cam].vK[iTakenFrame], P_);
		P = cvCloneMat(P_);

		cP.push_back(P);	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
		cvReleaseMat(&P_);
	}
	P = cvCreateMat(3,4,CV_32FC1);
	cvReleaseMat(&P);

	CvMat *X_ = cvCreateMat(visibleStructureID.size(), 3, CV_32FC1);

	for (int iFeature = 0; iFeature < visibleStructureID.size(); iFeature++)
	{
		cvSetReal2D(X_, iFeature, 0, dCameraParameter[13*cP.size()+3*iFeature]);
		cvSetReal2D(X_, iFeature, 1, dCameraParameter[13*cP.size()+3*iFeature+1]);
		cvSetReal2D(X_, iFeature, 2, dCameraParameter[13*cP.size()+3*iFeature+2]);
	}
	SetIndexedMatRowwise(X, visibleStructureID, X_);
	cvReleaseMat(&X_);
}


void RetrieveParameterFromGBA(double *dCameraParameter, CvMat *K, vector<Camera> &vCamera, vector<Theta> &vTheta, int nFrames)
{	
	CvMat *P;
	for (int cFrame = 0; cFrame < nFrames*vCamera.size(); cFrame++)
	{
		int iFrame = cFrame % nFrames;
		int iCamera = (int) cFrame / nFrames;
		vector<int>::const_iterator it = find(vCamera[iCamera].vTakenFrame.begin(), vCamera[iCamera].vTakenFrame.end(), iFrame);
		if (it == vCamera[iCamera].vTakenFrame.end())
			continue;
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *C = cvCreateMat(3,1,CV_32FC1);
		cvSetReal2D(q, 0, 0, dCameraParameter[7*cFrame]);
		cvSetReal2D(q, 1, 0, dCameraParameter[7*cFrame+1]);
		cvSetReal2D(q, 2, 0, dCameraParameter[7*cFrame+2]);
		cvSetReal2D(q, 3, 0, dCameraParameter[7*cFrame+3]);
		cvSetReal2D(C, 0, 0, dCameraParameter[7*cFrame+4]);
		cvSetReal2D(C, 1, 0, dCameraParameter[7*cFrame+5]);
		cvSetReal2D(C, 2, 0, dCameraParameter[7*cFrame+6]);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *P_ = cvCreateMat(3,4,CV_32FC1);
		P = cvCreateMat(3,4,CV_32FC1);
		Quaternion2Rotation(q, R);
		CreateCameraMatrix(R, C, K, P_);
		P = cvCloneMat(P_);

		int idx = (int) (it - vCamera[iCamera].vTakenFrame.begin());
		vCamera[iCamera].vP[idx] = P;
	
		cvReleaseMat(&q);
		cvReleaseMat(&C);
		cvReleaseMat(&R);
		cvReleaseMat(&P_);
	}
	P = cvCreateMat(3,4,CV_32FC1);
	cvReleaseMat(&P);

	CvMat *A = cvCreateMat(nFrames, nFrames, CV_32FC1);
	GetDCTMappingMatrix(A, nFrames);
	CvMat *A1 = cvCreateMat(1, nFrames, CV_32FC1);
	GetSubMatRowwise(A, 0, 0, A1);
	for (int iTheta = 0; iTheta < vTheta.size(); iTheta++)
	{
		int nBase = vTheta[iTheta].thetaX.size();
		if (vTheta[iTheta].isStatic)
		{
			CvMat *X = cvCreateMat(nFrames, 1, CV_32FC1);
			CvMat *theta1 = cvCreateMat(1,1,CV_32FC1);
			for (int iX = 0; iX < nFrames; iX++)
			{
				cvSetReal2D(X, iX, 0, dCameraParameter[7*nFrames*vCamera.size()+3*(1+nBase)*iTheta]);
			}
			cvMatMul(A1, X, theta1);
			vTheta[iTheta].thetaX[0] = cvGetReal2D(theta1, 0, 0);
			for (int iBase = 1; iBase < nBase; iBase++)
				vTheta[iTheta].thetaX[iBase] = 0;

			for (int iX = 0; iX < nFrames; iX++)
			{
				cvSetReal2D(X, iX, 0, dCameraParameter[7*nFrames*vCamera.size()+3*(1+nBase)*iTheta+1]);
			}
			cvMatMul(A1, X, theta1);
			vTheta[iTheta].thetaY[0] = cvGetReal2D(theta1, 0, 0);
			for (int iBase = 1; iBase < nBase; iBase++)
				vTheta[iTheta].thetaY[iBase] = 0;

			for (int iX = 0; iX < nFrames; iX++)
			{
				cvSetReal2D(X, iX, 0, dCameraParameter[7*nFrames*vCamera.size()+3*(1+nBase)*iTheta+2]);
			}
			cvMatMul(A1, X, theta1);
			vTheta[iTheta].thetaZ[0] = cvGetReal2D(theta1, 0, 0);
			for (int iBase = 1; iBase < nBase; iBase++)
				vTheta[iTheta].thetaZ[iBase] = 0;
		}
		else
		{
			for (int iBase = 0; iBase < nBase; iBase++)
				vTheta[iTheta].thetaX[iBase] = dCameraParameter[7*nFrames*vCamera.size()+3*(1+nBase)*iTheta+3+iBase];
			for (int iBase = 0; iBase < nBase; iBase++)
				vTheta[iTheta].thetaY[iBase] = dCameraParameter[7*nFrames*vCamera.size()+3*(1+nBase)*iTheta+3+nBase+iBase];
			for (int iBase = 0; iBase < nBase; iBase++)
				vTheta[iTheta].thetaZ[iBase] = dCameraParameter[7*nFrames*vCamera.size()+3*(1+nBase)*iTheta+3+2*nBase+iBase];
		}
	}
}

void Projection3Donto2D_MOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	CvMat *K = cvCreateMat(3,3,CV_32FC1);
	cvSetIdentity(K);
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	//double *intrinsic = (((AdditionalData *) adata)->vIntrinsic)[iCamera];
	cvSetReal2D(K, 0, 0, ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[0]);
	cvSetReal2D(K, 1, 1, ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[1]);
	cvSetReal2D(K, 0, 2, ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[2]);
	cvSetReal2D(K, 1, 2, ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[3]);
	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	CvMat *C = cvCreateMat(3,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	cvSetReal2D(q, 0, 0, rt[0]);
	cvSetReal2D(q, 1, 0, rt[1]);
	cvSetReal2D(q, 2, 0, rt[2]);
	cvSetReal2D(q, 3, 0, rt[3]);
	cvSetReal2D(C, 0, 0, rt[4]);
	cvSetReal2D(C, 1, 0, rt[5]);
	cvSetReal2D(C, 2, 0, rt[6]);
	Quaternion2Rotation(q, R);

	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	CvMat *P = cvCreateMat(3,4,CV_32FC1);
	cvMatMul(K, R, temp33);
	cvSetIdentity(P);
	ScalarMul(C, -1, C);
	SetSubMat(P, 0,3,C);
	cvMatMul(temp33, P, P);
	CvMat *X = cvCreateMat(4,1,CV_32FC1);
	cvSetReal2D(X, 0, 0, xyz[0]);
	cvSetReal2D(X, 1, 0, xyz[1]);
	cvSetReal2D(X, 2, 0, xyz[2]);
	cvSetReal2D(X, 3, 0, 1);
	CvMat *x = cvCreateMat(3,1,CV_32FC1);
	cvMatMul(P, X, x);

	if (j == 1)
		int k = 1;

	xij[0] = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	xij[1] = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

	cvReleaseMat(&K);
	cvReleaseMat(&X);
	cvReleaseMat(&x);
	cvReleaseMat(&P);
	cvReleaseMat(&temp33);
	cvReleaseMat(&q);
	cvReleaseMat(&C);
	cvReleaseMat(&R);
}

void Projection3Donto2D_MOTSTR_fast(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	int iFrame = vUsedFrame[j] % max_nFrames;
	if (j == 1)
		int k = 1;

	// Set intrinsic parameter
	double K11 = ((((AdditionalData *) adata)->vIntrinsic)[iFrame])[0];
	double K12 = 0;
	double K13 = ((((AdditionalData *) adata)->vIntrinsic)[iFrame])[2];
	double K21 = 0;
	double K22 = ((((AdditionalData *) adata)->vIntrinsic)[iFrame])[1];
	double K23 = ((((AdditionalData *) adata)->vIntrinsic)[iFrame])[3];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;
	
	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	//cout << R11 << " " << R12 << " " << R13 << endl;
	//cout << R21 << " " << R22 << " " << R23 << endl;
	//cout << R31 << " " << R32 << " " << R33 << endl;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];

	double X1 = xyz[0];
	double X2 = xyz[1];
	double X3 = xyz[2];

	// Building projection 
	double RX1 = R11*X1+R12*X2+R13*X3;
	double RX2 = R21*X1+R22*X2+R23*X3;
	double RX3 = R31*X1+R32*X2+R33*X3;

	double KRX1 = K11*RX1+K12*RX2+K13*RX3;
	double KRX2 = K21*RX1+K22*RX2+K23*RX3;
	double KRX3 = K31*RX1+K32*RX2+K33*RX3;

	double RC1 = R11*C1+R12*C2+R13*C3;
	double RC2 = R21*C1+R22*C2+R23*C3;
	double RC3 = R31*C1+R32*C2+R33*C3;

	double KRC1 = K11*RC1+K12*RC2+K13*RC3;
	double KRC2 = K21*RC1+K22*RC2+K23*RC3;
	double KRC3 = K31*RC1+K32*RC2+K33*RC3;

	double proj1 = KRX1-KRC1;
	double proj2 = KRX2-KRC2;
	double proj3 = KRX3-KRC3;

	xij[0] = proj1/proj3;
	xij[1] = proj2/proj3;
}

void Projection3Donto2D_MOTSTR_fast_Dome(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	int iFrame = vUsedFrame[j] % max_nFrames;
	if (j == 1)
		int k = 1;

	// Set intrinsic parameter
	double K11 = rt[7];
	double K12 = 0;
	double K13 = rt[9];
	double K21 = 0;
	double K22 = rt[8];
	double K23 = rt[10];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	//double k1 = rt[11];
	//double k2 = rt[12];

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;
	
	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	//cout << R11 << " " << R12 << " " << R13 << endl;
	//cout << R21 << " " << R22 << " " << R23 << endl;
	//cout << R31 << " " << R32 << " " << R33 << endl;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];

	double X1 = xyz[0];
	double X2 = xyz[1];
	double X3 = xyz[2];

	// Building projection 
	double RX1 = R11*X1+R12*X2+R13*X3;
	double RX2 = R21*X1+R22*X2+R23*X3;
	double RX3 = R31*X1+R32*X2+R33*X3;

	double KRX1 = K11*RX1+K12*RX2+K13*RX3;
	double KRX2 = K21*RX1+K22*RX2+K23*RX3;
	double KRX3 = K31*RX1+K32*RX2+K33*RX3;

	double RC1 = R11*C1+R12*C2+R13*C3;
	double RC2 = R21*C1+R22*C2+R23*C3;
	double RC3 = R31*C1+R32*C2+R33*C3;

	double KRC1 = K11*RC1+K12*RC2+K13*RC3;
	double KRC2 = K21*RC1+K22*RC2+K23*RC3;
	double KRC3 = K31*RC1+K32*RC2+K33*RC3;

	double proj1 = KRX1-KRC1;
	double proj2 = KRX2-KRC2;
	double proj3 = KRX3-KRC3;

	//double ik11 = 1/K11;	double ik12 = 0;	double ik13 = -K13/K11;
	//double ik21 = 0;		double ik22 = 1/K22;double ik23 = -K23/K22;
	//double ik31 = 0;		double ik32 = 0;	double ik33 = 1;
	
	double x = proj1/proj3;
	double y = proj2/proj3;

	//double xc = K13;
	//double yc = K23;

	//double nzc = (ik31*xc+ik32*yc+ik33);
	//double nxc = (ik11*xc+ik12*yc+ik13)/nzc; 
	//double nyc = (ik21*xc+ik22*yc+ik23)/nzc; 

	//double nz = (ik31*x+ik32*y+ik33);
	//double nx = (ik11*x+ik12*y+ik13)/nz; 
	//double ny = (ik21*x+ik22*y+ik23)/nz; 

	//double r = sqrt((nx-nxc)*(nx-nxc)+(ny-nyc)*(ny-nyc));
	//double L = 1 + k1*r + k2*r*r;
	//nx = nxc + 1/L*(nx - nxc);
	//ny = nyc + 1/L*(ny - nyc);

	//double z = (K31*nx+K32*ny+K33);
	//x = (K11*nx+K12*ny+K13)/z; 
	//y = (K21*nx+K22*ny+K23)/z;

	xij[0] = x;
	xij[1] = y;
}

void Projection3Donto2D_MOTSTR_fast_Distortion(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	double omega = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[4];
	double tan_omega_half_2 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[5];
	if (j == 1)
		int k = 1;
	if (j == 2)
		int k = 1;

	// Set intrinsic parameter
	double K11 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[0];
	double K12 = 0;
	double K13 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[2];
	double K21 = 0;
	double K22 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[1];
	double K23 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[3];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;

	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	//cout << R11 << " " << R12 << " " << R13 << endl;
	//cout << R21 << " " << R22 << " " << R23 << endl;
	//cout << R31 << " " << R32 << " " << R33 << endl;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];

	double X1 = xyz[0];
	double X2 = xyz[1];
	double X3 = xyz[2];

	// Building projection 
	double RX1 = R11*X1+R12*X2+R13*X3;
	double RX2 = R21*X1+R22*X2+R23*X3;
	double RX3 = R31*X1+R32*X2+R33*X3;

	double KRX1 = K11*RX1+K12*RX2+K13*RX3;
	double KRX2 = K21*RX1+K22*RX2+K23*RX3;
	double KRX3 = K31*RX1+K32*RX2+K33*RX3;

	double RC1 = R11*C1+R12*C2+R13*C3;
	double RC2 = R21*C1+R22*C2+R23*C3;
	double RC3 = R31*C1+R32*C2+R33*C3;

	double KRC1 = K11*RC1+K12*RC2+K13*RC3;
	double KRC2 = K21*RC1+K22*RC2+K23*RC3;
	double KRC3 = K31*RC1+K32*RC2+K33*RC3;

	double proj1 = KRX1-KRC1;
	double proj2 = KRX2-KRC2;
	double proj3 = KRX3-KRC3;

	double u = proj1/proj3;
	double v = proj2/proj3;

	double u_n = u/K11 - K13/K11;
	double v_n = v/K22 - K23/K22;

	double r_u = sqrt(u_n*u_n+v_n*v_n);
	double r_d = 1/omega*atan(r_u*tan_omega_half_2);

	double u_d_n = r_d/r_u * u_n;
	double v_d_n = r_d/r_u * v_n;

	double u_d = u_d_n*K11 + K13;
	double v_d = v_d_n*K22 + K23;

	xij[0] = u_d;
	xij[1] = v_d;
}

void Projection3Donto2D_MOTSTR_fast_Distortion_GoPro(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	double omega = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[4];
	double tan_omega_half_2 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[5];
	double princ_x1 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[6];
	double princ_y1 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[7];

	// Set intrinsic parameter
	double K11 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[0];
	double K12 = 0;
	double K13 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[2];
	double K21 = 0;
	double K22 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[1];
	double K23 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[3];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;

	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	//cout << R11 << " " << R12 << " " << R13 << endl;
	//cout << R21 << " " << R22 << " " << R23 << endl;
	//cout << R31 << " " << R32 << " " << R33 << endl;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];

	double X1 = xyz[0];
	double X2 = xyz[1];
	double X3 = xyz[2];

	// Building projection 
	double RX1 = R11*X1+R12*X2+R13*X3;
	double RX2 = R21*X1+R22*X2+R23*X3;
	double RX3 = R31*X1+R32*X2+R33*X3;

	double KRX1 = K11*RX1+K12*RX2+K13*RX3;
	double KRX2 = K21*RX1+K22*RX2+K23*RX3;
	double KRX3 = K31*RX1+K32*RX2+K33*RX3;

	double RC1 = R11*C1+R12*C2+R13*C3;
	double RC2 = R21*C1+R22*C2+R23*C3;
	double RC3 = R31*C1+R32*C2+R33*C3;

	double KRC1 = K11*RC1+K12*RC2+K13*RC3;
	double KRC2 = K21*RC1+K22*RC2+K23*RC3;
	double KRC3 = K31*RC1+K32*RC2+K33*RC3;

	double proj1 = KRX1-KRC1;
	double proj2 = KRX2-KRC2;
	double proj3 = KRX3-KRC3;

	double u = proj1/proj3;
	double v = proj2/proj3;

	//double u_n = u/K11 - K13/K11;
	//double v_n = v/K22 - K23/K22;

	if (princ_x1 == -1)
	{
		double u_n = u/K11 - K13/K11;
		double v_n = v/K22 - K23/K22;

		double r_u = sqrt(u_n*u_n+v_n*v_n);
		double r_d = 1/omega*atan(r_u*tan_omega_half_2);

		double u_d_n = r_d/r_u * u_n;
		double v_d_n = r_d/r_u * v_n;

		double u_d = u_d_n*K11 + K13;
		double v_d = v_d_n*K22 + K23;

		xij[0] = u_d;
		xij[1] = v_d;
	}
	else
	{
		double u_n = u - princ_x1;
		double v_n = v - princ_y1;

		double r_u = sqrt(u_n*u_n+v_n*v_n);
		double r_d = 1/omega*atan(r_u*tan_omega_half_2);

		double u_d_n = r_d/r_u * u_n;
		double v_d_n = r_d/r_u * v_n;

		double u_d = u_d_n + princ_x1;
		double v_d = v_d_n + princ_y1;

		xij[0] = u_d;
		xij[1] = v_d;
	}
}


void Projection3Donto2D_MOTSTR_fast_Distortion_ObstacleDetection(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	double omega = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[4];
	double tan_omega_half_2 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[5];
	double princ_x1 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[6];
	double princ_y1 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[7];

	// Set intrinsic parameter
	double K11 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[0];
	double K12 = 0;
	double K13 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[2];
	double K21 = 0;
	double K22 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[1];
	double K23 = ((((AdditionalData *) adata)->vIntrinsic)[iCamera])[3];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;

	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	//cout << R11 << " " << R12 << " " << R13 << endl;
	//cout << R21 << " " << R22 << " " << R23 << endl;
	//cout << R31 << " " << R32 << " " << R33 << endl;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];

	double X1 = xyz[0];
	double X2 = xyz[1];
	double X3 = xyz[2];

	// Building projection 
	double RX1 = R11*X1+R12*X2+R13*X3;
	double RX2 = R21*X1+R22*X2+R23*X3;
	double RX3 = R31*X1+R32*X2+R33*X3;

	double KRX1 = K11*RX1+K12*RX2+K13*RX3;
	double KRX2 = K21*RX1+K22*RX2+K23*RX3;
	double KRX3 = K31*RX1+K32*RX2+K33*RX3;

	double RC1 = R11*C1+R12*C2+R13*C3;
	double RC2 = R21*C1+R22*C2+R23*C3;
	double RC3 = R31*C1+R32*C2+R33*C3;

	double KRC1 = K11*RC1+K12*RC2+K13*RC3;
	double KRC2 = K21*RC1+K22*RC2+K23*RC3;
	double KRC3 = K31*RC1+K32*RC2+K33*RC3;

	double proj1 = KRX1-KRC1;
	double proj2 = KRX2-KRC2;
	double proj3 = KRX3-KRC3;

	double u = proj1/proj3;
	double v = proj2/proj3;

	//double u_n = u/K11 - K13/K11;
	//double v_n = v/K22 - K23/K22;

	double u_n = u - princ_x1;
	double v_n = v - princ_y1;

	double r_u = sqrt(u_n*u_n+v_n*v_n);
	double r_d = 1/omega*atan(r_u*tan_omega_half_2);

	double u_d_n = r_d/r_u * u_n;
	double v_d_n = r_d/r_u * v_n;

	double u_d = u_d_n + princ_x1;
	double v_d = v_d_n + princ_y1;

	xij[0] = u_d;
	xij[1] = v_d;
}

void Projection3Donto2D_MOTSTR_fast_Distortion_ObstacleDetection1(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	double omega = rt[11];
	double tan_omega_half_2 = 2*tan(omega/2);
	double princ_x1 = rt[12];
	double princ_y1 = rt[13];
	//cout << omega << endl;

	// Set intrinsic parameter
	double K11 = rt[7];
	double K12 = 0;
	double K13 = rt[9];
	double K21 = 0;
	double K22 = rt[8];;
	double K23 = rt[10];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;

	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	//cout << R11 << " " << R12 << " " << R13 << endl;
	//cout << R21 << " " << R22 << " " << R23 << endl;
	//cout << R31 << " " << R32 << " " << R33 << endl;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];

	double X1 = xyz[0];
	double X2 = xyz[1];
	double X3 = xyz[2];

	// Building projection 
	double RX1 = R11*X1+R12*X2+R13*X3;
	double RX2 = R21*X1+R22*X2+R23*X3;
	double RX3 = R31*X1+R32*X2+R33*X3;

	double KRX1 = K11*RX1+K12*RX2+K13*RX3;
	double KRX2 = K21*RX1+K22*RX2+K23*RX3;
	double KRX3 = K31*RX1+K32*RX2+K33*RX3;

	double RC1 = R11*C1+R12*C2+R13*C3;
	double RC2 = R21*C1+R22*C2+R23*C3;
	double RC3 = R31*C1+R32*C2+R33*C3;

	double KRC1 = K11*RC1+K12*RC2+K13*RC3;
	double KRC2 = K21*RC1+K22*RC2+K23*RC3;
	double KRC3 = K31*RC1+K32*RC2+K33*RC3;

	double proj1 = KRX1-KRC1;
	double proj2 = KRX2-KRC2;
	double proj3 = KRX3-KRC3;

	double u = proj1/proj3;
	double v = proj2/proj3;

	//double u_n = u/K11 - K13/K11;
	//double v_n = v/K22 - K23/K22;

	double u_n = u - princ_x1;
	double v_n = v - princ_y1;

	double r_u = sqrt(u_n*u_n+v_n*v_n);
	double r_d = 1/omega*atan(r_u*tan_omega_half_2);

	double u_d_n = r_d/r_u * u_n;
	double v_d_n = r_d/r_u * v_n;

	double u_d = u_d_n + princ_x1;
	double v_d = v_d_n + princ_y1;

	//cout << u << " " << v << " " << u_d << " " << v_d << endl;

	xij[0] = u_d;
	xij[1] = v_d;
}

void Projection3Donto2D_MOTSTR_fast_Distortion_iPhone(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	double k1 = rt[11];

	// Set intrinsic parameter
	double K11 = rt[7];
	double K12 = 0;
	double K13 = rt[9];
	double K21 = 0;
	double K22 = rt[8];
	double K23 = rt[10];
	double K31 = 0;
	double K32 = 0;
	double K33 = 1;

	// Set orientation
	double qw = rt[0];
	double qx = rt[1];
	double qy = rt[2];
	double qz = rt[3];
	double q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
	qw /= q_norm;
	qx /= q_norm;
	qy /= q_norm;
	qz /= q_norm;

	double R11 = 1.0-2*qy*qy-2*qz*qz; 		double R12 = 2*qx*qy-2*qz*qw;		double R13 = 2*qx*qz+2*qy*qw;
	double R21 = 2*qx*qy+2*qz*qw;			double R22 = 1.0-2*qx*qx-2*qz*qz;	double R23 = 2*qz*qy-2*qx*qw;
	double R31 = 2*qx*qz-2*qy*qw;			double R32 = 2*qy*qz+2*qx*qw;		double R33 = 1.0-2*qx*qx-2*qy*qy;

	//cout << R11 << " " << R12 << " " << R13 << endl;
	//cout << R21 << " " << R22 << " " << R23 << endl;
	//cout << R31 << " " << R32 << " " << R33 << endl;

	// Set translation
	double C1 = rt[4];
	double C2 = rt[5];
	double C3 = rt[6];

	double X1 = xyz[0];
	double X2 = xyz[1];
	double X3 = xyz[2];

	// Building projection 
	double RX1 = R11*X1+R12*X2+R13*X3;
	double RX2 = R21*X1+R22*X2+R23*X3;
	double RX3 = R31*X1+R32*X2+R33*X3;

	double KRX1 = K11*RX1+K12*RX2+K13*RX3;
	double KRX2 = K21*RX1+K22*RX2+K23*RX3;
	double KRX3 = K31*RX1+K32*RX2+K33*RX3;

	double RC1 = R11*C1+R12*C2+R13*C3;
	double RC2 = R21*C1+R22*C2+R23*C3;
	double RC3 = R31*C1+R32*C2+R33*C3;

	double KRC1 = K11*RC1+K12*RC2+K13*RC3;
	double KRC2 = K21*RC1+K22*RC2+K23*RC3;
	double KRC3 = K31*RC1+K32*RC2+K33*RC3;

	double proj1 = KRX1-KRC1;
	double proj2 = KRX2-KRC2;
	double proj3 = KRX3-KRC3;

	double u = proj1/proj3;
	double v = proj2/proj3;

	//double u_n = u/K11 - K13/K11;
	//double v_n = v/K22 - K23/K22;

	double u_u = (u-K13)/K11;
	double v_u = (v-K23)/K22;

	//cout << v << " " << v_u << " " << K11 << " " << K22 << " " << K23 << endl;

	double r_u = sqrt(u_u*u_u+v_u*v_u);
	double r_d = r_u*(1+k1*r_u*r_u);
	
	double u_d = r_d/r_u*u_u;
	double v_d = r_d/r_u*v_u;

	u_d = K11*u_d+K13;
	v_d = K22*v_d+K23;

	//cout << u << " " << v << " " << u_d << " " << v_d << " " << k1 << " " << r_u << endl;

	xij[0] = u_d;
	xij[1] = v_d;
}


void Projection3Donto2D_KMOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	CvMat *K = cvCreateMat(3,3,CV_32FC1);
	cvSetIdentity(K);
	//int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	//vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	//int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	//double *intrinsic = (((AdditionalData *) adata)->vIntrinsic)[iCamera];
	//cvSetReal2D(K, 0, 0, intrinsic[0]);
	//cvSetReal2D(K, 1, 1, intrinsic[1]);
	//cvSetReal2D(K, 0, 2, intrinsic[2]);
	//cvSetReal2D(K, 1, 2, intrinsic[3]);

	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	CvMat *C = cvCreateMat(3,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	cvSetReal2D(q, 0, 0, rt[0]);
	cvSetReal2D(q, 1, 0, rt[1]);
	cvSetReal2D(q, 2, 0, rt[2]);
	cvSetReal2D(q, 3, 0, rt[3]);
	cvSetReal2D(C, 0, 0, rt[4]);
	cvSetReal2D(C, 1, 0, rt[5]);
	cvSetReal2D(C, 2, 0, rt[6]);
	cvSetReal2D(K, 0, 0, rt[7]);
	cvSetReal2D(K, 1, 1, rt[8]);
	cvSetReal2D(K, 0, 2, rt[9]);
	cvSetReal2D(K, 1, 2, rt[10]);
	Quaternion2Rotation(q, R);

	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	CvMat *P = cvCreateMat(3,4,CV_32FC1);
	cvMatMul(K, R, temp33);
	cvSetIdentity(P);
	ScalarMul(C, -1, C);
	SetSubMat(P, 0,3,C);
	cvMatMul(temp33, P, P);
	CvMat *X = cvCreateMat(4,1,CV_32FC1);
	cvSetReal2D(X, 0, 0, xyz[0]);
	cvSetReal2D(X, 1, 0, xyz[1]);
	cvSetReal2D(X, 2, 0, xyz[2]);
	cvSetReal2D(X, 3, 0, 1);
	CvMat *x = cvCreateMat(3,1,CV_32FC1);
	cvMatMul(P, X, x);

	xij[0] = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	xij[1] = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

	cvReleaseMat(&K);
	cvReleaseMat(&X);
	cvReleaseMat(&x);
	cvReleaseMat(&P);
	cvReleaseMat(&temp33);
	cvReleaseMat(&q);
	cvReleaseMat(&C);
	cvReleaseMat(&R);
}

void ProjectionThetaonto2D_TEMPORAL(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	int cFrame = vUsedFrame[j] % max_nFrames;
	int nBase = ((AdditionalData*) adata)->nBase;
	vector<Theta> vTheta = ((AdditionalData *) adata)->vTheta;
	CvMat *P = cvCreateMat(3,4, CV_32FC1);
	for (int iP = 0; iP < 3; iP++)
		for (int jP = 0; jP < 4; jP++)
			cvSetReal2D(P, iP, jP, ((AdditionalData *) adata)->vdP[12*j+iP*4+jP]);
	double dt = rt[0];
	double t = cFrame + dt;

	CvMat *B = cvCreateMat(1, nBase, CV_32FC1);
	cvSetReal2D(B, 0, 0, sqrt(1.0/(double)max_nFrames));
	for (int iB = 1; iB < nBase; iB++)
		cvSetReal2D(B, 0, iB, sqrt(2.0/(double)max_nFrames)*cos((2*t-1)*(iB)*PI/2.0/(double)max_nFrames));

	//PrintMat(B);
	CvMat *thetaX = cvCreateMat(nBase,1, CV_32FC1);
	CvMat *thetaY = cvCreateMat(nBase,1, CV_32FC1);
	CvMat *thetaZ = cvCreateMat(nBase,1, CV_32FC1);
	for (int iTheta = 0; iTheta < nBase; iTheta++)
		cvSetReal2D(thetaX, iTheta, 0, xyz[iTheta]);
	for (int iTheta = 0; iTheta < nBase; iTheta++)
		cvSetReal2D(thetaY, iTheta, 0, xyz[nBase+iTheta]);
	for (int iTheta = 0; iTheta < nBase; iTheta++)
		cvSetReal2D(thetaZ, iTheta, 0, xyz[2*nBase+iTheta]);

	//PrintMat(thetaX);
	//PrintMat(thetaY);
	//PrintMat(thetaZ);

	//for (int iTheta = 0; iTheta < nBase; iTheta++)
	//	cout << xyz[iTheta] << " ";
	//cout << endl;

	CvMat *X3 = cvCreateMat(1,1,CV_32FC1);
	CvMat *Y3 = cvCreateMat(1,1,CV_32FC1);
	CvMat *Z3 = cvCreateMat(1,1,CV_32FC1);
	cvMatMul(B, thetaX, X3);
	cvMatMul(B, thetaY, Y3);
	cvMatMul(B, thetaZ, Z3);
	CvMat *X = cvCreateMat(4,1,CV_32FC1);
	cvSetReal2D(X, 0, 0, cvGetReal2D(X3,0,0));
	cvSetReal2D(X, 1, 0, cvGetReal2D(Y3,0,0));
	cvSetReal2D(X, 2, 0, cvGetReal2D(Z3,0,0));
	cvSetReal2D(X, 3, 0, 1);

	CvMat *x = cvCreateMat(3,1,CV_32FC1);
	cvMatMul(P, X, x);

	xij[0] = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	xij[1] = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);
	int error = 0;
	for (int iBase = 0; iBase < nBase; iBase++)
	{
		error += (xyz[iBase]-vTheta[i].thetaX[iBase])*(xyz[iBase]-vTheta[i].thetaX[iBase]);
	}
	for (int iBase = 0; iBase < nBase; iBase++)
	{
		error += (xyz[nBase+iBase]-vTheta[i].thetaY[iBase])*(xyz[nBase+iBase]-vTheta[i].thetaY[iBase]);
	}
	for (int iBase = 0; iBase < nBase; iBase++)
	{
		error += (xyz[2*nBase+iBase]-vTheta[i].thetaZ[iBase])*(xyz[2*nBase+iBase]-vTheta[i].thetaZ[iBase]);
	}

	xij[2] = 10*error;
	xij[3] = 10*rt[0];

	cvReleaseMat(&X);
	cvReleaseMat(&x);
	cvReleaseMat(&P);
	cvReleaseMat(&B);
	cvReleaseMat(&thetaX);
	cvReleaseMat(&thetaY);
	cvReleaseMat(&thetaZ);
	cvReleaseMat(&X3);
	cvReleaseMat(&Y3);
	cvReleaseMat(&Z3);
}


void Projection3Donto2D_KDMOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	CvMat *K = cvCreateMat(3,3,CV_32FC1);
	cvSetIdentity(K);
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);
	//double *intrinsic = (((AdditionalData *) adata)->vIntrinsic)[iCamera];
	//cvSetReal2D(K, 0, 0, intrinsic[0]);
	//cvSetReal2D(K, 1, 1, intrinsic[1]);
	//cvSetReal2D(K, 0, 2, intrinsic[2]);
	//cvSetReal2D(K, 1, 2, intrinsic[3]);

	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	CvMat *C = cvCreateMat(3,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	cvSetReal2D(q, 0, 0, rt[0]);
	cvSetReal2D(q, 1, 0, rt[1]);
	cvSetReal2D(q, 2, 0, rt[2]);
	cvSetReal2D(q, 3, 0, rt[3]);
	cvSetReal2D(C, 0, 0, rt[4]);
	cvSetReal2D(C, 1, 0, rt[5]);
	cvSetReal2D(C, 2, 0, rt[6]);
	cvSetReal2D(K, 0, 0, rt[7]);
	cvSetReal2D(K, 1, 1, rt[8]);
	cvSetReal2D(K, 0, 2, rt[9]);
	cvSetReal2D(K, 1, 2, rt[10]);
	Quaternion2Rotation(q, R);

	double k11 = rt[7];
	double k22 = rt[8];
	double px = rt[9];
	double py = rt[10];
	double k1 = rt[11];
	double k2 = rt[12];

	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	CvMat *P = cvCreateMat(3,4,CV_32FC1);
	cvMatMul(K, R, temp33);
	cvSetIdentity(P);
	ScalarMul(C, -1, C);
	SetSubMat(P, 0,3,C);
	cvMatMul(temp33, P, P);
	CvMat *X = cvCreateMat(4,1,CV_32FC1);
	cvSetReal2D(X, 0, 0, xyz[0]);
	cvSetReal2D(X, 1, 0, xyz[1]);
	cvSetReal2D(X, 2, 0, xyz[2]);
	cvSetReal2D(X, 3, 0, 1);
	CvMat *x = cvCreateMat(3,1,CV_32FC1);
	cvMatMul(P, X, x);

	xij[0] = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	xij[1] = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

	// taking into account distortion
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K, invK);
	double ik11 = cvGetReal2D(invK, 0, 0);	double ik12 = cvGetReal2D(invK, 0, 1);	double ik13 = cvGetReal2D(invK, 0, 2);
	double ik21 = cvGetReal2D(invK, 1, 0);	double ik22 = cvGetReal2D(invK, 1, 1);	double ik23 = cvGetReal2D(invK, 1, 2);
	double ik31 = cvGetReal2D(invK, 2, 0);	double ik32 = cvGetReal2D(invK, 2, 1);	double ik33 = cvGetReal2D(invK, 2, 2);
	double nz = (ik31*xij[0]+ik32*xij[1]+ik33);
	double nx = (ik11*xij[0]+ik12*xij[1]+ik13)/nz; 
	double ny = (ik21*xij[0]+ik22*xij[1]+ik23)/nz; 

	double r = sqrt((nx)*(nx)+(ny)*(ny));
	double L = 1 + k1*r + k2*r*r;
	nx = L*(nx);
	ny = L*(ny);

	xij[0] = (k11*nx+px); 
	xij[1] = (k22*ny+py); 

	cvReleaseMat(&K);
	cvReleaseMat(&invK);
	cvReleaseMat(&X);
	cvReleaseMat(&x);
	cvReleaseMat(&P);
	cvReleaseMat(&temp33);
	cvReleaseMat(&q);
	cvReleaseMat(&C);
	cvReleaseMat(&R);
}


void ProjectionThetaonto2D_MOTSTR(int j, int i, double *rt, double *xyz, double *xij, void *adata)
{
	//CvMat *K = cvCreateMat(3,3,CV_32FC1);
	//cvSetIdentity(K);
	//cvSetReal2D(K, 0, 0, ((AdditionalData *) adata)->intrinsic[0]);
	//cvSetReal2D(K, 1, 1, ((AdditionalData *) adata)->intrinsic[1]);
	//cvSetReal2D(K, 0, 2, ((AdditionalData *) adata)->intrinsic[2]);
	//cvSetReal2D(K, 1, 2, ((AdditionalData *) adata)->intrinsic[3]);
	//bool isStatic = ((AdditionalData *) adata)->isStatic[i];
	//CvMat *q = cvCreateMat(4,1,CV_32FC1);
	//CvMat *C = cvCreateMat(3,1,CV_32FC1);
	//CvMat *R = cvCreateMat(3,3,CV_32FC1);
	//cvSetReal2D(q, 0, 0, rt[0]);
	//cvSetReal2D(q, 1, 0, rt[1]);
	//cvSetReal2D(q, 2, 0, rt[2]);
	//cvSetReal2D(q, 3, 0, rt[3]);
	//cvSetReal2D(C, 0, 0, rt[4]);
	//cvSetReal2D(C, 1, 0, rt[5]);
	//cvSetReal2D(C, 2, 0, rt[6]);
	//Quaternion2Rotation(q, R);

	//CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *P = cvCreateMat(3,4,CV_32FC1);
	//cvMatMul(K, R, temp33);
	//cvSetIdentity(P);
	//ScalarMul(C, -1, C);
	//SetSubMat(P, 0,3,C);
	//cvMatMul(temp33, P, P);

	//if (!isStatic)
	//{
	//	int iFrame = j % ((AdditionalData *) adata)->nFrames;
	//	int nBase = ((AdditionalData *) adata)->nBase;
	//	int nFrames = ((AdditionalData *) adata)->nFrames;
	//	CvMat *x = cvCreateMat(2,1,CV_32FC1);
	//	CvMat *theta = cvCreateMat(3*nBase, 1, CV_32FC1);
	//	for (int iTheta = 0; iTheta < 3*nBase; iTheta++)
	//		cvSetReal2D(theta, iTheta, 0, xyz[iTheta]);
	//	DCTProjection(P, theta, nFrames, iFrame, nBase, x);
	//	xij[0] = cvGetReal2D(x, 0, 0);
	//	xij[1] = cvGetReal2D(x, 1, 0);
	//	cvReleaseMat(&x);
	//	cvReleaseMat(&theta);
	//}
	//else
	//{
	//	CvMat *X = cvCreateMat(4,1,CV_32FC1);
	//	cvSetReal2D(X, 0, 0, xyz[0]);
	//	cvSetReal2D(X, 1, 0, xyz[1]);
	//	cvSetReal2D(X, 2, 0, xyz[2]);
	//	cvSetReal2D(X, 3, 0, 1);
	//	CvMat *x = cvCreateMat(3,1,CV_32FC1);
	//	cvMatMul(P, X, x);
	//	

	//	xij[0] = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	//	xij[1] = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);
	//	cvReleaseMat(&X);
	//	cvReleaseMat(&x);
	//}

	//cvReleaseMat(&K);
	//cvReleaseMat(&P);
	//cvReleaseMat(&temp33);
	//cvReleaseMat(&q);
	//cvReleaseMat(&C);
	//cvReleaseMat(&R);
}

void ProjectionThetaonto2D_MOTSTR_LEVMAR(int j, int i, double *rt, double *xyz, double &xi, double &yi, void *adata)
{
	//CvMat *K = cvCreateMat(3,3,CV_32FC1);
	//cvSetIdentity(K);
	//cvSetReal2D(K, 0, 0, ((AdditionalData *) adata)->intrinsic[0]);
	//cvSetReal2D(K, 1, 1, ((AdditionalData *) adata)->intrinsic[1]);
	//cvSetReal2D(K, 0, 2, ((AdditionalData *) adata)->intrinsic[2]);
	//cvSetReal2D(K, 1, 2, ((AdditionalData *) adata)->intrinsic[3]);
	//bool isStatic = ((AdditionalData *) adata)->isStatic[i];
	//CvMat *q = cvCreateMat(4,1,CV_32FC1);
	//CvMat *C = cvCreateMat(3,1,CV_32FC1);
	//CvMat *R = cvCreateMat(3,3,CV_32FC1);
	//cvSetReal2D(q, 0, 0, rt[0]);
	//cvSetReal2D(q, 1, 0, rt[1]);
	//cvSetReal2D(q, 2, 0, rt[2]);
	//cvSetReal2D(q, 3, 0, rt[3]);
	//cvSetReal2D(C, 0, 0, rt[4]);
	//cvSetReal2D(C, 1, 0, rt[5]);
	//cvSetReal2D(C, 2, 0, rt[6]);
	//Quaternion2Rotation(q, R);

	//CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *P = cvCreateMat(3,4,CV_32FC1);
	//cvMatMul(K, R, temp33);
	//cvSetIdentity(P);
	//ScalarMul(C, -1, C);
	//SetSubMat(P, 0,3,C);
	//cvMatMul(temp33, P, P);

	//if (!isStatic)
	//{
	//	int iFrame = j % ((AdditionalData *) adata)->nFrames;
	//	int nBase = ((AdditionalData *) adata)->nBase;
	//	int nFrames = ((AdditionalData *) adata)->nFrames;
	//	CvMat *x = cvCreateMat(2,1,CV_32FC1);
	//	CvMat *theta = cvCreateMat(3*nBase, 1, CV_32FC1);
	//	for (int iTheta = 0; iTheta < 3*nBase; iTheta++)
	//		cvSetReal2D(theta, iTheta, 0, xyz[iTheta]);
	//	DCTProjection(P, theta, nFrames, iFrame, nBase, x);
	//	xi = cvGetReal2D(x, 0, 0);
	//	yi = cvGetReal2D(x, 1, 0);
	//	cvReleaseMat(&x);
	//	cvReleaseMat(&theta);
	//}
	//else
	//{
	//	CvMat *X = cvCreateMat(4,1,CV_32FC1);
	//	cvSetReal2D(X, 0, 0, xyz[0]);
	//	cvSetReal2D(X, 1, 0, xyz[1]);
	//	cvSetReal2D(X, 2, 0, xyz[2]);
	//	cvSetReal2D(X, 3, 0, 1);
	//	CvMat *x = cvCreateMat(3,1,CV_32FC1);
	//	cvMatMul(P, X, x);

	//	xi = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	//	yi = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);
	//	cvReleaseMat(&X);
	//	cvReleaseMat(&x);
	//}

	//cvReleaseMat(&K);
	//cvReleaseMat(&P);
	//cvReleaseMat(&temp33);
	//cvReleaseMat(&q);
	//cvReleaseMat(&C);
	//cvReleaseMat(&R);
}


void Projection3Donto2D_MOT(int j, int i, double *rt, double *xij, void *adata)
{
	//CvMat *K = cvCreateMat(3,3,CV_32FC1);
	//cvSetIdentity(K);
	//cvSetReal2D(K, 0, 0, ((AdditionalData *) adata)->intrinsic[0]);
	//cvSetReal2D(K, 1, 1, ((AdditionalData *) adata)->intrinsic[1]);
	//cvSetReal2D(K, 0, 2, ((AdditionalData *) adata)->intrinsic[2]);
	//cvSetReal2D(K, 1, 2, ((AdditionalData *) adata)->intrinsic[3]);
	//CvMat *q = cvCreateMat(4,1,CV_32FC1);
	//CvMat *C = cvCreateMat(3,1,CV_32FC1);
	//CvMat *R = cvCreateMat(3,3,CV_32FC1);
	//cvSetReal2D(q, 0, 0, rt[0]);
	//cvSetReal2D(q, 1, 0, rt[1]);
	//cvSetReal2D(q, 2, 0, rt[2]);
	//cvSetReal2D(q, 3, 0, rt[3]);
	//cvSetReal2D(C, 0, 0, rt[4]);
	//cvSetReal2D(C, 1, 0, rt[5]);
	//cvSetReal2D(C, 2, 0, rt[6]);
	//Quaternion2Rotation(q, R);

	//CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	//CvMat *P = cvCreateMat(3,4,CV_32FC1);
	//cvMatMul(K, R, temp33);
	//cvSetIdentity(P);
	//ScalarMul(C, -1, C);
	//SetSubMat(P, 0,3,C);
	//cvMatMul(temp33, P, P);
	//CvMat *X = cvCreateMat(4,1,CV_32FC1);
	//cvSetReal2D(X, 0, 0, ((AdditionalData *) adata)->XYZ[3*i]);
	//cvSetReal2D(X, 1, 0, ((AdditionalData *) adata)->XYZ[3*i+1]);
	//cvSetReal2D(X, 2, 0, ((AdditionalData *) adata)->XYZ[3*i+2]);
	//cvSetReal2D(X, 3, 0, 1);
	////PrintMat(X, "X");
	////PrintMat(C, "C");
	////cout << i << endl;
	////if (j == 1)
	////	int k = 1;

	//CvMat *x = cvCreateMat(3,1,CV_32FC1);
	//cvMatMul(P, X, x);
	//xij[0] = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	//xij[1] = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

	//cvReleaseMat(&K);
	//cvReleaseMat(&X);
	//cvReleaseMat(&x);
	//cvReleaseMat(&P);
	//cvReleaseMat(&temp33);
	//cvReleaseMat(&q);
	//cvReleaseMat(&C);
}

void Projection3Donto2D_KDMOT(int j, int i, double *rt, double *xij, void *adata)
{
	CvMat *K = cvCreateMat(3,3,CV_32FC1);
	cvSetIdentity(K);
	int max_nFrames = ((AdditionalData *) adata)->max_nFrames;
	vector<int> vUsedFrame = ((AdditionalData *) adata)->vUsedFrame;
	int iCamera = (int)((double)vUsedFrame[j]/max_nFrames);

	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	CvMat *C = cvCreateMat(3,1,CV_32FC1);
	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	cvSetReal2D(q, 0, 0, rt[0]);
	cvSetReal2D(q, 1, 0, rt[1]);
	cvSetReal2D(q, 2, 0, rt[2]);
	cvSetReal2D(q, 3, 0, rt[3]);
	cvSetReal2D(C, 0, 0, rt[4]);
	cvSetReal2D(C, 1, 0, rt[5]);
	cvSetReal2D(C, 2, 0, rt[6]);
	cvSetReal2D(K, 0, 0, rt[7]);
	cvSetReal2D(K, 1, 1, rt[8]);
	cvSetReal2D(K, 0, 2, rt[9]);
	cvSetReal2D(K, 1, 2, rt[10]);
	Quaternion2Rotation(q, R);

	double k11 = rt[7];
	double k22 = rt[8];
	double px = rt[9];
	double py = rt[10];
	double k1 = rt[11];
	double k2 = rt[12];

	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	CvMat *P = cvCreateMat(3,4,CV_32FC1);
	cvMatMul(K, R, temp33);
	cvSetIdentity(P);
	ScalarMul(C, -1, C);
	SetSubMat(P, 0,3,C);
	cvMatMul(temp33, P, P);
	CvMat *X = cvCreateMat(4,1,CV_32FC1);
	cvSetReal2D(X, 0, 0, ((AdditionalData *) adata)->XYZ[3*i]);
	cvSetReal2D(X, 1, 0, ((AdditionalData *) adata)->XYZ[3*i+1]);
	cvSetReal2D(X, 2, 0, ((AdditionalData *) adata)->XYZ[3*i+2]);
	cvSetReal2D(X, 3, 0, 1);
	CvMat *x = cvCreateMat(3,1,CV_32FC1);
	cvMatMul(P, X, x);

	xij[0] = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
	xij[1] = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

	// taking into account distortion
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K, invK);
	double ik11 = cvGetReal2D(invK, 0, 0);	double ik12 = cvGetReal2D(invK, 0, 1);	double ik13 = cvGetReal2D(invK, 0, 2);
	double ik21 = cvGetReal2D(invK, 1, 0);	double ik22 = cvGetReal2D(invK, 1, 1);	double ik23 = cvGetReal2D(invK, 1, 2);
	double ik31 = cvGetReal2D(invK, 2, 0);	double ik32 = cvGetReal2D(invK, 2, 1);	double ik33 = cvGetReal2D(invK, 2, 2);
	double nz = (ik31*xij[0]+ik32*xij[1]+ik33);
	double nx = (ik11*xij[0]+ik12*xij[1]+ik13)/nz; 
	double ny = (ik21*xij[0]+ik22*xij[1]+ik23)/nz; 

	double r = sqrt((nx)*(nx)+(ny)*(ny));
	double L = 1 + k1*r + k2*r*r;
	nx = L*(nx);
	ny = L*(ny);

	xij[0] = (k11*nx+px); 
	xij[1] = (k22*ny+py); 

	cvReleaseMat(&K);
	cvReleaseMat(&invK);
	cvReleaseMat(&X);
	cvReleaseMat(&x);
	cvReleaseMat(&P);
	cvReleaseMat(&temp33);
	cvReleaseMat(&q);
	cvReleaseMat(&C);
	cvReleaseMat(&R);
}

void GetParameterForSBA(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, CvMat *K, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFrames = cP.size(); 
	int nFeatures = X->rows;
	
	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		cvInvert(K, invK);
		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}

	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}

	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);

	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}
	

	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFrames = cP.size(); 
	int nFeatures = X->rows;

	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		cvInvert(vCamera[iCamera].vK[cFrame], invK);
		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}

	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}

	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);

	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				//cvSetReal2D(visibilityMask, iFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}

	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_Dome(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFrames = cP.size(); 
	int nFeatures = X->rows;

	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		cvInvert(vCamera[iCamera].vK[cFrame], invK);
		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cameraParameter.push_back(cvGetReal2D(vCamera[iCamera].vK[cFrame], 0, 0));
		cameraParameter.push_back(cvGetReal2D(vCamera[iCamera].vK[cFrame], 1, 1));
		cameraParameter.push_back(cvGetReal2D(vCamera[iCamera].vK[cFrame], 0, 2));
		cameraParameter.push_back(cvGetReal2D(vCamera[iCamera].vK[cFrame], 1, 2));
		//cameraParameter.push_back(vCamera[iCamera].vk1[cFrame]);
		//cameraParameter.push_back(vCamera[iCamera].vk2[cFrame]);

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}

	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}

	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);

	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vVisibleFrame.begin(),vFeature[iFeature].vVisibleFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vVisibleFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vVisibleFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vVisible_dis_x[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vVisible_dis_y[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				//cvSetReal2D(visibilityMask, iFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}

	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_Distortion(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFrames = cP.size(); 
	int nFeatures = X->rows;
	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		cvInvert(vCamera[iCamera].K, invK);
		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}
	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);
	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx_dis[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy_dis[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				//cvSetReal2D(visibilityMask, iFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}
	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}
void GetParameterForSBA_Distortion_Each(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask, vector<CvMat *> vK, 
						vector<double> vOmega, vector<double> vpx, vector<double> vpy)
{
	int nFrames = cP.size(); 
	int nFeatures = X->rows;
	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		cvInvert(vK[iFrame], invK);
		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 0, 0));
		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 1, 1));
		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 0, 2));
		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 1, 2));

		cameraParameter.push_back(vOmega[iFrame]);
		cameraParameter.push_back(vpx[iFrame]);
		cameraParameter.push_back(vpy[iFrame]);

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}
	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);
	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx_dis[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy_dis[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				//cvSetReal2D(visibilityMask, iFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}
	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_Distortion_Each_iPhone(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask, vector<CvMat *> vK, 
						vector<double> vk1)
{
	int nFrames = cP.size(); 
	int nFeatures = X->rows;
	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		cvInvert(vK[iFrame], invK);
		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 0, 0));
		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 1, 1));
		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 0, 2));
		cameraParameter.push_back(cvGetReal2D(vK[iFrame], 1, 2));

		cameraParameter.push_back(vk1[iFrame]);

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}
	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);
	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx_dis[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy_dis[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				//cvSetReal2D(visibilityMask, iFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}
	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_Distortion(vector<Feature> &vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask,
						vector<CvMat *> vK)
{
	int nFrames = cP.size(); 
	int nFeatures = X->rows;
	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		cvInvert(vK[iFrame], invK);
		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}
	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);
	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx_dis[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy_dis[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				//cvSetReal2D(visibilityMask, iFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}
	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_KRT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFrames = cP.size(); 
	int nFeatures = visibleStructureID.size();
	cout << "nFeatures: " << X->rows << " nFeatures_: " << nFeatures <<endl;

	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		int cFrame = vUsedFrame[iFrame];
		int takenFrame = vUsedFrame[iFrame] % max_nFrames;
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		vector<int>::const_iterator it = find(vCamera[iCamera].vTakenFrame.begin(), vCamera[iCamera].vTakenFrame.end(), takenFrame);
		if (it == vCamera[iCamera].vTakenFrame.end())
			return;
		int iTakenFrame = (int) (it - vCamera[iCamera].vTakenFrame.begin());
		cvInvert(vCamera[iCamera].vK[iTakenFrame], invK);
		double k11 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 0, 0);
		double k22 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 1, 1);
		double k13 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 0, 2);
		double k23 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 1, 2);

		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		Rotation2Quaternion(R, q);
		//PrintMat(R);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cameraParameter.push_back(k11);
		cameraParameter.push_back(k22);
		cameraParameter.push_back(k13);
		cameraParameter.push_back(k23);

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}

	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}

	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);

	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}


	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_KDRT(vector<Feature> vFeature, vector<int> vUsedFrame, vector<CvMat *> cP, CvMat *X, vector<Camera> vCamera, int max_nFrames, vector<int> visibleStructureID,
							vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFrames = cP.size(); 
	int nFeatures = visibleStructureID.size();

	for (int iFrame = 0; iFrame < vUsedFrame.size(); iFrame++)
	{
		PrintMat(cP[iFrame],"P");
		int cFrame = vUsedFrame[iFrame];
		int takenFrame = vUsedFrame[iFrame] % max_nFrames;
		int iCamera = (int) ((double)vUsedFrame[iFrame]/max_nFrames);
		CvMat *q = cvCreateMat(4,1,CV_32FC1);
		CvMat *R = cvCreateMat(3,3,CV_32FC1);
		CvMat *t = cvCreateMat(3,1,CV_32FC1);
		CvMat *invK = cvCreateMat(3,3,CV_32FC1);
		CvMat *invR = cvCreateMat(3,3,CV_32FC1);
		vector<int>::const_iterator it = find(vCamera[iCamera].vTakenFrame.begin(), vCamera[iCamera].vTakenFrame.end(), takenFrame);
		if (it == vCamera[iCamera].vTakenFrame.end())
			return;
		int iTakenFrame = (int) (it - vCamera[iCamera].vTakenFrame.begin());
		cvInvert(vCamera[iCamera].vK[iTakenFrame], invK);
		double k11 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 0, 0);
		double k22 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 1, 1);
		double k13 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 0, 2);
		double k23 = cvGetReal2D(vCamera[iCamera].vK[iTakenFrame], 1, 2);

		GetSubMatColwise(cP[iFrame], 0, 2, R);
		GetSubMatColwise(cP[iFrame], 3, 3, t);
		cvMatMul(invK, R, R);
		cvInvert(R, invR);
		cvMatMul(invK, t, t);
		cvMatMul(invR, t, t);
		ScalarMul(t, -1, t);
		Rotation2Quaternion(R, q);
		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
		cameraParameter.push_back(cvGetReal2D(q, 3, 0));

		cameraParameter.push_back(cvGetReal2D(t, 0, 0));
		cameraParameter.push_back(cvGetReal2D(t, 1, 0));
		cameraParameter.push_back(cvGetReal2D(t, 2, 0));

		cameraParameter.push_back(k11);
		cameraParameter.push_back(k22);
		cameraParameter.push_back(k13);
		cameraParameter.push_back(k23);

		cameraParameter.push_back(vCamera[iCamera].vk1[iTakenFrame]);
		cameraParameter.push_back(vCamera[iCamera].vk2[iTakenFrame]);

		cvReleaseMat(&R);
		cvReleaseMat(&t);
		cvReleaseMat(&q);
		cvReleaseMat(&invK);
		cvReleaseMat(&invR);
	}

	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 0));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 1));
		cameraParameter.push_back(cvGetReal2D(X, iFeature, 2));
	}

	CvMat *visibilityMask = cvCreateMat(visibleStructureID.size(), nFrames, CV_32FC1);

	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int cFeature = 0; cFeature < visibleStructureID.size(); cFeature++)
	{
		int iFeature = visibleStructureID[cFeature];
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy[idx]);
				cvSetReal2D(visibilityMask, cFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}


	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_TEMPORAL(vector<Feature> vFeature, vector<Theta> vTheta, vector<Camera> vCamera, int max_nFrames,
							 vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFrames = 0;
	vector<int> vUsedFrame;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			// Temporal error initialization
			cameraParameter.push_back(0);
			vUsedFrame.push_back(iCamera*max_nFrames+vCamera[iCamera].vTakenFrame[iFrame]);
			nFrames++;
		}
	}

	for (int iTheta = 0; iTheta < vTheta.size(); iTheta++)
	{
		for (int iTheta_x = 0; iTheta_x < vTheta[iTheta].thetaX.size(); iTheta_x++)
			cameraParameter.push_back(vTheta[iTheta].thetaX[iTheta_x]);
		for (int iTheta_x = 0; iTheta_x < vTheta[iTheta].thetaX.size(); iTheta_x++)
			cameraParameter.push_back(vTheta[iTheta].thetaY[iTheta_x]);
		for (int iTheta_x = 0; iTheta_x < vTheta[iTheta].thetaX.size(); iTheta_x++)
			cameraParameter.push_back(vTheta[iTheta].thetaZ[iTheta_x]);
	}

	CvMat *visibilityMask = cvCreateMat(vTheta.size(), nFrames, CV_32FC1);
	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);

			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy[idx]);
				feature2DParameter.push_back(0.0);
				feature2DParameter.push_back(0.0);
				cvSetReal2D(visibilityMask, iFeature, iVisibleFrame, 1);
				NZ++;
			}
		}
	}

	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForSBA_TEMPORAL_LEVMAR(vector<Feature> vFeature, vector<Theta> vTheta, vector<Camera> vCamera, int max_nFrames,
								 vector<double> &cameraParameter, vector<double> &feature2DParameter)
{
	int nFrames = 0;
	vector<int> vUsedFrame;
	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < vCamera[iCamera].vTakenFrame.size(); iFrame++)
		{
			// Temporal error initialization
			cameraParameter.push_back(0);
			vUsedFrame.push_back(iCamera*max_nFrames+vCamera[iCamera].vTakenFrame[iFrame]);
			nFrames++;
		}
	}

	for (int iTheta = 0; iTheta < vTheta.size(); iTheta++)
	{
		for (int iTheta_x = 0; iTheta_x < vTheta[iTheta].thetaX.size(); iTheta_x++)
			cameraParameter.push_back(vTheta[iTheta].thetaX[iTheta_x]);
		for (int iTheta_x = 0; iTheta_x < vTheta[iTheta].thetaX.size(); iTheta_x++)
			cameraParameter.push_back(vTheta[iTheta].thetaY[iTheta_x]);
		for (int iTheta_x = 0; iTheta_x < vTheta[iTheta].thetaX.size(); iTheta_x++)
			cameraParameter.push_back(vTheta[iTheta].thetaZ[iTheta_x]);
	}

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		for (int iVisibleFrame = 0; iVisibleFrame < vUsedFrame.size(); iVisibleFrame++)
		{
			vector<int>::iterator it = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(),vUsedFrame[iVisibleFrame]);
			if (it != vFeature[iFeature].vFrame.end())
			{
				int idx = int(it-vFeature[iFeature].vFrame.begin());
				feature2DParameter.push_back(vFeature[iFeature].vx[idx]);
				feature2DParameter.push_back(vFeature[iFeature].vy[idx]);
			}
			else
			{
				feature2DParameter.push_back(0);
				feature2DParameter.push_back(0);
			}
		}
	}
}





void GetParameterForGBA(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, vector<char> &vMask)
{
	int nFeatures = vFeature.size();

	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < nFrames; iFrame++)
		{
			vector<int>::const_iterator it = find(vCamera[iCamera].vTakenFrame.begin(), vCamera[iCamera].vTakenFrame.end(), iFrame);
			if (it == vCamera[iCamera].vTakenFrame.end())
			{
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);

				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
			}
			else
			{
				CvMat *q = cvCreateMat(4,1,CV_32FC1);
				CvMat *R = cvCreateMat(3,3,CV_32FC1);
				CvMat *t = cvCreateMat(3,1,CV_32FC1);
				CvMat *invK = cvCreateMat(3,3,CV_32FC1);
				CvMat *invR = cvCreateMat(3,3,CV_32FC1);
				cvInvert(K, invK);
				GetSubMatColwise(vCamera[iCamera].vP[iFrame], 0, 2, R);
				GetSubMatColwise(vCamera[iCamera].vP[iFrame], 3, 3, t);
				cvMatMul(invK, R, R);
				cvInvert(R, invR);
				cvMatMul(invK, t, t);
				cvMatMul(invR, t, t);
				ScalarMul(t, -1, t);
				Rotation2Quaternion(R, q);
				cameraParameter.push_back(cvGetReal2D(q, 0, 0));
				cameraParameter.push_back(cvGetReal2D(q, 1, 0));
				cameraParameter.push_back(cvGetReal2D(q, 2, 0));
				cameraParameter.push_back(cvGetReal2D(q, 3, 0));

				cameraParameter.push_back(cvGetReal2D(t, 0, 0));
				cameraParameter.push_back(cvGetReal2D(t, 1, 0));
				cameraParameter.push_back(cvGetReal2D(t, 2, 0));

				cvReleaseMat(&R);
				cvReleaseMat(&t);
				cvReleaseMat(&q);
				cvReleaseMat(&invK);
				cvReleaseMat(&invR);
			}
		}
	}

	for (int iTheta = 0; iTheta < vTheta.size(); iTheta++)
	{
		if (vTheta[iTheta].isStatic)
		{
			double IDCT = sqrt(1.0/(double)nFrames);
			double X = vTheta[iTheta].thetaX[0]*IDCT;
			double Y = vTheta[iTheta].thetaY[0]*IDCT;
			double Z = vTheta[iTheta].thetaZ[0]*IDCT;
			cameraParameter.push_back(X);
			cameraParameter.push_back(Y);
			cameraParameter.push_back(Z);
		}
		else
		{
			for (int iBase = 0; iBase < vTheta[iTheta].thetaX.size(); iBase++)
			{
				cameraParameter.push_back(vTheta[iTheta].thetaX[iBase]);
			}
			for (int iBase = 0; iBase < vTheta[iTheta].thetaX.size(); iBase++)
			{
				cameraParameter.push_back(vTheta[iTheta].thetaY[iBase]);
			}
			for (int iBase = 0; iBase < vTheta[iTheta].thetaX.size(); iBase++)
			{
				cameraParameter.push_back(vTheta[iTheta].thetaZ[iBase]);
			}
		}


	}

	CvMat *visibilityMask = cvCreateMat(vTheta.size(), vCamera.size()*nFrames, CV_32FC1);

	cvSetZero(visibilityMask);
	int NZ = 0;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		for (int iFrame = 0; iFrame < vFeature[iFeature].vFrame.size(); iFrame++)
		{
			feature2DParameter.push_back(vFeature[iFeature].vx[iFrame]);
			feature2DParameter.push_back(vFeature[iFeature].vy[iFrame]);
			cvSetReal2D(visibilityMask, iFeature, iFrame, 1);
		}
	}

	for (int iFeature = 0; iFeature < visibilityMask->rows; iFeature++)
	{
		for (int iFrame = 0; iFrame < visibilityMask->cols; iFrame++)
		{
			vMask.push_back(cvGetReal2D(visibilityMask, iFeature, iFrame));
		}
	}
	cvReleaseMat(&visibilityMask);
}

void GetParameterForGBA(vector<Feature> vFeature, vector<Camera> vCamera, vector<Theta> vTheta, CvMat *K, int nFrames,
						vector<double> &cameraParameter, vector<double> &feature2DParameter, CvMat &visibilityMask)
{
	int nFeatures = vFeature.size();

	for (int iCamera = 0; iCamera < vCamera.size(); iCamera++)
	{
		for (int iFrame = 0; iFrame < nFrames; iFrame++)
		{
			vector<int>::const_iterator it = find(vCamera[iCamera].vTakenFrame.begin(), vCamera[iCamera].vTakenFrame.end(), iFrame);
			if (it == vCamera[iCamera].vTakenFrame.end())
			{
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);

				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
				cameraParameter.push_back(0);
			}
			else
			{
				CvMat *q = cvCreateMat(4,1,CV_32FC1);
				CvMat *R = cvCreateMat(3,3,CV_32FC1);
				CvMat *t = cvCreateMat(3,1,CV_32FC1);
				CvMat *invK = cvCreateMat(3,3,CV_32FC1);
				CvMat *invR = cvCreateMat(3,3,CV_32FC1);
				cvInvert(K, invK);
				GetSubMatColwise(vCamera[iCamera].vP[iFrame], 0, 2, R);
				GetSubMatColwise(vCamera[iCamera].vP[iFrame], 3, 3, t);
				cvMatMul(invK, R, R);
				cvInvert(R, invR);
				cvMatMul(invK, t, t);
				cvMatMul(invR, t, t);
				ScalarMul(t, -1, t);
				Rotation2Quaternion(R, q);
				cameraParameter.push_back(cvGetReal2D(q, 0, 0));
				cameraParameter.push_back(cvGetReal2D(q, 1, 0));
				cameraParameter.push_back(cvGetReal2D(q, 2, 0));
				cameraParameter.push_back(cvGetReal2D(q, 3, 0));

				cameraParameter.push_back(cvGetReal2D(t, 0, 0));
				cameraParameter.push_back(cvGetReal2D(t, 1, 0));
				cameraParameter.push_back(cvGetReal2D(t, 2, 0));

				cvReleaseMat(&R);
				cvReleaseMat(&t);
				cvReleaseMat(&q);
				cvReleaseMat(&invK);
				cvReleaseMat(&invR);
			}
		}
	}

	for (int iTheta = 0; iTheta < vTheta.size(); iTheta++)
	{
		if (vTheta[iTheta].isStatic)
		{
			double IDCT = sqrt(1.0/(double)nFrames);
			double X = vTheta[iTheta].thetaX[0]*IDCT;
			double Y = vTheta[iTheta].thetaY[0]*IDCT;
			double Z = vTheta[iTheta].thetaZ[0]*IDCT;
			cameraParameter.push_back(X);
			cameraParameter.push_back(Y);
			cameraParameter.push_back(Z);
		}
		else
		{
			for (int iBase = 0; iBase < vTheta[iTheta].thetaX.size(); iBase++)
			{
				cameraParameter.push_back(vTheta[iTheta].thetaX[iBase]);
			}
			for (int iBase = 0; iBase < vTheta[iTheta].thetaX.size(); iBase++)
			{
				cameraParameter.push_back(vTheta[iTheta].thetaY[iBase]);
			}
			for (int iBase = 0; iBase < vTheta[iTheta].thetaX.size(); iBase++)
			{
				cameraParameter.push_back(vTheta[iTheta].thetaZ[iBase]);
			}
		}


	}

	visibilityMask = *cvCreateMat(vTheta.size(), vCamera.size()*nFrames, CV_32FC1);

	cvSetZero(&visibilityMask);
	int NZ = 0;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		for (int iFrame = 0; iFrame < vFeature[iFeature].vFrame.size(); iFrame++)
		{
			feature2DParameter.push_back(vFeature[iFeature].vx[iFrame]);
			feature2DParameter.push_back(vFeature[iFeature].vy[iFrame]);
			cvSetReal2D(&visibilityMask, iFeature, iFrame, 1);
		}
	}
}

//void GetCameraParameter(CvMat *P, CvMat *K, CvMat &R, CvMat &C)
//{
//	R = *cvCreateMat(3,3,CV_32FC1);
//	C = *cvCreateMat(3,1,CV_32FC1);
//	CvMat *temp34 = cvCreateMat(3,4,CV_32FC1);
//	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
//	cvInvert(K, invK);
//	cvMatMul(invK, P, temp34);
//	GetSubMatColwise(temp34, 0,2,&R);
//	GetSubMatColwise(temp34, 3,3,&C);
//	CvMat *invR = cvCreateMat(3,3,CV_32FC1);
//	cvInvert(&R, invR);
//	cvMatMul(invR, &C, &C);
//	ScalarMul(&C, -1, &C);
//
//	cvReleaseMat(&temp34);
//	cvReleaseMat(&invK);
//	cvReleaseMat(&invR);
//}
//
//void GetCameraParameter(CvMat *P, CvMat *K, CvMat *R, CvMat *C)
//{
//	//R = *cvCreateMat(3,3,CV_32FC1);
//	//C = *cvCreateMat(3,1,CV_32FC1);
//	CvMat *temp34 = cvCreateMat(3,4,CV_32FC1);
//	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
//	cvInvert(K, invK);
//	cvMatMul(invK, P, temp34);
//	GetSubMatColwise(temp34, 0,2,R);
//	GetSubMatColwise(temp34, 3,3,C);
//	CvMat *invR = cvCreateMat(3,3,CV_32FC1);
//	cvInvert(R, invR);
//	cvMatMul(invR, C, C);
//	ScalarMul(C, -1, C);
//
//	cvReleaseMat(&temp34);
//	cvReleaseMat(&invK);
//	cvReleaseMat(&invR);
//}

void CreateCameraMatrix(CvMat *R, CvMat *C, CvMat *K, CvMat &P)
{
	P = *cvCreateMat(3,4,CV_32FC1);
	cvSetIdentity(&P);
	ScalarMul(C, -1, C);
	SetSubMat(&P, 0,3, C);
	cvMatMul(R, &P, &P);
	cvMatMul(K, &P, &P);
}

void CreateCameraMatrix(CvMat *R, CvMat *C, CvMat *K, CvMat *P)
{
	cvSetIdentity(P);
	ScalarMul(C, -1, C);
	SetSubMat(P, 0,3, C);
	cvMatMul(R, P, P);
	cvMatMul(K, P, P);
}

int ExcludePointBehindCamera(CvMat *X, CvMat *P1, CvMat *P2, vector<int> featureID, vector<int> &excludedFeatureID, CvMat &cX)
{
	CvMat *H1 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH1 = cvCreateMat(4, 4, CV_32FC1);
	CvMat *HX1 = cvCreateMat(X->rows, X->cols, CV_32FC1);
	cvSetIdentity(H1);
	SetSubMat(H1, 0, 0, P1);
	cvInvert(H1, invH1);
	Pxx_inhomo(H1, X, HX1);

	CvMat *H2 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH2 = cvCreateMat(4, 4, CV_32FC1);
	CvMat *HX2 = cvCreateMat(X->rows, X->cols, CV_32FC1);
	cvSetIdentity(H2);
	SetSubMat(H2, 0, 0, P2);
	cvInvert(H2, invH2);
	Pxx_inhomo(H2, X, HX2);

	excludedFeatureID.clear();
	for (int i = 0; i < X->rows; i++)
	{
		if ((cvGetReal2D(HX1, i, 2) > 0) && (cvGetReal2D(HX2, i, 2) > 0))
			excludedFeatureID.push_back(featureID[i]);
	}
	if (excludedFeatureID.size() == 0)
		return 0;
	cX = *cvCreateMat(excludedFeatureID.size(),3, CV_32FC1);
	int k = 0;
	for (int i = 0; i < X->rows; i++)
	{
		if ((cvGetReal2D(HX1, i, 2) > 0) && (cvGetReal2D(HX2, i, 2) > 0))		
		{
			cvSetReal2D(&cX, k, 0, cvGetReal2D(X, i, 0));
			cvSetReal2D(&cX, k, 1, cvGetReal2D(X, i, 1));
			cvSetReal2D(&cX, k, 2, cvGetReal2D(X, i, 2));
			k++;
		}
	}
	return 1;
}

int ExcludePointBehindCamera_mem(CvMat *X, CvMat *P1, CvMat *P2, vector<int> featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX)
{
	CvMat *H1 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH1 = cvCreateMat(4, 4, CV_32FC1);
	CvMat *HX1 = cvCreateMat(X->rows, X->cols, CV_32FC1);
	cvSetIdentity(H1);
	SetSubMat(H1, 0, 0, P1);
	cvInvert(H1, invH1);
	Pxx_inhomo(H1, X, HX1);

	CvMat *H2 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH2 = cvCreateMat(4, 4, CV_32FC1);
	CvMat *HX2 = cvCreateMat(X->rows, X->cols, CV_32FC1);
	cvSetIdentity(H2);
	SetSubMat(H2, 0, 0, P2);
	cvInvert(H2, invH2);
	Pxx_inhomo(H2, X, HX2);

	excludedFeatureID.clear();
	for (int i = 0; i < X->rows; i++)
	{
		if ((cvGetReal2D(HX1, i, 2) > 0) && (cvGetReal2D(HX2, i, 2) > 0))
			excludedFeatureID.push_back(featureID[i]);
	}
	if (excludedFeatureID.size() == 0)
	{
		cvReleaseMat(&H1);
		cvReleaseMat(&HX1);
		cvReleaseMat(&H2);
		cvReleaseMat(&HX2);
		cvReleaseMat(&invH1);
		cvReleaseMat(&invH2);
		return 0;
	}
	//cX = *cvCreateMat(excludedFeatureID.size(),3, CV_32FC1);
	int k = 0;
	for (int i = 0; i < X->rows; i++)
	{
		if ((cvGetReal2D(HX1, i, 2) > 0) && (cvGetReal2D(HX2, i, 2) > 0))		
		{
			vector<double> cX_vec;
			cX_vec.push_back(cvGetReal2D(X, i, 0));
			cX_vec.push_back(cvGetReal2D(X, i, 1));
			cX_vec.push_back(cvGetReal2D(X, i, 2));

			cX.push_back(cX_vec);
			k++;
		}
	}
	cvReleaseMat(&H1);
	cvReleaseMat(&HX1);
	cvReleaseMat(&H2);
	cvReleaseMat(&HX2);
	cvReleaseMat(&invH1);
	cvReleaseMat(&invH2);
	return cX.size();
}

int ExcludePointBehindCamera_mem_fast(CvMat *X, CvMat *P1, CvMat *P2, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX)
{
	//CvMat *H1 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH1 = cvCreateMat(4, 4, CV_32FC1);
	//CvMat *HX1 = cvCreateMat(X->rows, X->cols, CV_32FC1);
	//cvSetIdentity(H1);
	//SetSubMat(H1, 0, 0, P1);
	//cvInvert(H1, invH1);
	//Pxx_inhomo(H1, X, HX1);

	//CvMat *H2 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH2 = cvCreateMat(4, 4, CV_32FC1);
	//CvMat *HX2 = cvCreateMat(X->rows, X->cols, CV_32FC1);
	//cvSetIdentity(H2);
	//SetSubMat(H2, 0, 0, P2);
	//cvInvert(H2, invH2);
	//Pxx_inhomo(H2, X, HX2);

	excludedFeatureID.clear();
	for (int i = 0; i < X->rows; i++)
	{
		CvMat *X_3d = cvCreateMat(4,1,CV_32FC1);
		cvSetReal2D(X_3d, 0, 0, cvGetReal2D(X,i,0));
		cvSetReal2D(X_3d, 1, 0, cvGetReal2D(X,i,1));
		cvSetReal2D(X_3d, 2, 0, cvGetReal2D(X,i,2));
		cvSetReal2D(X_3d, 3, 0, 1);

		CvMat *x1 = cvCreateMat(3,1,CV_32FC1);
		CvMat *x2 = cvCreateMat(3,1,CV_32FC1);

		cvMatMul(P1, X_3d, x1);
		cvMatMul(P2, X_3d, x2);
		
		if ((cvGetReal2D(x1, 2, 0) > 0) && (cvGetReal2D(x2, 2, 0) > 0))
		{
			excludedFeatureID.push_back(featureID[i]);

			vector<double> cX_vec;
			cX_vec.push_back(cvGetReal2D(X, i, 0));
			cX_vec.push_back(cvGetReal2D(X, i, 1));
			cX_vec.push_back(cvGetReal2D(X, i, 2));

			cX.push_back(cX_vec);
		}

		cvReleaseMat(&x1);
		cvReleaseMat(&x2);
		cvReleaseMat(&X_3d);
	}
	if (excludedFeatureID.size() == 0)
	{
		//cvReleaseMat(&H1);
		//cvReleaseMat(&HX1);
		//cvReleaseMat(&H2);
		//cvReleaseMat(&HX2);
		//cvReleaseMat(&invH1);
		//cvReleaseMat(&invH2);
		return 0;
	}
	//cvReleaseMat(&H1);
	//cvReleaseMat(&HX1);
	//cvReleaseMat(&H2);
	//cvReleaseMat(&HX2);
	//cvReleaseMat(&invH1);
	//cvReleaseMat(&invH2);
	return cX.size();
}


int ExcludePointAtInfinity(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K1, CvMat *K2, vector<int> featureID, vector<int> &excludedFeatureID, CvMat &cX)
{
	CvMat *q1 = cvCreateMat(4,1,CV_32FC1);
	CvMat *R1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *t1 = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR1 = cvCreateMat(3,3,CV_32FC1);

	CvMat *q2 = cvCreateMat(4,1,CV_32FC1);
	CvMat *R2 = cvCreateMat(3,3,CV_32FC1);
	CvMat *t2 = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR2 = cvCreateMat(3,3,CV_32FC1);

	GetSubMatColwise(P1, 0, 2, R1);
	GetSubMatColwise(P1, 3, 3, t1);
	cvInvert(K1, invK1);
	cvMatMul(invK1, R1, R1);
	cvInvert(R1, invR1);
	cvMatMul(invK1, t1, t1);
	cvMatMul(invR1, t1, t1);
	ScalarMul(t1, -1, t1);

	GetSubMatColwise(P2, 0, 2, R2);
	GetSubMatColwise(P2, 3, 3, t2);
	cvInvert(K2, invK2);
	cvMatMul(invK2, R2, R2);
	cvInvert(R2, invR2);
	cvMatMul(invK2, t2, t2);
	cvMatMul(invR2, t2, t2);
	ScalarMul(t2, -1, t2);
	
	double xC1 = cvGetReal2D(t1, 0, 0);
	double yC1 = cvGetReal2D(t1, 1, 0);
	double zC1 = cvGetReal2D(t1, 2, 0);
	double xC2 = cvGetReal2D(t2, 0, 0);
	double yC2 = cvGetReal2D(t2, 1, 0);
	double zC2 = cvGetReal2D(t2, 2, 0);

	excludedFeatureID.clear();
	vector<double> vInner;
	for (int i = 0; i < X->rows; i++)
	{
		double x3D = cvGetReal2D(X, i, 0);
		double y3D = cvGetReal2D(X, i, 1);
		double z3D = cvGetReal2D(X, i, 2);

		double v1x = x3D - xC1;		double v1y = y3D - yC1;		double v1z = z3D - zC1;
		double v2x = x3D - xC2;		double v2y = y3D - yC2;		double v2z = z3D - zC2;

		double nv1 = sqrt(v1x*v1x+v1y*v1y+v1z*v1z);
		double nv2 = sqrt(v2x*v2x+v2y*v2y+v2z*v2z);
		v1x /= nv1;		v1y /= nv1;		v1z /= nv1;
		v2x /= nv2;		v2y /= nv2;		v2z /= nv2;
		double inner = v1x*v2x+v1y*v2y+v1z*v2z;
		vInner.push_back(inner);
		if ((abs(inner) < cos(PI/180*3)) && (inner > 0))
			excludedFeatureID.push_back(featureID[i]);
	}
	if (excludedFeatureID.size() == 0)
		return 0;
	cX = *cvCreateMat(excludedFeatureID.size(),3, CV_32FC1);
	int k = 0;
	for (int i = 0; i < X->rows; i++)
	{
		if ((abs(vInner[i]) < cos(PI/180*3)) && (vInner[i] > 0))	
		{
			cvSetReal2D(&cX, k, 0, cvGetReal2D(X, i, 0));
			cvSetReal2D(&cX, k, 1, cvGetReal2D(X, i, 1));
			cvSetReal2D(&cX, k, 2, cvGetReal2D(X, i, 2));
			k++;
		}
	}
	return 1;
}

int ExcludePointAtInfinity_mem(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K1, CvMat *K2, vector<int> featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX)
{
	CvMat *q1 = cvCreateMat(4,1,CV_32FC1);
	CvMat *R1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *t1 = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR1 = cvCreateMat(3,3,CV_32FC1);

	CvMat *q2 = cvCreateMat(4,1,CV_32FC1);
	CvMat *R2 = cvCreateMat(3,3,CV_32FC1);
	CvMat *t2 = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR2 = cvCreateMat(3,3,CV_32FC1);

	GetSubMatColwise(P1, 0, 2, R1);
	GetSubMatColwise(P1, 3, 3, t1);
	cvInvert(K1, invK1);
	cvMatMul(invK1, R1, R1);
	cvInvert(R1, invR1);
	cvMatMul(invK1, t1, t1);
	cvMatMul(invR1, t1, t1);
	ScalarMul(t1, -1, t1);

	GetSubMatColwise(P2, 0, 2, R2);
	GetSubMatColwise(P2, 3, 3, t2);
	cvInvert(K2, invK2);
	cvMatMul(invK2, R2, R2);
	cvInvert(R2, invR2);
	cvMatMul(invK2, t2, t2);
	cvMatMul(invR2, t2, t2);
	ScalarMul(t2, -1, t2);

	double xC1 = cvGetReal2D(t1, 0, 0);
	double yC1 = cvGetReal2D(t1, 1, 0);
	double zC1 = cvGetReal2D(t1, 2, 0);
	double xC2 = cvGetReal2D(t2, 0, 0);
	double yC2 = cvGetReal2D(t2, 1, 0);
	double zC2 = cvGetReal2D(t2, 2, 0);

	excludedFeatureID.clear();
	vector<double> vInner;
	for (int i = 0; i < X->rows; i++)
	{
		double x3D = cvGetReal2D(X, i, 0);
		double y3D = cvGetReal2D(X, i, 1);
		double z3D = cvGetReal2D(X, i, 2);

		double v1x = x3D - xC1;		double v1y = y3D - yC1;		double v1z = z3D - zC1;
		double v2x = x3D - xC2;		double v2y = y3D - yC2;		double v2z = z3D - zC2;

		double nv1 = sqrt(v1x*v1x+v1y*v1y+v1z*v1z);
		double nv2 = sqrt(v2x*v2x+v2y*v2y+v2z*v2z);
		v1x /= nv1;		v1y /= nv1;		v1z /= nv1;
		v2x /= nv2;		v2y /= nv2;		v2z /= nv2;
		double inner = v1x*v2x+v1y*v2y+v1z*v2z;
		vInner.push_back(inner);
		if ((abs(inner) < cos(PI/180*3)) && (inner > 0))
			excludedFeatureID.push_back(featureID[i]);
	}
	if (excludedFeatureID.size() == 0)
	{
		cvReleaseMat(&q1);
		cvReleaseMat(&R1);
		cvReleaseMat(&t1);
		cvReleaseMat(&invK1);
		cvReleaseMat(&invR1);

		cvReleaseMat(&q2);
		cvReleaseMat(&R2);
		cvReleaseMat(&t2);
		cvReleaseMat(&invK2);
		cvReleaseMat(&invR2);
		return 0;
	}
	int k = 0;
	for (int i = 0; i < X->rows; i++)
	{
		if ((abs(vInner[i]) < cos(PI/180*3)) && (vInner[i] > 0))	
		{
			vector<double> cX_vec;
			cX_vec.push_back(cvGetReal2D(X, i, 0));
			cX_vec.push_back(cvGetReal2D(X, i, 1));
			cX_vec.push_back(cvGetReal2D(X, i, 2));
			cX.push_back(cX_vec);
			k++;
		}
	}

	cvReleaseMat(&q1);
	cvReleaseMat(&R1);
	cvReleaseMat(&t1);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invR1);

	cvReleaseMat(&q2);
	cvReleaseMat(&R2);
	cvReleaseMat(&t2);
	cvReleaseMat(&invK2);
	cvReleaseMat(&invR2);
	return cX.size();
}

int ExcludePointAtInfinity_mem_fast(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K1, CvMat *K2, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX)
{
	CvMat *q1 = cvCreateMat(4,1,CV_32FC1);
	CvMat *R1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *t1 = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK1 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR1 = cvCreateMat(3,3,CV_32FC1);

	CvMat *q2 = cvCreateMat(4,1,CV_32FC1);
	CvMat *R2 = cvCreateMat(3,3,CV_32FC1);
	CvMat *t2 = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK2 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR2 = cvCreateMat(3,3,CV_32FC1);

	GetSubMatColwise(P1, 0, 2, R1);
	GetSubMatColwise(P1, 3, 3, t1);
	cvInvert(K1, invK1);
	cvMatMul(invK1, R1, R1);
	cvInvert(R1, invR1);
	cvMatMul(invK1, t1, t1);
	cvMatMul(invR1, t1, t1);
	ScalarMul(t1, -1, t1);

	GetSubMatColwise(P2, 0, 2, R2);
	GetSubMatColwise(P2, 3, 3, t2);
	cvInvert(K2, invK2);
	cvMatMul(invK2, R2, R2);
	cvInvert(R2, invR2);
	cvMatMul(invK2, t2, t2);
	cvMatMul(invR2, t2, t2);
	ScalarMul(t2, -1, t2);

	double xC1 = cvGetReal2D(t1, 0, 0);
	double yC1 = cvGetReal2D(t1, 1, 0);
	double zC1 = cvGetReal2D(t1, 2, 0);
	double xC2 = cvGetReal2D(t2, 0, 0);
	double yC2 = cvGetReal2D(t2, 1, 0);
	double zC2 = cvGetReal2D(t2, 2, 0);

	excludedFeatureID.clear();
	vector<double> vInner;
	for (int i = 0; i < X->rows; i++)
	{
		double x3D = cvGetReal2D(X, i, 0);
		double y3D = cvGetReal2D(X, i, 1);
		double z3D = cvGetReal2D(X, i, 2);

		double v1x = x3D - xC1;		double v1y = y3D - yC1;		double v1z = z3D - zC1;
		double v2x = x3D - xC2;		double v2y = y3D - yC2;		double v2z = z3D - zC2;

		double nv1 = sqrt(v1x*v1x+v1y*v1y+v1z*v1z);
		double nv2 = sqrt(v2x*v2x+v2y*v2y+v2z*v2z);
		v1x /= nv1;		v1y /= nv1;		v1z /= nv1;
		v2x /= nv2;		v2y /= nv2;		v2z /= nv2;
		double inner = v1x*v2x+v1y*v2y+v1z*v2z;
		vInner.push_back(inner);
		if ((abs(inner) < cos(PI/180*3)) && (inner > 0))
		{			
			vector<double> cX_vec;
			cX_vec.push_back(x3D);
			cX_vec.push_back(y3D);
			cX_vec.push_back(z3D);
			cX.push_back(cX_vec);
			excludedFeatureID.push_back(featureID[i]);
		}
	}
	if (excludedFeatureID.size() == 0)
	{
		cvReleaseMat(&q1);
		cvReleaseMat(&R1);
		cvReleaseMat(&t1);
		cvReleaseMat(&invK1);
		cvReleaseMat(&invR1);

		cvReleaseMat(&q2);
		cvReleaseMat(&R2);
		cvReleaseMat(&t2);
		cvReleaseMat(&invK2);
		cvReleaseMat(&invR2);
		return 0;
	}

	cvReleaseMat(&q1);
	cvReleaseMat(&R1);
	cvReleaseMat(&t1);
	cvReleaseMat(&invK1);
	cvReleaseMat(&invR1);

	cvReleaseMat(&q2);
	cvReleaseMat(&R2);
	cvReleaseMat(&t2);
	cvReleaseMat(&invK2);
	cvReleaseMat(&invR2);
	return cX.size();
}



void ExcludePointHighReprojectionError(vector<Feature> vFeature, vector<CvMat *> cP, vector<int> vUsedFrame, vector<int> &visibleStrucrtureID, CvMat *X_tot)
{
	vector<bool> temp;
	temp.resize(visibleStrucrtureID.size(), true);
	for (int iP = 0; iP < cP.size(); iP++)
	{
		for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());
				CvMat *X = cvCreateMat(4, 1, CV_32FC1);
				CvMat *x = cvCreateMat(3, 1, CV_32FC1);
				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, visibleStrucrtureID[iVS], 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, visibleStrucrtureID[iVS], 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, visibleStrucrtureID[iVS], 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy[idx];
				double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);
				
				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 5)
				{
					cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					if(temp[iVS])
						temp[iVS] = false;
				}
				cvReleaseMat(&X);
				cvReleaseMat(&x);
			}
		}
	}
	vector<int> tempID;
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		if (temp[iVS])
		{
			tempID.push_back(visibleStrucrtureID[iVS]);
		}
	}
	visibleStrucrtureID = tempID;
}

void ExcludePointHighReprojectionError_mem(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, vector<int> &visibleStrucrtureID, CvMat *X_tot)
{
	vector<bool> temp;
	temp.resize(visibleStrucrtureID.size(), true);
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	for (int iP = 0; iP < cP.size(); iP++)
	{
		for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());

				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, visibleStrucrtureID[iVS], 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, visibleStrucrtureID[iVS], 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, visibleStrucrtureID[iVS], 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy[idx];
				double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 5)
				{
					cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					if(temp[iVS])
						temp[iVS] = false;
				}
			}
		}
	}
	cvReleaseMat(&X);
	cvReleaseMat(&x);
	vector<int> tempID;
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		if (temp[iVS])
		{
			tempID.push_back(visibleStrucrtureID[iVS]);
		}
	}
	visibleStrucrtureID = tempID;
}

int ExcludePointHighReprojectionError_mem_fast(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					//PrintMat(X);
					//PrintMat(cP[iP]);
					//PrintMat(x);
					

					double u0 = vFeature[iFeature].vx[idx];
					double v0 = vFeature[iFeature].vy[idx];
					double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 8)
					{
						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}
					
				}
			}

			vFeature[iFeature].nProj = nProj;
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}

int ExcludePointHighReprojectionError_mem_fast_Dome(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			//int nProj = 0;
			//for (int iP = 0; iP < cP.size(); iP++)
			//{
			//	vector<int>:: const_iterator it = find(vFeature[iFeature].vVisibleFrame.begin(), vFeature[iFeature].vVisibleFrame.end(), vUsedFrame[iP]);
			//	if (it != vFeature[iFeature].vFrame.end())
			//	{
			//		nProj++;
			//	}
			//}

			//if (nProj == 0)
			//	continue;

			if (vFeature[iFeature].vVisibleFrame.size() == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vVisibleFrame.begin(), vFeature[iFeature].vVisibleFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vVisibleFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vVisibleFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					//PrintMat(X);
					//PrintMat(cP[iP]);
					//PrintMat(x);
					

					double u0 = vFeature[iFeature].vVisible_dis_x[idx];
					double v0 = vFeature[iFeature].vVisible_dis_y[idx];
					double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 8)
					{
						vFeature[iFeature].vVisibleFrame.erase(vFeature[iFeature].vVisibleFrame.begin()+idx);
						vFeature[iFeature].vVisible_x.erase(vFeature[iFeature].vVisible_x.begin()+idx);
						vFeature[iFeature].vVisible_y.erase(vFeature[iFeature].vVisible_y.begin()+idx);
						vFeature[iFeature].vVisible_dis_x.erase(vFeature[iFeature].vVisible_dis_x.begin()+idx);
						vFeature[iFeature].vVisible_dis_y.erase(vFeature[iFeature].vVisible_dis_y.begin()+idx);
						//vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						
						if (vFeature[iFeature].vVisibleFrame.size() < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							vFeature[iFeature].vVisibleFrame.clear();
							vFeature[iFeature].vVisible_x.clear();
							vFeature[iFeature].vVisible_y.clear();

							vFeature[iFeature].vVisible_dis_x.clear();
							vFeature[iFeature].vVisible_dis_y.clear();
							break;
						}
					}
					
				}
			}
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}


int ExcludePointHighReprojectionError_mem_fast_Distortion(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, CvMat *K)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double tan_omega_half_2 = tan(omega/2)*2;

					double K11 = cvGetReal2D(K, 0, 0);
					double K22 = cvGetReal2D(K, 1, 1);
					double K13 = cvGetReal2D(K, 0, 2);
					double K23 = cvGetReal2D(K, 1, 2);

					double u_n = u/K11 - K13/K11;
					double v_n = v/K22 - K23/K22;

					double r_u = sqrt(u_n*u_n+v_n*v_n);
					double r_d = 1/omega*atan(r_u*tan_omega_half_2);

					double u_d_n = r_d/r_u * u_n;
					double v_d_n = r_d/r_u * v_n;

					double u1 = u_d_n*K11 + K13;
					double v1 = v_d_n*K22 + K23;

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 5)
					{
						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}

				}
			}
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}

int ExcludePointHighReprojectionError_mem_fast_GoPro(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot,
	vector<double> vOmega, vector<CvMat *> vK, vector<double> vpx1, vector<double> vpy1)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double u1, v1;

					if (vpx1[iP] == -1)
					{
						double tan_omega_half_2 = tan(vOmega[iP]/2)*2;

						double K11 = cvGetReal2D(vK[iP], 0, 0);
						double K22 = cvGetReal2D(vK[iP], 1, 1);
						double K13 = cvGetReal2D(vK[iP], 0, 2);
						double K23 = cvGetReal2D(vK[iP], 1, 2);

						double u_n = u/K11 - K13/K11;
						double v_n = v/K22 - K23/K22;

						double r_u = sqrt(u_n*u_n+v_n*v_n);
						double r_d = 1/vOmega[iP]*atan(r_u*tan_omega_half_2);

						double u_d_n = r_d/r_u * u_n;
						double v_d_n = r_d/r_u * v_n;

						u1 = u_d_n*K11 + K13;
						v1 = v_d_n*K22 + K23;
					}
					else
					{
						double tan_omega_half_2 = tan(vOmega[iP]/2)*2;

						double u_n = u - vpx1[iP];
						double v_n = v - vpy1[iP];

						double r_u = sqrt(u_n*u_n+v_n*v_n);
						double r_d = 1/vOmega[iP]*atan(r_u*tan_omega_half_2);

						double u_d_n = r_d/r_u * u_n;
						double v_d_n = r_d/r_u * v_n;

						u1 = u_d_n + vpx1[iP];
						v1 = v_d_n + vpy1[iP];
					}	

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 3)
					{

						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}

				}
			}
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}

int ExcludePointHighReprojectionError_mem_fast_Distortion_AD(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, CvMat *K, vector<vector<int> > &vvPointIndex)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double tan_omega_half_2 = tan(omega/2)*2;

					double K11 = cvGetReal2D(K, 0, 0);
					double K22 = cvGetReal2D(K, 1, 1);
					double K13 = cvGetReal2D(K, 0, 2);
					double K23 = cvGetReal2D(K, 1, 2);

					double u_n = u/K11 - K13/K11;
					double v_n = v/K22 - K23/K22;

					double r_u = sqrt(u_n*u_n+v_n*v_n);
					double r_d = 1/omega*atan(r_u*tan_omega_half_2);

					double u_d_n = r_d/r_u * u_n;
					double v_d_n = r_d/r_u * v_n;

					double u1 = u_d_n*K11 + K13;
					double v1 = v_d_n*K22 + K23;

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 3)
					{
						vector<int>::const_iterator it = find(vvPointIndex[iP].begin(), vvPointIndex[iP].end(), vFeature[iFeature].id);
						//if (it != vvPointIndex[iP].end())
						{
							int idx_point = (int) (it - vvPointIndex[iP].begin());
							vvPointIndex[iP].erase(vvPointIndex[iP].begin()+idx_point);
						}
						

						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}

				}
			}
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}

int ExcludePointHighReprojectionError_mem_fast_AD(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, CvMat *K, vector<vector<int> > &vvPointIndex)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					//double tan_omega_half_2 = tan(omega/2)*2;

					//double K11 = cvGetReal2D(K, 0, 0);
					//double K22 = cvGetReal2D(K, 1, 1);
					//double K13 = cvGetReal2D(K, 0, 2);
					//double K23 = cvGetReal2D(K, 1, 2);

					//double u_n = u/K11 - K13/K11;
					//double v_n = v/K22 - K23/K22;

					//double r_u = sqrt(u_n*u_n+v_n*v_n);
					//double r_d = 1/omega*atan(r_u*tan_omega_half_2);

					//double u_d_n = r_d/r_u * u_n;
					//double v_d_n = r_d/r_u * v_n;

					//double u1 = u_d_n*K11 + K13;
					//double v1 = v_d_n*K22 + K23;

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 3)
					{
						vector<int>::const_iterator it = find(vvPointIndex[iP].begin(), vvPointIndex[iP].end(), vFeature[iFeature].id);
						if (it != vvPointIndex[iP].end())
						{
							int idx_point = (int) (it - vvPointIndex[iP].begin());
							vvPointIndex[iP].erase(vvPointIndex[iP].begin()+idx_point);
						}
						

						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}

				}
			}
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}

int ExcludePointHighReprojectionError_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, vector<double> vOmega, vector<double> vprinc_x1, vector<double> vprinc_y1, vector<CvMat *> vK)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double tan_omega_half_2 = tan(vOmega[iP]/2)*2;

					//double K11 = cvGetReal2D(K, 0, 0);
					//double K22 = cvGetReal2D(K, 1, 1);
					//double K13 = cvGetReal2D(K, 0, 2);
					//double K23 = cvGetReal2D(K, 1, 2);

					//double u_n = u/K11 - K13/K11;
					//double v_n = v/K22 - K23/K22;

					double u_n = u - vprinc_x1[iP];
					double v_n = v - vprinc_y1[iP];

					double r_u = sqrt(u_n*u_n+v_n*v_n);
					double r_d = 1/vOmega[iP]*atan(r_u*tan_omega_half_2);

					double u_d_n = r_d/r_u * u_n;
					double v_d_n = r_d/r_u * v_n;

					double u1 = u_d_n + vprinc_x1[iP];
					double v1 = v_d_n + vprinc_y1[iP];

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 10)
					{
						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}

				}
			}
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}

int ExcludePointHighReprojectionError_mem_fast_Distortion_iPhone(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, vector<double> vk1, vector<CvMat *> vK)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double fx = cvGetReal2D(vK[iP], 0, 0);
					double fy = cvGetReal2D(vK[iP], 1, 1);
					double px = cvGetReal2D(vK[iP], 0, 2);
					double py = cvGetReal2D(vK[iP], 1, 2);

					double ud = (u-px)/fx;
					double vd = (v-py)/fy;

					double r = sqrt(ud*ud+vd*vd);
					double r_d = r*(1+vk1[iP]*r*r);
					ud = ud*r_d/r;
					vd = vd*r_d/r;

					double u1 = ud*fx+px;
					double v1 = vd*fy+py;

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 10)
					{
						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}
					


				}
			}
			vFeature[iFeature].nProj = nProj;
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}


int ExcludePointHighReprojectionError_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, double princ_x1, double princ_y1, CvMat *K)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1=0, count2=0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>:: const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int) (it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
					double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

					double tan_omega_half_2 = tan(omega/2)*2;

					//double K11 = cvGetReal2D(K, 0, 0);
					//double K22 = cvGetReal2D(K, 1, 1);
					//double K13 = cvGetReal2D(K, 0, 2);
					//double K23 = cvGetReal2D(K, 1, 2);

					//double u_n = u/K11 - K13/K11;
					//double v_n = v/K22 - K23/K22;

					double u_n = u - princ_x1;
					double v_n = v - princ_y1;

					double r_u = sqrt(u_n*u_n+v_n*v_n);
					double r_d = 1/omega*atan(r_u*tan_omega_half_2);

					double u_d_n = r_d/r_u * u_n;
					double v_d_n = r_d/r_u * v_n;

					double u1 = u_d_n + princ_x1;
					double v1 = v_d_n + princ_y1;

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1));
					//cout << dist << endl;

					if (dist > 3)
					{
						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin()+idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin()+idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin()+idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin()+idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin()+idx);
						vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin()+idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}

				}
			}
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}



bool ExcludePointHighReprojectionError_AddingFrame(vector<Feature> vFeature, vector<CvMat *> cP, vector<int> vUsedFrame
												  , vector<int> &visibleStrucrtureID, CvMat &X_tot
												  , vector<int> &visibleStrucrtureID_new, CvMat &X_tot_new)
{
	visibleStrucrtureID_new.clear();
	vector<double> vx, vy, vz;
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());
				CvMat *X = cvCreateMat(4, 1, CV_32FC1);
				CvMat *x = cvCreateMat(3, 1, CV_32FC1);
				cvSetReal2D(X, 0, 0, cvGetReal2D(&X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(&X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(&X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy[idx];
				double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 5)
				{
					//cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					isIn = false;
					break;
				}
				cvReleaseMat(&X);
				cvReleaseMat(&x);
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vx.push_back(cvGetReal2D(&X_tot, iVS, 0));
			vy.push_back(cvGetReal2D(&X_tot, iVS, 1));
			vz.push_back(cvGetReal2D(&X_tot, iVS, 2));
		}
	}
	if (visibleStrucrtureID_new.size() == 0)
		return false;
	X_tot_new = *cvCreateMat(visibleStrucrtureID_new.size(), 3, CV_32FC1);
	for (int ivx = 0; ivx < visibleStrucrtureID_new.size(); ivx++)
	{
		cvSetReal2D(&X_tot_new, ivx, 0, vx[ivx]);
		cvSetReal2D(&X_tot_new, ivx, 1, vy[ivx]);
		cvSetReal2D(&X_tot_new, ivx, 2, vz[ivx]);
	}
	
	return true;
}

bool ExcludePointHighReprojectionError_AddingFrame_mem(vector<Feature> &vFeature, vector<CvMat *> cP, vector<int> vUsedFrame
												   , vector<int> &visibleStrucrtureID, CvMat *X_tot
												   , vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new)
{
	visibleStrucrtureID_new.clear();
	vector<double> vx, vy, vz;
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());
				CvMat *X = cvCreateMat(4, 1, CV_32FC1);
				CvMat *x = cvCreateMat(3, 1, CV_32FC1);
				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy[idx];
				double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 5)
				{
					//cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					isIn = false;
					cvReleaseMat(&X);
					cvReleaseMat(&x);
					break;
				}
				cvReleaseMat(&X);
				cvReleaseMat(&x);
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vx.push_back(cvGetReal2D(X_tot, iVS, 0));
			vy.push_back(cvGetReal2D(X_tot, iVS, 1));
			vz.push_back(cvGetReal2D(X_tot, iVS, 2));
		}
	}
	if (visibleStrucrtureID_new.size() == 0)
		return false;

	for (int ivx = 0; ivx < visibleStrucrtureID_new.size(); ivx++)
	{
		vector<double> X_tot_new_vec;
		X_tot_new_vec.push_back(vx[ivx]);
		X_tot_new_vec.push_back(vy[ivx]);
		X_tot_new_vec.push_back(vz[ivx]);
		X_tot_new.push_back(X_tot_new_vec);
	}

	return true;
}

bool ExcludePointHighReprojectionError_AddingFrame_mem_fast(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
													   , vector<int> &visibleStrucrtureID, CvMat *X_tot
													   , vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new)
{
	visibleStrucrtureID_new.clear();
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());

				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy[idx];
				double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 8)
				{
					//cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					isIn = false;
					break;
				}
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vector<double> X_tot_new_vec;
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 0));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 1));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 2));
			X_tot_new.push_back(X_tot_new_vec);
		}
	}
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	if (visibleStrucrtureID_new.size() == 0)
		return false;

	return true;
}

bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
															double omega, CvMat *K)
{
	visibleStrucrtureID_new.clear();
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());

				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				double tan_omega_half_2 = tan(omega/2)*2;

				double K11 = cvGetReal2D(K, 0, 0);
				double K22 = cvGetReal2D(K, 1, 1);
				double K13 = cvGetReal2D(K, 0, 2);
				double K23 = cvGetReal2D(K, 1, 2);

				double u_n = u/K11 - K13/K11;
				double v_n = v/K22 - K23/K22;

				double r_u = sqrt(u_n*u_n+v_n*v_n);
				double r_d = 1/omega*atan(r_u*tan_omega_half_2);

				double u_d_n = r_d/r_u * u_n;
				double v_d_n = r_d/r_u * v_n;

				double u1 = u_d_n*K11 + K13;
				double v1 = v_d_n*K22 + K23;

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx_dis[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy_dis[idx];
				//double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				//double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				//cout << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 5)
				{
					//cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					isIn = false;
					break;
				}
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vector<double> X_tot_new_vec;
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 0));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 1));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 2));
			X_tot_new.push_back(X_tot_new_vec);
		}
	}
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	if (visibleStrucrtureID_new.size() == 0)
		return false;

	return true;
}

bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
															double omega, double princ_x1, double princ_y1, CvMat *K)
{
	visibleStrucrtureID_new.clear();
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());

				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				double tan_omega_half_2 = tan(omega/2)*2;

				//double K11 = cvGetReal2D(K, 0, 0);
				//double K22 = cvGetReal2D(K, 1, 1);
				//double K13 = cvGetReal2D(K, 0, 2);
				//double K23 = cvGetReal2D(K, 1, 2);

				//double u_n = u/K11 - K13/K11;
				//double v_n = v/K22 - K23/K22;

				double u_n = u - princ_x1;
				double v_n = v - princ_y1;

				double r_u = sqrt(u_n*u_n+v_n*v_n);
				double r_d = 1/omega*atan(r_u*tan_omega_half_2);

				double u_d_n = r_d/r_u * u_n;
				double v_d_n = r_d/r_u * v_n;

				double u1 = u_d_n + princ_x1;
				double v1 = v_d_n + princ_y1;

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx_dis[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy_dis[idx];
				//double u1 = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				//double v1 = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 5)
				{
					//cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					isIn = false;
					break;
				}
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vector<double> X_tot_new_vec;
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 0));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 1));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 2));
			X_tot_new.push_back(X_tot_new_vec);
		}
	}
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	if (visibleStrucrtureID_new.size() == 0)
		return false;

	return true;
}

bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion_ObstacleDetection(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
															vector<double> vOmega, vector<double> vPrinc_x1, vector<double> vPrinc_y1, vector<CvMat *> vK)
{
	visibleStrucrtureID_new.clear();
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());

				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				double tan_omega_half_2 = tan(vOmega[iP]/2)*2;

				//double K11 = cvGetReal2D(K, 0, 0);
				//double K22 = cvGetReal2D(K, 1, 1);
				//double K13 = cvGetReal2D(K, 0, 2);
				//double K23 = cvGetReal2D(K, 1, 2);

				//double u_n = u/K11 - K13/K11;
				//double v_n = v/K22 - K23/K22;

				double u_n = u - vPrinc_x1[iP];
				double v_n = v - vPrinc_y1[iP];

				double r_u = sqrt(u_n*u_n+v_n*v_n);
				double r_d = 1/vOmega[iP]*atan(r_u*tan_omega_half_2);

				double u_d_n = r_d/r_u * u_n;
				double v_d_n = r_d/r_u * v_n;

				double u1 = u_d_n + vPrinc_x1[iP];
				double v1 = v_d_n + vPrinc_y1[iP];

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx_dis[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy_dis[idx];

				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 10)
				{
					//cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					isIn = false;
					break;
				}
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vector<double> X_tot_new_vec;
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 0));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 1));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 2));
			X_tot_new.push_back(X_tot_new_vec);
		}
	}
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	if (visibleStrucrtureID_new.size() == 0)
		return false;

	return true;
}

bool ExcludePointHighReprojectionError_AddingFrame_mem_fast_Distortion_iPhone(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
															, vector<int> &visibleStrucrtureID, CvMat *X_tot
															, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new, 
															vector<double> vk1, vector<CvMat *> vK)
{
	visibleStrucrtureID_new.clear();
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>:: const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int) (it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());

				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u = cvGetReal2D(x, 0, 0)/cvGetReal2D(x, 2, 0);
				double v = cvGetReal2D(x, 1, 0)/cvGetReal2D(x, 2, 0);

				double fx = cvGetReal2D(vK[iP], 0, 0);
				double fy = cvGetReal2D(vK[iP], 1, 1);
				double px = cvGetReal2D(vK[iP], 0, 2);
				double py = cvGetReal2D(vK[iP], 1, 2);

				double ud = (u-px)/fx;
				double vd = (v-py)/fy;

				double r = sqrt(ud*ud+vd*vd);
				double r_d = r*(1+vk1[iP]*r*r);
				ud = ud*r_d/r;
				vd = vd*r_d/r;

				double u1 = ud*fx+px;
				double v1 = vd*fy+py;


				double u0 = vFeature[visibleStrucrtureID[iVS]].vx_dis[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy_dis[idx];

				if (sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) > 10)
				{
					//cout << visibleStrucrtureID[iVS] << "th 3D point erased " << sqrt((u0-u1)*(u0-u1)+(v0-v1)*(v0-v1)) << endl;
					isIn = false;
					break;
				}
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vector<double> X_tot_new_vec;
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 0));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 1));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 2));
			X_tot_new.push_back(X_tot_new_vec);
		}
	}
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	if (visibleStrucrtureID_new.size() == 0)
		return false;

	return true;
}

void OrientationRefinement(CvMat *R_1, CvMat *R_F, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat*> &vM, vector<CvMat*> &vm, vector<CvMat *> &vx1, vector<CvMat *> &vx2)
{
	//PrintAlgorithm("Orientation Refinement");
	//vector<double> cameraParameter, measurement;
	//AdditionalData adata;// focal_x focal_y princ_x princ_y
	////double intrinsic[4] = {cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1), cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2)};
	////adata.vIntrinsic.push_back(intrinsic);
	//adata.nFrames = vFrame1.size();

	//adata.vx1 = &vx1;
	//adata.vx2 = &vx2;

	//for (int iFrame = 0; iFrame < vFrame1.size(); iFrame++)
	//{
	//	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	//	Rotation2Quaternion(vM[iFrame], q);
	//	cameraParameter.push_back(cvGetReal2D(q, 0, 0));
	//	cameraParameter.push_back(cvGetReal2D(q, 1, 0));
	//	cameraParameter.push_back(cvGetReal2D(q, 2, 0));
	//	cameraParameter.push_back(cvGetReal2D(q, 3, 0));

	//	cameraParameter.push_back(cvGetReal2D(vm[iFrame], 0, 0));
	//	cameraParameter.push_back(cvGetReal2D(vm[iFrame], 1, 0));
	//	cameraParameter.push_back(cvGetReal2D(vm[iFrame], 2, 0));
	//	cvReleaseMat(&q);
	//}

	//CvMat *R_r = cvCreateMat(3,3,CV_32FC1);
	//CvMat *R_1_inv = cvCreateMat(3,3,CV_32FC1);
	//cvInvert(R_1, R_1_inv);
	//cvMatMul(R_F, R_1_inv, R_r);
	//CvMat *q_r = cvCreateMat(4,1,CV_32FC1);
	//Rotation2Quaternion(R_r, q_r);
	//measurement.push_back(cvGetReal2D(q_r, 0, 0));
	//measurement.push_back(cvGetReal2D(q_r, 1, 0));
	//measurement.push_back(cvGetReal2D(q_r, 2, 0));
	//measurement.push_back(cvGetReal2D(q_r, 3, 0));

	//for (int i = 0; i < vx1.size(); i++)
	//{
	//	if (vx1[i]->rows == 1)
	//		continue;
	//	//for (int j = 0; j < vx1[i]->rows; j++)	
	//	for (int j = 0; j < 10; j++)		
	//		measurement.push_back(0);
	//}
	//cvReleaseMat(&R_r);
	//cvReleaseMat(&R_1_inv);
	//cvReleaseMat(&q_r);

	//double *dmeasurement = (double *) malloc(measurement.size() * sizeof(double));
	//double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	//for (int i = 0; i < cameraParameter.size(); i++)
	//	dCameraParameter[i] = cameraParameter[i];
	//for (int i = 0; i < measurement.size(); i++)
	//	dmeasurement[i] = measurement[i];

	//double opt[5];
	//opt[0] = 1e-3;
	//opt[1] = 1e-12;
	//opt[2] = 1e-12;
	//opt[3] = 1e-12;
	//opt[4] = 0;
	//double info[12];

	//double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), measurement.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	//if(!work)
	//	fprintf(stderr, "memory allocation request failed in main()\n");

	//int ret ;//dlevmar_dif(ObjectiveOrientationRefinement, dCameraParameter, dmeasurement, cameraParameter.size(), measurement.size(),
	//	1e+2, opt, info, work, NULL, &adata);

	//PrintSBAInfo(info);

	//for (int iFrame = 0; iFrame < vFrame1.size(); iFrame++)
	//{
	//	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	//	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	//	cvSetReal2D(q, 0, 0, dCameraParameter[7*iFrame+0]);
	//	cvSetReal2D(q, 1, 0, dCameraParameter[7*iFrame+1]);
	//	cvSetReal2D(q, 2, 0, dCameraParameter[7*iFrame+2]);
	//	cvSetReal2D(q, 3, 0, dCameraParameter[7*iFrame+3]);
	//	Quaternion2Rotation(q, R);
	//	for (int i = 0; i < 3; i++)
	//	{
	//		for (int j = 0; j < 3; j++)
	//		{
	//			cvSetReal2D(vM[iFrame], i, j, cvGetReal2D(R, i, j));
	//		}
	//	}
	//	cvReleaseMat(&q);
	//	cvReleaseMat(&R);

	//	cvSetReal2D(vm[iFrame], 0, 0, dCameraParameter[7*iFrame+4]);
	//	cvSetReal2D(vm[iFrame], 1, 0, dCameraParameter[7*iFrame+5]);
	//	cvSetReal2D(vm[iFrame], 2, 0, dCameraParameter[7*iFrame+6]);
	//	
	//}

	//free(dmeasurement);
	//free(dCameraParameter);
	//free(work);
}

void OrientationRefinement1(CvMat *R_1, CvMat *R_F, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat*> &vM, vector<CvMat*> &vm, vector<CvMat *> &vx1, vector<CvMat *> &vx2, 
									               vector<int> vFrame1_r, vector<int> vFrame2_r, vector<CvMat*> &vM_r, vector<CvMat*> &vm_r, vector<CvMat *> &vx1_r, vector<CvMat *> &vx2_r)
{
	//PrintAlgorithm("Orientation Refinement for non-consecutive frame");
	//
	//// focal_x focal_y princ_x princ_y
	////double intrinsic[4] = {cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1), cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2)};
	////adata.vIntrinsic.push_back(intrinsic);

	//for (int iFrame = 0; iFrame < vFrame1_r.size(); iFrame++)
	//{
	//	vector<double> cameraParameter, measurement;
	//	AdditionalData adata;
	//	//cout << endl << "Orientation Refinement for non-consecutive frame: " << vFrame1_r[iFrame] << " " << vFrame2_r[iFrame] << endl; 
	//	cout << vFrame1_r[iFrame] << " ";
	//	cameraParameter.clear();
	//	measurement.clear();
	//	vector<double> vx1_a, vy1_a, vx2_a, vy2_a;
	//	for (int ix = 0; ix < vx1_r[iFrame]->rows; ix++)
	//	{
	//		vx1_a.push_back(cvGetReal2D(vx1_r[iFrame], ix, 0));
	//		vy1_a.push_back(cvGetReal2D(vx1_r[iFrame], ix, 1));

	//		vx2_a.push_back(cvGetReal2D(vx2_r[iFrame], ix, 0));
	//		vy2_a.push_back(cvGetReal2D(vx2_r[iFrame], ix, 1));

	//		measurement.push_back(0);
	//	}

	//	adata.vx1_a = &vx1_a;
	//	adata.vy1_a = &vy1_a;
	//	adata.vx2_a = &vx2_a;
	//	adata.vy2_a = &vy2_a;
	//	
	//	vector<int>::iterator it1 = find(vFrame1.begin(), vFrame1.end(), vFrame1_r[iFrame]);
	//	vector<int>::iterator it2 = find(vFrame2.begin(), vFrame2.end(), vFrame2_r[iFrame]);

	//	int idx1 = (int) (it1 - vFrame1.begin());
	//	int idx2 = (int) (it2 - vFrame2.begin());

	//	CvMat *R_r = cvCreateMat(3,3,CV_32FC1);
	//	cvSetIdentity(R_r);
	//	for (int iIdx = idx1; iIdx < idx2+1; iIdx++)
	//	{
	//		cvMatMul(vM[iIdx], R_r, R_r);
	//	}

	//	CvMat *q = cvCreateMat(4,1,CV_32FC1);
	//	Rotation2Quaternion(R_r, q);
	//	//Rotation2Quaternion(vM_r[iFrame], q);

	//	adata.qw = cvGetReal2D(q, 0, 0);
	//	adata.qx = cvGetReal2D(q, 1, 0);
	//	adata.qy = cvGetReal2D(q, 2, 0);
	//	adata.qz = cvGetReal2D(q, 3, 0);
	//	cvReleaseMat(&q);
	//	cameraParameter.push_back(cvGetReal2D(vm_r[iFrame], 0, 0));
	//	cameraParameter.push_back(cvGetReal2D(vm_r[iFrame], 1, 0));
	//	cameraParameter.push_back(cvGetReal2D(vm_r[iFrame], 2, 0));
	//	

	//	cvSetReal2D(vM_r[iFrame], 0, 0, cvGetReal2D(R_r, 0, 0));	cvSetReal2D(vM_r[iFrame], 0, 1, cvGetReal2D(R_r, 0, 1));	cvSetReal2D(vM_r[iFrame], 0, 2, cvGetReal2D(R_r, 0, 2));
	//	cvSetReal2D(vM_r[iFrame], 1, 0, cvGetReal2D(R_r, 1, 0));	cvSetReal2D(vM_r[iFrame], 1, 1, cvGetReal2D(R_r, 1, 1));	cvSetReal2D(vM_r[iFrame], 1, 2, cvGetReal2D(R_r, 1, 2));
	//	cvSetReal2D(vM_r[iFrame], 2, 0, cvGetReal2D(R_r, 2, 0));	cvSetReal2D(vM_r[iFrame], 2, 1, cvGetReal2D(R_r, 2, 1));	cvSetReal2D(vM_r[iFrame], 2, 2, cvGetReal2D(R_r, 2, 2));


	//	//for (int i = 0; i < vx1_r.size(); i++)
	//	//{
	//	//	for (int j = 0; j < vx1[i]->rows; j++)		
	//	//		measurement.push_back(0);
	//	//}

	//	double *dmeasurement = (double *) malloc(measurement.size() * sizeof(double));
	//	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));

	//	for (int i = 0; i < cameraParameter.size(); i++)
	//		dCameraParameter[i] = cameraParameter[i];
	//	for (int i = 0; i < measurement.size(); i++)
	//		dmeasurement[i] = measurement[i];

	//	double opt[5];
	//	opt[0] = 1e-3;
	//	opt[1] = 1e-12;
	//	opt[2] = 1e-12;
	//	opt[3] = 1e-12;
	//	opt[4] = 0;
	//	double info[12];
	//	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), measurement.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
	//	if(!work)
	//		fprintf(stderr, "memory allocation request failed in main()\n");

	//	int ret ;//dlevmar_dif(ObjectiveOrientationRefinement1, dCameraParameter, dmeasurement, cameraParameter.size(), measurement.size(),
	//		1e+2, opt, info, work, NULL, &adata);

	//	//PrintSBAInfo(info);
	//	cvReleaseMat(&R_r);

	//	cvSetReal2D(vm_r[iFrame], 0, 0, dCameraParameter[0]);
	//	cvSetReal2D(vm_r[iFrame], 1, 0, dCameraParameter[1]);
	//	cvSetReal2D(vm_r[iFrame], 2, 0, dCameraParameter[2]);

	//	free(dmeasurement);
	//	free(dCameraParameter);
	//	free(work);
	//}
	//cout << endl;
}

//void OrientationRefinement_sba(CvMat *R_1, CvMat *R_F, vector<int> vFrame1, vector<int> vFrame2, vector<CvMat*> &vM, vector<CvMat*> &vm, vector<CvMat *> &vx1, vector<CvMat *> &vx2, 
//							vector<int> vFrame1_r, vector<int> vFrame2_r, vector<CvMat*> &vM_r, vector<CvMat*> &vm_r, vector<CvMat *> &vx1_r, vector<CvMat *> &vx2_r)
//{
//	PrintAlgorithm("Orientation Refinement - sba");
//	vector<double> cameraParameter, measurement;
//	AdditionalData adata;// focal_x focal_y princ_x princ_y
//	//double intrinsic[4] = {cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1), cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2)};
//	//adata.vIntrinsic.push_back(intrinsic);
//	//adata.vFrame1 = vFrame1;
//	//adata.vFrame2 = vFrame2;
//	//adata.vFrame1_r = vFrame1_r;
//	//adata.vFrame2_r = vFrame2_r;
//
//	//adata.vx1 = &vx1;
//	//adata.vx2 = &vx2;
//
//	//adata.vx1_r = &vx1_r;
//	//adata.vx2_r = &vx2_r;
//
//	//for (int iFrame = 0; iFrame < vFrame1.size(); iFrame++)
//	//{
//	//	CvMat *q = cvCreateMat(4,1,CV_32FC1);
//	//	Rotation2Quaternion(vM[iFrame], q);
//	//	cameraParameter.push_back(cvGetReal2D(q, 0, 0));
//	//	cameraParameter.push_back(cvGetReal2D(q, 1, 0));
//	//	cameraParameter.push_back(cvGetReal2D(q, 2, 0));
//	//	cameraParameter.push_back(cvGetReal2D(q, 3, 0));
//	//	cvReleaseMat(&q);
//
//	//	cameraParameter.push_back(cvGetReal2D(vm[iFrame], 0, 0));
//	//	cameraParameter.push_back(cvGetReal2D(vm[iFrame], 1, 0));
//	//	cameraParameter.push_back(cvGetReal2D(vm[iFrame], 2, 0));
//	//}
//
//	//for (int iFrame = 0; iFrame < vFrame1_r.size(); iFrame++)
//	//{
//	//	cameraParameter.push_back(cvGetReal2D(vm_r[iFrame], 0, 0));
//	//	cameraParameter.push_back(cvGetReal2D(vm_r[iFrame], 1, 0));
//	//	cameraParameter.push_back(cvGetReal2D(vm_r[iFrame], 2, 0));
//	//}
//
//	CvMat *R_r = cvCreateMat(3,3,CV_32FC1);
//	CvMat *R_1_inv = cvCreateMat(3,3,CV_32FC1);
//	cvInvert(R_1, R_1_inv);
//	cvMatMul(R_F, R_1_inv, R_r);
//	CvMat *q_r = cvCreateMat(4,1,CV_32FC1);
//	Rotation2Quaternion(R_r, q_r);
//	//measurement.push_back(cvGetReal2D(q_r, 0, 0));
//	//measurement.push_back(cvGetReal2D(q_r, 1, 0));
//	//measurement.push_back(cvGetReal2D(q_r, 2, 0));
//	//measurement.push_back(cvGetReal2D(q_r, 3, 0));
//
//	for (int i = 0; i < vx1.size(); i++)
//	{
//		for (int j = 0; j < vx1[i]->rows; j++)	
//		//for (int j = 0; j < 7; j++)		
//			measurement.push_back(0);
//	}
//
//	//for (int i = 0; i < vx1_r.size(); i++)
//	//{
//	//	for (int j = 0; j < vx1[i]->rows; j++)	
//	//	//for (int j = 0; j < 3; j++)		
//	//		measurement.push_back(0);
//	//}
//
//	cameraParameter.push_back(cvGetReal2D(q_r, 0, 0));
//	cameraParameter.push_back(cvGetReal2D(q_r, 1, 0));
//	cameraParameter.push_back(cvGetReal2D(q_r, 2, 0));
//	cameraParameter.push_back(cvGetReal2D(q_r, 3, 0));
//
//	for (int iFrame = 0; iFrame < vFrame1.size(); iFrame++)
//	{
//		CvMat *q = cvCreateMat(4,1,CV_32FC1);
//		Rotation2Quaternion(vM[iFrame], q);
//		cameraParameter.push_back(cvGetReal2D(q, 0, 0));
//		cameraParameter.push_back(cvGetReal2D(q, 1, 0));
//		cameraParameter.push_back(cvGetReal2D(q, 2, 0));
//		cameraParameter.push_back(cvGetReal2D(q, 3, 0));
//		cvReleaseMat(&q);
//
//		cameraParameter.push_back(cvGetReal2D(vm[iFrame], 0, 0));
//		cameraParameter.push_back(cvGetReal2D(vm[iFrame], 1, 0));
//		cameraParameter.push_back(cvGetReal2D(vm[iFrame], 2, 0));
//	}
//
//
//
//	cvReleaseMat(&R_r);
//	cvReleaseMat(&R_1_inv);
//	cvReleaseMat(&q_r);
//
//	double *dmeasurement = (double *) malloc(measurement.size() * sizeof(double));
//	double *dCameraParameter = (double *) malloc(cameraParameter.size() * sizeof(double));
//
//	for (int i = 0; i < cameraParameter.size(); i++)
//		dCameraParameter[i] = cameraParameter[i];
//	for (int i = 0; i < measurement.size(); i++)
//		dmeasurement[i] = measurement[i];
//
//	double opt[5];
//	opt[0] = 1e-3;
//	opt[1] = 1e-5;
//	opt[2] = 1e-5;
//	opt[3] = 1e-5;
//	opt[4] = 0;
//	double info[12];
//
//	int ret = sba_mot_levmar()
//
//	double *work ;// (double*)malloc((LM_DIF_WORKSZ(cameraParameter.size(), measurement.size())+cameraParameter.size()*cameraParameter.size())*sizeof(double));
//	if(!work)
//		fprintf(stderr, "memory allocation request failed in main()\n");
//
//	int ret ;//dlevmar_dif(ObjectiveOrientationRefinement1, dCameraParameter, dmeasurement, cameraParameter.size(), measurement.size(),
//		1e+2, opt, info, work, NULL, &adata);
//
//	PrintSBAInfo(info);
//	vector<double> vR11, vR12, vR13, vR21, vR22, vR23, vR31, vR32, vR33;
//	for (int iFrame = 0; iFrame < vFrame1.size(); iFrame++)
//	{
//		CvMat *q = cvCreateMat(4,1,CV_32FC1);
//		CvMat *R = cvCreateMat(3,3,CV_32FC1);
//		cvSetReal2D(q, 0, 0, dCameraParameter[7*iFrame+0]);
//		cvSetReal2D(q, 1, 0, dCameraParameter[7*iFrame+1]);
//		cvSetReal2D(q, 2, 0, dCameraParameter[7*iFrame+2]);
//		cvSetReal2D(q, 3, 0, dCameraParameter[7*iFrame+3]);
//		Quaternion2Rotation(q, R);
//		for (int i = 0; i < 3; i++)
//		{
//			for (int j = 0; j < 3; j++)
//			{
//				cvSetReal2D(vM[iFrame], i, j, cvGetReal2D(R, i, j));
//			}
//		}
//
//		vR11.push_back(cvGetReal2D(R, 0, 0));	vR12.push_back(cvGetReal2D(R, 0, 1));	vR13.push_back(cvGetReal2D(R, 0, 2));
//		vR21.push_back(cvGetReal2D(R, 1, 0));	vR22.push_back(cvGetReal2D(R, 1, 1));	vR23.push_back(cvGetReal2D(R, 1, 2));
//		vR31.push_back(cvGetReal2D(R, 2, 0));	vR32.push_back(cvGetReal2D(R, 2, 1));	vR33.push_back(cvGetReal2D(R, 2, 2));
//		cvReleaseMat(&q);
//		cvReleaseMat(&R);
//
//		cvSetReal2D(vm[iFrame], 0, 0, dCameraParameter[7*iFrame+4]);
//		cvSetReal2D(vm[iFrame], 1, 0, dCameraParameter[7*iFrame+5]);
//		cvSetReal2D(vm[iFrame], 2, 0, dCameraParameter[7*iFrame+6]);
//	}
//
//	for (int iFrame_r = 0; iFrame_r < vFrame1_r.size(); iFrame_r++)
//	{
//		double R11 = 1;		double R12 = 0;		double R13 = 0;
//		double R21 = 0;		double R22 = 1;		double R23 = 0;
//		double R31 = 0;		double R32 = 0;		double R33 = 1;
//
//		for (int iFrame = vFrame1_r[iFrame_r]; iFrame < vFrame2_r[iFrame_r]; iFrame++)
//		{
//			R11 = vR11[iFrame]*R11 + vR12[iFrame]*R21 + vR13[iFrame]*R31;
//			R12 = vR11[iFrame]*R12 + vR12[iFrame]*R22 + vR13[iFrame]*R32;
//			R13 = vR11[iFrame]*R13 + vR12[iFrame]*R23 + vR13[iFrame]*R33;
//
//			R21 = vR21[iFrame]*R11 + vR22[iFrame]*R21 + vR23[iFrame]*R31;
//			R22 = vR21[iFrame]*R12 + vR22[iFrame]*R22 + vR23[iFrame]*R32;
//			R23 = vR21[iFrame]*R13 + vR22[iFrame]*R23 + vR23[iFrame]*R33;
//
//			R31 = vR31[iFrame]*R11 + vR32[iFrame]*R21 + vR33[iFrame]*R31;
//			R32 = vR31[iFrame]*R12 + vR32[iFrame]*R22 + vR33[iFrame]*R32;
//			R33 = vR31[iFrame]*R13 + vR32[iFrame]*R23 + vR33[iFrame]*R33;
//		}
//		cvSetReal2D(vM_r[iFrame_r], 0, 0, R11);	cvSetReal2D(vM_r[iFrame_r], 0, 1, R12);	cvSetReal2D(vM_r[iFrame_r], 0, 2, R13);
//		cvSetReal2D(vM_r[iFrame_r], 1, 0, R21);	cvSetReal2D(vM_r[iFrame_r], 1, 1, R22);	cvSetReal2D(vM_r[iFrame_r], 1, 2, R23);
//		cvSetReal2D(vM_r[iFrame_r], 2, 0, R31);	cvSetReal2D(vM_r[iFrame_r], 2, 1, R32);	cvSetReal2D(vM_r[iFrame_r], 2, 2, R33);
//
//		cvSetReal2D(vm_r[iFrame_r], 0, 0, dCameraParameter[7*vFrame1.size()+3*iFrame_r+0]);
//		cvSetReal2D(vm_r[iFrame_r], 1, 0, dCameraParameter[7*vFrame1.size()+3*iFrame_r+1]);
//		cvSetReal2D(vm_r[iFrame_r], 2, 0, dCameraParameter[7*vFrame1.size()+3*iFrame_r+2]);
//	}
//
//	free(dmeasurement);
//	free(dCameraParameter);
//	free(work);
//}

void DetectPOI(vector<CvMat *> vP, vector<CvMat *> vV, int nSegments, double range, double merging_threshold, vector<double> vBandwidth, vector<CvMat *> &vPOI,
			   double epsilon_cov, int nSegments_cov, vector<CvMat *> &v_a_cov, vector<CvMat *> &v_b_cov, vector<CvMat *> &v_l_cov, vector<double> &vf)
{

	vector<CvMat *> vPOI_temp;
	vector<vector<double> > vvWeight;
	for (int iP = 0; iP < vP.size(); iP++)
	{
		for (int iSeg = 0; iSeg < nSegments; iSeg++)
		{
			double y1, y2, y3;
			y1 = cvGetReal2D(vP[iP], 0, 0)+(range/(double)nSegments)*(iSeg+1)*cvGetReal2D(vV[iP], 0, 0);
			y2 = cvGetReal2D(vP[iP], 1, 0)+(range/(double)nSegments)*(iSeg+1)*cvGetReal2D(vV[iP], 1, 0);
			y3 = cvGetReal2D(vP[iP], 2, 0)+(range/(double)nSegments)*(iSeg+1)*cvGetReal2D(vV[iP], 2, 0);

			bool isBad = false;
			int nIter = 0;
			vector<double> vWeight;
			vector<double> vWeight_temp;
			while (1)
			{
				double yp1 = y1, yp2 = y2, yp3 = y3;
				vWeight.clear();
				MeanShift_Gaussian_Cone(y1, y2, y3, vP, vV, vBandwidth, vWeight);
				double normDiff = sqrt((y1-yp1)*(y1-yp1)+(y2-yp2)*(y2-yp2)+(y3-yp3)*(y3-yp3));
				if (normDiff < 1e-5)
				{
					break;
				}
				nIter++;
				if (nIter > 2000)
				{
					isBad = true;
					break;
				}
			}

			if (isBad)
			{
				continue;
			}
			double sumw = 0;
			for (int iw = 0; iw < vWeight.size(); iw++)
			{
				sumw += vWeight[iw];
			}
			for (int iw = 0; iw < vWeight.size(); iw++)
			{
				vWeight[iw] /= sumw;
			}

			vWeight_temp = vWeight;

			sort(vWeight.begin(), vWeight.end());
			if (vWeight[vWeight.size()-2]/vWeight[vWeight.size()-1] > 0.01)
			{
				if (vPOI_temp.empty())
				{
					CvMat *poi = cvCreateMat(3,1,CV_32FC1);
					cvSetReal2D(poi, 0, 0, y1);
					cvSetReal2D(poi, 1, 0, y2);
					cvSetReal2D(poi, 2, 0, y3);
					vPOI_temp.push_back(poi);
					vvWeight.push_back(vWeight_temp);
				}
				else
				{
					bool isIn = false;
					for (int iPoi = 0; iPoi < vPOI_temp.size(); iPoi++)
					{
						double c1 = cvGetReal2D(vPOI_temp[iPoi], 0, 0) - y1;
						double c2 = cvGetReal2D(vPOI_temp[iPoi], 1, 0) - y2;
						double c3 = cvGetReal2D(vPOI_temp[iPoi], 2, 0) - y3;
						
						if (sqrt(c1*c1+c2*c2+c3*c3) < merging_threshold)
						{
							isIn = true;
							break;
						}	
					}

					if (!isIn)
					{
						CvMat *poi = cvCreateMat(3,1,CV_32FC1);
						cvSetReal2D(poi, 0, 0, y1);
						cvSetReal2D(poi, 1, 0, y2);
						cvSetReal2D(poi, 2, 0, y3);
						vPOI_temp.push_back(poi);
						vvWeight.push_back(vWeight_temp);
					}
				}
			}
		}
	}
	
	for (int iPOI = 0; iPOI < vPOI_temp.size(); iPOI++)
	{
		vector<double> v_a, v_b, v_l;
		bool isGood = POICovariance(vP, vV, vBandwidth, vPOI_temp[iPOI], epsilon_cov, nSegments_cov, v_a, v_b, v_l);
		if (!isGood)
			continue;

		CvMat *a = cvCreateMat(v_a.size(), 1, CV_32FC1);
		CvMat *b = cvCreateMat(v_b.size(), 1, CV_32FC1);
		CvMat *l = cvCreateMat(v_b.size(), 1, CV_32FC1);

		for (int ia = 0; ia < v_a.size(); ia++)
		{
			cvSetReal2D(a, ia, 0, v_a[ia]);
			cvSetReal2D(b, ia, 0, v_b[ia]);
			cvSetReal2D(l, ia, 0, v_l[ia]);
		}

		v_a_cov.push_back(a);
		v_b_cov.push_back(b);
		v_l_cov.push_back(l);

		double f0 = EvaulateDensityFunction(vP, vV, vBandwidth, vPOI_temp[iPOI]);
		vf.push_back(f0);

		vPOI.push_back(cvCloneMat(vPOI_temp[iPOI]));


		//CvMat *U = cvCreateMat(3,3,CV_32FC1);
		//CvMat *Radius = cvCreateMat(3,1, CV_32FC1);
		//PrintMat(vPOI[iPOI]);
		//POICovariance(vP, vV, vvWeight[iPOI], U, Radius);
		//vU.push_back(U);
		//vRadius.push_back(Radius);
	}
	for (int i = 0; i < vPOI.size(); i++)
	{
		cvReleaseMat(&vPOI_temp[i]);
	}
	vPOI_temp.clear();
}

double EvaulateDensityFunction(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, CvMat *x)
{
	double f = 0;
	for (int ip = 0; ip < vP.size(); ip++)
	{
		if (vBandwidth[ip] < 0)
			continue;
		double norm_v = sqrt(cvGetReal2D(vV[ip], 0, 0)*cvGetReal2D(vV[ip], 0, 0)+cvGetReal2D(vV[ip], 1, 0)*cvGetReal2D(vV[ip], 1, 0)+cvGetReal2D(vV[ip], 2, 0)*cvGetReal2D(vV[ip], 2, 0));
		ScalarMul(vV[ip], 1/norm_v, vV[ip]);
		CvMat *xmp = cvCreateMat(3,1,CV_32FC1);
		cvSub(x, vP[ip], xmp);
		CvMat *dot_product = cvCreateMat(1,1,CV_32FC1);
		CvMat *v_t = cvCreateMat(1,3,CV_32FC1);
		cvTranspose(vV[ip], v_t);
		cvMatMul(v_t, xmp, dot_product);
		CvMat *dx = cvCreateMat(3,1,CV_32FC1);
		CvMat *dot_ray = cvCreateMat(3,1,CV_32FC1);
		ScalarMul(vV[ip], cvGetReal2D(dot_product, 0, 0), dot_ray);
		cvSub(xmp, dot_ray, dx);
		if (cvGetReal2D(dot_product, 0, 0) > 0)
		{
			double dist1 = sqrt(cvGetReal2D(dx, 0, 0)*cvGetReal2D(dx, 0, 0)+cvGetReal2D(dx, 1, 0)*cvGetReal2D(dx, 1, 0)+cvGetReal2D(dx, 2, 0)*cvGetReal2D(dx, 2, 0));
			double dist2 = cvGetReal2D(dot_product, 0, 0);

			f+= 1/vBandwidth[ip] * exp(-(dist1/dist2)*(dist1/dist2)/2/vBandwidth[ip]/vBandwidth[ip]);
		}

		cvReleaseMat(&xmp);
		cvReleaseMat(&dot_product);
		cvReleaseMat(&v_t);
		cvReleaseMat(&dx);
		cvReleaseMat(&dot_ray);
	}
	f /= vP.size();
	return f;
}

double EvaulateDensityFunction(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, CvMat *x, vector<double> &vWeight)
{
	double f = 0;
	for (int ip = 0; ip < vP.size(); ip++)
	{
		if (vBandwidth[ip] < 0)
		{
			vWeight.push_back(0);
			continue;
		}
		double norm_v = sqrt(cvGetReal2D(vV[ip], 0, 0)*cvGetReal2D(vV[ip], 0, 0)+cvGetReal2D(vV[ip], 1, 0)*cvGetReal2D(vV[ip], 1, 0)+cvGetReal2D(vV[ip], 2, 0)*cvGetReal2D(vV[ip], 2, 0));
		ScalarMul(vV[ip], 1/norm_v, vV[ip]);
		CvMat *xmp = cvCreateMat(3,1,CV_32FC1);
		cvSub(x, vP[ip], xmp);
		CvMat *dot_product = cvCreateMat(1,1,CV_32FC1);
		CvMat *v_t = cvCreateMat(1,3,CV_32FC1);
		cvTranspose(vV[ip], v_t);
		cvMatMul(v_t, xmp, dot_product);
		CvMat *dx = cvCreateMat(3,1,CV_32FC1);
		CvMat *dot_ray = cvCreateMat(3,1,CV_32FC1);
		ScalarMul(vV[ip], cvGetReal2D(dot_product, 0, 0), dot_ray);
		cvSub(xmp, dot_ray, dx);
		if (cvGetReal2D(dot_product, 0, 0) > 0)
		{
			double dist1 = sqrt(cvGetReal2D(dx, 0, 0)*cvGetReal2D(dx, 0, 0)+cvGetReal2D(dx, 1, 0)*cvGetReal2D(dx, 1, 0)+cvGetReal2D(dx, 2, 0)*cvGetReal2D(dx, 2, 0));
			double dist2 = cvGetReal2D(dot_product, 0, 0);

			f+= 1/vBandwidth[ip] * exp(-(dist1/dist2)*(dist1/dist2)/2/vBandwidth[ip]/vBandwidth[ip]);
			vWeight.push_back(1/vBandwidth[ip] * exp(-(dist1/dist2)*(dist1/dist2)/2/vBandwidth[ip]/vBandwidth[ip]));
		}
		else
		{
			vWeight.push_back(0);
		}

		cvReleaseMat(&xmp);
		cvReleaseMat(&dot_product);
		cvReleaseMat(&v_t);
		cvReleaseMat(&dx);
		cvReleaseMat(&dot_ray);
	}
	f /= vP.size();
	return f;
}

void DetectPOI(vector<CvMat *> vP, vector<CvMat *> vV, int nSegments, double range, double merging_threshold, vector<double> vBandwidth, vector<CvMat *> &vPOI,
			   double epsilon_cov, int nSegments_cov, vector<CvMat *> &v_a_cov, vector<CvMat *> &v_b_cov, vector<CvMat *> &v_l_cov, vector<double> &vf, 
			   vector<vector<CvMat *> > &vvMeanTrajectory)
{

	vector<CvMat *> vPOI_temp;
	vector<vector<double> > vvWeight;
	for (int iP = 0; iP < vP.size(); iP++)
	{
		for (int iSeg = 0; iSeg < nSegments; iSeg++)
		{
			double y1, y2, y3;
			y1 = cvGetReal2D(vP[iP], 0, 0)+(range/(double)nSegments)*(iSeg+1)*cvGetReal2D(vV[iP], 0, 0);
			y2 = cvGetReal2D(vP[iP], 1, 0)+(range/(double)nSegments)*(iSeg+1)*cvGetReal2D(vV[iP], 1, 0);
			y3 = cvGetReal2D(vP[iP], 2, 0)+(range/(double)nSegments)*(iSeg+1)*cvGetReal2D(vV[iP], 2, 0);

			bool isBad = false;
			int nIter = 0;
			vector<double> vWeight;
			vector<double> vWeight_temp;

			vector<double> mean_x;
			vector<double> mean_y;
			vector<double> mean_z;
			while (1)
			{
				mean_x.push_back(y1);
				mean_y.push_back(y2);
				mean_z.push_back(y3);
				double yp1 = y1, yp2 = y2, yp3 = y3;
				vWeight.clear();
				MeanShift_Gaussian_Cone(y1, y2, y3, vP, vV, vBandwidth, vWeight);
				double normDiff = sqrt((y1-yp1)*(y1-yp1)+(y2-yp2)*(y2-yp2)+(y3-yp3)*(y3-yp3));
				if (normDiff < 1e-5)
				{
					break;
				}
				nIter++;
				if (nIter > 2000)
				{
					isBad = true;
					break;
				}
			}

			if (isBad)
			{
				continue;
			}
			double sumw = 0;
			for (int iw = 0; iw < vWeight.size(); iw++)
			{
				sumw += vWeight[iw];
			}
			for (int iw = 0; iw < vWeight.size(); iw++)
			{
				vWeight[iw] /= sumw;
			}

			vWeight_temp = vWeight;

			sort(vWeight.begin(), vWeight.end());
			if (vWeight[vWeight.size()-2]/vWeight[vWeight.size()-1] > 0.1)
			{
				vector<CvMat *> vMean;
				for (int ii = 0; ii < mean_x.size(); ii++)
				{
					CvMat *mm = cvCreateMat(3,1,CV_32FC1);
					cvSetReal2D(mm, 0, 0, mean_x[ii]);
					cvSetReal2D(mm, 1, 0, mean_y[ii]);
					cvSetReal2D(mm, 2, 0, mean_z[ii]);
					vMean.push_back(mm);
				}
				vvMeanTrajectory.push_back(vMean);

				if (vPOI_temp.empty())
				{
					CvMat *poi = cvCreateMat(3,1,CV_32FC1);
					cvSetReal2D(poi, 0, 0, y1);
					cvSetReal2D(poi, 1, 0, y2);
					cvSetReal2D(poi, 2, 0, y3);
					vPOI_temp.push_back(poi);
					vvWeight.push_back(vWeight_temp);
				}
				else
				{
					bool isIn = false;
					for (int iPoi = 0; iPoi < vPOI_temp.size(); iPoi++)
					{
						double c1 = cvGetReal2D(vPOI_temp[iPoi], 0, 0) - y1;
						double c2 = cvGetReal2D(vPOI_temp[iPoi], 1, 0) - y2;
						double c3 = cvGetReal2D(vPOI_temp[iPoi], 2, 0) - y3;

						if (sqrt(c1*c1+c2*c2+c3*c3) < merging_threshold)
						{
							isIn = true;
							break;
						}	
					}

					if (!isIn)
					{
						CvMat *poi = cvCreateMat(3,1,CV_32FC1);
						cvSetReal2D(poi, 0, 0, y1);
						cvSetReal2D(poi, 1, 0, y2);
						cvSetReal2D(poi, 2, 0, y3);
						vPOI_temp.push_back(poi);
						vvWeight.push_back(vWeight_temp);
					}
				}
			}
		}
	}

	for (int iPOI = 0; iPOI < vPOI_temp.size(); iPOI++)
	{
		vector<double> v_a, v_b, v_l;
		bool isGood = POICovariance(vP, vV, vBandwidth, vPOI_temp[iPOI], epsilon_cov, nSegments_cov, v_a, v_b, v_l);
		if (!isGood)
		{
			vector<vector<CvMat *> > vvMeanTraj_temp;
			for (int iTraj = 0; iTraj < vvMeanTrajectory.size(); iTraj++)
			{
				double traj_x = cvGetReal2D(vvMeanTrajectory[iTraj][vvMeanTrajectory[iTraj].size()-1], 0, 0);
				double traj_y = cvGetReal2D(vvMeanTrajectory[iTraj][vvMeanTrajectory[iTraj].size()-1], 1, 0);
				double traj_z = cvGetReal2D(vvMeanTrajectory[iTraj][vvMeanTrajectory[iTraj].size()-1], 2, 0);
				double error = sqrt((traj_x-cvGetReal2D(vPOI_temp[iPOI], 0, 0))*(traj_x-cvGetReal2D(vPOI_temp[iPOI], 0, 0))+
									(traj_y-cvGetReal2D(vPOI_temp[iPOI], 1, 0))*(traj_y-cvGetReal2D(vPOI_temp[iPOI], 1, 0))+
									(traj_z-cvGetReal2D(vPOI_temp[iPOI], 2, 0))*(traj_z-cvGetReal2D(vPOI_temp[iPOI], 2, 0)));
				if (error < 1e-3)
					continue;
				vvMeanTraj_temp.push_back(vvMeanTrajectory[iTraj]);
			}
			vvMeanTrajectory = vvMeanTraj_temp;
			continue;
		}

		CvMat *a = cvCreateMat(v_a.size(), 1, CV_32FC1);
		CvMat *b = cvCreateMat(v_b.size(), 1, CV_32FC1);
		CvMat *l = cvCreateMat(v_b.size(), 1, CV_32FC1);

		for (int ia = 0; ia < v_a.size(); ia++)
		{
			cvSetReal2D(a, ia, 0, v_a[ia]);
			cvSetReal2D(b, ia, 0, v_b[ia]);
			cvSetReal2D(l, ia, 0, v_l[ia]);
		}

		v_a_cov.push_back(a);
		v_b_cov.push_back(b);
		v_l_cov.push_back(l);

		double f0 = EvaulateDensityFunction(vP, vV, vBandwidth, vPOI_temp[iPOI]);
		vf.push_back(f0);

		vPOI.push_back(cvCloneMat(vPOI_temp[iPOI]));


		//CvMat *U = cvCreateMat(3,3,CV_32FC1);
		//CvMat *Radius = cvCreateMat(3,1, CV_32FC1);
		//PrintMat(vPOI[iPOI]);
		//POICovariance(vP, vV, vvWeight[iPOI], U, Radius);
		//vU.push_back(U);
		//vRadius.push_back(Radius);
	}
	for (int i = 0; i < vPOI.size(); i++)
	{
		cvReleaseMat(&vPOI_temp[i]);
	}
	vPOI_temp.clear();
}


bool POICovariance(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, CvMat *poi, double epsilon, int nSegments, vector<double> &v_a, vector<double> &v_b, vector<double> &v_l)
{
	double f0 = EvaulateDensityFunction(vP, vV, vBandwidth, poi);
	double phi_step = 2*PI/nSegments;
	double theta_step = PI/(nSegments/2);

	bool isGood = true;

	for (int iphi = 0; iphi < nSegments; iphi++)
	{
		for (int itheta = 0; itheta < nSegments/2+1; itheta++)
		{
			CvMat *v = cvCreateMat(3,1,CV_32FC1);
			cvSetReal2D(v, 0, 0, cos(iphi*phi_step)*sin(itheta*theta_step));
			cvSetReal2D(v, 1, 0, sin(iphi*phi_step)*sin(itheta*theta_step));
			cvSetReal2D(v, 2, 0, cos(itheta*theta_step));
			CvMat *v_temp = cvCreateMat(3,1,CV_32FC1);

			CvMat *x = cvCreateMat(3,1,CV_32FC1);
			ScalarMul(v, epsilon, v_temp);
			cvAdd(poi, v_temp, x);
			double f = EvaulateDensityFunction(vP, vV, vBandwidth, x);
			if ((f0-f)/(-epsilon) > -5e-2)
			{				
				int k = 2;
				while (1)
				{
					ScalarMul(v, k*epsilon, v_temp);
					k++;
					cvAdd(poi, v_temp, x);
					f = EvaulateDensityFunction(vP, vV, vBandwidth, x);

					if (f < f0)
					{
						isGood = true;
						break;
					}
					if (k > 100)
					{
						isGood = false;
						break;
					}
				}			
				if (!isGood)
					v_a.push_back(-5e-2);
				else
					v_a.push_back((f0-f)/(-epsilon));
			}
			else
			{
				v_a.push_back((f0-f)/(-epsilon));
			}
			v_b.push_back(f0);
			

			double alpha = 0;
			int iter = 0;
			while (1)
			{
				alpha = alpha + 0.1;
				ScalarMul(v, epsilon+alpha, v_temp);
				cvAdd(poi, v_temp, x);
				f = EvaulateDensityFunction(vP, vV, vBandwidth, x);
				double y = (f0-f)/(-epsilon) * (epsilon+alpha) + f0;
				if (abs(f-y) > 2e-1)
				{
					v_l.push_back(alpha+epsilon);
					break;
				}
				iter++;

				if ((!isGood) && (iter == 2))
				{
					v_l.push_back(alpha+epsilon);
					break;
				}

				if (iter == 10)
				{
					v_l.push_back(alpha+epsilon);
					break;
				}
			}

			cvReleaseMat(&v);
			cvReleaseMat(&x);
		}
	}
	return isGood;
}

void POICovariance(vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vWeight, CvMat *U, CvMat *Radius)
{
	CvMat *A = cvCreateMat(3*vP.size(), 3, CV_32FC1);
	CvMat *b = cvCreateMat(3*vP.size(), 1, CV_32FC1);

	for (int iCamera = 0; iCamera < vP.size(); iCamera++)
	{
		CvMat *Ai = cvCreateMat(3,3, CV_32FC1);
		CvMat *bi = cvCreateMat(3,1, CV_32FC1);

		Vec2Skew(vV[iCamera], Ai);
		ScalarMul(Ai, vWeight[iCamera], Ai);
		cvMatMul(Ai, vP[iCamera], bi);

		SetSubMat(A, 3*iCamera, 0, Ai);
		SetSubMat(b, 3*iCamera, 0, bi);

		cvReleaseMat(&Ai);
		cvReleaseMat(&bi);
	}

	double meanb = 0;
	for (int ib = 0; ib < b->rows; ib++)
	{
		meanb += cvGetReal2D(b, ib, 0);
	}
	meanb /= b->rows;

	double varianceb = 0;
	for (int ib = 0; ib < b->rows; ib++)
	{
		varianceb += (cvGetReal2D(b, ib, 0)-meanb)*(cvGetReal2D(b, ib, 0)-meanb);
	}
	varianceb /= b->rows;

	CvMat *At = cvCreateMat(A->cols, A->rows, CV_32FC1);
	cvTranspose(A, At);
	CvMat *Q = cvCreateMat(3,3,CV_32FC1);
	cvMatMul(At, A, Q);
	cvInvert(Q, Q);
	ScalarMul(Q, varianceb, Q);

	CvMat *D = cvCreateMat(3,3,CV_32FC1);
	CvMat *V = cvCreateMat(3,3,CV_32FC1);

	cvSVD(Q, D, U, V);
	cvSetReal2D(Radius, 0, 0, sqrt(cvGetReal2D(D, 0, 0)));
	cvSetReal2D(Radius, 1, 0, sqrt(cvGetReal2D(D, 1, 1)));
	cvSetReal2D(Radius, 2, 0, sqrt(cvGetReal2D(D, 2, 2)));


	cvReleaseMat(&D);
	cvReleaseMat(&V);
	cvReleaseMat(&Q);
	cvReleaseMat(&A);
	cvReleaseMat(&At);
	cvReleaseMat(&b);
}

void MeanShift_Gaussian_Cone(double &y1, double &y2, double &y3, vector<CvMat *> vP, vector<CvMat *> vV, vector<double> vBandwidth, vector<double> &vWeight)
{
	double sum1 = 0, sum2 = 0, sum3 = 0;
	double normalization = 0;

	for (int iP = 0; iP < vP.size(); iP++)
	{
		double dist, gradient1, gradient2, gradient3;
		DistanceBetweenConeAndPoint(cvGetReal2D(vP[iP], 0, 0), cvGetReal2D(vP[iP], 1, 0), cvGetReal2D(vP[iP], 2, 0), 
			cvGetReal2D(vV[iP], 0, 0), cvGetReal2D(vV[iP], 1, 0), cvGetReal2D(vV[iP], 2, 0),	
			y1, y2, y3, dist);

		double xmp1 = y1-cvGetReal2D(vP[iP], 0, 0);
		double xmp2 = y2-cvGetReal2D(vP[iP], 1, 0);
		double xmp3 = y3-cvGetReal2D(vP[iP], 2, 0);
		double v1 = cvGetReal2D(vV[iP], 0, 0);
		double v2 = cvGetReal2D(vV[iP], 1, 0);
		double v3 = cvGetReal2D(vV[iP], 2, 0);
		double v_norm = sqrt(v1*v1+v2*v2+v3*v3);
		
		v1 /= v_norm;
		v2 /= v_norm;
		v3 /= v_norm;

		double vxmp1 = v1*xmp1;
		double vxmp2 = v2*xmp2;
		double vxmp3 = v3*xmp3;

		double vxmp = vxmp1+vxmp2+vxmp3;
		double x_tilde1 = cvGetReal2D(vP[iP], 0, 0) + (vxmp + dist*dist*vxmp)*v1;
		double x_tilde2 = cvGetReal2D(vP[iP], 1, 0) + (vxmp + dist*dist*vxmp)*v2;
		double x_tilde3 = cvGetReal2D(vP[iP], 2, 0) + (vxmp + dist*dist*vxmp)*v3;

		double weight = exp(-dist*dist/vBandwidth[iP]/vBandwidth[iP]/2)/(vBandwidth[iP]*vBandwidth[iP]*vBandwidth[iP]*vxmp*vxmp);
		sum1 += weight * x_tilde1;
		sum2 += weight * x_tilde2;
		sum3 += weight * x_tilde3;
		normalization += weight;
		vWeight.push_back(weight);
	}

	y1 = sum1/normalization;
	y2 = sum2/normalization;
	y3 = sum3/normalization;
}

void MeanShift_Gaussian_Ray(double &y1, double &y2, double &y3, vector<CvMat *> vP, vector<CvMat *> vV, double bandwidth, vector<double> &vWeight)
{
	double sum1 = 0, sum2 = 0, sum3 = 0;
	double normalization = 0;

	for (int iP = 0; iP < vP.size(); iP++)
	{
		double dist, gradient1, gradient2, gradient3;
		DistanceBetweenRayAndPoint(cvGetReal2D(vP[iP], 0, 0), cvGetReal2D(vP[iP], 1, 0), cvGetReal2D(vP[iP], 2, 0), 
								   cvGetReal2D(vV[iP], 0, 0), cvGetReal2D(vV[iP], 1, 0), cvGetReal2D(vV[iP], 2, 0),	
								   y1, y2, y3, dist, gradient1, gradient2, gradient3);
		double dir1, dir2, dir3;
		dir1 = y1-cvGetReal2D(vP[iP], 0, 0);
		dir2 = y2-cvGetReal2D(vP[iP], 1, 0);
		dir3 = y3-cvGetReal2D(vP[iP], 2, 0);

		double dot_product = dir1*cvGetReal2D(vV[iP], 0, 0)+dir2*cvGetReal2D(vV[iP], 1, 0)+dir3*cvGetReal2D(vV[iP], 2, 0);

		if (dot_product < 0)
		{
			dist = 1e+6;
		}
		
		sum1 += gradient1*exp(-dist*dist/bandwidth/bandwidth);
		sum2 += gradient2*exp(-dist*dist/bandwidth/bandwidth);
		sum3 += gradient3*exp(-dist*dist/bandwidth/bandwidth);	

		normalization += exp(-dist*dist/bandwidth/bandwidth);	
		vWeight.push_back(exp(-dist*dist/bandwidth/bandwidth));
	}

	y1 = sum1/normalization;
	y2 = sum2/normalization;
	y3 = sum3/normalization;
}

void DistanceBetweenRayAndPoint(double p1, double p2, double p3, double v1, double v2, double v3, double x1, double x2, double x3,
								double &d, double &g1, double &g2, double &g3)
{
	double v_norm = sqrt(v1*v1+v2*v2+v3*v3);
	v1 /= v_norm;
	v2 /= v_norm;
	v3 /= v_norm;
	double xmp1 = x1-p1;
	double xmp2 = x2-p2;
	double xmp3 = x3-p3;

	double xmp_dot_v1 = xmp1*v1;
	double xmp_dot_v2 = xmp1*v2;
	double xmp_dot_v3 = xmp1*v3;

	double d1 = xmp1-xmp_dot_v1*v1;
	double d2 = xmp2-xmp_dot_v2*v2;
	double d3 = xmp3-xmp_dot_v3*v3;

	d = sqrt(d1*d1+d2*d2+d3*d3);
	
	g1 = p1 + xmp_dot_v1*v1;
	g2 = p2 + xmp_dot_v2*v2;
	g3 = p3 + xmp_dot_v3*v3;
}

void DistanceBetweenConeAndPoint(double p1, double p2, double p3, double v1, double v2, double v3, double x1, double x2, double x3,
								double &d)
{
	double v_norm = sqrt(v1*v1+v2*v2+v3*v3);
	v1 /= v_norm;
	v2 /= v_norm;
	v3 /= v_norm;
	double xmp1 = x1-p1;
	double xmp2 = x2-p2;
	double xmp3 = x3-p3;

	double xmp_dot_v1 = xmp1*v1;
	double xmp_dot_v2 = xmp2*v2;
	double xmp_dot_v3 = xmp3*v3;

	double direction = xmp_dot_v1+xmp_dot_v2+xmp_dot_v3;

	if (direction > 0)
	{
		double upper1 = xmp1 - direction*v1;
		double upper2 = xmp2 - direction*v2;
		double upper3 = xmp3 - direction*v3;

		d = sqrt(upper1*upper1+upper2*upper2+upper3*upper3)/direction;
	}
	else
	{
		d = 1e+6;
	}
}
