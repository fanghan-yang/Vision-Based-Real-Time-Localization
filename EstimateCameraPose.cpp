#include "EstimateCameraPose.h"
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

int DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierIndex)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	//vector<int> vOutlierIndex;
	//vInlierIndex.clear();
	//vOutlierIndex.clear();

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	vector<CvMat*> result_P(ransacMaxIter);
	vector<vector<int> > result_inlier(ransacMaxIter);
 	//cout<<"start Ransac"<<endl;

	//#pragma omp parallel for
	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{
		//cout<<iRansacIter<<endl;		
		CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
		CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
		CvMat *randP = cvCreateMat(3,4,CV_32FC1);
		int *randIdx = (int *) malloc(min_set * sizeof(int));
		CvMat *reproj = cvCreateMat(3,1,CV_32FC1);
		CvMat *homo_X = cvCreateMat(4,1,CV_32FC1);

		//cout<<iRansacIter<<" random"<<endl;		
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand()%X->rows;

		//cout<<iRansacIter<<" set point"<<endl;		
		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		//cout<<iRansacIter<<" epnp"<<endl;		
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		//cout<<iRansacIter<<" got inliner"<<endl;		
  		vector<int> vInlier, vOutlier;
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
		result_P[iRansacIter] = cvCreateMat(3,4,CV_32FC1);
		result_inlier[iRansacIter] = vInlier;
		cvCopy(result_P[iRansacIter], randP);
		//SetSubMat(result_P[iRansacIter], 0, 0, randP);

		//cout<<iRansacIter<<" release"<<endl;		
		cvReleaseMat(&reproj);
		cvReleaseMat(&homo_X);
		free(randIdx);
		cvReleaseMat(&randx);
		cvReleaseMat(&randX);
		cvReleaseMat(&randP);

		//cout<<iRansacIter<<" done"<<endl;		
		//if (vInlier.size() > X->rows * 0.8)
		//{
		//	break;
		//}
	}
 
 	//cout<<"finish Ransac"<<endl;
	int maxInlier = 0;
 	for (int i = 0; i < ransacMaxIter; i++)
	{		
 		if (result_inlier[i].size() > maxInlier)
		{
			maxInlier = result_inlier[i].size();
			SetSubMat(P, 0, 0, result_P[i]);
			vInlierIndex = result_inlier[i];
			//vOutlierIndex = vOutlier;
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

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&x_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	//if (vInlierIndex.size() < 20)
	//	return 0;
	//cout << "Number of features to do ePNP camera pose estimation: " << vInlierIndex.size() << endl;
	return vInlierIndex.size();
}
