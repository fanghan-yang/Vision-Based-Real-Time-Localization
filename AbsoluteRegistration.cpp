// StaticReconstruction.cpp : Defines the entry point for the console application.
//

//#include <flann/flann.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <boost/filesystem.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <omp.h>
#include "DataUtility.h"
#include "MultiviewGeometryUtility.h"
//#include "EstimateCameraPose.h"
#include "MathUtility.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

#define DLT_RANSAC_THRESHOLD 2
#define DLT_RANSAC_MAX_ITER 100

#define SCALAR 5
#define DEBUG_FLOW 0
#define DEBUG_KNN 0
#define DEBUG_INLIER 0

#define FILE_PATH "./"
#define INPUT_FOLDER "reconstruction/"
#define OUTPUT_FOLDER "absolute/"

// global variable
string path = FILE_PATH;
string cameraFile;
vector<string> vFilename;
//double focal_x, focal_y, princ_x, princ_y, omega, k1;
CvMat *K, *invK;
double k1;
bool isSIFT = false;
//CvMat *invK;
int frameID = -1;
int last_frame = -1;
cv::Mat current_img;
cv::Mat detection_img;
vector<Correspondence2D3D> last_corr;
vector<double> scales;
cv::Scalar color_red = cv::Scalar(0, 0, 255);
cv::Scalar color_green = cv::Scalar(0, 255, 0);
cv::Scalar color_blue = cv::Scalar(255, 0, 0);
cv::Scalar color_yellow = cv::Scalar(0, 255, 255);

// pthread
pthread_rwlock_t image_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_t tracking_thread;

void Undistortion(CvMat *K, CvMat *invK, double k1, vector<double> &vx,  vector<double> &vy)
{
    double ik11 = cvGetReal2D(invK, 0, 0);  double ik12 = cvGetReal2D(invK, 0, 1);  double ik13 = cvGetReal2D(invK, 0, 2);
    double ik21 = cvGetReal2D(invK, 1, 0);  double ik22 = cvGetReal2D(invK, 1, 1);  double ik23 = cvGetReal2D(invK, 1, 2);
    double ik31 = cvGetReal2D(invK, 2, 0);  double ik32 = cvGetReal2D(invK, 2, 1);  double ik33 = cvGetReal2D(invK, 2, 2);
    double k11 = cvGetReal2D(K, 0, 0);  double k12 = cvGetReal2D(K, 0, 1);  double k13 = cvGetReal2D(K, 0, 2);
    double k21 = cvGetReal2D(K, 1, 0);  double k22 = cvGetReal2D(K, 1, 1);  double k23 = cvGetReal2D(K, 1, 2);
    double k31 = cvGetReal2D(K, 2, 0);  double k32 = cvGetReal2D(K, 2, 1);  double k33 = cvGetReal2D(K, 2, 2);

    for (int iPoint = 0; iPoint < vx.size(); iPoint++)
    {
        double x = vx[iPoint];
        double y = vy[iPoint];
        double xc = k13;
        double yc = k23;
        double nzc = (ik31*xc+ik32*yc+ik33);
        double nxc = (ik11*xc+ik12*yc+ik13)/nzc;
        double nyc = (ik21*xc+ik22*yc+ik23)/nzc;
 
        double nz = (ik31*x+ik32*y+ik33);
        double nx = (ik11*x+ik12*y+ik13)/nz;
        double ny = (ik21*x+ik22*y+ik23)/nz;
 
        double r_d = sqrt((nx-nxc)*(nx-nxc)+(ny-nyc)*(ny-nyc));
        //r_d = 0.01;
 
        double c = 1/k1;
        double d = -c*r_d;
 
        double Q = c/3;
        double R = -d/2;
 
        double Delta = Q*Q*Q + R*R;
        double r_u, r_u_1, r_u_2, r_u_3;
        if (Delta >= 0)
        {
            double RDelta = R + sqrt(Delta);
    double signedRDelta = RDelta;
    if (RDelta < 0)
       signedRDelta = -RDelta;

            signedRDelta = pow(signedRDelta, 1.0/3);
    if (RDelta < 0)
       RDelta = -signedRDelta;
            r_u = RDelta + Q/RDelta;
//cout << Delta << " " << R << " " << RDelta << " " << Q << endl;
        }
        else
        {
            //double S = sqrt(R*R-Delta);
            //S = pow(S, 1.0/3);
            //double T = 1.0/3*atan(sqrt(-Delta)/R);
            //r_u = -S * cos(T) + S*sqrt(3.0)*sin(T);
        Q = -Q;
        double S = sqrt(Q);
        double T = acos(R/S/Q);
        r_u_1 = 2*S*cos(T/3);
        r_u_2 = 2*S*cos((T+2*M_PI)/3);
        r_u_3 = 2*S*cos((T-2*M_PI)/3);
//cout << "T " << T << " " << r_u_1 << " " << r_u_2 << " " << r_u_3 << endl; 
        
//        if (abs(r_u_1-r_d)<abs(r_u_2-r_d) && abs(r_u_1-r_d)<abs(r_u_3-r_d) )
//        r_u = r_u_1;
//        else if (abs(r_u_2-r_d)<abs(r_u_3-r_d))
//        r_u = r_u_2;
//        else
        r_u = r_u_3;
        }
 
        //double r_d1 = r_u * (1+k1*r_u*r_u);
        //double r_d1 = r_u_1 * (1+k1*r_u_1*r_u_1);
        //double r_d2 = r_u_2 * (1+k1*r_u_2*r_u_2);
        //double r_d3 = r_u_3 * (1+k1*r_u_3*r_u_3);
 
        nx = nxc + abs(r_u)/r_d * (nx - nxc);
        ny = nyc + abs(r_u)/r_d * (ny - nyc);

        //cout << Delta << " "<< r_d << " " << r_d1 << " " << r_u_1 << " " << r_d2 << " " << r_u_2 << " " << r_d3 << " " << r_u_3 << endl;
        //cout << Delta << " "<< r_d << " " << r_d1 << " " << r_u << endl;
        double z = (k31*nx+k32*ny+k33);
        x = (k11*nx+k12*ny+k13)/z;
        y = (k21*nx+k22*ny+k23)/z;
 
        vx[iPoint] = x;
        vy[iPoint] = y;
 
        //cout << x << " " << y << " " << z << endl;
    }
}

void Undistortion_Radial(CvMat* K, CvMat* invK, double k1, vector<double> &vx,  vector<double> &vy)
{
    double ik11 = cvGetReal2D(invK, 0, 0);  double ik12 = cvGetReal2D(invK, 0, 1);  double ik13 = cvGetReal2D(invK, 0, 2);
    double ik21 = cvGetReal2D(invK, 1, 0);  double ik22 = cvGetReal2D(invK, 1, 1);  double ik23 = cvGetReal2D(invK, 1, 2);
    double ik31 = cvGetReal2D(invK, 2, 0);  double ik32 = cvGetReal2D(invK, 2, 1);  double ik33 = cvGetReal2D(invK, 2, 2);
    double k11 = cvGetReal2D(K, 0, 0);  double k12 = cvGetReal2D(K, 0, 1);  double k13 = cvGetReal2D(K, 0, 2);
    double k21 = cvGetReal2D(K, 1, 0);  double k22 = cvGetReal2D(K, 1, 1);  double k23 = cvGetReal2D(K, 1, 2);
    double k31 = cvGetReal2D(K, 2, 0);  double k32 = cvGetReal2D(K, 2, 1);  double k33 = cvGetReal2D(K, 2, 2);
    for (int iPoint = 0; iPoint < vx.size(); iPoint++)
    {
        double x = vx[iPoint];
        double y = vy[iPoint];
        double xc = k13;
        double yc = k23;
        double nzc = (ik31*xc+ik32*yc+ik33);
        double nxc = (ik11*xc+ik12*yc+ik13)/nzc;
        double nyc = (ik21*xc+ik22*yc+ik23)/nzc;
 
        double nz = (ik31*x+ik32*y+ik33);
        double nx = (ik11*x+ik12*y+ik13)/nz;
        double ny = (ik21*x+ik22*y+ik23)/nz;
 
        double r_d = sqrt((nx-nxc)*(nx-nxc)+(ny-nyc)*(ny-nyc));
        //r_d = 0.01;
 
        double c = 1/k1;
        double d = -c*r_d;
 
        double Q = c/3;
        double R = -d/2;
 
        double Delta = Q*Q*Q + R*R;
        double r_u, r_u_1, r_u_2, r_u_3;
        if (Delta >= 0)
        {
            double RDelta = R + sqrt(Delta);
            RDelta = pow(RDelta, 1.0/3);
            r_u = RDelta + Q/RDelta;
        }
        else
        {
            //double S = sqrt(R*R-Delta);
            //S = pow(S, 1.0/3);
            //double T = 1.0/3*atan(sqrt(-Delta)/R);
            //r_u = -S * cos(T) + S*sqrt(3.0)*sin(T);
	    Q = -Q;
            double S = sqrt(Q);
	    double T = acos(R/S/Q);
	    //r_u_1 = 2*S*cos(T/3);
	    //r_u_2 = 2*S*cos((T+2*M_PI)/3);
	    //r_u_3 = 2*S*cos((T-2*M_PI)/3);
	    r_u = 2*S*cos((T-2*M_PI)/3);
        }
 
        //double r_d1 = r_u * (1+k1*r_u*r_u);
        //double r_d1 = r_u_1 * (1+k1*r_u_1*r_u_1);
        //double r_d2 = r_u_2 * (1+k1*r_u_2*r_u_2);
        //double r_d3 = r_u_3 * (1+k1*r_u_3*r_u_3);
 
        nx = nxc + abs(r_u)/r_d * (nx - nxc);
        ny = nyc + abs(r_u)/r_d * (ny - nyc);

        //cout << Delta << " "<< r_d << " " << r_d1 << " " << r_u_1 << " " << r_d2 << " " << r_u_2 << " " << r_d3 << " " << r_u_3 << endl;
        //cout << Delta << " "<< r_d << " " << r_d1 << " " << r_u << endl;
        double z = (k31*nx+k32*ny+k33);
        x = (k11*nx+k12*ny+k13)/z;
        y = (k21*nx+k22*ny+k23)/z;
 
        vx[iPoint] = x;
        vy[iPoint] = y;
 
        //cout << x << " " << y << endl;
    }
}

void Distortion_Radial(CvMat* K, CvMat* invK, double k1, vector<double> &vx,  vector<double> &vy)
{
    double ik11 = cvGetReal2D(invK, 0, 0);  double ik12 = cvGetReal2D(invK, 0, 1);  double ik13 = cvGetReal2D(invK, 0, 2);
    double ik21 = cvGetReal2D(invK, 1, 0);  double ik22 = cvGetReal2D(invK, 1, 1);  double ik23 = cvGetReal2D(invK, 1, 2);
    double ik31 = cvGetReal2D(invK, 2, 0);  double ik32 = cvGetReal2D(invK, 2, 1);  double ik33 = cvGetReal2D(invK, 2, 2);
    double k11 = cvGetReal2D(K, 0, 0);  double k12 = cvGetReal2D(K, 0, 1);  double k13 = cvGetReal2D(K, 0, 2);
    double k21 = cvGetReal2D(K, 1, 0);  double k22 = cvGetReal2D(K, 1, 1);  double k23 = cvGetReal2D(K, 1, 2);
    double k31 = cvGetReal2D(K, 2, 0);  double k32 = cvGetReal2D(K, 2, 1);  double k33 = cvGetReal2D(K, 2, 2);
    for (int iPoint = 0; iPoint < vx.size(); iPoint++)
    {
        double x = vx[iPoint];
        double y = vy[iPoint];
	
	double nx = (x-k13)/k11;
	double ny = (y-k23)/k22;
 
        double r_u = sqrt(nx*nx+ny*ny);
	double u = nx * (1 + k1 * r_u * r_u);
	double v = ny * (1 + k1 * r_u * r_u);

	u = k11*u+k13;
	v = k22*v+k23;
 
        vx[iPoint] = u;
        vy[iPoint] = v;
 
        //cout << x << " " << y << endl;
    }
}

void printTime(string text, timeval t1, timeval t2){
	time_t sec = t2.tv_sec-t1.tv_sec;
	suseconds_t usec = t2.tv_usec-t1.tv_usec;
	cout<<text<<(sec*1000000+usec)/1000000.0<<"s"<<endl;
}

void UpdateCameraDataRealTime(string filename, CvMat * vP, int vFrame, CvMat *K, bool file_exist)
{
	ofstream fout;
	if (file_exist) {
		fout.open(filename.c_str(), ios_base::app);
	} else {
		fout.open(filename.c_str(), ios_base::out);
	}

	CvMat *R = cvCreateMat(3,3,CV_32FC1);
	CvMat *t = cvCreateMat(3,1,CV_32FC1);
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K, invK);
	CvMat *temp34 = cvCreateMat(3,4,CV_32FC1);
	CvMat *temp33 = cvCreateMat(3,3,CV_32FC1);
	CvMat *invR = cvCreateMat(3,3,CV_32FC1);

	cvMatMul(invK, vP, temp34);
	GetSubMatColwise(temp34, 0, 2, R);
	GetSubMatColwise(temp34, 3, 3, t);
	cvInvert(R, invR);
	cvMatMul(invR, t, t);

	fout << "0 " << vFrame << endl;
	fout << -cvGetReal2D(t, 0, 0) << " " << -cvGetReal2D(t, 1, 0) << " " << -cvGetReal2D(t, 2, 0) << endl;
	fout << cvGetReal2D(R, 0, 0) << " " << cvGetReal2D(R, 0, 1) << " " << cvGetReal2D(R, 0, 2) << endl;
	fout << cvGetReal2D(R, 1, 0) << " " << cvGetReal2D(R, 1, 1) << " " << cvGetReal2D(R, 1, 2) << endl;
	fout << cvGetReal2D(R, 2, 0) << " " << cvGetReal2D(R, 2, 1) << " " << cvGetReal2D(R, 2, 2) << endl;

	cvReleaseMat(&R);
	cvReleaseMat(&t);
	cvReleaseMat(&invK);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&invR);

	fout.close();
}

void* Optical_Flow(void* param) {
	cout << "Tracking Thread starts ..." << endl;
	timeval t1, t2;
	vector<CvMat *> vP;
	vector<int> vFrame;
	vP.resize(vFilename.size());
	vFrame.resize(vFilename.size(), -1);

	InitCuda(0);
	cout << vFilename.size() << " images." << endl;
	int registered_frame = 0;
	cv::Mat img_prev, img_curr, img_debug;
	vector<Correspondence2D3D> vCorr;
	vector<double> vCorr_scales;

	timeval t_prev, t_curr;
	vector<cv::Point2f> feature0;
	vector<cv::Point2f> feature1;
	int nFeatures;
	
	bool isInitialized = false;
	int file_id = 0;
	bool file_exist = false;
	for (int iFile = 0; iFile < vFilename.size(); iFile++)
	{
		bool new_feature_flag = false;
		cv::imread(path+vFilename[iFile], 0).convertTo(img_curr, CV_8U);

		pthread_rwlock_wrlock(&image_lock);
		current_img = img_curr.clone();
		pthread_rwlock_unlock(&image_lock);

		cout << "  <flow> Load file " << iFile << endl;
		pthread_rwlock_rdlock(&image_lock);
		bool isSIFT_loc = isSIFT;
		pthread_rwlock_unlock(&image_lock);

		if (isSIFT_loc)
		{
			pthread_rwlock_rdlock(&image_lock);
			cv::Mat im_d = detection_img.clone();			
			img_prev = detection_img.clone();
			registered_frame = last_frame;
			vCorr = last_corr;
			if (DEBUG_FLOW)
			{
				vCorr_scales = scales;
			}
			pthread_rwlock_unlock(&image_lock);

			new_feature_flag = true;		

			nFeatures = vCorr.size();
			feature0.resize(nFeatures);
			for (int iF = 0; iF < nFeatures; iF++)
	  		{
	  			feature0[iF].x = vCorr[iF].u;
	  			feature0[iF].y = vCorr[iF].v;
	  		}
			
			feature1 = feature0;
			isInitialized = true;			
			
			//img_curr.convertTo(current_img, CV_32FC1);
			
			cout<< "    <Tracking> Got frame: "<< registered_frame << endl;

			if (flase) {
				//cv::Mat im_d = img_prev.clone();
				for (int i = 0; i < vCorr.size(); i++)
				{
					cv::Point2f p1, p2;
					p1.x = vCorr[i].u;
					p1.y = vCorr[i].v;
					//line(img_debug, feature1[i], p1, color_blue, 1);
					circle(im_d, p1, 5, color_green, -1);	
					if (i == 0)
					cout << "optical: " << last_corr[i].u << " " << last_corr[i].v << endl;
				}
			
				char temp[1000];
				sprintf(temp, "./result/first%04d.jpg", registered_frame);
				cout << temp << endl;
				imwrite(temp, im_d);
			}
		}

		pthread_rwlock_wrlock(&image_lock);
		frameID = iFile;
		pthread_rwlock_unlock(&image_lock);
		
		if (iFile==0)
     			usleep(500000);

		if (!isInitialized)
			continue;

		pthread_rwlock_wrlock(&image_lock);
		isSIFT = false;
		pthread_rwlock_unlock(&image_lock);

		//pthread_rwlock_rdlock(&feature_lock);
		//if (last_frame>registered_frame){
		//	registered_frame = last_frame;
		//	vCorr = last_corr;
		//}
		//pthread_rwlock_unlock(&feature_lock);
		
		//////////////////////////////////////////////////////////////////////////////////////
		///////////////////// 		Optical Flow
		//////////////////////////////////////////////////////////////////////////////////////
		vector<uchar> optical_flow_found_feature(nFeatures);
		vector<float> optical_flow_feature_error(nFeatures);

		cv::TermCriteria optical_flow_termination_criteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, .3 );

		cv::Size optical_flow_window(50, 50);
		
		if (iFile == 0 || registered_frame == -1) {
			img_prev = img_curr.clone();
			continue;
		}			
		cout << "    <Tracking> Flow feature num: " << nFeatures << endl;
		gettimeofday(&t1, NULL);
		//img_next = current_img.clone();	
		
		cv::calcOpticalFlowPyrLK(img_prev, img_curr, feature0, feature1,
					 optical_flow_found_feature, 
					 optical_flow_feature_error,
					 optical_flow_window, 15,
					 optical_flow_termination_criteria,
					 cv::OPTFLOW_USE_INITIAL_FLOW);

		if (DEBUG_FLOW) {
			char output_file_name[100];
			sprintf(output_file_name, "./result/flow_feature_number_%0d.txt", iFile);
			ofstream fout(output_file_name);
			fout << iFile << " " << feature0.size() << " " << feature1.size() << endl;
			fout.close();
		}

		gettimeofday(&t2,NULL);
		printTime("    <Tracking> Flow time: ",t1,t2);
		//img_prev = img_curr.clone();

		vector<double> vx(feature1.size()),  vy(feature1.size());

		// plot flow before undistortion
		if (DEBUG_FLOW)
			cvtColor(img_curr, img_debug, CV_GRAY2RGB);
		vx.clear();
		vy.clear();
		for (int i = 0; i < feature1.size(); i++)
		{
			vx.push_back(feature1[i].x);
			vy.push_back(feature1[i].y);

			// plot flow tracking
			if (DEBUG_FLOW) {
				string myStr;          //The string
				ostringstream myTemp;  //temp as in temporary
				myTemp << iFile;
				myStr = myTemp.str();
				cv::Point textPos;
				textPos.x = 35;
				textPos.y = 35;
				putText(img_debug, myStr, textPos, 2, 0.8, color_red);
				if (new_feature_flag) {
					if ((int)optical_flow_found_feature[i] == 1)
						circle(img_debug, feature1[i], 1.4 * vCorr_scales[i], color_green, 2);
					else
						circle(img_debug, feature0[i], 1.4 * vCorr_scales[i], color_red, 2);
				} else {
					if ((int)optical_flow_found_feature[i] == 1)
						circle(img_debug, feature1[i], 1.4 * vCorr_scales[i], color_blue, 2);
					else
						circle(img_debug, feature0[i], 1.4 * vCorr_scales[i], color_red, 2);
				}
			}
		}

		Undistortion_Radial(K, invK, k1, vx, vy);

		/////////////////////////////////////////////////////////////////
		/////////////////////		EPnP
		/////////////////////////////////////////////////////////////////
		gettimeofday(&t1,NULL);
		int num_tracked = 0;
		for (int iPoint = 0; iPoint < nFeatures; iPoint++){
			num_tracked += (optical_flow_found_feature[iPoint]);
		}	
		cout<< "    <Tracking> Tracked points:" << num_tracked << endl;

		CvMat *cX = cvCreateMat(num_tracked, 3, CV_32FC1);
		CvMat *cx = cvCreateMat(num_tracked, 2, CV_32FC1);	
		int tracked_id = 0;
		for (int iPoint = 0; iPoint < nFeatures; iPoint++)
			if (optical_flow_found_feature[iPoint]){
				cvSetReal2D(cX, tracked_id , 0, vCorr[iPoint].x);
				cvSetReal2D(cX, tracked_id , 1, vCorr[iPoint].y);
				cvSetReal2D(cX, tracked_id , 2, vCorr[iPoint].z);

				cvSetReal2D(cx, tracked_id , 0, vx[iPoint]);
				cvSetReal2D(cx, tracked_id , 1, vy[iPoint]);
				//cout<<"<Tracking>pnp:"<<cvGetReal2D(cX,tracked_id,0)<<" "<<cvGetReal2D(cX,tracked_id,1)<<" "<<cvGetReal2D(cX,tracked_id,2)<<" "<<cvGetReal2D(cx,tracked_id,0)<<" "<<cvGetReal2D(cx,tracked_id,1)<<endl;
				tracked_id++;
			}
    //cout<<"<Tracking>"<<cvGetReal2D(K,0,0)<<" "<<cvGetReal2D(K,0,1)<<" "<<cvGetReal2D(K,0,2)<<endl;
    //cout<<"<Tracking>"<<cvGetReal2D(K,1,0)<<" "<<cvGetReal2D(K,1,1)<<" "<<cvGetReal2D(K,1,2)<<endl;
    //cout<<"<Tracking>"<<cvGetReal2D(K,2,0)<<" "<<cvGetReal2D(K,2,1)<<" "<<cvGetReal2D(K,2,2)<<endl;
		
		vector<int> vInlier;

		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		//EPNP_ExtrinsicCameraParamEstimation(cX, cx, K, P);	
		//for (int i = 0; i < vCorr.size(); i++)
		//	vInlier.push_back(i);

		PnP_Opencv(cX, cx, K, P, 5, 200, vInlier);
		
		//DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs(cX, cx, K, P, 5, 200, vInlier);	
		
		//////////////////////////////////////////
		//Refinement
		CvMat *cX_ransac = cvCreateMat(vInlier.size(), 3, CV_32FC1);
		CvMat *cx_ransac = cvCreateMat(vInlier.size(), 2, CV_32FC1);

		for (int iPoint = 0; iPoint < vInlier.size(); iPoint++)
		{
			cvSetReal2D(cX_ransac, iPoint, 0, vCorr[vInlier[iPoint]].x);
			cvSetReal2D(cX_ransac, iPoint, 1, vCorr[vInlier[iPoint]].y);
			cvSetReal2D(cX_ransac, iPoint, 2, vCorr[vInlier[iPoint]].z);

			cvSetReal2D(cx_ransac, iPoint, 0, vCorr[vInlier[iPoint]].u);
			cvSetReal2D(cx_ransac, iPoint, 1, vCorr[vInlier[iPoint]].v);
		}
		//EPNP_ExtrinsicCameraParamEstimation(cX_ransac, cx_ransac, K, P);
		
		// if (vP.size() > 0)
			// P = cvCloneMat(vP[vP.size()-1]);

		gettimeofday(&t1,NULL);
		//AbsoluteCameraPoseRefinement_Jacobian(cX_ransac, cx_ransac, P, K, 40);
		gettimeofday(&t2,NULL);
		printTime("    <Tracking> Refinement time: ", t1, t2);
		cvReleaseMat(&cX_ransac);
		cvReleaseMat(&cx_ransac);
		//////////////////////////////////////////

		if (DEBUG_FLOW) {
			CvMat *R = cvCreateMat(3,3,CV_32FC1);
			CvMat *C = cvCreateMat(3,1,CV_32FC1);
			GetCameraParameter(P, K, R, C);

			vector<double> vu, vv, vu1, vv1;
			CvMat *X4 = cvCreateMat(4,1,CV_32FC1);
			CvMat *x4 = cvCreateMat(3,1,CV_32FC1);
			for (int i = 0; i < cX->rows; i++)
			{
				cvSetReal2D(X4, 0, 0, cvGetReal2D(cX, i, 0));
				cvSetReal2D(X4, 1, 0, cvGetReal2D(cX, i, 1));
				cvSetReal2D(X4, 2, 0, cvGetReal2D(cX, i, 2));
				cvSetReal2D(X4, 3, 0, 1);
			
				cvMatMul(P, X4, x4);
				vu.push_back(cvGetReal2D(x4, 0, 0)/cvGetReal2D(x4, 2, 0));
				vv.push_back(cvGetReal2D(x4, 1, 0)/cvGetReal2D(x4, 2, 0));
				
				vu1.push_back(cvGetReal2D(cx, i, 0));
				vv1.push_back(cvGetReal2D(cx, i, 1));

			}
			//vector<double> vu1, vv1;
			//vu1 = vu;
			//vv1 = vv;
			Distortion_Radial(K, invK, k1, vu, vv);
			Distortion_Radial(K, invK, k1, vu1, vv1);
			//Undistortion_Radial(K, invK, k1, vu, vv);

			//for (int i = 0; i < vInlier.size(); i++)
			//{
			//	cv::Point2f p1, p2;
		//		p1.x = vu[vInlier[i]];
		//		p1.y = vv[vInlier[i]];
		//		//line(img_debug, feature1[vInlier[i]], p1, color_blue, 1);
		//		circle(img_debug, feature1[vInlier[i]], 3, color_blue, -1);
		//	}

			for (int i = 0; i < vCorr.size(); i++)
			{
				cv::Point2f p1, p2;
				p1.x = vu[i];
				p1.y = vv[i];
			
				p2.x = vu1[i];
				p2.y = vv1[i];
				line(img_debug, p1, p2, color_blue, 1);
				//circle(img_debug, feature1[vInlier[i]], 3, color_blue, -1);
			}

			//Distortion_Radial(K, invK, k1, vx, vy);
			for (int i = 0; i < vx.size(); i++)
			{
				cv::Point2f p1, p2;
				p1.x = vu1[i];
				p1.y = vv1[i];
				//line(img_debug, feature1[i], p1, color_blue, 1);
				circle(img_debug, p1, 5, color_green, -1);
			}


			double origin_x = cvGetReal2D(C, 0, 0) + 5*cvGetReal2D(R, 2, 0);
			double origin_y = cvGetReal2D(C, 1, 0) + 5*cvGetReal2D(R, 2, 1);
			double origin_z = cvGetReal2D(C, 2, 0) + 5*cvGetReal2D(R, 2, 2);

			double X_x = origin_x + 1;
			double X_y = origin_y + 0;
			double X_z = origin_z + 0;

			double Y_x = origin_x + 0;
			double Y_y = origin_y + 1;
			double Y_z = origin_z + 0;

			double Z_x = origin_x + 0;
			double Z_y = origin_y + 0;
			double Z_z = origin_z + 1;

			CvMat *O_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(O_, 0, 0, origin_x);
			cvSetReal2D(O_, 1, 0, origin_y);
			cvSetReal2D(O_, 2, 0, origin_z);
			cvSetReal2D(O_, 3, 0, 1);

			CvMat *X_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(X_, 0, 0, X_x);
			cvSetReal2D(X_, 1, 0, X_y);
			cvSetReal2D(X_, 2, 0, X_z);
			cvSetReal2D(X_, 3, 0, 1);

			CvMat *Y_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(Y_, 0, 0, Y_x);
			cvSetReal2D(Y_, 1, 0, Y_y);
			cvSetReal2D(Y_, 2, 0, Y_z);
			cvSetReal2D(Y_, 3, 0, 1);

			CvMat *Z_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(Z_, 0, 0, Z_x);
			cvSetReal2D(Z_, 1, 0, Z_y);
			cvSetReal2D(Z_, 2, 0, Z_z);
			cvSetReal2D(Z_, 3, 0, 1);

			CvMat *o_ = cvCreateMat(3,1,CV_32FC1);
			CvMat *x_ = cvCreateMat(3,1,CV_32FC1);
			CvMat *y_ = cvCreateMat(3,1,CV_32FC1);
			CvMat *z_ = cvCreateMat(3,1,CV_32FC1);

			cvMatMul(P, O_, o_);
			cvMatMul(P, X_, x_);
			cvMatMul(P, Y_, y_);
			cvMatMul(P, Z_, z_);

			cv::Point2f op, xp, yp, zp;
			op.x = cvGetReal2D(o_, 0, 0)/cvGetReal2D(o_, 2, 0) + 200;
			op.y = cvGetReal2D(o_, 1, 0)/cvGetReal2D(o_, 2, 0) + 100;

			xp.x = cvGetReal2D(x_, 0, 0)/cvGetReal2D(x_, 2, 0) + 200;
			xp.y = cvGetReal2D(x_, 1, 0)/cvGetReal2D(x_, 2, 0) + 100;

			yp.x = cvGetReal2D(y_, 0, 0)/cvGetReal2D(y_, 2, 0) + 200;
			yp.y = cvGetReal2D(y_, 1, 0)/cvGetReal2D(y_, 2, 0) + 100;

			zp.x = cvGetReal2D(z_, 0, 0)/cvGetReal2D(z_, 2, 0) + 200;
			zp.y = cvGetReal2D(z_, 1, 0)/cvGetReal2D(z_, 2, 0) + 100;

			line(img_debug, op, xp, color_red, 2);
			line(img_debug, op, yp, color_green, 2);
			line(img_debug, op, zp, color_blue, 2);

			cvReleaseMat(&O_);
			cvReleaseMat(&X_);
			cvReleaseMat(&Y_);
			cvReleaseMat(&Z_);

			cvReleaseMat(&o_);
			cvReleaseMat(&x_);
			cvReleaseMat(&y_);
			cvReleaseMat(&z_);
			cvReleaseMat(&C);
			cvReleaseMat(&R);

			char output_file_name[100];
			sprintf(output_file_name, "./result/flow_image_%03d.jpg", iFile);
			imwrite(output_file_name, img_debug);
			
			if (false) {
				char temp[1000];
				sprintf(temp, "./result/flow%04d.jpg", file_id);
				file_id++;
				imwrite(temp, img_debug);
			}
		}

		if (false) {
			//static_cast<unsigned>(optical_flow_found_feature);
			char output_file_name[100];
			sprintf(output_file_name, "./result/cX_%0d.txt", iFile);
			ofstream fout(output_file_name);
			for (int i = 0; i < num_tracked; i++)
			{
				fout << cvGet2D(cX, i, 0).val[0] << " " << cvGet2D(cX, i, 1).val[0] << " " << cvGet2D(cX, i, 2).val[0] << endl;
			}
			fout.close();
		}

		if (vInlier.size() < 20)
		{
			cout << "    <Tracking> No ePNP solution " << vInlier.size() << endl;
			cvReleaseMat(&cX);
			cvReleaseMat(&cx);
			cvReleaseMat(&P);
			continue;
		}

		gettimeofday(&t2,NULL);
		printTime("    <Tracking> Epnp time: ", t1, t2);
		cvReleaseMat(&cX);
		cvReleaseMat(&cx);
		//cout<<"<Tracking>pnp inlier:"<<vInlier.size()<<endl; 
		
		vP[iFile] = P;
		vFrame[iFile] = iFile;
		
		UpdateCameraDataRealTime(cameraFile, P, iFile, K, file_exist);
		if (!file_exist)
			file_exist = true;

		cout<<"    <Tracking> Finish frame:"<<iFile<<endl;
		t_prev = t_curr;
		gettimeofday(&t_curr,NULL);
		printTime("    <Tracking> Finish time: ",t_prev,t_curr); 
	}
	pthread_rwlock_wrlock(&image_lock);
	frameID = -2;
	pthread_rwlock_unlock(&image_lock);
/*
	vector<CvMat *> vP1;
	vector<int> vFrame1;

	for (int i = 0; i < vFrame.size(); i++)
	{
		if (vFrame[i] < 0)
			continue;
		vP1.push_back(vP[i]);
		vFrame1.push_back(vFrame[i]);
	}

	SaveAbsoluteCameraData(cameraFile, vP1, vFrame1, vFilename.size(), K);
*/
}



int main ( int argc, char * * argv )
{	
	// Load data
//	string path = FILE_PATH;
	string middleFolder = INPUT_FOLDER;
	string outputFolder = OUTPUT_FOLDER;				
	// Input file
	string structureFile = path + middleFolder + "structure.txt";
	string descriptorFile = path + middleFolder + "descriptors.txt";
	string infoFile = path + "imagelist.list";
	// Output file
	string outputPath = path + outputFolder;
	string cameraFile_match = path + outputFolder + "camera_matching.txt";
	cameraFile = path + outputFolder + "camera.txt";
	string cameraFile1 = path + outputFolder + "camera_AD.txt";
	string usedSIFTFile = path + outputFolder + "correspondence2D3D.txt";
	string usedSIFTFile_ransac = path + outputFolder + "correspondence2D3D_ransac.txt";
	//mkdir (outputPath.c_str());
	//boost::filesystem::create_directories(outputPath.c_str())20;
	//vector<string> vFilename;
	LoadFileListData(infoFile, vFilename);
	
	// Load calibration data
	ifstream fin_cal;
	string calibfile = path + "calib_fisheye.txt";
	fin_cal.open(calibfile.c_str(), ifstream::in);
	string dummy;
	int im_width, im_height;
	double focal_x, focal_y, princ_x, princ_y, omega;//, k1;
	double distCtrX, distCtrY;
	fin_cal >> dummy >> im_width;
	fin_cal >> dummy >> im_height;
	fin_cal >> dummy >> focal_x;
	fin_cal >> dummy >> focal_y;
	fin_cal >> dummy >> princ_x;
	fin_cal >> dummy >> princ_y;
	fin_cal >> dummy >> omega;
	fin_cal >> dummy >> distCtrX;
	fin_cal >> dummy >> distCtrY;

	//CvMat *K = cvCreateMat(3,3,CV_32FC1);
	K = cvCreateMat(3,3,CV_32FC1);
	cvSetIdentity(K);
	cvSetReal2D(K, 0, 0, focal_x);
	cvSetReal2D(K, 0, 2, princ_x);
	cvSetReal2D(K, 1, 1, focal_y);
	cvSetReal2D(K, 1, 2, princ_y);
	//CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	invK = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K, invK);
	k1 = distCtrX;
	fin_cal.close();
		
	vector<double> vX, vY, vZ;
	vector<int> vID;
	vector<vector<int> > vvDesc;
	LoadStructureData(structureFile, vID, vX, vY, vZ);
	LoadDescriptorData(descriptorFile, vvDesc);
		
	//Retrieve 3D descriptor
	vector<vector<int> > vvDesc_temp;
	for (int iDesc = 0; iDesc < vID.size(); iDesc++)
	{
		vvDesc_temp.push_back(vvDesc[vID[iDesc]]);
	}
	vvDesc = vvDesc_temp;
	vvDesc_temp.clear();

	// Construct 3D KD tree
	cv::Mat descriptors(vvDesc.size(), vvDesc[0].size(), CV_32F);
	for(int i=0; i<descriptors.rows; ++i)
	     for(int j=0; j<descriptors.cols; ++j)
		  descriptors.at<float>(i, j) = vvDesc[i][j];
	std::vector<cv::Mat> tmpDesc(1, descriptors.clone());
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	matcher->add(tmpDesc);
	matcher->train();	
/*	
	vector<std::vector<cv::Mat> > tmpDescs(4);
	vector<cv::Ptr<cv::DescriptorMatcher> > matchers(4);
	for (int i=0; i<4; i++) {
		tmpDescs[i] = vector<cv::Mat>(1, descriptors.clone());
		matchers[i] = cv::DescriptorMatcher::create("FlannBased");
		matchers[i]->add(tmpDescs[i]);
		matchers[i]->train();		
	}
*/
	////////////////////////////////////////////////////////////////////////////
	// Absolute Registration
	//vector<CvMat *> vP;
	//vector<int> vFrame;
	//vector<vector<int> > vvInlier;
	//vP.resize(vFilename.size());
	//vFrame.resize(vFilename.size(), -1);
	//vvInlier.resize(vFilename.size());
	//omp_set_num_threads(50);
	//#pragma omp parallel for

	pthread_create(&tracking_thread, NULL, Optical_Flow, NULL);

	vector<CvMat *> vP;
	vector<int> vFrame;
	vP.resize(vFilename.size());
	vFrame.resize(vFilename.size(), -1);


	//InitCuda(0);
	timeval t1, t2;
	cv::Mat img_prev, img_next;
	int currentID = -1;
	timeval t_prev, t_curr;

	cv::Mat img_debug;
	while(true) {
		cv::Mat img; 

		

		pthread_rwlock_rdlock(&image_lock);
		int frameID_loc = frameID;
		cv::Mat img_temp = current_img.clone();
		
		pthread_rwlock_unlock(&image_lock);

		if (frameID_loc == -2) {
			cout << "<Matching> Thread ends ..." << endl;			
			break;
		}
		
		if (currentID >= frameID_loc) {
			continue;
		}
		currentID = frameID_loc;
		img_temp.convertTo(img, CV_32FC1);

		cout<< "<Matching> Got frame: "<< currentID << ": "<< img.cols <<","<<img.rows << endl;
		//extract feature
		vector<SIFT_Descriptor> vSift_desc;

	  	unsigned int w = img.cols;
	  	unsigned int h = img.rows;

	  	// Perform some initial blurring (if needed)
		cv::GaussianBlur(img, img, cv::Size(5,5), 1.0);
		
	 	// Initial Cuda images and download images to device
	  	CudaImage img_cuda;
	  	img_cuda.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)img.data);
	  	img_cuda.Download();
		
		////////////////////////////////////////////////////////////////////////
	  	//////////////////		Sift extraction
		////////////////////////////////////////////////////////////////////////
		gettimeofday(&t1,NULL);
	  	SiftData siftData;
	  	//float initBlur = 0.0f;
	  	//float thresh = 3.0f;
	  	//InitSiftData(siftData, 1200, true, true);
	  	//ExtractSift(siftData, img_cuda, 3, initBlur, thresh, 2.0f);

	float initBlur = 0.0f;
	float thresh = 2.0f;
	InitSiftData(siftData, 4096, true, true);
	ExtractSift(siftData, img_cuda, 5, initBlur, thresh, 0.0f);

		gettimeofday(&t2,NULL);
		printTime("<Matching> Extraction time: ",t1,t2); 

		vector<double> vx,  vy;
		for (int i = 0; i < siftData.numPts; i++)
		{
			if (siftData.h_data[i].scale > SCALAR) {
				continue;
			}
			vx.push_back(siftData.h_data[i].xpos);
			vy.push_back(siftData.h_data[i].ypos);
		}
		vector<double> vx_d = vx,  vy_d = vy;
		Undistortion_Radial(K, invK, k1, vx, vy);
		for (int i = 0; i < siftData.numPts; i++)
		{
			if (siftData.h_data[i].scale > SCALAR)
				continue;
			SIFT_Descriptor sift_desc;
			sift_desc.x = vx[vSift_desc.size()];
			sift_desc.y = vy[vSift_desc.size()];
			sift_desc.scale = siftData.h_data[i].scale;
			sift_desc.orientation = siftData.h_data[i].orientation/360*M_PI;
			float *sift = siftData.h_data[i].data;
			for (int j = 0; j < 128; j++)
			{
				sift_desc.vDesc.push_back(int((double)sift[j]*1000.0));
			}  
			vSift_desc.push_back(sift_desc);
		}

		FreeSiftData(siftData);
		cout << "<Matching> Sift number: " << vSift_desc.size() << endl;
		
		////////////////////////////////////////////////////////////////////////
	  	//////////////////		FLANN
		////////////////////////////////////////////////////////////////////////	
		gettimeofday(&t1,NULL);
		cv::Mat descs(vSift_desc.size(), 128, CV_32F);
		for(int i=0; i<descs.rows; ++i)
		    for(int j=0; j<descs.cols; ++j)
			  descs.at<float>(i, j) = vSift_desc[i].vDesc[j];

		vector<vector<cv::DMatch> > all_matches;
		matcher->knnMatch(descs, all_matches, 2);
		
		vector<pair<float, float> > distort_feature;
		vector<Correspondence2D3D> vCorr, vCorr_ransac;
		vector<double> vCorr_scales;
		for (int iDesc = 0; iDesc < all_matches.size(); iDesc++)
		{
			//float dist1 = dist[iDesc][0];
			//float dist2 = dist[iDesc][1];
			float dist1 = all_matches[iDesc][0].distance;
			float dist2 = all_matches[iDesc][1].distance;

			if (dist1/dist2 < 0.7)
			{
				Correspondence2D3D corr;
				corr.u = vSift_desc[iDesc].x;
				corr.v = vSift_desc[iDesc].y;

				corr.x = vX[all_matches[iDesc][0].trainIdx];
				corr.y = vY[all_matches[iDesc][0].trainIdx];
				corr.z = vZ[all_matches[iDesc][0].trainIdx];

				corr.id_2D = vSift_desc[iDesc].id;
				corr.id_3D = vID[all_matches[iDesc][0].trainIdx];

				vCorr.push_back(corr);
				distort_feature.push_back(make_pair(vx_d[iDesc], vy_d[iDesc]));
				if (DEBUG_FLOW)
					vCorr_scales.push_back(vSift_desc[iDesc].scale);
			}
		}

		gettimeofday(&t2,NULL);
		printTime("<Matching> Matching time: ",t1,t2);
		
		// plot matching result
		if (DEBUG_KNN) {
			cvtColor(img, img_debug, CV_GRAY2RGB);
			for (int i = 0; i < vSift_desc.size(); i++) {
				cv::Point2f point;
				point.x = vSift_desc[i].x;
				point.y = vSift_desc[i].y;
				circle(img_debug, point, 5, color_blue, 2);
			}
			for (int i = 0; i < vCorr.size(); i++) {
				cv::Point2f point;
				point.x = vCorr[i].u;
				point.y = vCorr[i].v;
				circle(img_debug, point, 5, color_red, 2);
			}
			char output_file_name[100];
			sprintf(output_file_name, "./result/corr_%0d.jpg", currentID);
			imwrite(output_file_name, img_debug);
		}

		if (vCorr.size() < 20)
		{
			cout << "<Matching> No corresondence" << endl;
			pthread_rwlock_rdlock(&image_lock);
			if (frameID==-2){
				pthread_rwlock_unlock(&image_lock);
				break;
			}
			pthread_rwlock_unlock(&image_lock);

			//pthread_rwlock_wrlock(&image_lock);
			//frameID = last_frame;
			//pthread_rwlock_unlock(&image_lock);

			continue;
		}
		cout << "<Matching> Number of correspondences: " << vCorr.size() << endl;    
		
		////////////////////////////////////////////////////////////////////////
	  	//////////////////		EPnP
		////////////////////////////////////////////////////////////////////////
		gettimeofday(&t1,NULL);
		CvMat *cX = cvCreateMat(vCorr.size(), 3, CV_32FC1);
		CvMat *cx = cvCreateMat(vCorr.size(), 2, CV_32FC1);

		for (int iPoint = 0; iPoint < vCorr.size(); iPoint++)
		{
			cvSetReal2D(cX, iPoint, 0, vCorr[iPoint].x);
			cvSetReal2D(cX, iPoint, 1, vCorr[iPoint].y);
			cvSetReal2D(cX, iPoint, 2, vCorr[iPoint].z);

			cvSetReal2D(cx, iPoint, 0, vCorr[iPoint].u);
			cvSetReal2D(cx, iPoint, 1, vCorr[iPoint].v);
			//cout<<"<Matching>pnp:"<<cvGetReal2D(cX,iPoint,0)<<" "<<cvGetReal2D(cX,iPoint,1)<<" "<<cvGetReal2D(cX,iPoint,2)<<" "<<cvGetReal2D(cx,iPoint,0)<<" "<<cvGetReal2D(cx,iPoint,1)<<endl;
		}
		
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
		vector<int> vInlier;
		PnP_Opencv(cX, cx, K, P, 5, 200, vInlier);
		if (vInlier.size() < 20)
		{
			if (DEBUG_FLOW) {
				char output_file_name[100];
				sprintf(output_file_name, "./result/inlier_number_%0d.txt", currentID);
				ofstream fout(output_file_name);
				fout << currentID << " " << vInlier.size() << endl;
				fout.close();
			}

			cout << "<Matching> No ePNP solution " << vInlier.size() << endl;
			cvReleaseMat(&cX);
			cvReleaseMat(&cx);
			cvReleaseMat(&P);

			if (frameID_loc==-2){
				break;
			}

			continue;
		}

		if (DEBUG_FLOW) {
			char output_file_name[100];
			sprintf(output_file_name, "./result/inlier_number_%0d.txt", currentID);
			ofstream fout(output_file_name);
			fout << currentID << " " << vInlier.size() << endl;
			fout.close();
		}
		
		gettimeofday(&t2,NULL);
		printTime("<Matching> Epnp time: ",t1,t2); 
		cvReleaseMat(&cX);
		cvReleaseMat(&cx);	
		cout<<"<Matching> pnp inlier number:"<<vInlier.size()<<endl; 
		//cout<<"<Matching>"<<cvGetReal2D(P,0,0)<<" "<<cvGetReal2D(P,0,1)<<" "<<cvGetReal2D(P,0,2)<<endl;
		//cout<<"<Matching>"<<cvGetReal2D(P,1,0)<<" "<<cvGetReal2D(P,1,1)<<" "<<cvGetReal2D(P,1,2)<<endl;
		//cout<<"<Matching>"<<cvGetReal2D(P,2,0)<<" "<<cvGetReal2D(P,2,1)<<" "<<cvGetReal2D(P,2,2)<<endl;
		//cout<<"<Matching>"<<cvGetReal2D(P,0,3)<<" "<<cvGetReal2D(P,1,3)<<" "<<cvGetReal2D(P,2,3)<<endl;
		
		//////////////////////////////////////////
		//Refinement
		CvMat *cX_ransac = cvCreateMat(vInlier.size(), 3, CV_32FC1);
		CvMat *cx_ransac = cvCreateMat(vInlier.size(), 2, CV_32FC1);

		for (int iPoint = 0; iPoint < vInlier.size(); iPoint++)
		{
			cvSetReal2D(cX_ransac, iPoint, 0, vCorr[vInlier[iPoint]].x);
			cvSetReal2D(cX_ransac, iPoint, 1, vCorr[vInlier[iPoint]].y);
			cvSetReal2D(cX_ransac, iPoint, 2, vCorr[vInlier[iPoint]].z);

			cvSetReal2D(cx_ransac, iPoint, 0, vCorr[vInlier[iPoint]].u);
			cvSetReal2D(cx_ransac, iPoint, 1, vCorr[vInlier[iPoint]].v);
		}
		
		// if (vP.size() > 0)
			// P = cvCloneMat(vP[vP.size()-1]);
		
		gettimeofday(&t1,NULL);
		//AbsoluteCameraPoseRefinement_Jacobian(cX_ransac, cx_ransac, P, K, 80);
		gettimeofday(&t2,NULL);
		printTime("<Matching> Refinement time: ", t1, t2);
		cvReleaseMat(&cX_ransac);
		cvReleaseMat(&cx_ransac);

		//////////////////////////////////////////
		
		
		
		//restore inliner
		pthread_rwlock_wrlock(&image_lock);
		detection_img = img_temp.clone();
		last_corr.resize(vInlier.size());
		last_frame = currentID;
		scales.resize(vInlier.size()); // debug
		for (int iF = 0; iF < vInlier.size(); iF++)
		{
			last_corr[iF] = vCorr[vInlier[iF]];
			last_corr[iF].u = distort_feature[vInlier[iF]].first;
			last_corr[iF].v = distort_feature[vInlier[iF]].second;
			if (DEBUG_FLOW)
				scales[iF] = vCorr_scales[vInlier[iF]];

			if (iF == 0)
				cout << last_corr[iF].u << " " << last_corr[iF].v << endl;
		}
		pthread_rwlock_unlock(&image_lock);
		//sleep(100);

		t_prev = t_curr;
		gettimeofday(&t_curr,NULL);
		printTime("<Matching> Frame time: ",t_prev,t_curr);

		// plot inliers
		if (DEBUG_INLIER) {
			pthread_rwlock_rdlock(&image_lock);
			vector<Correspondence2D3D> corr_temp = last_corr;
			pthread_rwlock_unlock(&image_lock);
			cvtColor(img, img_debug, CV_GRAY2RGB);
			for (int i = 0; i < vCorr.size(); i++) {
				cv::Point2f point;
				point.x = vCorr[i].u;
				point.y = vCorr[i].v;
				circle(img_debug, point, 5, color_blue, 2);
			}
			for (int i = 0; i < vInlier.size(); i++) {
				cv::Point2f point;
				point.x = corr_temp[i].u;
				point.y = corr_temp[i].v;
				circle(img_debug, point, 5, color_red, 2);
			}

			CvMat *R = cvCreateMat(3,3,CV_32FC1);
			CvMat *C = cvCreateMat(3,1,CV_32FC1);
			GetCameraParameter(P, K, R, C);

			double origin_x = cvGetReal2D(C, 0, 0) + 5*cvGetReal2D(R, 2, 0);
			double origin_y = cvGetReal2D(C, 1, 0) + 5*cvGetReal2D(R, 2, 1);
			double origin_z = cvGetReal2D(C, 2, 0) + 5*cvGetReal2D(R, 2, 2);

			double X_x = origin_x + 1;
			double X_y = origin_y + 0;
			double X_z = origin_z + 0;

			double Y_x = origin_x + 0;
			double Y_y = origin_y + 1;
			double Y_z = origin_z + 0;

			double Z_x = origin_x + 0;
			double Z_y = origin_y + 0;
			double Z_z = origin_z + 1;

			CvMat *O_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(O_, 0, 0, origin_x);
			cvSetReal2D(O_, 1, 0, origin_y);
			cvSetReal2D(O_, 2, 0, origin_z);
			cvSetReal2D(O_, 3, 0, 1);

			CvMat *X_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(X_, 0, 0, X_x);
			cvSetReal2D(X_, 1, 0, X_y);
			cvSetReal2D(X_, 2, 0, X_z);
			cvSetReal2D(X_, 3, 0, 1);

			CvMat *Y_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(Y_, 0, 0, Y_x);
			cvSetReal2D(Y_, 1, 0, Y_y);
			cvSetReal2D(Y_, 2, 0, Y_z);
			cvSetReal2D(Y_, 3, 0, 1);

			CvMat *Z_ = cvCreateMat(4,1,CV_32FC1);
			cvSetReal2D(Z_, 0, 0, Z_x);
			cvSetReal2D(Z_, 1, 0, Z_y);
			cvSetReal2D(Z_, 2, 0, Z_z);
			cvSetReal2D(Z_, 3, 0, 1);

			CvMat *o_ = cvCreateMat(3,1,CV_32FC1);
			CvMat *x_ = cvCreateMat(3,1,CV_32FC1);
			CvMat *y_ = cvCreateMat(3,1,CV_32FC1);
			CvMat *z_ = cvCreateMat(3,1,CV_32FC1);

			cvMatMul(P, O_, o_);
			cvMatMul(P, X_, x_);
			cvMatMul(P, Y_, y_);
			cvMatMul(P, Z_, z_);

			cv::Point2f op, xp, yp, zp;
			op.x = cvGetReal2D(o_, 0, 0)/cvGetReal2D(o_, 2, 0) + 200;
			op.y = cvGetReal2D(o_, 1, 0)/cvGetReal2D(o_, 2, 0) + 100;

			xp.x = cvGetReal2D(x_, 0, 0)/cvGetReal2D(x_, 2, 0) + 200;
			xp.y = cvGetReal2D(x_, 1, 0)/cvGetReal2D(x_, 2, 0) + 100;

			yp.x = cvGetReal2D(y_, 0, 0)/cvGetReal2D(y_, 2, 0) + 200;
			yp.y = cvGetReal2D(y_, 1, 0)/cvGetReal2D(y_, 2, 0) + 100;

			zp.x = cvGetReal2D(z_, 0, 0)/cvGetReal2D(z_, 2, 0) + 200;
			zp.y = cvGetReal2D(z_, 1, 0)/cvGetReal2D(z_, 2, 0) + 100;

			line(img_debug, op, xp, color_red, 2);
			line(img_debug, op, yp, color_green, 2);
			line(img_debug, op, zp, color_blue, 2);

			cvReleaseMat(&O_);
			cvReleaseMat(&X_);
			cvReleaseMat(&Y_);
			cvReleaseMat(&Z_);

			cvReleaseMat(&o_);
			cvReleaseMat(&x_);
			cvReleaseMat(&y_);
			cvReleaseMat(&z_);

			char output_file_name[100];
			sprintf(output_file_name, "./result/inlier_%0d.jpg", currentID);
			imwrite(output_file_name, img_debug);

			cvReleaseMat(&C);
			cvReleaseMat(&R);
		}

		

		if (false) {
			char output_file_name[100];
			sprintf(output_file_name, "./result/feature_%0d.txt", currentID);
			ofstream fout(output_file_name);
			for (int i = 0; i < vInlier.size(); i++)
			{
				fout<<vCorr[vInlier[i]].u << " " << vCorr[vInlier[i]].v << " " << vCorr_scales[vInlier[i]]<<endl;
			}
			fout.close();
		}

		vP[currentID] = P;
		vFrame[currentID] = currentID;

		pthread_rwlock_wrlock(&image_lock);
		isSIFT = true;
		pthread_rwlock_unlock(&image_lock);
	}

	vector<CvMat *> vP1;
	vector<int> vFrame1;

	for (int i = 0; i < vFrame.size(); i++)
	{
		if (vFrame[i] < 0)
			continue;
		vP1.push_back(vP[i]);
		vFrame1.push_back(vFrame[i]);
	}

	SaveAbsoluteCameraData(cameraFile_match, vP1, vFrame1, vFilename.size(), K);

	(void) pthread_join(tracking_thread, NULL);
	return 0;
}
