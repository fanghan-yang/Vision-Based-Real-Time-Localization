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
#define DEBUG 1

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
vector<Correspondence2D3D> last_corr;
vector<double> scales;
cv::Scalar color_red = cv::Scalar(0, 0, 255);
cv::Scalar color_green = cv::Scalar(0, 255, 0);
cv::Scalar color_blue = cv::Scalar(255, 0, 0);
cv::Scalar color_yellow = cv::Scalar(255, 255, 0);

// pthread
pthread_rwlock_t image_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_rwlock_t feature_lock = PTHREAD_RWLOCK_INITIALIZER;
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

/*
void Undistortion(double omega, double DistCtrX, double DistCtrY, vector<double> &vx,  vector<double> &vy)
{
  if (abs(omega) < 1e-6)
    return;
	for (int iPoint = 0; iPoint < vx.size(); iPoint++)
	{
		double x = vx[iPoint]-DistCtrX;
		double y = vy[iPoint]-DistCtrY;
		double r_d = sqrt(x*x+y*y);
		double r_u = tan(r_d*omega)/2/tan(omega/2); 
		double x_u = r_u/r_d*x;
		double y_u = r_u/r_d*y;
		vx[iPoint] = x_u+DistCtrX;
		vy[iPoint] = y_u+DistCtrY;
	}
}
*/

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

void printTime(string text, timeval t1, timeval t2){
	time_t sec = t2.tv_sec-t1.tv_sec;
	suseconds_t usec = t2.tv_usec-t1.tv_usec;
	cout<<text<<(sec*1000000+usec)/1000000.0<<"s"<<endl;
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
	for (int iFile = 0; iFile < vFilename.size(); iFile++)
	{
		bool new_feature_flag = false;
		cv::imread(path+vFilename[iFile], 0).convertTo(img_curr, CV_8U);

		pthread_rwlock_wrlock(&image_lock);
		current_img = img_curr.clone();
		pthread_rwlock_unlock(&image_lock);

		pthread_rwlock_rdlock(&feature_lock);
		cout << "  <flow> Load file " << iFile << endl;
		if (/*last_frame==frameID*/isSIFT){	
			new_feature_flag = true;
			registered_frame = last_frame;
			vCorr = last_corr;
			if (DEBUG)
				vCorr_scales = scales;
			nFeatures = vCorr.size();
			feature0.resize(nFeatures);
			for (int iF = 0; iF < nFeatures; iF++)
	  		{
	  			feature0[iF].x = vCorr[iF].u;
	  			feature0[iF].y = vCorr[iF].v;
	  		}
			
			feature1 = feature0;
			isInitialized = true;
/*
			char output_file_name[100];
			sprintf(output_file_name, "./result/match_%4d_%4d.bmp", frameID, iFile);
*/			
			
			img_prev = img_curr.clone();
			//img_curr.convertTo(current_img, CV_32FC1);
			
			cout<< "    <Tracking> Got frame: "<< registered_frame << endl;
/*			
			//debug
			cv::Mat img_debug = img_prev.clone();
			for (int i = 0; i < feature0.size(); i++)
			{
				circle(img_debug, feature0[i],5,color);
			}
			imwrite(output_file_name, img_debug);
*/		}
		pthread_rwlock_unlock(&feature_lock);

		pthread_rwlock_wrlock(&image_lock);
		frameID = iFile;
		pthread_rwlock_unlock(&image_lock);
		
		if (iFile==0)
     			usleep(500000);

		if (!isInitialized)
			continue;

		pthread_rwlock_wrlock(&feature_lock);
		isSIFT = false;
		pthread_rwlock_unlock(&feature_lock);

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

		cv::Size optical_flow_window(30, 30);
		
		if (iFile == 0 || registered_frame == -1) {
			img_prev = img_curr.clone();
			continue;
		}			
		cout << "    <Tracking> Flow feature num: " << nFeatures << endl;
		gettimeofday(&t1, NULL);
		//img_next = current_img.clone();
		
		if (false) {
			char output_file_name[100];
			sprintf(output_file_name, "./result/feature_before_flow_%0d.txt", iFile);
			ofstream fout(output_file_name);
			for (int i = 0; i < feature0.size(); i++)
			{
				fout << "( " << feature0[i].x << " " << feature0[i].y << " ) ( " <<feature1[i].x << " " << feature1[i].y << " )" << endl;
			}
			fout.close();
		}		
		
		cv::calcOpticalFlowPyrLK(img_prev, img_curr, feature0, feature1,
					 optical_flow_found_feature, 
					 optical_flow_feature_error,
					 optical_flow_window, 15,
					 optical_flow_termination_criteria,
					 cv::OPTFLOW_USE_INITIAL_FLOW);
		if (false) {
			//static_cast<unsigned>(optical_flow_found_feature);
			char output_file_name[100];
			sprintf(output_file_name, "./result/feature_after_flow_%0d.txt", iFile);
			ofstream fout(output_file_name);
			for (int i = 0; i < optical_flow_found_feature.size(); i++)
			{
				fout << (int)optical_flow_found_feature[i] << " ( " << feature1[i].x << " " << feature1[i].y << " ) error: " << optical_flow_feature_error[i] << endl;
			}
			fout.close();
		}

		if (DEBUG) {
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
		//cout<<iFile<<endl;
		//for (int i = 0; i < feature0.size(); i++)
		//{
		//	cout<<feature0[i].x<<" "<<feature0[i].y<<endl;
		//}

		// plot flow before undistortion
		if (DEBUG)
			cvtColor(img_curr, img_debug, CV_GRAY2RGB);
		for (int i = 0; i < feature1.size(); i++)
		{
			vx[i] = feature1[i].x;
			vy[i] = feature1[i].y;

			// plot flow tracking
			if (DEBUG) {
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

		if (DEBUG) {
			char output_file_name[100];
			sprintf(output_file_name, "./result/flow_image_%0d.bmp", iFile);
			imwrite(output_file_name, img_debug);
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
		
		CvMat *P = cvCreateMat(3,4,CV_32FC1);
//		EPNP_ExtrinsicCameraParamEstimation(cX, cx, K, P);	

		vector<int> vInlier;
		DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs(cX, cx, K, P, 10, 50, vInlier);	

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
		
		//update corr
/*		vector<Correspondence2D3D> next_corr(vCorr.size());
		for (int iF = 0; iF < vCorr.size(); iF++)
		{
			next_corr[iF] = vCorr[iF];
			next_corr[iF].u = feature1[iF].x;
			next_corr[iF].v = feature1[iF].y;
		}
*/
/*
		vector<Correspondence2D3D> next_corr(vInlier.size());
		for (int iF = 0; iF < vInlier.size(); iF++)
		{
			next_corr[iF] = vCorr[vInlier[iF]];
			next_corr[iF].u = feature1[vInlier[iF]].x;
			next_corr[iF].v = feature1[vInlier[iF]].y;
		}
    		vCorr = next_corr;
*/
		//cout<<"<Tracking>"<<cvGetReal2D(P,0,0)<<" "<<cvGetReal2D(P,0,1)<<" "<<cvGetReal2D(P,0,2)<<endl;
		//cout<<"<Tracking>"<<cvGetReal2D(P,1,0)<<" "<<cvGetReal2D(P,1,1)<<" "<<cvGetReal2D(P,1,2)<<endl;
		//cout<<"<Tracking>"<<cvGetReal2D(P,2,0)<<" "<<cvGetReal2D(P,2,1)<<" "<<cvGetReal2D(P,2,2)<<endl;
		//cout<<"<Tracking>"<<cvGetReal2D(P,0,3)<<" "<<cvGetReal2D(P,1,3)<<" "<<cvGetReal2D(P,2,3)<<endl;
		vP[iFile] = P;
		vFrame[iFile] = iFile;

		cout<<"    <Tracking> Finish frame:"<<iFile<<endl;
		t_prev = t_curr;
		gettimeofday(&t_curr,NULL);
		printTime("    <Tracking> Finish time: ",t_prev,t_curr); 
	}
	pthread_rwlock_wrlock(&image_lock);
	frameID = -2;
	pthread_rwlock_unlock(&image_lock);

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
		if (frameID == -2) {
			cout << "<Matching> Thread ends ..." << endl;
			pthread_rwlock_unlock(&image_lock);
			break;
		}
		
		if (currentID >= frameID) {
			pthread_rwlock_unlock(&image_lock);
			continue;
		}
		currentID = frameID;

		current_img.convertTo(img, CV_32FC1);
		pthread_rwlock_unlock(&image_lock);

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
	  	float initBlur = 0.0f;
	  	float thresh = 3.0f;
	  	InitSiftData(siftData, 1200, true, true);
	  	ExtractSift(siftData, img_cuda, 3, initBlur, thresh, 2.0f);
		gettimeofday(&t2,NULL);
		printTime("<Matching> Extraction time: ",t1,t2); 

		vector<double> vx,  vy;
		vector<double> vx_large_scale, vy_large_scale;
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
/*		int num_block = vSift_desc.size()/4;
		if (vSift_desc.size()%4!=0) num_block++;
		vector<cv::Mat> descs(4);
		#pragma omp parallel for
		for (int index = 0; index<4; index++){
		    int max_I;
		    if (num_block>vSift_desc.size()-index*num_block)
			max_I = vSift_desc.size()-index*num_block;
		    else
			max_I = num_block;
		    cv::Mat Desc(max_I, 128, CV_32F);
		    for (int i=0; i<max_I; ++i)
			for(int j=0; j<128; ++j)finish frame:34
			    Desc.at<float>(i, j) = vSift_desc[i+index*num_block].vDesc[j];
		    //descs.push_back(Desc);
		    descs[index] = Desc;
		}
		//cout<<"extract"<<endl;
		//index = 1;last_corr
		//    for (int i=0; i<200; ++i)
		//	for(int j=0; j<128; ++j)
		//	    descs[index].at<float>(i, j) = vSift_desc[i+index*200].vDesc[j];
		vector<vector<vector<cv::DMatch> > > all_matches(4);
		//cout<<"before matching"<<endl;
		#pragma omp parallel for
		for (int i=0; i<4; ++i)
		    matchers[i]->knnMatch(descs[i], all_matches[i], 2);
		//cout<<"after matching"<<endl;

		//Get inliners
		vector<Correspondence2D3D> vCorr, vCorr_ransac;
		for (int i = 0; i < vSift_desc.size(); i++)
		{
			int blockID = i/num_block;
			int iDesc = i%num_block;
			//float dist1 = dist[iDesc][0];
			//float dist2 = dist[iDesc][1];
			//cout<<blockID<<" "<<all_matches[blockID].size()<<" "<<iDesc<<endl;
			float dist1 = all_matches[blockID][iDesc][0].distance;
			float dist2 = all_matches[blockID][iDesc][1].distance;
			//cout<<i<<" "<<all_matches[blockID][iDesc][0].trainIdx<<" "<<all_matches[blockID][iDesc][1].trainIdx<<endl;

			if (dist1/dist2 < 0.7)
			{
				Correspondence2D3D corr;
				corr.u = vSift_desc[i].x;
				corr.v = vSift_desc[i].y;

				corr.x = vX[all_matches[blockID][iDesc][0].trainIdx];
				corr.y = vY[all_matches[blockID][iDesc][0].trainIdx];
						current_corr.resize(vInlier.size());
		for (int iF = 0; iF < nFeatures; iF++)
		{
			current_corr[iF] = vCorr[vInlier[iF]];
		}
corr.z = vZ[all_matches[blockID][iDesc][0].trainIdx];

				corr.id_2D = vSift_desc[i].id;
				corr.vector<cv::Point2f> fid_3D = vID[all_matches[blockID][iDesc][0].trainIdx];

				vCorr.push_back(corr);
			}
		}
*/
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
				if (DEBUG)
					vCorr_scales.push_back(vSift_desc[iDesc].scale);
			}
		}

		gettimeofday(&t2,NULL);
		printTime("<Matching> Matching time: ",t1,t2);
		
		// plot matching result
		if (DEBUG) {
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
			sprintf(output_file_name, "./result/corr_%0d.bmp", currentID);
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
		if (DLT_ExtrinsicCameraParamEstimationWRansac_EPNP_mem_abs(cX, cx, K, P, 10, 100, vInlier) < 20)
		{
			if (DEBUG) {
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

		if (DEBUG) {
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
		
		//restore inliner
		pthread_rwlock_wrlock(&feature_lock);
		last_frame = currentID;
		last_corr.resize(vInlier.size());
		scales.resize(vInlier.size()); // debug
		for (int iF = 0; iF < vInlier.size(); iF++)
		{
			last_corr[iF] = vCorr[vInlier[iF]];
			last_corr[iF].u = distort_feature[vInlier[iF]].first;
			last_corr[iF].v = distort_feature[vInlier[iF]].second;
			if (DEBUG)
				scales[iF] = vCorr_scales[vInlier[iF]];
		}
		pthread_rwlock_unlock(&feature_lock);
		//sleep(100);

		t_prev = t_curr;
		gettimeofday(&t_curr,NULL);
		printTime("<Matching> Frame time: ",t_prev,t_curr);

		// plot inliers
		if (DEBUG) {
			pthread_rwlock_rdlock(&feature_lock);
			vector<Correspondence2D3D> corr_temp = last_corr;
			pthread_rwlock_unlock(&feature_lock);
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
			char output_file_name[100];
			sprintf(output_file_name, "./result/inlier_%0d.bmp", currentID);
			imwrite(output_file_name, img_debug);
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

		pthread_rwlock_wrlock(&feature_lock);
		isSIFT = true;
		pthread_rwlock_unlock(&feature_lock);
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
