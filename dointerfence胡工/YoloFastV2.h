#pragma once

#include <opencv.hpp>

#include <dnn.hpp>
#include <sstream>
#include<iostream>
#include<fstream>
using namespace std;
using namespace cv;
using namespace dnn;


class YoloFastV2
{
public:
	YoloFastV2();
	void InitialModel(string modelPath, float objThreshold, float confThreshold, float nmsThreshlod);
	void Detect(Mat& srcImg);
	void Detect(Mat& srcImg, int* topX, int* topY, int* width, int* height, int* classID, float* score, int& count);
	void VideoDetect();
	void ImageDetect();
	void ImageDetect2();
	void ImageDetect3();
	void SetConfidence(float boxConf, float classConf);
	void  SetAnchors(float anchors[2][6] );

private :
	 float anchors[2][6]= { {31.91,67.56, 36.61,88.01, 39.28,67.42},
							{48.00,68.21, 49.08,91.21, 57.30,70.45} };
	const float stride[3]= { 16.0, 32.0 };
	const int inputWidth = 352;
	const int inputHeight = 352;
	const int numStage = 2;
	const int anchorNum = 3;

	float objThreshold;
	float confThreshold;
	float nmsThreshold;
	vector<string> classes = {"1","2"};
	int num_class;
	Net net;

	TickMeter tm;

private:
	void drawPred(Mat& frame,int classId, float conf, int left, int top, int right, int bottom,double fps);

};
