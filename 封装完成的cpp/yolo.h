#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace dnn;


class Yolo {
public:
	Yolo();
	void readModel(string netPath);
	void Detect(Mat& SrcImg,  int* topX, int* topY, int* width, int* height, int* classID, float* score, int& count);

private:
	Net net;
	const float netAnchors[3][6] = { { 10,13, 16,30, 33,23 },{ 30,61, 62,45, 59,119 },{ 116,90, 156,198, 373,326 } };

	const int netWidth = 640;   //ONNX图片输入宽度
	const int netHeight = 640;  //ONNX图片输入高度
	const int strideSize = 3;   //stride size

	const float netStride[4] = { 8, 16.0,32,64 };

	float boxThreshold = 0.25;
	float classThreshold = 0.25;

	float nmsThreshold = 0.45;
	float nmsScoreThreshold = boxThreshold * classThreshold;

	std::vector<std::string> classes = { "0", "1"};

private:
	void drawPred(Mat& img, int classId, float conf, int left, int top, int right, int bottom, double fps);
};