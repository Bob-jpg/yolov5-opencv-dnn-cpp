
#include "yolo.h"
#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>
#include <string>
#include<ctime>


using namespace std;
using namespace cv;
using namespace dnn;

static Yolo yoloModel1;
static Yolo yoloModel2;
static Yolo yoloModel3;


struct DetecResult
{
public:
	int TopX[50];
	int TopY[50];
	int Width[50];
	int Height[50];
	float Score[50];
	int ClassID[50];
	int Count;
};
//C:/Users/86166/source/repos/ConsoleApplication1/ConsoleApplication1/x64/Debug
extern "C" _declspec(dllexport) bool LoadModel(const char* filename)
{
	try
	{
		
		yoloModel1.readModel(filename);
		yoloModel2.readModel(filename);
		yoloModel3.readModel(filename);
		return true;
	}
	catch (exception ex)
	{
		return false;
	}

}


extern "C" _declspec(dllexport) void Detect(unsigned char* data,const int height, const int width, const int stride, DetecResult & result, const int index)
{


	int count = 0;
	try
	{
		cv::Mat img;
		if (stride / width == 3)
			img = cv::Mat(height, width, CV_8UC3, data, stride);
		if (stride / width == 1)
		{
			img = cv::Mat(height, width, CV_8UC1, data, stride);
			cv::cvtColor(img, img, COLOR_GRAY2BGR);
		}
		if (stride / width == 4)
			img = cv::Mat(height, width, CV_8UC4, data, stride);

		int topX[50];
		int topY[50];
		int bwidth[50];
		int bheight[50];
		float score[50];
		int classID[50];

		if (index == 1)
		{
			count = 0;
			yoloModel1.Detect(img, topX, topY, bwidth, bheight, classID, score, count);

		}
		else if (index == 2)
		{
			count = 0;
			yoloModel2.Detect(img, topX, topY, bwidth, bheight, classID, score, count);
		}
		else if (index == 3)
		{
			count = 0;
			yoloModel3.Detect(img, topX, topY, bwidth, bheight, classID, score, count);
		}

		result.Count = count;
		for (int i = 0; i < count; i++)
		{
			result.TopX[i] = topX[i];
			result.TopY[i] = topY[i];
			result.Width[i] = bwidth[i];
			result.Height[i] = bheight[i];
			result.Score[i] = score[i];
			result.ClassID[i] = classID[i];
		}
	}
	catch (exception& e)
	{
		result.Count = count;
	}

}


int main()
{	
	Net net;
	LoadModel( "JBox.onnx");

	int topX[50];
	int topY[50];
	int bwidth[50];
	int bheight[50];
	float score[50];
	int classID[50];

	cv::Mat img = imread("JB1.jpg");

	int count = 0;
	yoloModel1.Detect(img,topX, topY, bwidth, bheight, classID, score, count);

	waitKey(0);
	destroyAllWindows();
	std::cout << "Hello World!\n";

}