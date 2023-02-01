// OpenCVTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include<opencv2/opencv.hpp>
#include <direct.h>
#include "YoloFastV2.h"
using namespace cv;

static YoloFastV2 yoloModel1;
static YoloFastV2 yoloModel2;
static YoloFastV2 yoloModel3;

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

extern "C" _declspec(dllexport) bool LoadModel(const char* filename)
{
	try
	{
		yoloModel1.InitialModel(filename, 0.3, 0.3, 0.3);
		yoloModel2.InitialModel(filename, 0.3, 0.3, 0.3);
		yoloModel3.InitialModel(filename, 0.3, 0.3, 0.3);
		return true;
	}
	catch (exception ex)
	{
		return false;
	}
	
}


extern "C" _declspec(dllexport) void Detect(unsigned char* data,const int height, const int width, const int stride,DetecResult& result,const int index,float anchors[2][6])
{

	yoloModel1.SetAnchors(anchors);
	yoloModel2.SetAnchors(anchors);
	yoloModel3.SetAnchors(anchors);

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
		for (int i = 0;i < count;i++)
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


extern "C" _declspec(dllexport) void SetAnchors(float* data, const int num)
{
	
			//yoloModel1.SetAnchors(data, num);
			//yoloModel2.SetAnchors(data, num);
			//yoloModel3.SetAnchors(data, num);
		
}


//static std::string CurrentWorkDirectory()
//{
//	char buff[250];
//	_getcwd(buff, 250);
//	std::string directory(buff);
//	return directory;
//}


int main()
{
	LoadModel("JBox.onnx");

	int topX[50];
	int topY[50];
	int bwidth[50];
	int bheight[50];
	float score[50];
	int classID[50];

	cv::Mat img = imread("JB1.jpg");

	int count = 0;
	yoloModel1.Detect(img, topX, topY, bwidth, bheight, classID, score, count);

	waitKey(0);
	destroyAllWindows();
    std::cout << "Hello World!\n";
}