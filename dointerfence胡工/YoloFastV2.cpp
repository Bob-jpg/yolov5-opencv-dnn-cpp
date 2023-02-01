#include"YoloFastV2.h"

YoloFastV2::YoloFastV2()
{

}


void YoloFastV2::InitialModel(string modelPath, float objThreshold, float confThreshold, float nmsThreshlod)
{
	this->objThreshold = objThreshold;
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshlod;
	this->net = readNet(modelPath);
}


void YoloFastV2::drawPred(Mat& frame,int classId, float conf, int left, int top, int right, int bottom,double fps)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);
	string label = format("%.2f", conf);
	label = this->classes[classId] + ":" + label;

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);

	putText(frame, label, Point(left, top-5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

	string sFps = format("Detect Time: %.2f ms", fps);
	putText(frame,sFps, Point(labelSize.height, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
}

void YoloFastV2::Detect(Mat& srcImg)
{	
	tm.reset();
	tm.start();
	Mat blob;
	blobFromImage(srcImg, blob, 1 / 255.0, Size(this->inputWidth, this->inputHeight));
	this->net.setInput(blob);
	vector<Mat> outs;
	vector<string> layerNames = this->net.getUnconnectedOutLayersNames();
	this->net.forward(outs, layerNames);

	vector<int> classesIds;
	vector<float> confidences;
	vector<Rect> boxes;

	float ratioh = (float)srcImg.rows / this->inputHeight;
	float ratiow = (float)srcImg.cols / this->inputWidth;

	int nout = this->anchorNum * 5 + this->classes.size(), row_ind = 0;
	float* pdata = (float*)outs[0].data;


	for (int n = 0; n < this->numStage; n++)
	{
		int numGridX = (int)(this->inputWidth / this->stride[n]);
		int numGridY = (int)(this->inputHeight / this->stride[n]);
		for (int i = 0; i < numGridY; i++)
		{
			for (int j = 0; j < numGridX; j++)
			{
				Mat scores = outs[0].row(row_ind).colRange(this->anchorNum * 5, outs[0].cols);
				Point classIdPoint;
				double maxClassScore;

				minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
				for (int q = 0; q < this->anchorNum; q++)
				{
					const float anchorW = this->anchors[n][q * 2];
					const float anchorH = this->anchors[n][q * 2 + 1];

					float boxScore = pdata[4 * this->anchorNum + q];
					if (boxScore > this->objThreshold && maxClassScore > this->confThreshold)
					{
						float cx = (pdata[4 * q] * 2.f - 0.5f + j)*this->stride[n];
						float cy = (pdata[4 * q + 1] * 2.f - 0.5f + i)*this->stride[n];
						float w = powf(pdata[4 * q + 2] * 2.f, 2.f)*anchorW;
						float h = powf(pdata[4 * q + 3] * 2.f, 2.f)*anchorH;

						int left = (cx - 0.5*w)*ratiow;
						int top = (cy - 0.5*h)*ratioh;

						classesIds.push_back(classIdPoint.x);
						confidences.push_back(boxScore*maxClassScore);
						boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
					}
				}
				row_ind++;
				pdata += nout;
			}
		}		
	}	
	tm.stop();
	double fps=tm.getAvgTimeMilli();

	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(srcImg, classesIds[idx], confidences[idx],
			box.x, box.y, box.x + box.width, box.y + box.height,fps);
	}
}


void YoloFastV2::Detect(Mat& srcImg,int *topX,int *topY,int *width,int *height,int *classID,float *score,int& count)
{
	


	tm.reset();
	tm.start();
	Mat blob;
	blobFromImage(srcImg, blob, 1 / 255.0, Size(this->inputWidth, this->inputHeight));
	this->net.setInput(blob);
	vector<Mat> outs;
	vector<string> layerNames = this->net.getUnconnectedOutLayersNames();
	this->net.forward(outs, layerNames);

	vector<int> classesIds;
	vector<float> confidences;
	vector<Rect> boxes;

	float ratioh = (float)srcImg.rows / this->inputHeight;
	float ratiow = (float)srcImg.cols / this->inputWidth;

	int nout = this->anchorNum * 5 + this->classes.size(), row_ind = 0;
	float* pdata = (float*)outs[0].data;


	for (int n = 0; n < this->numStage; n++)
	{
		int numGridX = (int)(this->inputWidth / this->stride[n]);
		int numGridY = (int)(this->inputHeight / this->stride[n]);
		for (int i = 0; i < numGridY; i++)
		{
			for (int j = 0; j < numGridX; j++)
			{
				Mat scores = outs[0].row(row_ind).colRange(this->anchorNum * 5, outs[0].cols);
				Point classIdPoint;
				double maxClassScore;

				minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
				for (int q = 0; q < this->anchorNum; q++)
				{
					const float anchorW = this->anchors[n][q * 2];
					const float anchorH = this->anchors[n][q * 2 + 1];

					float boxScore = pdata[4 * this->anchorNum + q];
					if (boxScore > this->objThreshold && maxClassScore > this->confThreshold)
					{
						float cx = (pdata[4 * q] * 2.f - 0.5f + j) * this->stride[n];
						float cy = (pdata[4 * q + 1] * 2.f - 0.5f + i) * this->stride[n];
						float w = powf(pdata[4 * q + 2] * 2.f, 2.f) * anchorW;
						float h = powf(pdata[4 * q + 3] * 2.f, 2.f) * anchorH;

						int left = (cx - 0.5 * w) * ratiow;
						int top = (cy - 0.5 * h) * ratioh;

						classesIds.push_back(classIdPoint.x);
						confidences.push_back(boxScore * maxClassScore);
						boxes.push_back(Rect(left, top, (int)(w * ratiow), (int)(h * ratioh)));
					}
				}
				row_ind++;
				pdata += nout;
			}
		}
	}
	tm.stop();
	double fps = tm.getAvgTimeMilli();

	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	
	count = indices.size();
	for (size_t i = 0; i < count; ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		topX[i] = box.x;
		topY[i] = box.y;
		width[i] = box.width;
		height[i] = box.height;
		classID[i] = classesIds[idx];
		score[i] = confidences[idx];

		//this->drawPred(srcImg, classesIds[idx], confidences[idx],
		//	box.x, box.y, box.x + box.width, box.y + box.height, fps);
	}
	//imwrite("result.jpg", srcImg);
}


void YoloFastV2::SetConfidence(float boxConf, float classConf)
{
	this->confThreshold = classConf;
	this->objThreshold = boxConf;
}


void YoloFastV2::SetAnchors(float anchors[2][6])
{
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j <6; j++)
		{
			anchors[i][j] = anchors[i][j];
		}
	}
	
}



void YoloFastV2::ImageDetect()
{
	//VideoCapture capture;
	//capture.open(0);
	namedWindow("ImageDetectResult");
	while (1)
	{
		Mat frame=imread("JB1.jpg");	
		Detect(frame);
		imshow("ImageDetectResult", frame);
		waitKey(1);
	}
}

void YoloFastV2::ImageDetect2()
{
	//VideoCapture capture;
	//capture.open(0);
	namedWindow("ImageDetectResult2");
	while (1)
	{
		Mat frame = imread("dog.jpg");
		Detect(frame);
		imshow("ImageDetectResult2", frame);
		waitKey(1);
	}
}

void YoloFastV2::ImageDetect3()
{
	//VideoCapture capture;
	//capture.open(0);
	namedWindow("ImageDetectResult3");
	while (1)
	{
		Mat frame = imread("dog.jpg");
		Detect(frame);
		imshow("ImageDetectResult3", frame);
		waitKey(1);
	}
}


void YoloFastV2::VideoDetect()
{
	VideoCapture capture;
	capture.open(0);
	namedWindow("ShowResult");
	while (1)
	{
		Mat frame;
		capture >> frame;
		Detect(frame);
		imshow("ShowResult", frame);
		waitKey(1);
	}
}