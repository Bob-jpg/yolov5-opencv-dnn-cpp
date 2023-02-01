#include"yolo.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;


Yolo::Yolo()
{

}

void Yolo::readModel(string netPath) {
	this->net = readNet(netPath);
	//cpu
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}


void Yolo::Detect(Mat& SrcImg, int* topX, int* topY, int* width, int* height, int* classID, float* score, int& count) {
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
	//如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
	//blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
	this->net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
	this->net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	std::sort(netOutputImg.begin(), netOutputImg.end(), [](Mat& A, Mat& B) {return A.size[1] > B.size[1]; });     //////////////opencv4.6
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = classes.size() + 5;  //输出的网络宽度是类别数+5
	float* pdata = (float*)netOutputImg[0].data;
	for (int stride = 0; stride < strideSize; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) {	//anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_x; j++) {
					float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
					if (box_score >= boxThreshold) {
						cv::Mat scores(1, classes.size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = (float)max_class_socre;
						if (max_class_socre >= classThreshold) {
							//rect [x,y,w,h]
							float x = pdata[0];  //x
							float y = pdata[1];  //y
							float w = pdata[2];  //w
							float h = pdata[3];  //h
							int left = (x - 0.5 * w) * ratio_w;
							int top = (y - 0.5 * h) * ratio_h;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
						}
					}
					pdata += net_width;//下一行
				}
			}
		}
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
	count=nms_result.size();
	for (int i = 0; i <count; i++) {
		int idx = nms_result[i];
		Rect box = boxes[idx];


		topX[i] = box.x;
		topY[i] = box.y;
		width[i] = box.width;
		height[i] = box.height;
		classID[i] = classIds[idx];
		score[i] = confidences[idx];
		
	}
}

void Yolo::drawPred(Mat& img, int classId, float conf, int left, int top, int right, int bottom, double fps) {
	rectangle(img, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);
	string label = format("%.2f", conf);
	label = this->classes[classId] + ":" + label;

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	string sFps = format("Detect Time: %.2f ms", fps);
	putText(img, sFps, Point(labelSize.height, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
}





