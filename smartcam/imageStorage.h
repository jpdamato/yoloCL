#pragma once

#include <iostream>
//#include <opencv\cv.h>
//#include <opencv\highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

class imageStorage
{
public:
	imageStorage(string path);
	Mat image;
	VideoCapture vc;
	size_t numFrame;

	bool getNextMat();
	static int compressMatToJPEG(Mat& frame, Rect region, ofstream& ofs);
	static int writeToFile(Mat& frame, std::string filename, int objectID, int last);
};