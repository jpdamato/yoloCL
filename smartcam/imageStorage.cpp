#include "imageStorage.h"

// #include <jpge.h>

#include <iostream>
#include <sstream>
#include <fstream>


imageStorage::imageStorage(string path)//:vc(path)
{
	vc = VideoCapture(path);
	vc >> image;
	numFrame=0;
}

bool imageStorage::getNextMat(){
	++numFrame;
	return vc.retrieve(image);
}

cv::Point getBestResize(int w, int h)
{
	double rw = std::log(w) / std::log(2);
	double rh = std::log(h) / std::log(2);

	int coef = MAX(1, MIN(MAX(std::ceill(rw), std::ceill(rh)), 9));

	rw = std::pow(2, coef);
	rh = std::pow(2, coef);



	return Point(MIN(1024, (int)rw), MIN(1024, (int)rh));
}

int imageStorage::writeToFile(Mat& frame, std::string filename, int objectID, int last)
{
	

	//cv::Point resize = getBestResize(frame.cols, frame.rows);

	//cv::resize(frame, frame, resize);

	return cv::imwrite(filename, frame);
	// IFDEF JPGE
	//cv::cvtColor(frame, frame, CV_BGR2RGB);
	//jpge::params myParams;
	//myParams.m_quality = 50;
	//return jpge::compress_image_to_jpeg_file(filename.c_str(), frame.cols, frame.rows, frame.channels(), frame.data);
	
}

int imageStorage::compressMatToJPEG(Mat& frame, Rect region, ofstream& ofs)
{
	unsigned char *srcData, *dstData;
	int cmp_len;
	int _width = region.width;
	int _height = region.height;
	int resultBufSize = _width * _height * 3;

	srcData = new unsigned char[resultBufSize];
	dstData = new unsigned char[resultBufSize];

	int index = 0;

	//jpge::params myParams;

	//myParams.m_quality = 50;

	// guardo los pixeles
	for (int y = 0; y <region.height; y++)
		for (int x = 0; x < region.width; x++)
		{
			if (x + region.x >= frame.cols) break;
			if (y + region.y >= frame.rows) break;
			cv::Vec3b pix0 = frame.at<cv::Vec3b>(MAX(0, y + region.y), MAX(0, x + region.x));
			srcData[index] = pix0[0]; index++;
			srcData[index] = pix0[1]; index++;
			srcData[index] = pix0[2]; index++;
		}

	bool temp = NULL;//compress_image_to_jpeg_file_in_memory(dstData, resultBufSize, _width, _height, 3, srcData, myParams);


	ofs.write((char*)&resultBufSize, sizeof(int));
	if (temp)
	{
		ofs.write((char*)dstData, resultBufSize);
	}
	free(srcData);
	free(dstData);
	return resultBufSize;
}