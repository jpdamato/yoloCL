#pragma once

#include <opencv2/core.hpp>   // Include OpenCV API
#include <opencv2/highgui.hpp>   // Include OpenCV API
#include <opencv2/imgcodecs.hpp>   // Include OpenCV API
#include <opencv2/imgproc.hpp>   // Include OpenCV API
#include "opencvUtils.h"



cv::Mat GetSquareImage(const cv::Mat& img, int target_width )
{
	int width = img.cols,
		height = img.rows;

	cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

	int max_dim = (width >= height) ? width : height;
	float scale = ((float)target_width) / max_dim;
	cv::Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}

	cv::resize(img, square(roi), roi.size());

	return square;
}

image mat_to_image(cv::Mat& m, image& im)
{
	IplImage ipl = m;
	IplImage* src = &ipl;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	unsigned char *data = (unsigned char *)src->imageData;
	int step = src->widthStep;
	int i, j, k;

	for (i = 0; i < h; ++i) {
		for (k = 0; k < c; ++k) {
			for (j = 0; j < w; ++j) {
				im.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.;
			}
		}
	}
	return im;
}

void update_detections(cv::Mat& im)
{
	
	for (int i = 0; i < im.cols; i++)
		for (int j = 0; j < im.rows; j++)
		{
			im.at<float>(j, i) *= 0.98f;
		}
}

void accum_detections(cv::Mat& im, int _class, double x, double y, double w, double h)
{
	x = x - w / 2;
	y = y - h / 2;

	int ix = x * im.cols;
	int iy = y * im.rows;
	int iw = w * im.cols;
	int ih = h * im.rows;
	

	for (int i = 0; i < iw; i++)
		for (int j = 0; j < ih; j++)
		{
			double dX = (1.0*i - iw / 2.0) / (0.5*iw);
			double dY = (1.0*j - ih / 2.0) / (0.5*ih);
			if (ix + i < 0 ) continue;
			if (iy + j < 0 ) continue;

			if ((iy + j < im.rows) && (ix + i < im.cols))
			{
				double val = im.at<float>(iy + j, ix + i);
				im.at<float>(iy + j, ix + i) = std::min(val + std::max(0.0, 1.0 - sqrt(dX*dX + dY*dY)) , 255.0);
			}
		}
	

}

float* _imData;
image _sized;

float* frameToCNNImage(network *net, cv::Mat& mM)
{

	// set input
	//Take image data
	if (!_imData)
	{
		_imData = (float*)calloc(net->w * net->h * net->c, sizeof(float));
		_sized = make_image(net->w, net->h, mM.channels());
	}

	if (mM.cols != net->w)
	{
		cv::Mat  squared = GetSquareImage(mM, net->w);
		mat_to_image(squared, _sized);
	}
	else
	{
		mat_to_image(mM, _sized);
	}
	//Resize for network
	return _sized.data;
}


void cl_draw_detections(cv::Mat& im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
	int i, j;

	for (i = 0; i < num; ++i)
	{
		char labelstr[4096] = { 0 };
		int _class = -1;
		float prob = 0.0f;
		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > thresh)
			{
				if (_class < 0) {
					strcat(labelstr, names[j]);
					_class = j;
				}
				else {
					strcat(labelstr, ", ");
					strcat(labelstr, names[j]);
				}

				prob = dets[i].prob[j] * 100;

			}
		}
		if (_class >= 0)
		{
			int width = im.rows * .006;


			//printf("%d %s: %.0f%%\n", i, names[class], prob*100);
			int offset = _class * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = dets[i].bbox;
		

			int left = (b.x - b.w / 2.)*im.cols;
			int right = (b.x + b.w / 2.)*im.cols;
			int top = (b.y - b.h / 2.)*im.rows;
			int bot = (b.y + b.h / 2.)*im.rows;

			if (left < 0) left = 0;
			if (right > im.cols - 1) right = im.cols - 1;
			if (top < 0) top = 0;
			if (bot > im.rows - 1) bot = im.rows - 1;

			std::string labels(labelstr);
			labels += std::to_string(prob);

			cv::rectangle(im, cv::Rect(left, top, right - left, bot - top), cv::Scalar(red * 255, green * 255, blue * 255),2);
			if (alphabet) {

				cv::putText(im, labels, cv::Point(left, top), 3, 1, cv::Scalar(red * 255, green * 255, blue * 255));

			}

		}
	}

}
