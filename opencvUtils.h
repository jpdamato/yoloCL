#pragma once

#include <opencv2/core.hpp>   // Include OpenCV API
#include <opencv2/highgui.hpp>   // Include OpenCV API
#include <opencv2/imgcodecs.hpp>   // Include OpenCV API
#include <opencv2/imgproc.hpp>   // Include OpenCV API
#include "../src/image.h"


cv::Mat GetSquareImage(const cv::Mat& img, int target_width = 500);
image mat_to_image(cv::Mat& m, image& im);
void cl_draw_detections(cv::Mat& im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);
void accum_detections(cv::Mat& im, int _class, double x, double y, double w, double h);
void update_detections(cv::Mat& im);
float* frameToCNNImage(network *net, cv::Mat& mM);

