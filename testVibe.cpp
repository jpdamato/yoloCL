

#include <thread>
#include <mutex>

#pragma warning( disable : 4244)

#include <opencv2/core.hpp>   // Include OpenCV API
#include <opencv2/highgui.hpp>   // Include OpenCV API
#include <opencv2/imgcodecs.hpp>   // Include OpenCV API
#include <opencv2/imgproc.hpp>   // Include OpenCV API
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>

#include <time.h>
#include <winsock.h>
#include <vector>
#include <thread>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "Yolo/image.h"
#include "gpuUtils/cl_utils.h"
#include "gpuUtils/cl_MemHandler.h"
#include "u_ProcessTime.h"

#include "gpuUtils/clVibe.h"


int mainVibe(int argc, char **argv)
{

	std::string dirEXE = "";
	cv::VideoCapture cap;
			

	dirEXE =  "D:\\Sdks\\Yolo\\darknet\\x64\\Release\\";

	std::string videoFN(argv[4]);
	// Create a VideoCapture object and use camera to capture the video
	cap.open("D:\\Resources\\video\\orco\\2019-03-11_18-37-55.mp4");
	//cap.open(videoFN); // "D:\\resources\\video\\avellaneda y santamarina\\2019-07-09_10-00-00.mp4");
					   //cv::VideoCapture cap("D:\\Resources\\video\\Usina 2019-06\\2019-06-05_00-29-39.mp4"); // D:\\Resources\\video\\Pladema\\2019 - 06 - 27_11 - 00 - 00.mp4");

					   // Check if camera opened successfully
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	gpu_index = find_int_arg(argc, argv, "-i", 0);

	size_t totalAvailableMem = find_int_arg(argc, argv, "-MB", 0);

	if (totalAvailableMem > 0)
	{
		gpuMemHandler::setMaxUsedMem(totalAvailableMem * 1024 * 1024);
	}


	if (find_arg(argc, argv, "-nogpu")) {
		gpu_index = -1;
	}
	clUtils::initDevice(gpu_index, 0);
	gpuMemHandler::verifyMemMode();

	cl_bgs clV;
	
	
	
	cv::Mat frame;
	cap >> frame;
	cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
	clV.initialize("", frame,gpu_index, 0);

	cv::Mat backFrame;
	int nframe = 0;

	startProcess("ALL");

	while (1)
	{
		int64 t0 = cv::getTickCount();
		
		cap >> frame;		
		
		if (frame.empty()) break;
		cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

		if (backFrame.cols == 0)
		  backFrame.create(frame.rows, frame.cols, CV_8UC1);

		clV.operate(frame, backFrame, false);
		clV.getBack(backFrame);
		
		int64 t1 = cv::getTickCount();
		double secs = (t1 - t0) / cv::getTickFrequency();

		cv::putText(frame, "fps:" + to_string(1/secs), cv::Point(50, 50), 1, 2, cv::Scalar(2550, 0, 0));

		cv::imshow("frame", frame);
		cv::imshow("back", backFrame);
		cv::waitKey(1);

		nframe++;

		

		if (nframe % 100 == 0)
		{
			std::cout << "frame " << nframe << "\n";
		
			cv::waitKey(1);
			showProcessTime();
		}

	}

	endProcess("ALL");
	
	cv::waitKey(-1);


}