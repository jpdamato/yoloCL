// =================================================================================================
// Project: 
// AIS - Video detector and Tracking
// 
// File information:
// File for testing different configurations
// Institution.... ais.pladema.net
// Author......... Juan D'Amato
// Changed at..... 2019-11-10
// License........ MIT license
// =================================================================================================



#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

#include "mainApps.h"

#include "Yolo/network.h"
#include "blobs/blobsHistory.h"
#include "gpuUtils/yoloopenCL.h"
#include "gpuUtils/cl_utils.h"
#include "gpuUtils/cl_MemHandler.h"
#include "gpuUtils/clVibe.h"

#include "Yolo/opencvUtils.h"

#include "cnnInstance.h"
#include "u_ProcessTime.h"

#include <iostream>
#include <ctype.h>



using namespace std;
using namespace cv;

using namespace trackingLib;




BlobsByFrame* _detectOnFrame(cv::Mat& mM, cv::Mat& mask, int frameNumber)
{

	BlobsByFrame* bFs = new BlobsByFrame();
	bFs->findContours(mM, mask, frameNumber);
	return bFs;
}

int mainTrackAlgs(int argc, char** argv) {
	// show help
	//! [help]
	if (argc < 2) {
		cout <<
			" Usage: tracker <video_name>\n"
			" examples:\n"
			" example_tracking_kcf Bolt/img/%04d.jpg\n"
			" example_tracking_kcf faceocc2.webm\n"
			<< endl;
		return 0;
	}
	//! [help]

	std::string videoFN;
	int vIndex = find_arg(argc, argv, "-v");
	if (vIndex > 0)
	{
		videoFN = argv[vIndex];
	}
	else
	{
		videoFN = "D:\\Resources\\video\\orco\\2019-03-11_18-47-54.mp4";
	}

	VideoCapture cap;
	cap.open(videoFN);

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}
	// declares all required variables
	//! [vars]
	Rect2d roi;
	Mat frame;
	//! [vars]

	// create a tracker object
	//! [create]
	Ptr<Tracker> tracker = TrackerMedianFlow::create();
	//! [create]

	int gpu_index = find_int_arg(argc, argv, "-i", 0);
	int devince_index = 1;

	clUtils::initDevice(gpu_index, 0);
	gpuMemHandler::verifyMemMode();
	
	//! [setvideo]

	// get bounding box
	//! [getframe]
	cap >> frame;
	//! [getframe]
	
	/////////////////////////////////////////
	/////// INIT CNN
	cl_bgs clV;
	clV.initialize("D:\\Sdks\\Yolo\\darknet\\src\\bgsocl_v2.cl", frame, gpu_index, devince_index);

	// perform the tracking process
	printf("Start the tracking process, press ESC to quit.\n");

	cv::Mat frameD, backFrame;

	backFrame = frame.clone();
	BlobsByFrame* bbfs = NULL;

	MultiTracker* multi = new MultiTracker();

	for (;; ) {
		// get frame from the video
		cap >> frame;

		// stop the program if no more images
		if (frame.rows == 0 || frame.cols == 0)
			break;

		// update the tracking result
		//! [update]
		tracker->update(frame, roi);
		//! [update]

		//! [visualization]
		// draw the tracked object
		rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);

		// show image with the tracked object
		imshow("tracker", frame);
		//! [visualization]
		char wk = waitKey(1);
		//quit on ESC button
		if (wk == 'q') break;

		if (wk == 'r')
		{
			roi = selectROI("tracker", frame);
			tracker = TrackerKCF::create();
			tracker->init(frame, roi);
		}
	}
	
	/*
	cv::cvtColor(frame, frameD, CV_BGR2GRAY);
		clV.operate(frameD, backFrame, false);
		clV.getBack(backFrame);

		int pW, pB;
		pixelBWCount(backFrame, pW, pB);
			if (pW > pB * 0.03)


		if (bbfs)
		{
			bbfs->drawOnFrame(frame);
		}
		imshow("tracker", frame);
		imshow("back", backFrame);
		//! [visualization]
		char wk = waitKey(1);
		//quit on ESC button
		if (wk == 'q') break;

		if (wk == 'r')
		{
			//roi = selectROI("tracker", frame);
			//tracker = TrackerKCF::create();
			//tracker->init(frame, roi);
			bbfs = NULL;
		}
	}
	*/
	return 0;
}
