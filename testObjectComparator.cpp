#include <thread>
#include <mutex>

#include <time.h>
#include <winsock.h>
#include <vector>
#include <thread>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#pragma warning( disable : 4244)
#pragma warning( disable : 4190)

#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>

#include "blobs/colorClassifier.h"
#include "Yolo/network.h"
#include "blobs/blobsHistory.h"
#include "gpuUtils/yoloopenCL.h"
#include "gpuUtils/cl_utils.h"
#include "gpuUtils/cl_MemHandler.h"
#include "gpuUtils/clVibe.h"
#include "Yolo/opencvUtils.h"
#include "blobs/Export.h"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "cnnInstance.h"
#include "u_ProcessTime.h"

using namespace trackingLib;


Mat getLayerFromNET(network *net, int lIndex)
{
	int lw = net->layers[lIndex].out_w * net->layers[lIndex].out_c;
	int lh = net->layers[lIndex].out_h;
	float* output = net->layers[lIndex].output;

	cv::Mat layerW;
	layerW.create(lw, lh, CV_32F);

	gpuBuffer* c = gpuMemHandler::getBuffer(output);
	gpuMemHandler::ReadBuffer(c, CL_TRUE, 0, true);

	layerW.data = (uchar*)output;
	return layerW.clone();
}

void showLayer(network *net, int lIndex, std::string name)
{
	cv::Mat layerW = getLayerFromNET(net, lIndex);

	double min = -1.0f;
	double max = 1.0f;
	//cv::minMaxIdx(layerW, &min, &max);
	cv::Mat adjMap;
	// expand your range to 0..255. Similar to histEq();
	layerW.convertTo(adjMap, CV_8UC1, 255 / (max - min), -min);

	// this is great. It converts your grayscale image into a tone-mapped one, 
	// much more pleasing for the eye
	// function is found in contrib module, so include contrib.hpp 
	// and link accordingly
	cv::Mat falseColorsMap;
	applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);

	cv::imshow(name, falseColorsMap);

}

cv::Mat stored;


///------- Localizing the best match with minMaxLoc ------------------------------------------------------------------------
///------- tracking by template location --------------------------------------------------------------------------------------------------------
cv::Point trackbyTPL(cv::Mat& img, cv::Mat& mytemplate)
{
	cv::Mat result;
	if (img.channels() == 3)
	{
		cv::cvtColor(img, img, CV_RGB2GRAY);
	}
	if (mytemplate.channels() == 3)
	{
		cv::cvtColor(mytemplate, mytemplate, CV_RGB2GRAY);
	}
	matchTemplate(img, mytemplate, result, CV_TM_SQDIFF_NORMED);
	double minVal, maxVal;
	Point  minLoc, maxLoc, matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	matchLoc = minLoc;

	return matchLoc;
}


// tracking using previous computed layers
cv::Point trackbyCNN(cv::Mat& img, cv::Rect templateR, network* net, bool first)
{
	// set input

	//Resize for network
	int key = 0;
	image _sized;
	float* _imData = NULL;

	mat_to_image(img, _sized);
	net->input = _sized.data;
	net->truth = 0;
	net->train = 0;
	net->delta = 0;

	// predict
	int origLayers = net->n;
	int lIndex = 0;
	net->n = 4;
	int nc = 32;
	// store previous frame

	if (first)
	{
		//stored = getLayerFromNET(net, lIndex);
	}

	// process again
	YoloCL::predictYoloCL(net, 1, 0);

	cv::Mat l0;// = getLayerFromNET(net, lIndex);
	std::vector<cv::Point> matchs;
	cv::Rect r = templateR;

	//cv::imshow("stored", stored);

	cv::imshow("l0", l0);

	cv::Point mean(0, 0);
	int cnt = 0;
	for (int i = 0; i < nc; i++)
	{
		int yoff = i * r.height;
		cv::Mat subImg;
		subImg = l0(cv::Rect(0, yoff, img.cols, img.rows));

		r.y = templateR.y + yoff;
		cv::Mat templateB;
		//templateB = stored(r);

		cv::Point match = trackbyTPL(subImg, templateB);
		mean.x += match.x;
		mean.y += match.y;
		matchs.push_back(match);
		cnt++;


	}
	mean.x = mean.x / cnt;
	mean.y = mean.y / cnt;


	//showLayer(net, 1, "ll");
	//showLayer(net, 2, "l2");
	//showLayer(net, 3, "l3");

	net->n = origLayers;

	return mean;
}

cv::Point trackByFeatures()
{
	cv::Point cvs;

	return cvs;
}

void trackBlobs(cv::Mat& frame1, BlobsByFrame* bb1, int frameNumber, network* net, int skipFrames, int templateAlgorithm)
{
	BlobsByFrame* bbN = new BlobsByFrame();

	startProcess("templateMatchig");
	// use previously computed CNNs
	if (templateAlgorithm == 0)
	{
		for (auto b : bb1->trackBlobs)
		{
			if (b->speed[0] > 0.0)
			{
				cv::Point match = trackbyCNN(frame1, b->getRect(), net, frameNumber % skipFrames == 1) + cv::Point(b->getRect().width / 2, b->getRect().height / 2);
				b->history.push_back(match);

				break;
			}
		}


	}
	// use template matching
	else
		if (templateAlgorithm == 1)
		{

			stored = getLayerFromNET(net, 39);
			cv::imshow("stored", stored);

			for (auto b : bb1->trackBlobs)
			{
				if (b->speed[0] > 0.0)
				{
				//	cv::Point match = trackbyTPL(frame1, b->cutImg) + cv::Point(b->cutImg.cols / 2, b->cutImg.rows / 2);
				//	b->history.push_back(match);
				}
			}
		}
		else
		{

		}
	endProcess("templateMatchig");
}

BlobsByFrame* detectOnFrame(cv::Mat& mM, cv::Mat& mask, int frameNumber)
{

	BlobsByFrame* bFs = new BlobsByFrame();

	bFs->findContours(mM, mask, frameNumber);

	//std::cout << "potential blobs" << bFs->trackBlobs.size() << "\n";

	return bFs;
}

float computeDisplacement(cv::Rect r, cv::Mat& backFrame)
{
	int cnt = 0;

	if (backFrame.rows == 0)
		return 1.0f;

	for (int x = 0; x < r.width; x++)
		for (int y = 0; y < r.height; y++)
		{
			if (backFrame.at<uchar>(y + r.y, x + r.x) > 0)
			{
				cnt++;
			}
		}

	return (1.0f * cnt) / (r.width * r.height);
}



void drawOnFrame(cv::Mat& frame, BlobsByFrame* bFs)
{

	std::vector<Blob*> tBs = bFs->trackBlobs;

	for (int i = 0; i < tBs.size(); i++)
	{
		cv::rectangle(frame, tBs[i]->getRect(), cvScalar(250, 100, 250), 1);
		cv::String s = std::to_string(i) + ":" + tBs[i]->classes[0].first;
		putText(frame, s, tBs[i]->getRect().tl(),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
	}

}


BlobsByFrame* bb1 = NULL;
BlobsByFrame* bb2 = NULL;
Blob* _selectedBlob = NULL;


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;


		if (BlobsHistoryFWD::getInstance(NULL) && (BlobsHistoryFWD::getInstance(NULL)->activeblobs.size() > 0))
		{

			for (int i = 0; i < BlobsHistoryFWD::getInstance(NULL)->activeblobs.size(); i++)
			{
				Blob* b = BlobsHistoryFWD::getInstance(NULL)->activeblobs[i]->getLast();
				cv::Rect2d r = b->getRect();
				if ((x > r.x) && (x < r.x + r.width) &&
					(y > r.y) && (y < r.y + r.height))
				{
					std::cout << "blob " << i << endl;
				//	if (b->cutImg.rows == 0) continue;
				//	cv::rectangle(b->cutImg, cv::Rect(0, 0, b->cutImg.cols, b->cutImg.rows), cv::Scalar(255, 0, 0));
				//	cv::imshow("blob", b->cutImg);
					//CalculatePaletteProportions(b->cutImg, b->colorPalette);


					for (int j = 0; j < 17; j++)
					{
						if (b->colorPalette[j] > 0.025)
						{
							std::cout << COLOR_PALETTE_NAMES[j] << ":" << b->colorPalette[j] << "..";
						}
					}
					std::cout << endl;

					if (_selectedBlob )
					{
						std::cout << " Weight " << weight(_selectedBlob  , b) << "\n";
					}
					else
						_selectedBlob  = b;
					if (event == EVENT_RBUTTONDOWN)
					{
						_selectedBlob = b;
					}

				}
			}
		}
	}


}

void mainObjectMatching(int argc, char **argv)
{
	cv::VideoCapture cap;
	int gpu_index = 0;
	std::string videoFN = "D:\\Resources\\video\\orco\\2019-03-11_18-37-55.mp4";


	std::string _dirEXE = "";


	int vIndex = find_arg(argc, argv, "-v");
	if (vIndex > 0)
	{
		videoFN = argv[vIndex];
	}

	///////////////////
	/// TEST
	//////////////////////////
	BlobsHistory* bb = BlobsHistory::getInstance();

	// Create a VideoCapture object and use camera to capture the video
	cap.open(videoFN);
	// Take 2 frames from video
	cv::Mat frame;

	_dirEXE = "D:\\Sdks\\Yolo\\darknet\\x64\\Release\\";

	gpu_index = find_int_arg(argc, argv, "-i", 0);
	int devince_index = 1;

	clUtils::initDevice(gpu_index, 0);
	gpuMemHandler::verifyMemMode();
	/////////////////////////////////////////
	/////// INIT CNN
	YoloCL::initYoloCL(_dirEXE, 128, gpu_index, devince_index);

	char *cfgfile = argv[2];
	char *weightfile = argv[3];
	network *net = buildNet(_dirEXE, cfgfile, weightfile);

	// Process first frame and discard
	cap >> frame;

	BlobsHistoryFWD* bH = BlobsHistoryFWD::getInstance(net);

	cl_bgs clV;
	clV.initialize("D:\\Sdks\\Yolo\\darknet\\src\\bgsocl_v2.cl", frame, gpu_index, devince_index);
	cv::Mat backFrame;

	frame = GetSquareImage(frame, 608);
	//cv::resize(frame1, frame1, cv::Size(), l0W * 1.0f / frame1.cols, l0H * 1.0f / frame1.rows);

	cv::Mat frame0, frame1, frame2, bckFrame1, bckFrame2;
	// Process first frame 
	cap >> frame;

	//Create a window
	cv::namedWindow("ImageDisplay", 1);

	//set the callback function for any mouse event
	cv::setMouseCallback("ImageDisplay", CallBackFunc, NULL);

	int frameNumber = 0;
	int templateAlgorithm = 2;
	while (true)
	{
		clearTimers();
		startProcess("frame process");
		cv::Mat frameOrig, frameRender;
		cap >> frameOrig;

		frameRender = GetSquareImage(frameOrig, 608);
		frame1 = GetSquareImage(frameOrig, 608);


		if (backFrame.cols == 0)
		{
			backFrame.create(frameOrig.rows, frameOrig.cols, CV_8UC1);
		}


		cv::Mat frameD = frameOrig.clone();
		// extract movement
		clV.operate(frameD, backFrame, false);
		clV.getBack(backFrame);




		/*
		cv::Mat back3;
		cv::erode(back3, back3, Mat(), Point(-1, -1), 2, 1, 1);
		cv::dilate(back3, back3, Mat(), Point(-1, -1), 20, 1, 1);

		cv::cvtColor(backFrame, back3, CV_GRAY2BGR);

		// apply and operation
		cv::bitwise_and(frameRender, back3, frame1);
		*/
		//discard first frame
		bH->findAndAddBlobs(frameOrig, backFrame, frameNumber);

		endProcess("frame process");

		//bb1->drawOnFrame(frameRender);
		bH->drawOnFrame(frameOrig);
		auto start = std::chrono::system_clock::now();
		bH->updateBlobsState(frameNumber, start);
		cv::imshow("ImageDisplay", frameOrig);
		cv::imshow("BS + CNN", frame1);
		cv::imshow("BS", backFrame);

		//cv::imshow("frame 2", frame2);

		cv::waitKey(20);



		//mapper->draw(blobsCoords);

		if (frameNumber % 21 == 0)
		{
			showProcessTime();
			std::cout << "Actual frame " << frameNumber << ".. " << clV.updateStep << "\n";




		}
		frameNumber++;

	}
}