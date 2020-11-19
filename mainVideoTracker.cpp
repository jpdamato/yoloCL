

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
#include "gpuUtils/cnnInstance.h"
#include "gpuUtils/cl_MemHandler.h"
#include "gpuUtils/yoloopenCL.h"
#include "gpuUtils/cl_utils.h"
#include "scheduledMultitracker.h"

#include "../include/darknet.h"
#include "../src/yolo_layer.h"
#include "../src/softmax_layer.h"
#include "../src/reorg_layer.h"
#include "../src/region_layer.h"
#include "../src/parser.h"
#include "../src/network.h"
#include "../src/option_list.h"

#include "trackRender.h"



char** __names;

network* buildNet(std::string _dirEXE, std::string cfgfile, std::string weightfile, std::string namesSrc )
{
	network *net = load_network_custom((char*)cfgfile.c_str(), (char*)weightfile.c_str(), 1, 1);
	set_batch_network(net, 1);

	std::string  datacfg = _dirEXE + "cfg/coco.data";
	list *options = read_data_cfg((char*)datacfg.c_str());
	
	if (namesSrc != "")
	{
		__names = get_labels((char*)namesSrc.c_str());
	}
	else
	{
		char *name_list = option_find_str(options, (char*) "names", (char*)"data/names.list");

		std::string snameList = _dirEXE + name_list;
		__names = get_labels((char*)snameList.c_str());
	}
	char* _alphabet = NULL; // load_alphabet();

	return net;

}

void drawBlobsOnFrame(std::vector<trackingLib::Blob*>& tBs, cv::Mat& frame, int nFrame, cv::Scalar blobColor,trackingLib::clMultiTracker* _multiTracker)
{
	RNG rng(12345);


	for (auto b : tBs)
	{
		if (b->active <= 0) continue;
		if (b->classes.size() == 0) continue;

		trackingLib::classToTrack* ctk = _multiTracker->getClass(b->classes[0].second);
		// if class is not been tracked, continue
		if (!ctk || !ctk->active) continue;

		cv::Rect blobRect = b->getRect();

		float length = 0;
		int minFrame = 1000000;
		for (int i = 1; i < b->before.size() - 1; i++)
		{
			trackingLib::Blob* bF = b->before[i];
			if (bF->frameNumber == 0) continue;
			minFrame = min(minFrame, bF->frameNumber);
		}
		//it is not ready to be render
		if (minFrame > nFrame) continue;

		if (b->before.size() > 0)
		{
			for (int i = 1; i < b->before.size() - 1; i++)
			{

				cv::line(frame, b->before[i]->getCenter(), b->before[i + 1]->getCenter(), b->color, 2);

				trackingLib::Blob* bF = b->before[i];
				if (bF->frameNumber == nFrame)
				{
					blobRect = bF->getRect();
					break;
				}
			}
		}
		//int bV = getVariation(b);
		int bV = b->tag;
		if (b->tag > -250)
		{
			cv::rectangle(frame, blobRect, b->color, 5);
		}
		else
		{
			cv::rectangle(frame, blobRect, cv::Scalar(150, 150, 150), 5);
		}
		if (b->contour.size() > 0)
		{
			//Scalar color = Scalar(0, 255, 0);	
			//cv::polylines(frame, b->contour, true, color, 2);
		}

		cv::String s = "";

		if (b->classes.size() > 0)
		{
			s = std::to_string(b->id) + ":" + b->classes[0].second + ":" + to_string(b->life) + ":" + to_string(b->tag);
		}
		else
		{
			s = std::to_string(b->id) + ": DOE " + ":" + to_string(b->life);
		}
		//////////////////////////////////////////////////////////////////////////////////
		// Render TEXT
		cv::Point textPos = blobRect.tl();
		textPos.y -= 20;
		cv::Point textPos2 = blobRect.tl();
		textPos2.x = blobRect.br().x;
		cv::rectangle(frame, textPos, textPos2, b->color, -1);
		int width = (int)blobRect.width;

		putText(frame, s, cv::Point(textPos.x + 5, textPos2.y - 2),
			FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255.0, 255.0, 255.0), 1, CV_AA);


	}
}

std::string getExePath()
{
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}


int testImageDetection(std::string dirEXE)
{
	//////////////////////////////////////////
 /// GPU Initialization
 /////////////////////////////////////////////////
	clUtils::initDevice(gpu_index, 0);
	gpuMemHandler::verifyMemMode();
	YoloCL::setAllocationScheme(1, 2048);
	YoloCL::initYoloCL(dirEXE, 256, gpu_index, 1);

	///////////////////////////////////////////////
	//// CNN PARAMETERS
	//////////////////////////////////////////////
	std::string cfgfile = dirEXE +"CNN\\yolov4.cfg";
	std::string weightfile = dirEXE + "CNN\\yolov4.weights";

	cnnInstance* cnn = new cnnInstance(buildNet(dirEXE, cfgfile, weightfile, ""), 1);
	cnn->names = __names;
	if (!cnn)
	{
		std::cout << " Failed to load CNN Files. Now exit" << "\n";
		return -1;
	}

	cv::VideoCapture cap( "D:\\googleDrive\\Projects\\DataVia\\Videos\\Hospital\\Hospital Consultory Room Contamination.mp4");

	gpu_index = 0;
	
	for (int i = 0; i < 500; i++)
	{
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) break;
		cnn->setFrame(frame);
		cnn->updateInput();
		cnn->predict(false);
		cnn->readResults();

		trackingLib::BlobsByFrame* bb0 = cnn->readResults(true,0.25);

		bb0->drawOnFrame(frame);

		cv::imshow("frame", frame);
		cv::waitKey(25);
	}
}


int main(int argc, char **argv) try
{
	///// File Input
/*
	std::string videoFileIn = argv[1];
	cv::VideoCapture vi("D:\\googleDrive\\Projects\\DataVia\\videos\\2019-01-23_13-43-08.mp4");
	cv::Mat frame;
	vi >> frame;
	cv::VideoWriter* vw = 	new cv::VideoWriter("D:\\googleDrive\\Projects\\DataVia\\videos\\grossery.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 20, frame.size());;

	bool isSaving = false;
	while (!frame.empty())
	{
		//cv::rotate(frame, frame, ROTATE_180);
		if (isSaving)
		{
			
			vw->write(frame);
			cv::circle(frame, cv::Point(50, 50), 10, cv::Scalar(0, 0, 255));
		}
		cv::imshow("video", frame);
		int k = cv::waitKey(40);

		if (k == 's') isSaving = !isSaving;
		if (k == 'q') break;
		vi >> frame;
	}

	vw->release();
	*/

	std::string videoFN = "";
	cv::VideoCapture cap;

	std::string dirEXE = "D:\\Sdks\\YoloV4\\x64\\Release\\";

	if (argc == 1)
	{
		return testImageDetection(dirEXE);
	}

	

	//dirEXE = "D:/Sdks/videodetectorandtracker_v2/Bin/x64/Release/";


	int vIndex = find_arg(argc, argv, (char*)"-v");
	if (vIndex > 0)
	{
		videoFN = argv[vIndex];
	}
	else
	{
		videoFN = "D:\\googleDrive\\Projects\\DataVia\\Videos\\Office Stock FootageUOlRBc_1080p.mp4";
	}

	// Create a VideoCapture object and use camera to capture the video
	if (videoFN == "1")
	{
		cap.open(0);
	}
	else
	  cap.open(videoFN);
	//cap.open(videoFN); // "D:\\resources\\video\\avellaneda y santamarina\\2019-07-09_10-00-00.mp4");
					   //cv::VideoCapture cap("D:\\Resources\\video\\Usina 2019-06\\2019-06-05_00-29-39.mp4"); // D:\\Resources\\video\\Pladema\\2019 - 06 - 27_11 - 00 - 00.mp4");

					   // Check if camera opened successfully
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	gpu_index = find_int_arg(argc, argv, (char*)"-i", 0);

	size_t totalAvailableMem = 2500;


	//////////////////////////////////////////
	 /// GPU Initialization
	 /////////////////////////////////////////////////
	clUtils::initDevice(gpu_index, 0);
	gpuMemHandler::verifyMemMode();
	YoloCL::setAllocationScheme(1, totalAvailableMem);
	YoloCL::initYoloCL(dirEXE, 256, gpu_index, 1);

	///////////////////////////////////////////////
	//// CNN PARAMETERS
	//////////////////////////////////////////////
	//std::string cfgfile = "D:\\Resources\\CNNs\\Yolo\\cross-hands-tiny.cfg";
	//std::string weightfile = "D:\\Resources\\CNNs\\Yolo\\cross-hands-tiny.weights";

	std::string cfgfile = "CNN/yolov4.cfg";
	std::string weightfile = "CNN/yolov4.weights";

    std::string  datacfg = dirEXE + "cfg/coco.data";
	list *options = read_data_cfg((char*)datacfg.c_str());
	char *name_list = option_find_str(options, (char*)"names", (char*)"data/names.list");

	
	cnnInstance* cnn = new cnnInstance(buildNet(dirEXE, cfgfile, weightfile), 1);
	cnn->names = __names;
	if (!cnn)
	{
		std::cout << " Failed to load CNN Files. Now exit" << "\n";
		return -1;
	}
	////////////////////////////////////////////////
		// force 1 prediction
	cv::Mat firstFrameRGB, firstFrameGray;
	if (!getFrameFromVid(cap, firstFrameRGB, firstFrameGray))
	{
		std::cout << " Could not get a frame from videoSource. NOW EXIT" << "\n";
		return -1;
	}


	int nframe = 0;
	if (cnn)
	{
		cnn->setFrame(firstFrameRGB);
		cnn->updateInput();
		cnn->predict(false);
	}

	trackingLib::clMultiTracker* multiTracker = new trackingLib::clMultiTracker();
	multiTracker->cnn = cnn;
	if (find_arg(argc, argv, (char*)"-a") > 0)
	{
		multiTracker->applyPolicy = true;
	}
	else
	{
		multiTracker->applyPolicy = false;
	}
	initBGS(dirEXE, firstFrameRGB, 0, "MOG2");
	trackingLib::BlobsHistory* bH = new trackingLib::BlobsHistory(30, firstFrameRGB.cols, firstFrameRGB.rows, "scene0");

	
	/////////////////////////////////
	/////////////////////////////////////////////////////////////////////////
	// Multi-Thread mode
	std::thread readVideoThread, renderThread, cnnThread, trackerThread, storeThread;
	trackingLib::tlThreadManager threadMgr;
	/////////////////////////////////////////////////////////////////////////////

	
	readVideoThread = std::thread(asyncrchonousREAD, "1", bH, multiTracker, cap, &threadMgr);
	renderThread = std::thread(asyncrchonousRENDERLite, "", firstFrameRGB.cols, firstFrameRGB.rows, bH, multiTracker, &threadMgr);
	trackerThread = std::thread(asyncrchonousTRACK, "2", &threadMgr, multiTracker);
	cnnThread = std::thread(asyncrchonousCNN, cnn, &threadMgr);
	
	/// Finish all threads
	readVideoThread.join();
	renderThread.join();
	cnnThread.join();
	storeThread.join();
	threadMgr.finishAllQueues();
	trackerThread.join();
	

	cap.release();

}
catch (std::system_error & e)
{
	std::cerr << "Exception :: " << e.what();
}