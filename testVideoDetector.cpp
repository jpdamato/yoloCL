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
// =================================================================================================
#include <thread>
#include <mutex>

#pragma warning( disable : 4244)
#pragma warning( disable : 4190)
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <time.h>
#include <winsock.h>
#include <vector>
#include <thread>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>


#include "Yolo/network.h"
#include "blobs/blobsHistory.h"
#include "gpuUtils/yoloopenCL.h"
#include "gpuUtils/cl_utils.h"
#include "gpuUtils/cl_MemHandler.h"
#include "gpuUtils/clVibe.h"
#include "cnnInstance.h"

#include "u_ProcessTime.h"

#include "Yolo/opencvUtils.h"

#include "opencv2/xfeatures2d.hpp"


#define MAX_FRAME_TO_ANALYZE 500000

#define CNN_INSTANCES 1

//cl_bgs vibe;


std::string dirEXE = "";
cv::Mat frameGray;
cv::Mat backFrame;
cv::Mat heatMap;
bool renderOut = true;
bool computeHeatmap = false;
int jump = 10;

bool useBackGroundSub = false;
std::string outputDir = "";
int blastVersion = 1;

int frameNumber = 0;
double fps = 0;
std::chrono::time_point< std::chrono::steady_clock> last_ReadTime;

trackingLib::BlobsHistory* bb;

cl_bgs clV;


image build_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}


struct SURFDetector
{
	Ptr<Feature2D> surf;
	SURFDetector(double hessian = 800.0)
	{
		surf = cv::xfeatures2d::SURF::create(hessian);
	}
	
	void operator()(const cv::Mat& in, const cv::Mat& mask, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors, bool useProvided = false)
	{
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

trackingLib::BlobsByFrame* converToBlobs(std::vector<object_detected> detections, cv::Mat input, int frameN)
{

	trackingLib::BlobsByFrame* blobs = new trackingLib::BlobsByFrame();
	//instantiate detectors/matchers
	SURFDetector  surf;

	//declare input/output
	std::vector<KeyPoint> keypoints1, keypoints2;
	
	for (int i = 0; i < detections.size(); i++)
	{
		trackingLib::Blob* b = trackingLib::Blob::getInstance(frameN);

		box f = detections[i].reg;
		cv::Rect r(f.x * input.cols, f.y * input.rows, f.w* input.cols, f.h* input.rows);
		b->updatePos(r, input.size());	

		blobs->trackBlobs.push_back(b);

	}

	blobs->enqueue(input, input, "", 0, 10);
	
	return blobs;
}





void detectOnFrame(std::vector<cnnInstance*> &cnns)
{
	
	last_ReadTime = std::chrono::steady_clock::now();

	int evalFrame = 1;

	auto lastProcessTime = std::chrono::steady_clock::now();

	

	while (1)
	{
		clearTimers();


		float lapsedTime = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - lastProcessTime).count();
		if (lapsedTime < jump) continue;
		evalFrame++;
		startProcess("CNN");
		for (auto& cnn : cnns)
		{
			if (cnn->mM.cols == 0)
			{
				continue;
			}

			cnn->mtxImg.lock();
			if (useBackGroundSub)
			{

				cv::Mat frameD = cnn->mM.clone();
				if (cnn->foreGround.cols == 0)
				{ 
					cnn->foreGround = frameD.clone();
					cv::cvtColor(cnn->foreGround, cnn->foreGround, CV_BGR2GRAY);
				}

				clV.operate(frameD, cnn->foreGround, false);
				clV.getBack(cnn->foreGround);

				erode(cnn->foreGround, cnn->foreGround, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1), 2, BORDER_CONSTANT , 1);
				dilate(cnn->foreGround, cnn->foreGround, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1), 15, BORDER_CONSTANT, 1);

				cv::Rect roi(0, 0, cnn->foreGround.cols, 50);

				rectangle(cnn->foreGround, roi, Scalar(0, 0, 0),-1);
				cv::Mat back3;
				cvtColor(cnn->foreGround, back3, CV_GRAY2BGR);

				
				// apply and operation
				bitwise_and(cnn->mM, back3, cnn->mM);

			}
			cnn->updateInput();
			cnn->mtxImg.unlock();

			auto actualTime = std::chrono::steady_clock::now();
			
			cnn->predict(useBackGroundSub);

			float layerProcTime = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - actualTime).count();
			float gpuUsedMem = (float)gpuMemHandler::getMaxUsedMem();;
			//clUtils::exportCSV(dirEXE + "yoloData.csv", { layerProcTime ,gpuUsedMem }, cnn->nframe);

			cnn->mtxM.lock();
			cnn->readResults();
			cnn->mtxM.unlock();

			///
			layer l = cnn->net->layers[cnn->net->n - 1];
			std::vector<object_detected> detections = YoloCL::makeDetections(cnn->dets, cnn->nboxes, 0.45, getnames(), l.classes);

			if ((outputDir != "") && detections.size() > 0)
			{
				std::stringstream ss;
				ss << outputDir << "frame_" << cnn->nframe << ".jpg";
				
				cv::Mat exporterJPG;
				cv::cvtColor(cnn->frame, exporterJPG, CV_BGR2RGB);
				

				for (auto od : detections)
				{
					std::vector<std::string> rows;
					rows.push_back(std::to_string(cnn->nframe));
					//rectangle(exporterJPG,Rect( (od.reg.x - od.reg.w/2)  * exporterJPG.cols, (od.reg.y - od.reg.h/2) * exporterJPG.rows,
					//	                     od.reg.w * exporterJPG.cols,od.reg.h * exporterJPG.rows),cv::Scalar(255,0,0));
					rows.push_back(std::to_string(od.reg.x));
					rows.push_back(std::to_string(od.reg.y));
					rows.push_back(std::to_string(od.reg.w));
					rows.push_back(std::to_string(od.reg.h));
					for (int j = 0; j < od.classes.size(); j++)
					{
						rows.push_back(od.classes[j].first);
						rows.push_back(to_string(od.classes[j].second));

					}

					clUtils::exportCSV(outputDir + "videoClassifications.csv", rows, frameNumber);
				}

				cv::imwrite(ss.str(), exporterJPG);

			}
			if (cnn->nframe > MAX_FRAME_TO_ANALYZE) break;

			cnn->nframe++;
		}

		double cnnTime = endProcess("CNN");
		fps = 1000 / cnnTime;
		
		auto actualTime = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<chrono::seconds>(actualTime - last_ReadTime).count() > 2)
		{
			last_ReadTime = actualTime;
			showProcessTime();
		}

		lastProcessTime = std::chrono::steady_clock::now();

		

	}

}


int nFrame = 0;
bool detecting = true;
cv::VideoCapture cap;
int videoMode = 1;
int lastFrameNumber = 1;
bool applyROI = false;

cv::Mat cropFrame(cv::Mat& frame, cv::Rect roi)
{
	cv::Mat cropped = frame(roi);
	cv::Mat cropRes;
	cv::resize(cropped, cropRes, cv::Size(), frame.cols / roi.width, frame.rows / roi.height);
	return cropRes;
}

cv::Mat getNextFrame()
{
	cv::Mat frame;
	if (videoMode == 0)
	{
		frame = cv::imread("D:\\Resources\\video\\screens2\\1 (" + std::to_string(frameNumber)+").jpg");
	}
	else
	{
		cap >> frame;

		cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
		frameNumber++;
	}

	return frame;
}



void renderVideo(std::vector<cnnInstance*> &cnns)
{
	cv::Mat frameM;
	
	bool play = true;

	cv::Mat lastFrame;


	lastFrame = getNextFrame();

	int evalFrame = 0;


	while (1)
	{
		cv::Mat m;

		if (play)
		{

			frameM = getNextFrame();
			if (frameM.empty()) break;
			lastFrame = frameM.clone();
		}
		else
		{
			frameM = lastFrame.clone();
			
		}

	

		
		for (auto& cnn : cnns)
		{

			// If the frame is empty, break immediately
			if (frameM.empty())
				break;

			cnn->mtxImg.lock();
			m = frameM.clone();
			cv::cvtColor(m, cnn->mM, CV_BGR2RGB);
			cnn->mtxImg.unlock();

			cnn->mtxM.lock();
			if (cnn->dets)
			{
				// 
				//std::cout << " exporting frame" << frameNumber << "\n";
				
				
				if (computeHeatmap)
				{
					if (heatMap.cols == 0)
					{
						heatMap.create(cv::Size(frameM.cols, frameM.rows), CV_32F);
					}

					update_detections(heatMap);

					for (int i = 0; i < cnn->nboxes; ++i)
					{
						box bb = cnn->dets[i].bbox;
						accum_detections(heatMap, cnn->dets[i].classes, bb.x, bb.y, bb.w, bb.h);
					}
				}
				layer l = cnn->net->layers[cnn->net->n - 1];
			
				cl_draw_detections(m, cnn->dets, cnn->nboxes, 0.45, getnames(), getalphabet(), l.classes);

				cv::putText(m, "#:"+std::to_string(cnn->nframe), cv::Point(50, 30), 1, 1.5,cv::Scalar(0,120,255),3);
				cv::putText(m, "blast:" + std::to_string(blastVersion), cv::Point(50, 50), 1, 1.5, cv::Scalar(0, 120, 255), 3);
				cv::putText(m, "fps:"+std::to_string(fps), cv::Point(50, 70), 1, 1.5, cv::Scalar(0, 120, 255), 3);

				

				
			}

			if (heatMap.cols)
			{
				double min = 0.0;
				double max = 300.0;
				//cv::minMaxIdx(heatMap, &min, &max);
				cv::Mat adjMap;
				// Histogram Equalization
				float scale = 255 / (max - min);
				heatMap.convertTo(adjMap, CV_8UC1, scale, -min*scale);

				// this is great. It converts your grayscale image into a tone-mapped one, 
				// much more pleasing for the eye
				// function is found in contrib module, so include contrib.hpp 
				// and link accordingly
				cv::Mat falseColorsMap;
				applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);

				//cv::imshow("HeatMap", falseColorsMap);

				//Apply thresholding
				//cv::threshold(adjMap, adjMap, 10, 255, cv::THRESH_BINARY);

				//cv::imshow("mask", adjMap);
				//cv::cvtColor(adjMap, adjMap, CV_GRAY2BGR);

				//cv::bitwise_and(adjMap, falseColorsMap, falseColorsMap);
				/*
				for (int i = 0; i < m.cols;i++)
					for (int j = 0; j < m.rows; j++)
					{
						int index = (j * m.cols + i) ;
						if (heatMap.at<float>(j, i) > 0.05)
						{
							float w = 0.5;
							m.data[index * 3 + 0] = m.data[index * 3 + 0] * (1-w) + falseColorsMap.data[index * 3+0] * w;
							m.data[index * 3 + 1] = m.data[index * 3 + 1] * (1 - w) + falseColorsMap.data[index * 3+1] * w;
							m.data[index * 3 + 2] = m.data[index * 3 + 2] * (1 - w) + falseColorsMap.data[index * 3+2] * w;
						}
					}

					*/
				//cv::addWeighted(m, 0.8, falseColorsMap, 0.2, 0, m);
				cv::add(frameM, falseColorsMap, frameM);
				if (renderOut)
				{
					cv::imshow("HeatMap", frameM);

					cv::imshow("predictions", m);
				}

			}
			else
			{
				if (renderOut)
				{
					cv::imshow("predictions", m);
				}
			}
			if (cnn->squared.rows)
			{
				//cv::imshow("region" + std::to_string(cnn->id), cnn->squared);
			}

			if (useBackGroundSub && cnn->foreGround.cols > 0)
			{
				cv::imshow("bgs", cnn->foreGround);
			}

			cnn->mtxM.unlock();

			int key = cv::waitKey(100);

		//	if (key == 'Y') yoloMode = !yoloMode;
			if (key == 'B') useBackGroundSub = !useBackGroundSub;
			if (key == 'O') blastVersion = (blastVersion + 1) % 3;
			if (key == 'P') 	{ 				play = !play;			}

			if (key == 'r') 			{ 				applyROI = !applyROI;				
			}
			//if (key == 'n') frameNumber++;
			//if (key == 'm') frameNumber--;
			if (key == 'v')
			{
				visualize_network(cnn->net,cnn->net->n-2);
				play = !play;
			}
			if (cnn->nframe > MAX_FRAME_TO_ANALYZE) return;
		}
	}

}


int mainVideoDetector(int argc, char **argv)
{

	//test_resize("data/bad.jpg");
	//test_box();
	//test_convolutional_layer();
	if (argc < 2) {
		fprintf(stderr, "usage: %s <function>\n", argv[0]);
		return 0;
	}

	dirEXE = ExePath() + "\\";// "D:\\Sdks\\Yolo\\darknet\\x64\\Release\\";

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

	int oIndex = find_arg(argc, argv, "-o");
	if (oIndex > 0)
	{
		outputDir = argv[oIndex];
	}
	else
	{
		outputDir = "";
	}

	bb = trackingLib::BlobsHistory::getInstance();
	// Create a VideoCapture object and use camera to capture the video
	cap.open(videoFN);
	//cap.open(videoFN);
	//cv::VideoCapture cap("D:\\Resources\\video\\Usina 2019-06\\2019-06-05_00-29-39.mp4"); // D:\\Resources\\video\\Pladema\\2019 - 06 - 27_11 - 00 - 00.mp4");

	// Check if camera opened successfully
	if (!cap.isOpened())
	{
		std::cout << "Error opening video stream" << std::endl;
		return -1;
	}

	jump = find_int_arg(argc, argv, "-j", 10);


	gpu_index = find_int_arg(argc, argv, "-i", 0);

	size_t totalAvailableMem = find_int_arg(argc, argv, "-MB", 0);

	if (totalAvailableMem > 0)
	{
		gpuMemHandler::setMaxUsedMem(totalAvailableMem * 1024 * 1024);
	}

	
	if (find_arg(argc, argv, "-nogpu")) {
		gpu_index = -1;
	}

	if (gpu_index > 0)
	{
		gpuMemHandler::setDeviceHasOwnMem(false);
	}

	int width = 1280;
	int height = 720;

	cv::Mat frameGray(width, height, CV_8UC1);

	

#ifdef OPENCL	
	clUtils::initDevice(gpu_index, 0);
	gpuMemHandler::verifyMemMode();

	//vibe.initialize(frameGray);


	clV.initialize("D:\\Sdks\\Yolo\\darknet\\src\\bgsocl_v2.cl", getNextFrame(), gpu_index, 0);

#endif
	YoloCL::initYoloCL(dirEXE, 256, gpu_index, 0);
		
	std::string  datacfg = dirEXE + "cfg/coco.data";
	char *cfgfile = argv[2];
	char *weightfile = argv[3];

	
	
	std::thread detect;
	std::thread render;



	std::vector<cnnInstance*> cnnInstances;

	for (int i = 0; i < CNN_INSTANCES; i++)
	{
		network *net = buildNet(dirEXE,cfgfile, weightfile);

		size_t totalMemSize = 0;

		totalMemSize += _msize(net->input);
		totalMemSize += _msize(net->workspace);
		int biggerLayer = 0;
		size_t biggerlayerSize = 0;

		for (int i = 0; i < net->n; i++)
		{
			layer l = net->layers[i];
			size_t layerSize = 0;
			if (l.output) 	layerSize += _msize(l.output);
			if (l.scales) layerSize += _msize(l.scales);
			if (l.biases) layerSize += _msize(l.biases);
			if (l.weights) layerSize += _msize(l.weights);
			if (l.indexes) layerSize += _msize(l.indexes);
			if (l.rolling_variance) layerSize += _msize(l.rolling_variance);
			if (l.rolling_mean) layerSize += _msize(l.rolling_mean);
		//	if (l.delta) free(l.delta); //not used in this

			totalMemSize += layerSize;
			if (biggerlayerSize < layerSize)
			{
				biggerLayer = i;
				biggerlayerSize = layerSize;
			}

		}
		std::cout << "Bigger layer" << biggerLayer << " Size in MB" << biggerlayerSize / (1024.0*1024.0) << "\n";
		std::cout << "Mean layer size " << (totalMemSize / net->n) / (1024.0*1024.0) << "\n";
		std::cout << "Total req mem in MB" << totalMemSize / (1024.0*1024.0) << "\n";

		set_batch_network(net, 1);
		//srand(2222222);

		cnnInstance* cnn = new cnnInstance(net, cnnInstances.size());

		if (CNN_INSTANCES > 1)
		{
			int x = i % 2;
			int y = i / 2;
			cnn->roi = cv::Rect(x*width / 2, y*height / 2, width / 2, height / 2);
		}
		else
		{
			cnn->roi = cv::Rect(0, 0, width , height );
		}
		cnnInstances.push_back(cnn);
		
	}
	detect = std::thread(detectOnFrame, cnnInstances);
	render = std::thread(renderVideo, cnnInstances);

	detect.join();
	render.join();
	return 0;
}

#include "mainApps.h"

int mode = 2;

int main(int argc, char **argv)
{

	int mode = find_int_arg(argc, argv, "-M", 2);

	if (mode == 0)
		mainVideoDetector(argc, argv);
	else
		if (mode == 1)
			return mainClassify(argc, argv);
		else
			if (mode == 2)
				return mainVibe(argc, argv);
			else
				if (mode == 3)
				{

				}
				//	run_yolo(argc, argv);
				else if (mode == 4)
				{
					mainTrackAlgs(argc, argv);
					
				}
				else if (mode == 5)
				{
					mainOpticalFlow(argc, argv);

				}
				else
				{
					mainObjectMatching(argc, argv);
				}


}