
#ifndef TRACKING_LIB_H
#define TRACKING_LIB_H

#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>


#include <opencv2/opencv.hpp>


#include "contourDetection.h"
#include "../blobs/blobsHistory.h"
#include "eventDetectors.h"

#include "tinyxml.h"

#define M_MOG  1
#define M_MOG2 2
#define OURS 3
#define VIBE 4
#define OURS_PLUS 5
#define UPDATE_FRAME_RATE 4

#define APP_SOURCE_DLL 1
#define APP_SOURCE_EXE 2

#define SOURCE_VIDEO_FILE 1
#define SOURCE_STREAMING 2

#define ACTIVE_BLOBS 1
#define DEAD_BLOBS 2

using namespace trackingLib;

class TiXmlDocument;

class AlgorithmParameters
{
public :

	// GPU Processing
	int deviceProcessingIndex = 0;
	int platformProcessingIndex = 0;
	int gpuWasInitialized = 0;

	int minBlobArea = 500;
	int minBlobHistory = 5;
	int tracking = 1;
	Mat  firstFrame;
	Mat _orig, orig_warped, tmp;
	cv::Size originalFrameSize;
	cv::Size processRes;

	VideoCapture cap;

	int BGS_modelUpdate = 1;
	bool frameLocked = false;

	int datasourceType = SOURCE_VIDEO_FILE;
	std::string frameInfo;
	int ID = 0;
	bool compareBGS = false;
	int idFrame = 0;
	int ommitFrames = 100;
	std::chrono::time_point<std::chrono::system_clock> startDate;
	int step = 1;
	Mat backgroundMask;
	std::vector<string> filesToProcess;
	int attemptingToReconnect = 0;

	int sleepTime = 0;
	int appSource;
	string processingResolutions[5] = { "320x180", "640x360", "1280x720","1920x1080", "original" };
	
	string idSceneDetector;
	std::vector<BlobsDetector*> detectors;

	std::string imageExtension;
	std::string imageFormat;

	string lastMessage;
	/*Por ejemplo de la forma: C:\Temporal*/
	string mediaPath = "";
	/*Viene de la forma: idSmartcam\IdCamera\*/
	string outputDir = "";
	string MODE = "";
	string method = "ours";
	int show = 0;
	int frameRate = 30;
	AlgorithmParameters();
	void clear();

	Mat getBackFrame();
	void setBackFrame(Mat& frame, string info);
	void lock();
	void unlock();
	std::chrono::time_point<std::chrono::system_clock> computeTime(int frameN, int frameRate);

};


void parseParametersBgs(AlgorithmParameters* params, TiXmlElement* root);
void parseParametersDetectors(AlgorithmParameters* params, TiXmlElement* root);


int initialize( AlgorithmParameters* params, Mat &frame, int sourceApp);
bool initModelBGS(Mat imgGray, int& result);
void InitMask(Mat& frame, Mat& imgGray);
bool fileExist(const std::string& name);

char* storeStringInGlobalMem(string filename);

BlobsHistory* getBlobsHistoryInstance();
float relativePositionX(int x);
float relativePositionY(int y);
AlgorithmParameters* getAlgorithmParameters();

bool endsWith(const std::string &mainStr, const std::string &toMatch);
#endif

