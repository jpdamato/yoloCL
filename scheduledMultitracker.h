#pragma once
#ifndef MULTI_TRACKER_H
#define MULTI_TRACKER_H

#include <iostream>
#include <fstream>

#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>

#include "gpuUtils/cnnInstance.h"

#include "blobs/blobsHistory.h"
#include "gpuUtils/yoloopenCL.h"
#include "gpuUtils/cl_utils.h"
#include "gpuUtils/cl_MemHandler.h"
#include "gpuUtils/clVibe.h"
#include "opencvUtils.h"


#include "u_ProcessTime.h"
#include <iostream>
#include <ctype.h>
#include <queue>


const int HIGH_FREQ_MOVEMENT = 2;
const int MID_FREQ_MOVEMENT = 1;
const int LOW_FREQ_MOVEMENT = 0;

namespace trackingLib
{
	enum trackerType  { ttBOOSTING, ttMIL, ttKCF, ttTLD,ttMedianFLOW,ttGoTURN,ttMOSSE,ttCSRT , ttTPL ,ttUnassigned};

	
	struct frameData
	{
		cv::Mat rgb;
		cv::Mat gray;
		int nframe;
		std::vector<Blob*> blobs;
	};

	class classToTrack
	{
	public:
		std::string className;
		bool active;
		int trackerId;
		int counter;
		classToTrack(std::string nm, bool state, int trackerID) { this->className = nm; this->active = state;  trackerId = trackerID; counter = 0; }
	};

	class FrameQueue
	{
	protected :
		int _qsize;
	public:
		bool active;
		FrameQueue();
		~FrameQueue();

		// Add an element to the queue.
		void queue(cv::Mat &rgb, cv::Mat &gray, int nframe, std::vector<Blob*>& bs);

		// Add an element to the queue.
		void queue(cv::Mat &rgb, cv::Mat &gray, int nframe);

		int queue_size();

		int firstFrame();
		// Get the "front"-element.
		// If the queue is empty, wait till a element is avaiable.
		frameData dequeue(void);

	private:
		std::queue<frameData> qdata;
		mutable std::mutex m;
		std::condition_variable c;
	};

	struct DummyModel : TrackerModel
	{
		virtual void modelUpdateImpl() CV_OVERRIDE {}
		virtual void modelEstimationImpl(const std::vector<Mat>&) CV_OVERRIDE {}
	};

	class TPLTracker : public virtual Tracker
	{
	public:
		cv::Mat mytemplate;
		Rect2d myRect;
		virtual void read(const FileNode& fn) {}
		virtual void write(FileStorage& fs) const {}
		TPLTracker() { isInit = false; }
		static Ptr<TPLTracker> create() { return Ptr<TPLTracker>(new TPLTracker()); }

		bool init(InputArray image, const Rect2d& boundingBox)
		{
			isInit = true;
			return initImpl(image.getMat(), boundingBox);
		}

	protected:

		virtual bool initImpl(const Mat& image, const Rect2d& boundingBox)
		{
			model = makePtr<DummyModel>();
			cv::Mat roi(image(boundingBox));
			mytemplate = roi.clone();
			myRect = boundingBox;
			return true;
		}
		virtual bool updateImpl(const Mat& img, Rect2d& boundingBox)
		{
			cv::Mat result;
			matchTemplate(img, mytemplate, result, CV_TM_SQDIFF_NORMED);
			double minVal, maxVal;
			Point  minLoc, maxLoc, matchLoc;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
			matchLoc = minLoc;

			if (sqrt((minLoc.x - myRect.x) * (minLoc.x - myRect.x) + (minLoc.y - myRect.y)*(minLoc.y - myRect.y)) > 100)
			{
				return false;
			}

			myRect.x = minLoc.x;
			myRect.y = minLoc.y;

			boundingBox = myRect;

			return true;

		}
	};


	class clMultiTracker
	{
	public:

		int windowTime = 2;
		bool applyPolicy = false;
		bool store = false;
		trackerType defaultTraker;
		bool accelerateDeath = false;
		cnnInstance* cnn;
		std::vector<classToTrack*> classesToTrack;
		double alpha = 0.25, beta = 0.10, gamma = 0.2, delta = 0.45;
		int nBins = 3;
		//std::vector<Ptr<Tracker>> trackers1;
		void clear();
		classToTrack* getClass(std::string cn);
		double computePriority(Blob* b, cv::Mat& frame, cv::Mat& mask);
		void update(cv::Mat& frame, cv::Mat& backFrame, std::vector<Blob*>& activeblobs, int nframe);
		void initialize(int nTraker, cv::Mat& frame, std::vector<Blob*>& activeblobs);
	};

	class tlThreadManager
	{
	public:
		bool finishAll;
		FrameQueue inputVideoQ, trackerQ, renderQ, cnnQ;
		void finishAllQueues();
	};

}

int getVariation(trackingLib::Blob* b);
cv::VideoCapture initCapturer(std::string path);
void updateState(std::vector<trackingLib::Blob*>& bbs, int nroFrame);
bool getFrameFromVid(VideoCapture& cap, cv::Mat& frame, cv::Mat& gray);

void asyncrchonousREAD(string idSceneDetector, trackingLib::BlobsHistory* _bH, trackingLib::clMultiTracker* multiT, 
	                    cv::VideoCapture cap, trackingLib::tlThreadManager* tM);
void asyncrchonousCNN(cnnInstance* cnn, trackingLib::tlThreadManager* tM);
void asyncrchonousTRACK(string idSceneDetector, trackingLib::tlThreadManager* tM, trackingLib::clMultiTracker* multiT );
void asyncrchonousNON_CNN(trackingLib::tlThreadManager* tM);

void showTextOnImage(cv::Mat& frame, std::vector<std::string>& textOut);
void updateBGS(cv::Mat& gray, cv::Mat& backFrame);
void initBGS(std::string dirEXE, cv::Mat& frame, int gpu_index, std::string algorithm);

#endif
