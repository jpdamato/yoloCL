#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>

#include <iostream>
#include <ctype.h>

#include "blobs/blobsHistory.h"
#include "scheduledMultitracker.h"
#include "blobs/Export.h"


using namespace trackingLib;


int maxFRAMES_TO_EVALUATE = 6000000;

BlobsHistory* _bH;
clMultiTracker* _multiTracker;
std::vector<std::string> videoFiles;
int activeVideoIndex = 0;
bool trackAllMovements = false;
std::mutex gpuMutex;

///////////////////////////////////////
// Init BGS
cl_bgs clVibe;
Ptr<BackgroundSubtractor> pBackSub;


std::vector< FrameQueue*> _all_queues;

///////////////////////////////////////////////////////////////////////////////////////
/////////// CAPTURE METHODS
///////////////////////////////////////////////////////////////////////////////////////


std::vector<std::string> get_all_files_names_within_folder(std::string folder)
{
	std::vector<std::string> names;

	/*
	string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(folder + "//" + fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	*/
	return names;
}

cv::VideoCapture initCapturer(std::string path)
{
	
	
	if ((path.substr(path.find_last_of(".") + 1) == "mp4") || ((path.substr(path.find_last_of(".") + 1) == "avi")))
	{
		videoFiles.push_back(path);

	}
	else
	{
		videoFiles = get_all_files_names_within_folder(path);

	}
	VideoCapture cap;
	activeVideoIndex = 0;
	cap.open(videoFiles[activeVideoIndex]);
	return cap;
}

bool getFrameFromVid(cv::VideoCapture& cap, cv::Mat& frame, cv::Mat& gray)
{
	//frame = imread("D:\\Resources\\Andenes\\10 - BOULOGNE_CAM 10 PLAT ASC_2020_05_21_13_13_22.jpg");
	// Cvt to Gray
	//cv::cvtColor(frame, gray, CV_BGR2GRAY);

	//return true;
	cap >> frame;
	if (frame.empty())
	{
		activeVideoIndex++;
		cap.release();
		if (activeVideoIndex < videoFiles.size())
		{
			// could not open file
			if (!cap.open(videoFiles[activeVideoIndex]))
			{
				return false;
			}
			cap.set(CV_CAP_PROP_POS_FRAMES, 0);
			cap >> frame;
		}
		else
		{
			return false;
		}
	}
	
	
	/// Resize
	//if (frame.cols > 1200)
	{

		double sz = std::min(1000.0 / frame.cols, 800.0 / frame.rows);
	//	cv::resize(frame, frame, cv::Size(), 1.0, 1.5);

	}

	if (frame.cols > 1000)
	{
	//	frame = frame(cv::Rect(0, 0, min(frame.cols, frame.rows) - 1, min(frame.cols, frame.rows) - 1));
	}
	// Cvt to Gray
	cv::cvtColor(frame, gray, CV_BGR2GRAY);
	
	return true;
}


/////////////////////////////////////////////////////////
//////////// FrameQueue
/////////////////////////////////////////////////////////
void tlThreadManager::finishAllQueues()
{
	inputVideoQ.active = false;
	trackerQ.active = false;
	renderQ.active = false;
	cnnQ.active = false;
}

FrameQueue::FrameQueue()
{
	active = true;
	_all_queues.push_back(this);
}

FrameQueue::~FrameQueue()
{}

// Add an element to the queue.
void FrameQueue::queue(cv::Mat &rgb, cv::Mat &gray, int nframe, std::vector<Blob*>& bs)
{
	std::lock_guard<std::mutex> lock(m);
	frameData fD;
	fD.rgb = rgb;
	fD.gray = gray;
	fD.nframe = nframe;
	for (auto b : bs)
	{
		fD.blobs.push_back(b);
	}
	qdata.push(fD);
	c.notify_one();
}

// Add an element to the queue.
void FrameQueue::queue(cv::Mat &rgb, cv::Mat &gray, int nframe)
{
	std::lock_guard<std::mutex> lock(m);
	frameData fD;
	fD.rgb = rgb;
	fD.gray = gray;
	fD.nframe = nframe;
	qdata.push(fD);
	_qsize = qdata.size();
	c.notify_one();
}

int FrameQueue::queue_size()
{
	return qdata.size();
}

int FrameQueue::firstFrame()
{
	if (qdata.size() == 0) return -1;

	return qdata.front().nframe;
}
// Get the "front"-element.
// If the queue is empty, wait till a element is avaiable.
frameData FrameQueue::dequeue(void)
{
	std::unique_lock<std::mutex> lock(m);
	while (qdata.empty())
	{
		// release lock as long as the wait and reaquire it afterwards.
		c.wait(lock);
		if (!active)
		{
			break;
		}
	}
	frameData val = qdata.front();
	qdata.pop();
	return val;
}


/////////////////////////////////////////////////////////
//////////// BGS
/////////////////////////////////////////////////////////
void initBGS(std::string dirEXE, cv::Mat& frame, int gpu_index, std::string algorithm)
{
	if (algorithm == "VIBE")
	{
		clVibe.initialize(dirEXE + "bgsocl_v2.cl", frame, gpu_index, 0);
	}
	else
		//create Background Subtractor objects
		if (algorithm == "MOG2")
			pBackSub = createBackgroundSubtractorMOG2();
		else
			pBackSub = createBackgroundSubtractorKNN();

}

void updateBGS(cv::Mat& gray, cv::Mat& backFrame)
{
	if (pBackSub)
	{
		pBackSub->apply(gray, backFrame);

	}
	else
	{
		if (backFrame.cols == 0)
		{
			backFrame = gray.clone();
		}
		
		clVibe.operate(gray, backFrame, false);
		clVibe.getBack(backFrame);
	
	}

	//cv::erode(backFrame, backFrame, Mat(), Point(-1, -1), 1, 1, 1);
	//cv::dilate(backFrame, backFrame, Mat(), Point(-1, -1), 3, 1, 1);

	//fill up line with black
	cv::Rect r(0, 0, backFrame.cols - 1, 25);
	cv::rectangle(backFrame, r, cv::Scalar(0, 0, 0), -1);
	for (int y = 0; y < 15; y++)
		for (int x = 0; x < backFrame.cols - 1; x++)
		{
			backFrame.data[y * backFrame.cols + x] = 0;
		}
}


int getVariation(Blob* b)
{
	if (b->before.size() < 4)
	{
		return HIGH_FREQ_MOVEMENT;
	}
	else
	{
		cv::Point2d cn(0, 0);
		int sz = b->before.size();
		for (int i = 0; i < sz; i++)
		{
			cn.x += b->before[i]->getRect().x;
			cn.y += b->before[i]->getRect().y;
		}

		cn.x /= sz;
		cn.y /= sz;

		cv::Point2d disp(0, 0);

		for (int i = 0; i < sz; i++)
		{
			disp.x += abs(b->before[i]->getRect().x - cn.x);
			disp.y += abs(b->before[i]->getRect().y - cn.y);
		}

		return max(disp.x, disp.y) / sz;
	}
}

/////////////////////////////////////////////////////////
//////////// clMultiTracker
/////////////////////////////////////////////////////////
double clMultiTracker::computePriority(Blob* b, cv::Mat& frame, cv::Mat& mask)
{
	double w0, w1, w2, w3;
	//not active. Discard
	if (!b->active) return -1000.0;
	///////////////////////////////////////
	// check if class has to be tracked
	

	// Classes weight . between 0 and 0.75
	if (b->classes.size() > 0)
	{
		// check if this class is active
		classToTrack* ctk = this->getClass(b->classes[0].second);
		if (!ctk || !ctk->active) return -10000;

		if (b->classes[0].second == "person") w0 = 0.25;
		else if (b->classes[0].second == "bicycle") w0 = 0.45;
		else if (b->classes[0].second == "motorbike") w0 = 0.60;
		else if (b->classes[0].second == "car") w0 = 0.70;
		else if (b->classes[0].second == "truck") w0 = 0.50;
		else if (b->classes[0].second == "bus") w0 = 0.50;
		else if (b->classes[0].second == "bike") w0 = 0.50;
		else if (b->classes[0].second == "dog") w0 = 0.50;
		else return -1000.0;
	}
	else
	{
		w0 = 0.1;
	}
	// between 0 and 1
	w1 = sqrt(b->getRect().y / frame.rows);

	// time in seconds since las update
	int64 t1 = cv::getTickCount();
	double msecs = 0.1 * (t1 - b->updateTracker) / cv::getTickFrequency();
	w2 = msecs;

	// Compute movement = amount of non-zero pixels on mask
	float cnz = countNonZero(mask(b->getRect()));
	float area = b->getRect().area();
	if ((1.0f * cnz / area) > 0.3) w3 = 1.0;
	else if ((1.0f * cnz / area) > 0.1) w3 = -1.0;
	else w3 = -3.0;

	return alpha * w0 + beta * w1 + gamma * w2 + delta * w3;
}

classToTrack* clMultiTracker::getClass(std::string cn)
{
	for (int i = 0; i < this->classesToTrack.size(); i++)
	{
		if (classesToTrack[i]->className == cn)
		{
			return classesToTrack[i];
		}
	}

	return NULL;
}

void clMultiTracker::initialize(int nTraker, cv::Mat& frame, std::vector<Blob*>& activeblobs)
{
	if (classesToTrack.size() == 0)
	{
		classesToTrack.push_back(new classToTrack("person", true, ttMIL));
		classesToTrack.push_back(new classToTrack("bicycle", false, ttMedianFLOW));
		classesToTrack.push_back(new classToTrack("motorbike", false, ttMedianFLOW));
		classesToTrack.push_back(new classToTrack("car", false, ttMedianFLOW));
		classesToTrack.push_back(new classToTrack("truck", false, ttMedianFLOW));
		classesToTrack.push_back(new classToTrack("bus", false, ttMedianFLOW));
		classesToTrack.push_back(new classToTrack("bike", false, ttMedianFLOW));
		classesToTrack.push_back(new classToTrack("dog", false, ttMedianFLOW));
	}
	
	int cntActive = 0;
	int cntNew = 0;
	int cntDelete = 0;
	// Count how many objects have been detected
	for (auto b : activeblobs)
	{
		if (b->before.size() == 0)
		{
			// has no history
			cntNew++;
		}
		else
			if (b->active)
			{
				cntActive++;
			}
			else
			{
				cntDelete++;
			}
	}
	std::cout << "active detections  " << cntActive << " New Detections" << cntNew << " To remove " << cntDelete << "\n";
	// Now, generate trackers
	for (int j = 0; j < activeblobs.size(); j++)
	{
		//not longer active
		if (!activeblobs[j]->active) continue;

		trackingLib::Blob* b = activeblobs[j];
		cv::Rect2d r = b->getRect();
		if ((r.width >= frame.cols) || (r.width >= frame.rows))
		{
			activeblobs[j]->active = false;
			continue;
		}


		if (!b->tracker.empty())
		{
			b->tracker.release();
		}
		// it is a new one
		Ptr<Tracker> tracker;

		classToTrack* classTC = getClass(b->classes[0].second);
		// Check if this class is active
		if (classTC != NULL)
		{
			if (!classTC->active) continue;
		}

		if ((classTC != NULL) && (classTC->trackerId != ttUnassigned))
		{
			b->trackerType = classTC->trackerId;
		}
		else
		{
			b->trackerType = nTraker;
		}
		
		if (b->trackerType == ttBOOSTING) tracker = TrackerBoosting::create();
		else if (b->trackerType == ttMIL) 	tracker = TrackerMIL::create();
		else if (b->trackerType == ttKCF)  tracker = TrackerKCF::create();
		else if (b->trackerType == ttTLD) 	tracker = TrackerTLD::create();
		else if (b->trackerType == ttMedianFLOW) 	tracker = TrackerMedianFlow::create();
		else if (b->trackerType == ttGoTURN)	tracker = TrackerGOTURN::create();
		else if (b->trackerType == ttMOSSE)	tracker = TrackerMOSSE::create();
		else if (b->trackerType == ttCSRT)	tracker = TrackerCSRT::create();
		else if (b->trackerType == ttTPL)  tracker = TPLTracker::create();

		b->tracker = tracker;
		
		tracker->init(frame, r);
		///////////////////////////////
			// failed to update tracker
			//////////////////////////////
		std::vector<Point> points;
		b->updatePos(r, frame, cv::Mat(), points, this->store,0);
	}
}

// Compares two intervals according to staring times. 
bool comparePriorities(trackingLib::Blob* i1, trackingLib::Blob* i2)
{
	return (i1->tag > i2->tag);

}


void clMultiTracker::update(cv::Mat& frame, cv::Mat& backFrame, std::vector<Blob*>& blobs, int nframe)
{
	int lostCounter = 0;
	startProcess((char*)"computePriorities");
	// Compute contours
	//BlobsByFrame* bbfCnt = new BlobsByFrame();
	//bbfCnt->findContours(frame, backFrame, nframe, 1000);
	std::vector<Blob*> priorities;
	///////////////////
	/// Compute priorities
	double meanP = 0.0;
	for (int j = 0; j < blobs.size(); j++)
	{
		if (!blobs[j]->active) continue;
		double p = computePriority(blobs[j], frame, backFrame);
		blobs[j]->tag = (int)(p * 255);
		priorities.push_back(blobs[j]);
		meanP += p;

	}
	meanP = meanP / blobs.size();
	// sort the intervals in increasing order of 
	// start time 
	if (applyPolicy)
	{
		sort(priorities.begin(), priorities.end(), comparePriorities);
	}
	endProcess((char*)"computePriorities");

	startProcess((char*)"updateTrackers");
	for (int i = 0; i < priorities.size(); i++)
	{
		if (applyPolicy)
		{
			if (i > nBins) break;

			//Not enough priority

		}

		if (priorities[i]->tag < -250)
		{
			continue;
		}

		if (!priorities[i]->active) continue;
		cv::Rect2d r = priorities[i]->getRect();

		////////////////////////////////////////
		// Compute speed
		int cNZ = countNonZero(backFrame(r));

		if (accelerateDeath)
		{

			if (1.0*cNZ / (r.width*r.height) < 0.05)
			{
				priorities[i]->life -= 2;
				std::cout << nframe << " : obj leave scene" << i << "\n";
			}
		}

		std::vector<Point> points;
		// if this image almost not contains pixels in movemnt
		if (!priorities[i]->tracker) continue;
		if (priorities[i]->tracker->update(frame, r))
		{
			priorities[i]->updateTracker = cv::getTickCount();
			//	float rnd = rand() / rando
			priorities[i]->updatePos(r, frame, backFrame, points , this->store, nframe);
			
		}
		else
		{
			priorities[i]->life -= 5;
			//	bLost.push_back(activeblobs[j]);
			lostCounter++;

		}
		
	}

	endProcess((char*)"updateTrackers");
	if (lostCounter > 0)
	{
		std::cout << nframe << " : objects lost" << lostCounter << "\n";
		//exportCNNData(bLost, "d:\\lostPoints.csv", nframe);
	}
}
void clMultiTracker::clear()
{

}



void updateState(std::vector<trackingLib::Blob*>& bbs, int nroFrame)
{
	////
	std::vector<Blob*> _active;
	BlobsHistory* _bH = BlobsHistory::getInstance();

	for (auto b : bbs)
	{
		if (!b->active)
		{
			
			continue;
		}
		b->life--;
		if (b->life < 0)
		{
			auto start = std::chrono::system_clock::now();
			if (b->guid == "")
			{
				b->guid = randomString();
			}
			if (b->classes.size() > 0)
			{
				b->guid = b->classes[0].second + "_" + b->guid;
			}
			else
			{
				b->guid = "unknown_" + b->guid;
			}

			_bH->onObjectDead(b, nroFrame, start);

		}
		else
		{
			_active.push_back(b);
		}
	}


	bbs.clear();
	bbs.swap(_active);
}


void asyncrchonousREAD(string idSceneDetector, BlobsHistory* bH, clMultiTracker* multiT, cv::VideoCapture cap, tlThreadManager* tM)
{
	int nframe = 0;

	std::cout << " Thread READ starting" << "\n";
	////////////////////////////////////////////////////
	//// TRACKERS
	_bH = bH;
	_multiTracker = multiT;
	int trackerIsActive = 0;
	while (true)
	{
		if (tM->finishAll) break;

		if (nframe >= maxFRAMES_TO_EVALUATE)
		{
			std::cout << "Max Frames evaluated" << "\n";
			
			tM->finishAll = true;
			break;
		}

		if (tM->inputVideoQ.queue_size() > multiT->windowTime * 3)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(15));
			continue;
		}

		cv::Mat frame, gray, backFrame;

		if (!getFrameFromVid(cap, frame, gray))
		{
			std::cout << "Frame could not be captured" << "\n";
			tM->finishAll = true;
			break;
		}


		if (multiT->cnn->roi.area() > 0)
		{
			frame = frame(multiT->cnn->roi);
			gray = gray(multiT->cnn->roi);
		}

		
		// extract movement
		updateBGS(gray, backFrame);
		
		
		int cNZ = countNonZero(backFrame);
		float area = backFrame.cols * backFrame.rows;
		if (1.0 * cNZ / (backFrame.cols * backFrame.rows) > 0.0)
		{
			trackerIsActive = 30;
		}

		if (trackerIsActive > 0) 
		{
			tM->inputVideoQ.queue(frame, backFrame, nframe);
			if (nframe % multiT->windowTime == 0)
			{
				tM->cnnQ.queue(frame, backFrame, nframe);
			}
			
			nframe++;

		}
		else
		{
			tM->renderQ.queue(frame, backFrame, nframe);
		}
		trackerIsActive--;
		std::this_thread::sleep_for(std::chrono::milliseconds(5));

		

	}
}

void asyncrchonousNON_CNN(tlThreadManager* tM)
{
	
	while (true)
	{
		if (tM->finishAll) break;

		// first update current blobs
		if (tM->cnnQ.queue_size() == 0)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}


		cv::Mat frame, backFrame;
		frameData fD = tM->cnnQ.dequeue();

		startProcess((char*)"Blobs");
		frame = fD.rgb;
		backFrame = fD.gray;

		//  Get objects to be tracked
		// Initialize By CNN
		BlobsByFrame* bbfCnt = new BlobsByFrame();
	    bbfCnt->findContours(frame, backFrame, fD.nframe, 1000);
		
		endProcess((char*)"Blobs");

		tM->trackerQ.queue(frame, backFrame, fD.nframe, bbfCnt->trackBlobs);


	}
}

void asyncrchonousCNN(cnnInstance* cnn, tlThreadManager* tM)
{
	int nframe, objectsDetected;
	while (true)
	{
		if (tM->finishAll) break;


		// first update current blobs
		if (tM->cnnQ.queue_size() == 0)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}

		
		cv::Mat frame, backFrame;
		frameData fD = tM->cnnQ.dequeue();

		if (fD.nframe >= maxFRAMES_TO_EVALUATE) break;

		std::cout << "Processing CNN" << "\n";

		startProcess((char*)"CNN");
		frame = fD.rgb;
		backFrame = fD.gray;

		//  Get objects to be tracked
		// Initialize By CNN
		cnn->setFrame(frame);
		cnn->predict(false);
		double timeCNN = endProcess((char*)"CNN");

#ifdef TRACKING_LIB
		BlobsByFrame* bbfsN0 = cnn->readResults();

		tM->trackerQ.queue(frame, backFrame, fD.nframe, bbfsN0->trackBlobs);
		std::cout << "Reading results CNN. Time"<< timeCNN << "ms \n";
#endif
	}
}

void asyncrchonousTRACK(string idSceneDetector, tlThreadManager* tM, clMultiTracker* multiT )
{
	
	std::cout << " Thread TRACK starting" << "\n";

	//! [update]
	while (true)
	{
	
		if (tM->finishAll) break;
	
		if ((tM->trackerQ.queue_size() == 0) || (tM->inputVideoQ.queue_size() == 0))
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(25));
			continue;
		}
	
		cv::Mat frame, backFrame;
		// if I have a frame to initiliaze
		frameData fDCNN = tM->trackerQ.dequeue();
		frame = fDCNN.rgb;
		backFrame = fDCNN.gray;
		auto start = std::chrono::system_clock::now();
		BlobsByFrame bF;
		bF.trackBlobs = fDCNN.blobs;
		// Merge existing blobs with new ones
	
		std::vector<Blob*> newBlobs = _bH->addBlobs(&bF, fDCNN.nframe, start, frame.size(), frame.size());
		///////////////////////////////////
		// Compute class statistics of new blobs 
		for (auto b : newBlobs)
		{
				if (b->classes.size() > 0)
				{
					// check if this class is active
					classToTrack* ctk = multiT->getClass(b->classes[0].second);
					if (ctk) 		ctk->counter++;

				}
		}
	
		// initialize with detections
		_multiTracker->initialize(multiT->defaultTraker, frame, _bH->activeblobs);

		frameData fInput = tM->inputVideoQ.dequeue();
		if (!tM->inputVideoQ.active)
		{
			break;
		}
		/// Advance trackers during several frames
		while (fInput.nframe < fDCNN.nframe + multiT->windowTime - 1)
		{
			if (tM->finishAll) break;
			startProcess((char*)"updateTrackers");
				_multiTracker->update(fInput.rgb, fInput.gray, _bH->activeblobs, fInput.nframe);
			endProcess((char*)"updateTrackers");

			startProcess((char*)"updateSTate");
				updateState(_bH->activeblobs, fInput.nframe);
			endProcess((char*)"updateSTate");

			tM->renderQ.queue(fInput.rgb, fInput.gray, fInput.nframe, _bH->activeblobs);
			fInput = tM->inputVideoQ.dequeue();

			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	//	objectsDetected = bH->activeblobs.size();

	std::cout << "Leaving tracker" << "\n";
}



