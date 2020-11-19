#ifndef TRACK_RENDER_H
#define TRACK_RENDER_H

#include <iostream>
#include <fstream>
#include <Windows.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "blobs/blobsHistory.h"
#include "scheduledMultitracker.h"

#include <iostream>
#include <ctype.h>

enum postProcessingFilters {pfNone, pfCloseDistance, pfLineCross,pfSecurityElements , pfMachinnery};

class blobLink
{
public :
	trackingLib::Blob* b0;
	trackingLib::Blob* b1;
	int life;
	int id;
	
	cv::Mat cut;
	int64 detectionTime;
	std::string name;
	blobLink(trackingLib::Blob* _b0, trackingLib::Blob* _b1);
	float getLength();
	void release();
	cv::Rect getBoundRect();
};
void enablePostProcessingFilters(postProcessingFilters pf);
void asyncrchonousRENDERLite(std::string gtFile, int fW, int fH, trackingLib::BlobsHistory* _bH, trackingLib::clMultiTracker* _multiTracker, trackingLib::tlThreadManager *tM);
void innerdrawBlobsOnFrame(std::vector<trackingLib::Blob*>& tBs, cv::Mat& frame, int nFrame, cv::Scalar blobColor,
	trackingLib::clMultiTracker* _multiTracker);
#endif