#include <iostream>
#include <fstream>
#include <Windows.h>
#include <ctype.h>
#include <cstdint>
#include <string>
#include <iostream>

#include <stdlib.h>
#include <string.h>

#include "blobs/blobsHistory.h"
#include "u_ProcessTime.h"
#include "trackRender.h"
#include "scheduledMultitracker.h"
#include "smartcam/eventDetectors.h"

int linkId = 0;
std::vector< blobLink*> alerts;
cv::Mat logo1, logo2, iconsSecurityH, iconsSecurityC;
std::vector<trackingLib::LineCrossDetector*> lineCrossEvents;
cv::Mat blobImage;
int activeFilters = 0;

std::string outputFile =  "d:\\rd_videoConstruccion1.mp4";
VideoWriter* vidOut;

void drawSmoothRectangle(cv::Mat&m, cv::Rect r, cv::Scalar color, bool drawBorder, std::string text)
{
	if (drawBorder) cv::rectangle(m, r, color, 3);
	//cv::rectangle(m, r, color * 0.5, -1);

	if (text != "")
	{
		cv::Point src = cv::Point(r.tl().x + 10 * text.length(), r.tl().y + 25);
		cv::rectangle(m, cv::Rect(r.tl(), src), color, -1);
		cv::putText(m, text, cv::Point(r.tl().x +2, r.tl().y + 12), 1, 1, cv::Scalar(255, 255, 255));
	}

}


void writeFrame(cv::Mat& frame, int frameIndex)
{
	
	if (outputFile != "" && !frame.empty())
	{
		if (!vidOut) {
			std::cout << "Init writter " << outputFile << std::endl;
			vidOut = new cv::VideoWriter(outputFile, cv::VideoWriter::fourcc('H', '2', '6', '4'), 20, frame.size());
		}
		vidOut->write(frame);

		//cv::imwrite("frame" + std::to_string(frameIndex) + ".jpg", frame);
	}
}

double pointDistance(cv::Point2d p0, cv::Point2d p1)
{
	double dst = sqrt((p1.x - p0.x) * (p1.x - p0.x) + (p1.y - p0.y) * (p1.y - p0.y));

	return dst;
}

void overlayImage(cv::Mat& src, cv::Mat& overlay, const cv::Point& location)
{
	for (int y = max(location.y, 0); y < src.rows; ++y)
	{
		int fY = y - location.y;

		if (fY >= overlay.rows)
			break;

		for (int x = max(location.x, 0); x < src.cols; ++x)
		{
			int fX = x - location.x;

			if (fX >= overlay.cols)
				break;

			double opacity = ((double)overlay.data[fY * overlay.step + fX * overlay.channels() + 3]) / 255;

			for (int c = 0; opacity > 0 && c < src.channels(); ++c)
			{
				unsigned char overlayPx = overlay.data[fY * overlay.step + fX * overlay.channels() + c];
				unsigned char srcPx = src.data[y * src.step + x * src.channels() + c];
				src.data[y * src.step + src.channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
			}
		}
	}
}

/////////////////////////////////////////////
/// LINE CROSS COUNTER
/////////////////////////////////////////////
void lineCrossCounter(cv::Mat& frame, std::vector<trackingLib::Blob*>& blobsToRender, int nFrame)
{
	std::chrono::time_point<std::chrono::system_clock> now;

	std::vector<trackingLib::Event> events;

	for (auto lineCross : lineCrossEvents)
	{

		lineCross->detect(blobsToRender, frame, nFrame, now);

		lineCross->drawOn(frame);

		for (auto e : lineCross->detectedEvents)
		{
			events.push_back(e);
		}
	}

	int step = 0;
	int height = 100;

	for (auto e : events)
	{
		/////////////////////////
		/// MOSTRAR FOTO EN VEZ DE VIDEO!!!
		/////////////////////////
		cv::Rect rdst(step, frame.rows - height, 100, height);
		Mat dst_roi = frame(rdst);

		cv::Rect rsrc = e.boundingBox;
		//cut.copyTo(dst_roi);
		cv::rectangle(frame, rdst, cv::Scalar(150, 50, 50), -1);
		cv::resize(e.frame, e.frame, cv::Size(98, 98));
		e.frame.copyTo(frame(cv::Rect(rdst.x + 1, rdst.y + 1, e.frame.cols, e.frame.rows)));
		step += 100;
	}
}

///////////////////////////////////////////////
/// SECURITY ELEMENTS RENDERING
///////////////////////////////////////////////
void matchSecurityColors(cv::Mat& frame, bool& foundHelmet, bool &foundCoat)
{
	int low_H = 0;
	int low_S = 117;
	int low_V = 145;
	int high_H = 44;
	int high_S = 255;
	int high_V = 255;
	cv::Mat frame_HSV, frame_threshold, frame_threshold2;

	// Convert from BGR to HSV colorspace
	cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
	// Detect the object based on HSV Range Values
	inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
	std::vector<std::vector<Point> > contoursOrange;
	findContours(frame_threshold, contoursOrange, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//cvtColor(frame_threshold, frame_threshold, COLOR_GRAY2BGR);
	//cv::bitwise_and(frame, frame_threshold, frameComb);

	foundHelmet = false;
	foundCoat = false;
	for (size_t i = 0; i < contoursOrange.size(); i++)
	{
		Rect br0 = boundingRect(contoursOrange[i]);
		if (br0.area() >  (frame.cols * frame.rows) * 0.1f)
		{
			foundCoat = true;
			//drawContours(frame, contoursOrange, static_cast<int>(i), Scalar(0, 255, 255), 3);
		}
	}

	


	//////////////////////////////////////////////////////////////////////////////////////
	// find white color
	////////////////////////////////////////////////
	inRange(frame, Scalar(200, 200, 200), Scalar(255, 255, 255), frame_threshold2);
	std::vector<std::vector<Point> > contoursWhite;
	findContours(frame_threshold2, contoursWhite, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	// Draw the foreground markers
	for (size_t i = 0; i < contoursWhite.size(); i++)
	{
		Rect br0 = boundingRect(contoursWhite[i]);
		if (br0.area() > (frame.cols * frame.rows) * 0.05f)
		{
			foundHelmet = true;
		//	drawContours(frame, contoursWhite, static_cast<int>(i), Scalar(255, 0, 0), 3);
		}
	}

}

void detectAndRenderSecurityElements(cv::Mat& frame, std::vector<trackingLib::Blob*>& blobsToRender, int nFrame)
{
	/// RENDER BUILDING
///// RENDER CLOSE FILTER!!
	if (blobsToRender.size() > 0)
	{
		for (auto b : blobsToRender)
		{
			if (b->cutImg.rows > 75)
			{
				std::string st = "id" + std::to_string(b->id);
				cv::Mat frameHSV = b->cutImg.clone();

				bool foundHelmet, foundCoat;
				matchSecurityColors(frameHSV, foundHelmet, foundCoat);
				if (foundHelmet)
				{
					overlayImage(frame, iconsSecurityH, cv::Point(b->getRect().tl().x, b->getRect().tl().y));
				}

				if (foundCoat)
				{
					overlayImage(frame, iconsSecurityC, cv::Point(b->getRect().tl().x, b->getRect().tl().y + iconsSecurityC.rows));

				}

				//	cv::imshow(st.c_str(), frameHSV);

			}
		}
	}

	

}

///////////////////////////////////////////////
/// DISTANCE DETECTION RENDERING
///////////////////////////////////////////////////////////
blobLink::blobLink(trackingLib::Blob* _b0, trackingLib::Blob* _b1)
{
	this->b0 = _b0;
	this->b1 = _b1;
	this->life = 2;
	this->id = linkId;
	linkId++;
}

void blobLink::release()
{
	
	this->cut.release();
}

float blobLink::getLength()
{
	Point2d p0(b0->getRect().x, b0->getRect().y);
	Point2d p1(b1->getRect().x, b1->getRect().y);

	return pointDistance(p0,p1);
}

cv::Rect blobLink::getBoundRect()
{
	vector<Point> contour;
	contour.push_back(b0->getRect().tl());
	contour.push_back(b0->getRect().br());
	contour.push_back(b1->getRect().tl());
	contour.push_back(b1->getRect().br());

	Rect roiRect = boundingRect(contour);
	return roiRect;

}
///////////////////////////
std::vector<std::vector<cv::Rect>> groundTruth;
const double ACTIVATION_SIZE = 100.0 / (1024 * 768);
bool exportResults = false;
std::vector<blobLink*> blobLinks;
blobLink* existLink(trackingLib::Blob* b0, trackingLib::Blob* b1)
{
	for (auto link : blobLinks)
	{
		if (((link->b0 == b0) && (link->b1 == b1)) ||
			((link->b0 == b1) && (link->b1 == b0)))
		{
			return link;
		}
	}

	return NULL;
}
void updateLinks(std::vector<trackingLib::Blob*>& blobsToRender, Mat& frame)
{
	// update links
	std::vector<blobLink*> tmpLinks;
	float minDistance = 50;

	for (auto link : blobLinks)
	{
		link->life--;

		if (link->getLength() > 2 * minDistance)
		{
			// objects comes too far
			link->release();
		}
		else
			if (link->getBoundRect().width > 3 * minDistance)
			{
				// objects comes too far
				link->release();
		   }
			else
		if (link->life > 0)
		{
			tmpLinks.push_back(link);
		}

		
	}

	blobLinks.clear();
	blobLinks.swap(tmpLinks);
		
	// Generate new Links
	for (auto b : blobsToRender)
	{
		float minD = 100000;
		trackingLib::Blob* closeB = NULL;

		for (auto b2 : blobsToRender)
		{
			if (b == b2) continue;
			if (b->life < 0) continue;
			if (b2->life < 0) continue;

			Point2d p0 = b->getCenter();
			Point2d p1 = b2->getCenter();


			float d = pointDistance(p0,p1);
			if (d < minD)
			{
				closeB = b2;
				minD = d;
			}
		}

		if (minD < 50)
		{
			blobLink* link = existLink(b, closeB);

			if (link)
			{
				link->life++;
				// add this to the queue
				if ((link->life > 15) && (link->cut.cols == 0))
				{
					cv::Rect rsrc = link->getBoundRect();
					if (rsrc.x < 0) rsrc.x = 0;
					if (rsrc.y < 0) rsrc.y = 0;
					rsrc.width = min(rsrc.width, frame.cols - rsrc.x - 1);
					rsrc.height = min(rsrc.height, frame.rows - rsrc.y - 1);
					cv::Mat cut = frame(rsrc).clone();
					cv::resize(cut, cut, cv::Size(), 2.0, 2.0);
					link->cut = cut;
					link->detectionTime = cv::getTickCount();
					cv::getTickCount();
					alerts.push_back(link);
				}
			}
			else
			{
				link = new blobLink(b, closeB);
				link->name = "link" + to_string(link->id);
				blobLinks.push_back(link);
			}
		}
	}



}

cv::Mat renderMachines(cv::Mat& frame2, std::vector<trackingLib::Blob*>& blobsToRender, int nFrame)
{
	for (auto b : blobsToRender)
	{
		cv::Rect blobRect = b->getRect();

		//it is not ready to be render
		if (b->frameNumber > nFrame) continue;

		//int bV = getVariation(b);
		//cv::rectangle(frame2, blobRect, cv::Scalar(0, 255, 0), 1);
		if (b->classes.size() == 0) continue;
		std::string sname = b->classes.at(0).second;
		drawSmoothRectangle(frame2, blobRect, cv::Scalar(0, 255, 0),true, sname);

	}

	return frame2;
}

cv::Mat renderAlerts(cv::Mat& frame2, std::vector<trackingLib::Blob*>& blobsToRender , int nFrame)
{
	int step = 0;
	int height = 100;	
	///// RENDER CLOSE FILTER!!
	if (blobsToRender.size() > 0)
	{
		updateLinks(blobsToRender, frame2);
		for (auto b : blobsToRender)
		{

			cv::Rect blobRect = b->getRect();

			//it is not ready to be render
			if (b->frameNumber > nFrame) continue;

			//int bV = getVariation(b);
			cv::rectangle(frame2, blobRect, cv::Scalar(0, 255, 0), 1);

		}
		/// draw linked blobs
		int index = 0;
		for (auto link : blobLinks)
		{
			if (link->life > 0)
			{
				Point2d p0 = link->b0->getCenter();
				Point2d p1 = link->b1->getCenter();

				cv::line(frame2, p0, p1, cv::Scalar(50, 50, 50 + link->life * 20), min(5, link->life));

				if (link->life > 10)
				{
					cv::Rect blobRect = link->b0->getRect();
					cv::rectangle(frame2, blobRect, cv::Scalar(0, 0, 255), 1);
					blobRect = link->b1->getRect();
					cv::rectangle(frame2, blobRect, cv::Scalar(0, 0, 255), 1);
				}

			}

			index++;
		}
	}

	
	
	for (int i = alerts.size() - 1; i >= 0; i--)
	{
		if (step + 100 >= frame2.cols) break;
		blobLink* link = alerts[i];
		if (link->cut.cols == 0) continue;
		cv::Rect rdst(step, frame2.rows - height, 100, height);
		Mat dst_roi = frame2(rdst);
		/////////////////////////
		/// MOSTRAR FOTO EN VEZ DE VIDEO!!!
		/////////////////////////
		cv::Rect rsrc = link->getBoundRect();
		//cut.copyTo(dst_roi);
		cv::rectangle(frame2, rdst, cv::Scalar(150, 50, 50), -1);
		cv::resize(link->cut, link->cut, cv::Size(98, 98));

		int secs = (int)(cv::getTickCount() - link->detectionTime) / cv::getTickFrequency();
		std::string s = to_string(secs) + "s";


		link->cut.copyTo(frame2(cv::Rect(rdst.x + 1, rdst.y + 1, link->cut.cols, link->cut.rows)));

		cv::putText(frame2, s, cv::Point(rdst.x + 1, rdst.y + 12), 1, 1, cv::Scalar(255, 220, 255), 2);
		step += 100;

	}

	return frame2;
}

////////////////////////////////////////////////////////
static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void exportCNNData(std::vector<trackingLib::Blob*>& detections, std::string outFile, int nframe)
{
	for (auto od : detections)
	{
		std::vector<std::string> rows;
		rows.push_back(std::to_string(nframe));
		//rectangle(exporterJPG,Rect( (od.reg.x - od.reg.w/2)  * exporterJPG.cols, (od.reg.y - od.reg.h/2) * exporterJPG.rows,
		//	                     od.reg.w * exporterJPG.cols,od.reg.h * exporterJPG.rows),cv::Scalar(255,0,0));
		rows.push_back(std::to_string(od->getRect().x));
		rows.push_back(std::to_string(od->getRect().y));
		rows.push_back(std::to_string(od->getRect().width));
		rows.push_back(std::to_string(od->getRect().height));
		for (int j = 0; j < od->classes.size(); j++)
		{
			rows.push_back(od->classes[j].second);
			rows.push_back(to_string(od->classes[j].first));

		}

		clUtils::exportCSV(outFile, rows, nframe);
	}
}

// for string delimiter
double computePreccision(std::vector<cv::Rect>& gt, std::vector<trackingLib::Blob*>& tBs)
{
	int matchCount = 0;
	for (auto gb : gt)
	{
		for (auto b : tBs)
		{

			if (b->active <= 0) continue;

			int bV = b->tag;
			//if (b->tag > -2500)
			{
				double as = areaSuperposed(gb, b->getRect());

				if (as > 0.5)
				{
					matchCount++;
					break;
				}

			}
		}
	}
	if (gt.size() == 0)
	{
		return 1.0;
	}
	else
		return (1.0 * matchCount) / gt.size();
}


bool showDemoWin = false;
bool showBack = true;

// Input Quadilateral or Image plane coordinates
Point2f* inputQuad = NULL;
// Output Quadilateral or World plane coordinates
Point2f* outputQuad = NULL;
trackingLib::Blob* selectedBlob = NULL;


/////////////////////////////////////////////////////////////////////////////////////
// renderStartParameters
//  render on imGui Windows common parameters, when starting the app
/////////////////////////////////////////////////////////////////////////////////////
void enablePostProcessingFilters(postProcessingFilters pf)
{
	activeFilters = activeFilters | (int)(1 << pf);
}
void asyncrchonousRENDERLite(std::string gtFile, int fW, int fH, trackingLib::BlobsHistory* _bH, 
	trackingLib::clMultiTracker* _multiTracker, trackingLib::tlThreadManager *tM)
{
	std::cout << " Thread RENDERlite starting" << "\n";
	/// time since start processing
	int64 t1 = cv::getTickCount();
	int frameCount = 1;
	
	double prec = 0.0;
	int cnt = 1;
	auto last_time = std::chrono::high_resolution_clock::now();
	// Maximum angle for the rotation of the pointcloud
	const double max_angle = 15.0;
	// We'll use rotation_velocity to rotate the pointcloud for a better view of the filters effects
	float rotation_velocity = 0.3f;

	cv::Mat firstFrame;

	logo1 = imread("D:\\Resources\\plademaLogo.png", IMREAD_UNCHANGED);
	cv::resize(logo1, logo1, cv::Size(), 0.2, 0.2);

	logo2 = imread("D:\\Resources\\Logo-AIS.png", IMREAD_UNCHANGED);
	cv::resize(logo2, logo2, cv::Size(), 0.25, 0.25);

	iconsSecurityH= imread("D:\\Resources\\iconsSecurity1.png", IMREAD_UNCHANGED);
	cv::resize(iconsSecurityH, iconsSecurityH, cv::Size(48,48), 0.2, 0.2);

	iconsSecurityC = imread("D:\\Resources\\iconsSecurity2.png", IMREAD_UNCHANGED);
	cv::resize(iconsSecurityC, iconsSecurityC, cv::Size(48,48), 0.2, 0.2);

	std::vector<std::string> messages;
	cv::Mat frame2, backFrame;
	int64 startRendertime = cv::getTickCount();

	//enablePostProcessingFilters(pfLineCross);
	//enablePostProcessingFilters(pfCloseDistance);
	//enablePostProcessingFilters(pfSecurityElements);
	enablePostProcessingFilters(pfMachinnery);

	lineCrossEvents.push_back( new trackingLib::LineCrossDetector(0, cv::Point2f(0.5, 0.7), cv::Point2f(0.95, 0.7)));

	lineCrossEvents.push_back(new trackingLib::LineCrossDetector(0, cv::Point2f(0.1, 0.8), cv::Point2f(0.45, 0.8)));
	
	while (true)
	{
		if (tM->finishAll) break;

		trackingLib::frameData fD = tM->renderQ.dequeue();
		cv::Mat frame = fD.rgb;
		cv::Mat back = fD.gray;
		/// BUGGG
		


		if (firstFrame.empty())
		{
			firstFrame = frame.clone();
		}

		if (fD.nframe % 100 == 1)
		{
			t1 = cv::getTickCount();
			frameCount = 1;
		}

		if (!frame.empty())
		{
			messages.clear();
			frame2 = frame.clone();
			backFrame = fD.gray.clone();

			double msecs = abs(t1 - cv::getTickCount()) / cv::getTickFrequency();
			msecs = msecs / (frameCount);

			float fps = 1.0f / msecs;
			messages.push_back("fps:" + std::to_string(fps));

			innerdrawBlobsOnFrame(fD.blobs, frame2, fD.nframe, cv::Scalar(255, 0, 0), _multiTracker);

			cv::resize(frame2, frame2, cv::Size(), 1.5, 1.5);

			imshow("Main frame", frame2);

			imshow("back", back);
			writeFrame(frame2, fD.nframe);
			int k = cv::waitKey(5);

			if (k == 't')
			{
				showProcessTime();
				clearTimers();
			}
			if (k == 'q')
			{
				break;
			}
		}
	}

	if (vidOut) vidOut->release();
}
void showTextOnImage(cv::Mat& frame, std::vector<std::string>& textOut)
{
	int yOffset = 0;
	////////////////////////////
	for (auto s : textOut)
	{
		cv::putText(frame, s, cv::Point(40, 40 + yOffset), 1, 1, cv::Scalar(0, 120, 255), 1);
		yOffset += 30;
	}

}

void innerdrawBlobsOnFrame(std::vector<trackingLib::Blob*>& tBs, cv::Mat& frame,int nFrame, cv::Scalar blobColor, 
	trackingLib::clMultiTracker* _multiTracker)
{
	RNG rng(12345);

	std::vector<trackingLib::Blob*> blobsToRender;
	for (auto b : tBs)
	{
		if (b->active <= 0) continue;

		if (b->classes.size() == 0) continue;

		trackingLib::classToTrack* ctk = _multiTracker->getClass(b->classes[0].second);
		// if class is not been tracked, continue
		//if (!ctk || !ctk->active) continue;

		blobsToRender.push_back(b);
	}

	if (activeFilters & (int)(1 << pfMachinnery))
	{
		renderMachines(frame, blobsToRender, nFrame);
	}

	if (activeFilters & (int)(1 << pfCloseDistance))
	{
		renderAlerts(frame, blobsToRender, nFrame);
	}
	
	if (activeFilters & (int)(1 << pfSecurityElements))
	{
		detectAndRenderSecurityElements(frame, blobsToRender, nFrame);
	}

	if (activeFilters & (int)(1 << pfLineCross))
	{
		lineCrossCounter(frame, blobsToRender, nFrame);
	}

	if (logo1.cols > 0) 	overlayImage(frame, logo1, cv::Point(frame.cols - 180, 20));
	if (logo2.cols > 0)		overlayImage(frame, logo2, cv::Point(50, 20));
}
