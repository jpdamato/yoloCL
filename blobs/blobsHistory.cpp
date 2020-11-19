
#include <filesystem>
#include <unordered_map>
#include <queue>
#include <stdio.h>
#include <chrono>
#include <cmath>        // std::abs

#include "colorClassifier.h"
#include "blobsHistory.h" 
#include "Export.h"
#include "../smartcam/trackingLib.h" 
#include "../smartcam/contourDetection.h"
#include "../smartcam/imageStorage.h"
#include "../smartcam/MaxRectsBinPack.h"
#include "../smartcam/scImage.h"
#include "../u_ProcessTime.h"


using namespace trackingLib;

// Variables globales
BlobsHistory* _bhInstance = NULL;





//
// PARSE THE NAME OF FILENAME
//

std::string getFileName(const string& s) {

	char sep = '/';

#ifdef _WIN32
	sep = '\\';
#endif

	size_t i = s.rfind(sep, s.length());
	if (i != string::npos) {
		return(s.substr(i + 1, s.length() - i));
	}

	return("");
}


//
// GET THE NAME OF FILENAME
//
std::string getFileNameWithoutExtension(const string& str) {

	string s = getFileName(str);
	char sep = '.';


	size_t i = s.rfind(sep, s.length());
	if (i != string::npos) {
		return(s.substr(0, i));
	}

	return("");
}

std::chrono::time_point<std::chrono::system_clock> initTM(string filepath)
{
	auto params = getAlgorithmParameters();
	std::chrono::time_point<std::chrono::system_clock> v;
	if (params->datasourceType == SOURCE_VIDEO_FILE)
	{
		string filename = getFileNameWithoutExtension(filepath);
		/*
		std::chrono::time_point<std::chrono::system_clock> res;
		std::vector<std::string> fn = splitString(filename, '_');
		std::vector<std::string> dmy = splitString(fn[0], '-');  //year-month-day
		std::vector<std::string> hms = splitString(fn[1], '-');  //hour-min-sec-milisec
		*/
		std::tm tm = {};
		//Sample 2018-08-05_16-39-46
		std::stringstream ss(filename);
		ss >> std::get_time(&tm, "%Y-%m-%d_%H-%M-%S");// "%YYYY_%mm_%ddY %H:%M:%S");
		//could not parse the year
		if (tm.tm_year == 0)
		{
			
			v = std::chrono::system_clock::now() ;
		}
		else
		{
			v = std::chrono::system_clock::from_time_t(std::mktime(&tm));
		}
	}
	else
	{
		auto t3h = std::chrono::hours(3);
		v = std::chrono::system_clock::now() - t3h;
	}


	return v;
}


double areaSuperposed(Rect r1, Rect r2)
{
	int XA2 = r1.x + r1.width;
	int XB2 = r2.x + r2.width;
	int XA1 = r1.x;
	int XB1 = r2.x;

	int YA2 = r1.y + r1.height;
	int YB2 = r2.y + r2.height;
	int YA1 = r1.y;
	int YB1 = r2.y;

	double	SI = MAX(0, MAX(XA2, XB2) - MIN(XA1, XB1)) * MAX(0, MAX(YA2, YB2) - MIN(YA1, YB1));
	//From there you compute the area of the union :

	double SU = r1.area() + r2.area() - SI;
	//And you can consider the ratio

	return SI / SU;
}


bool valueInRange(int value, int min, int max)
{
	return (value >= min) && (value <= max);
}



bool PointInPolygon(Point point, vector<Point>& points) {

	int i, j, nvert = (int)points.size();
	bool c = false;

	for (i = 0, j = nvert - 1; i < nvert; j = i++) {
		if (((points[i].y >= point.y) != (points[j].y >= point.y)) &&
			(point.x <= (points[j].x - points[i].x) * (point.y - points[i].y) / (points[j].y - points[i].y) + points[i].x)
			)
			c = !c;
	}

	return c;
}

bool collision(vector<Point>& vertices, vector<Point>& poly)
{
	bool status = false; //Collision status
	int x = 0, y = 0;

	for (int i = 0; i < vertices.size(); i++) // All edges of this polygon
	{
		if (PointInPolygon(vertices[i], poly)) return true;
	}

	for (int i = 0; i < poly.size(); i++) // All edges of this polygon
	{
		if (PointInPolygon(poly[i], vertices)) return true;
	}
	return false;
}

bool rectOverlap(Rect A, Rect B)
{
	bool xOverlap = valueInRange(A.x, B.x, B.x + B.width) ||
		valueInRange(B.x, A.x, A.x + A.width);

	bool yOverlap = valueInRange(A.y, B.y, B.y + B.height) ||
		valueInRange(B.y, A.y, A.y + A.height);

	return xOverlap && yOverlap;
}

struct less_than_key
{
	inline bool operator() (const vector<Point>& struct1, const vector<Point>& struct2)
	{
		Rect br1 = boundingRect(struct1);
		Rect br2 = boundingRect(struct2);

		return (br1.width > br2.width);
	}
};

// Calcula cuantos pixeles son de mascara y cuantos del objeto
void computeMaskWeight(Mat& mask, trackedBlob& tbs, int step)
{
	// Los combino si se solapan
	tbs.backgroundPixels = tbs.objectPixels = tbs.probPixels = 0;

	for (int x = tbs.r.x; x <= MIN(mask.cols - 1, tbs.r.width + tbs.r.x); x = x + step)
		for (int y = tbs.r.y; y <= MIN(mask.rows - 1, tbs.r.height + tbs.r.y); y = y + step)
		{
			uchar val = mask.at<uchar>(y, x);
			if (val <= 10) tbs.backgroundPixels++;
			else if (val >= 250) tbs.objectPixels++;
			else
				tbs.probPixels++;


		}

	tbs.probPixels *= step * step;
	tbs.backgroundPixels *= step * step;
	tbs.objectPixels *= step * step;



}


int mergeContours(Mat& frame, Mat& mask, vector<vector<Point> >& contours, std::vector<Blob*>& rectangles, int minArea, int frameN)
{
	int size = 0;
	int minDistanceCenter = (int)(frame.cols * 0.005);
	std::sort(contours.begin(), contours.end(), less_than_key());
	// Unifico los contornos si uno es interno

	for (int i = 0; i < contours.size(); i++)
	{
		Rect br0 = boundingRect(contours[i]);
		if (br0.width + br0.x >= frame.cols) br0.width = frame.cols - br0.x - 1;
		if (br0.height + br0.y >= frame.rows) br0.height = frame.rows - br0.y - 1;

	}

	// Los combino si se solapan

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() == 0) continue; //ya matchee con otro
		Rect br0 = boundingRect(contours[i]);
		Rect2f newR(br0);

		Blob* tb = Blob::getInstance(frameN);
		tb->contour = contours[i];
		std::vector<Point> newContour = contours[i];
		// Aado los puntos

		bool mixed = false;
		Point2f centerR1;
		//	float radiusR1 , radiusR2;

		//	minEnclosingCircle(contours[i], centerR1, radiusR1);
		Point2f centerR2;

		Rect br1;
		for (int j = i + 1; j < contours.size(); j++)
		{
			if (contours[j].size() == 0) continue; //ya mache con otro
			br1 = boundingRect(contours[j]);
			/*	minEnclosingCircle(contours[j], centerR2, radiusR2);
			int distancex = pow((static_cast<int>(centerR2.x) - static_cast<int>(centerR1.x)),2);
			int distancey = pow((static_cast<int>(centerR2.y) - static_cast<int>(centerR1.y)),2);
			int distanceCenter = sqrt(distancex + distancey);*/
			double areaSup = areaSuperposed(newR, br1);
			//distanceCenter < minDistanceCenter || 
			if (areaSup > BLOB_AREA_OVERLAP)
			{
				newR.width = MAX(newR.x + newR.width, br1.x + br1.width) - MIN(newR.x, br1.x);
				newR.height = MAX(newR.y + newR.height, br1.y + br1.height) - MIN(newR.y, br1.y);

				newR.x = MIN(newR.x, br1.x);
				newR.y = MIN(newR.y, br1.y);

				// combino los otros contornos
				newContour.insert(newContour.end(), contours[j].begin(), contours[j].end());

				contours[j].clear();
				mixed = true;
			}
		}

		if (mixed)
		{
			// Calculo nuevamente el convex Hull
			/// Find the convex hull object for each contour
			vector<Point> hull;
			convexHull(Mat(newContour), hull, false);

			tb->contour.swap(hull);
			Rect br1 = boundingRect(tb->contour);
			
		}
		size += (int)(newR.width * newR.height);
		newR = boundingRect(tb->contour);
		
		rectangles.push_back(tb);
	}

	return size;
}


std::string IntToString2(int value)
{
	char convC[10];
	_itoa_s(value, convC, 10);
	std::string convert(convC);
	return convert;
}

ContourDetection* cd = NULL;
//cl_canny* _canny = NULL;
/// Generate grad_x and grad_y
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
#define MIN_BLOB_SIZE 900


bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

std::string encodeDateTimeStr(int frameN, int frameRate)
{
	auto tp = getAlgorithmParameters()->computeTime(frameN, frameRate);
	// Reduce verbosity but let you know what is in what namespace
	//	namespace C = std::chrono;
	//namespace D = date;
	//	auto dp = D::floor<D::days>(tp);  // dp is a sys_days, which is a  type alias for a C::time_point
	//auto ymd = D::year_month_day{ dp };
	//auto time = D::make_time(C::duration_cast<C::milliseconds>(tp - dp));

	std::time_t tt = std::chrono::system_clock::to_time_t(tp);
	std::string sTp = std::asctime(std::gmtime(&tt));
	char buffer[80];
	struct tm * timeinfo;
	timeinfo = localtime(&tt);

	strftime(buffer, 80, "%G-%m-%d_%I-%M-%S", timeinfo);
	//strftime(buffer, 80, "%G-%m-%d", timeinfo);
	return buffer;
}


/*Retorna ovla fecha dada por parametros en forma de tick para usarse por ej como nombre de archivos*/
std::string encodeDateTime(int frameN, int frameRate)
{
	auto tp = getAlgorithmParameters()->computeTime(frameN, frameRate);
	// Reduce verbosity but let you know what is in what namespace
	//	namespace C = std::chrono;
	//namespace D = date;

	//	auto dp = D::floor<D::days>(tp);  // dp is a sys_days, which is a  type alias for a C::time_point



	//auto ymd = D::year_month_day{ dp };
	//auto time = D::make_time(C::duration_cast<C::milliseconds>(tp - dp));

	std::time_t tt = std::chrono::system_clock::to_time_t(tp);
	std::string sTp = std::asctime(std::gmtime(&tt));
	char buffer[80];
	struct tm * timeinfo;
	timeinfo = localtime(&tt);

	strftime(buffer, 80, "%G-%m-%d_%I-%M-%S",timeinfo);
	//strftime(buffer, 80, "%G-%m-%d", timeinfo);
	return buffer;
}
/*Retorna la fecha dada por parametros en forma de y-m-d para usarse por ej como nombre de archivos*/
std::string encodeDate(int frameN, int frameRate)
{
	auto tp = getAlgorithmParameters()->computeTime(frameN, frameRate);
	std::time_t tt = std::chrono::system_clock::to_time_t(tp);
	std::tm now_tm = *std::localtime(&tt);
	char buffer[80];
	strftime(buffer, 80, "%G-%m-%d", &now_tm);
	return buffer;

}


string encodeDateBlob(Blob* b)
{
	/*char chDate[120];
	strftime(chDate, 120, "%H%M%S%d%m%Y", &b->dateTime);
	string s(chDate);*/
	string ms = to_string(b->ms);
	if (ms.length() == 1)
	{
		ms = "00" + ms;
	}
	else if (ms.length() == 2)
	{
		ms = "0" + ms;
	}
	string s = std::to_string(b->day) + '-' + std::to_string(b->month) + '-' + std::to_string(b->year) + '_' + std::to_string(b->hour) + '-' + std::to_string(b->minute) + '-' + std::to_string(b->second) + '-' + ms;
	return s;
}

//**********************************************
// CLASS frameReferenceManager
//**********************************************
frameReferenceManager* _instanceFR = NULL;

void frameReference::release()
{
	_instanceFR->releaseRef(this);
	
}

frameReferenceManager* frameReferenceManager::getInstance()
{
	if (_instanceFR == NULL)
		_instanceFR = new frameReferenceManager();
	return _instanceFR;
}
frameReference* frameReferenceManager::addReference(cv::Mat& matRef)
{
	for (int i = 0; i < frameRef.size(); i++)
	{
		if (frameRef[i]->allocatedFrame.data == matRef.data)
		{
			frameRef[i]->referenceCounter++;
			return frameRef[i];
		}

	}
	frameReference* nfr = new frameReference();
	nfr->allocatedFrame = matRef;
	nfr->referenceCounter = 1;
	frameRef.push_back(nfr);
	return nfr;
}
void frameReferenceManager::releaseRef(frameReference* ref)
{
	if (ref == NULL) return;
	ref->referenceCounter--;
	if (ref->referenceCounter == 0)
	{
		ref->allocatedFrame.release();
	}

}

//**********************************************
// CLASS BLOB
//**********************************************
int blobsCounter = 0;
std::vector<Blob*> generatedBlobs;
std::vector<Blob*> availableBlobs;

int Blob::getNumberOfBlobs()
{
	return generatedBlobs.size();
}



static bool blobComparator(Blob* a, Blob* b)
{
	
	return a->creationTimePoint < b->creationTimePoint ;
}

Blob* Blob::getInstance(int frameN)
{
	//std::sort(generatedBlobs.begin(), generatedBlobs.end(), blobComparator);

	if (availableBlobs.size() > 200)
	{
		Blob* b = availableBlobs[availableBlobs.size()-1];

		availableBlobs.pop_back();
		
		b->frameNumber = frameN;
		b->life = INITIAL_BLOB_LIFE;
		b->active = 1; 
		b->lived = 0; 
		b->saved = false;
		b->savedBlobIndex = 0; 
		b->locked = false; 
		b->parent = NULL; 
		b->color = cv::Scalar((int)(255.0*rand()) / RAND_MAX, (int)(255.0*rand()) / RAND_MAX, (int)(255.0*rand()) / RAND_MAX);
	
		return b;
	}
	else
	{
		blobsCounter++;
		
		Blob* b = new Blob(frameN);
		
		generatedBlobs.push_back(b);
		
		return b;
	}
}

void Blob::updatePos(cv::Rect2d r, cv::Size processingSize)
{
	// Update Position
	this->historyR.push_back(r);
	this->r = r;
	this->widthPercent = r.width * 100 / (float)processingSize.width;
	this->heightPercent = r.height * 100 / (float)processingSize.height;
	this->xPercent = r.x / (float)processingSize.width;
	this->yPercent = r.y / (float)processingSize.height;
}

bool Blob::updatePos(cv::Rect2d newR, cv::Mat& frame, cv::Mat mask, std::vector<Point>& contour, bool saveCut, int frameN)
{
	float dist = sqrt((r.x - newR.x) * (r.x - newR.x) + (r.y - newR.y) * (r.y - newR.y));
	
	if (dist > 200)
	{
		//too far away. discard
		return false;
	}
	// Update Position
	newR.x = MAX(0, newR.x);
	newR.y = MAX(0, newR.y);
	newR.width = MIN(r.width, frame.cols - newR.x);
	newR.height = MIN(r.height, frame.rows - newR.y);
	// non-valid rectangle
	if (newR.width <= 0) return false;
	if (newR.height <= 0) return false;

	// create and ADD as child
	Blob* child = Blob::getInstance(frameN);
	child->updatePos( newR, frame.size());
	child->parent = this;
	if (saveCut)
	{
		cv::Mat croppedRef(frame, newR);
		child->cutImg = croppedRef.clone();
	}

	if (contour.size() > 0)
	{
		child->contour.swap(contour);
	}

	this->contour = child->contour;
	this->before.push_back(child);
	this->r = newR;
	return true;
}

cv::Rect2d Blob::getRect()
{
	return this->r;

}

string Blob::getBlobCreationDate() {
	return this->blobCreationDate;
}

string Blob::getBlobCreationTime()
{
	return this->blobCreationTime;
}


string Blob::getFilename()
{
	std::string fn;
//auto params = getAlgorithmParameters();
/*	string mediaPath = params->mediaPath;
	if (endsWith(mediaPath, "\\") == false)
		mediaPath.append("\\");
	string folder = params->outputDir;
	if (endsWith(folder, "\\") == false)
		folder.append("\\");
	string blobDir = this->addDateTick(this->guid);*/
	//if (this->parent == NULL)
	//{		
	//	fn = folder + blobDir + "\\" + this->guid + "_" + IntToString(this->id) + "_" + this->getBlobCreationTime() + "_" + this->getBlobCreationDate() + params->imageExtension;
	//	//fn = folder + this->getBlobDir()+ "\\"+ "obj" + IntToString(this->id) + "_" + IntToString(this->id) + "_" + this->getBlobCreationTime() + "_" + this->getBlobCreationDate() + params->imageExtension;
	//}
	//else
	//{
	//	fn = folder + blobDir + "\\" + this->guid + '_' + IntToString(this->id) + '_' + this->getBlobCreationTime() + '_' + this->getBlobCreationDate() + params->imageExtension;
	//}
	fn = this->guid + '_' + std::to_string(this->id) + '_' + this->getBlobCreationTime() + '_' + this->getBlobCreationDate() + ".jpg";
	return fn;
}



Blob* Blob::getLast()
{
	if (before.size() > 0)
		return before[before.size() - 1];
	else
		return this;

}

Point Blob::getCenter()
{
	return Point((int)(r.x + r.width / 2), (int)(r.y + r.height / 2));

}

void Blob::release()
{
	for (int i = 0; i < before.size(); i++)
	{

	//	before[i]->cutImg.release();
		if (before[i]->parent) before[i]->release();
		
	}
	
	this->active = 0;
	this->before.clear();
	this->history.clear();
	this->contour.clear();
	this->historyR.clear();
	this->classes.clear();
	this->parent = NULL;
	this->hierarchy.clear();

	if (this->tracker)
	{
		this->tracker->clear();
		this->tracker.release();
		
	}

	availableBlobs.push_back(this);
}

double Blob::getMaxArea()
{
	Rect r(0, 0, 0, 0);
	for (int i = 0; i < before.size(); i++)
	{
		if (before[i]->r.area() > r.area())
		{
			r = before[i]->r;
		}
	}

	return r.area();
}

double Blob::getMeanArea()
{
	double meanArea = 0.0;

	if (before.size() == 0) return meanArea;

	for (int i = 0; i < before.size(); i++)
	{
		meanArea += before[i]->r.area();

	}

	return meanArea / before.size();
}

float Blob::centerX()
{
	return  r.x + (float)r.width/ (float) 2;
}


float Blob::centerY()
{
	return r.y + (float)r.height / (float) 2;
}


///Clona información relacionada a la imagen, necesaria para el spritesheet
Blob* Blob::clone()
{
	Blob* tTB = new Blob(this->frameNumber);	
	tTB->id = this->id;
	tTB->guid = this->guid;
	tTB->contour.swap(this->contour);
	tTB->r = this->r;
	tTB->xPercent = this->xPercent;
	tTB->yPercent = this->yPercent;
	tTB->widthPercent = this->widthPercent;
	tTB->heightPercent = this->heightPercent;	

	//date
	tTB->day = this->day;
	tTB->month = this->month;
	tTB->year = this->year;

	tTB->blobCreationDate = this->blobCreationDate;

	//time
	tTB->hour = this->hour;
	tTB->minute = this->minute;
	tTB->second = this->second;
	tTB->ms = this->ms;

	if (this->cutImg.cols > 0)
	{
		tTB->cutImg = this->cutImg.clone();
	}

	for (auto cl : this->classes)
	{
		tTB->classes.push_back(cl);
	}
	tTB->blobCreationTime = this->blobCreationTime;

	for (int i = 0; i<this->before.size(); i++)
	{
		Blob* b = this->before[i];

		Blob* tempTB = new Blob(b->frameNumber);
		tempTB->id = b->id;
		tempTB->guid = b->guid;
		tempTB->contour.swap(b->contour);
		tempTB->r = b->r;

		//date
		tempTB->day = b->day;
		tempTB->month = b->month;
		tempTB->year = b->year;

		//time
		tempTB->hour = b->hour;
		tempTB->minute = b->minute;
		tempTB->second = b->second;
		tempTB->ms = b->ms;

		tTB->before.push_back(tempTB);
	}
	
	return tTB;
}

cv::Vec2f Blob::getDir()
{
	cv::Vec2f dif;
	dif[0] = 0;
	dif[1] = 0;
	if (before.size() > 2)
	{
		int i0 = (int)before.size() - 1;
		int i1 = (int)before.size() - 2;


		dif[0] = (float)before[i0]->centerX() - before[i1]->centerX();
		dif[1] = (float)before[i0]->centerY() - before[i1]->centerY();
	}

	return dif;
}

int uniqueBlobID = 0;
Blob::Blob(int frameN) 
{ life = INITIAL_BLOB_LIFE;
  active = 1; lived = 0; saved = false;
          savedBlobIndex = 0; locked = false; parent = NULL; id = uniqueBlobID; uniqueBlobID++;
		  this->color = cv::Scalar((int)(255.0*rand())/RAND_MAX, (int)(255.0*rand()) / RAND_MAX, (int)(255.0*rand()) / RAND_MAX);
		  this->frameNumber = frameN;
}

std::pair<std::string, float> Blob::getTopColor()
{
	 std::pair<std::string, float> topColor;
	 float topC = 0;
	 int topI = 0;

	 for (int i = 0; i < PALETTE_COUNT; i++)
	 {
		 if (!this->colorPalette) continue;
		 if (topC < this->colorPalette[i])
		 {
			 topC = colorPalette[i];
			 topI = i;
		 }
		 
	 }


	return std::make_pair(COLOR_PALETTE_NAMES[topI], topC);
}



//**********************************************
// CLASS BLOBS by FRAME
//**********************************************
BlobsByFrame::BlobsByFrame()
{
	mainColors.push_back(cv::Vec3b(255, 0, 0)); // Rojo
	mainColors.push_back(cv::Vec3b(255, 255, 255)); // white
	mainColors.push_back(cv::Vec3b(128, 128, 128)); // light gray
	mainColors.push_back(cv::Vec3b(0, 255, 0)); // light blue
	mainColors.push_back(cv::Vec3b(0, 0, 255)); // light blue
	mainColors.push_back(cv::Vec3b(0, 0, 0)); // light blue

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;


}



void BlobsByFrame::computeMainColors(Mat &src)
{
	/*	const int width = src.cols;
	Mat hsv;
	vector<Mat> channels;
	cvtColor(src, hsv, CV_BGR2HSV);

	split(hsv, channels);


	imshow("hue", channels[0]);
	imshow("sat", channels[1]);
	imshow("val", channels[2]);
	*/

}

void BlobsByFrame::clear()
{
	contours.clear();
	trackBlobs.clear();

}

/*dir viene de la forma c:\Temporal\idSmartcam\idCamera\*/
void BlobsByFrame::enqueue(Mat& frame, Mat& mask, string dir, int firstID, int minSize)
{
	for (int i = 0; i < trackBlobs.size(); i++)
	{
		// Descarto los BLOBS que no cumplen el tamao minimo
		if (trackBlobs[i]->getRect().area() < minSize) continue;
		if (trackBlobs[i]->getRect().area() > frame.rows * frame.cols) continue;
		
		cv::Rect r = trackBlobs[i]->getRect();
		r.x = MAX(r.x, 0);
		r.y = MAX(r.y, 0);

		r.width = MIN(r.width, frame.cols - r.x-1);
		r.height = MIN(r.height, frame.rows - r.y-1);

		// Muchos pixeles son inciertos.. Sirve para eliminar algunos BLOBs
		//computeMaskWeight(mask, trackBlobs[i], 3);
		//if ( trackBlobs[i].objectPixels < 60 ) continue;

		trackBlobs[i]->assigned = 1;
		trackBlobs[i]->id = firstID + i;
		trackBlobs[i]->xPercent = (1.0f*r.x) / frame.cols;
		trackBlobs[i]->yPercent = (1.0f*r.y) / frame.rows;

		// Setup a rectangle to define your region of interest
		cv::Rect myROI = r;
		// BUG !!! CONSUME MUCHA CPU
		cv::Mat croppedRef(frame, myROI);
		trackBlobs[i]->cutImg = croppedRef.clone();
		
		//CalculatePaletteProportions(trackBlobs[i]->cutImg, trackBlobs[i]->colorPalette);

		//Asignar el nombre al blob

	}
}

void BlobsByFrame::findContours(Mat& frame, Mat& mask, int nFrame, int minArea)
{

	int thresh = 100;
	int max_thresh = 255;
	//	RNG rng(12345);

	//	double fpsA[5];
	if (cd == NULL)
		cd = new ContourDetection(frame.cols, frame.rows);

	int64 start = cv::getTickCount();
	/// Detect edges using canny

	Mat canny_output;
	canny_output = mask.clone();
	cv::Canny(mask, canny_output, thresh, thresh * 2, 3);

	cv::Mat dilateElement = cv::getStructuringElement(0, cv::Size(3, 3), cv::Point(-1, -1));
	cv::dilate(canny_output, canny_output, dilateElement, cv::Point(-1, -1), 3);

	// Own method to find contours
	//cd->fillContours(mask, 50, 1, nFrame);
	//contours = cd->contours;
	/// Find contours
	cv::findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		Rect br0 = boundingRect(contours[i]);
		if (br0.area() < minArea)
		{
			contours[i].clear();
			continue;
		}
		convexHull(Mat(contours[i]), hull[i], false);
	}
	//fpsA[3] = 1000 * (cv::getTickCount() - start) / cv::getTickFrequency(); start = cv::getTickCount();
	contours.clear();
	for (int i = 0; i < hull.size(); i++)
	{
		if (hull[i].size() > 0)
		{
			contours.push_back(hull[i]);
		}
	}

	blobsMemSize = mergeContours(frame, mask, contours, trackBlobs, 100, nFrame);

	/*if (nFrame % 20 == 0)
		std::cout << " ------FIND CONTOURS -" << "(" << 1000 * (cv::getTickCount() - start) / cv::getTickFrequency() << ")-------" << std::endl;
*/
}

void BlobsByFrame::drawOnFrame(Mat& frame)
{
	RNG rng(12345);
	std::vector<Blob*> tBs = this->trackBlobs;

	for (auto b : trackBlobs)
	{
		if (b->speed[0] == 0.0f)
		{
			//cv::rectangle(frame, b->getRect(), cvScalar(125, 125, 125), 3);
		}
		else
		{
			cv::rectangle(frame, b->getRect(), cvScalar(250, 100, 250), 3);
		}
		
		if (b->classes.size() > 0)
		{
			for (int j = 0; j < b->classes.size(); j++)
			{
				cv::Rect r = b->getRect();
				r.x += j * 5;
				r.y += j * 5;

				cv::String s = std::to_string(b->id) + ":" + b->classes[j].second + "("+std::to_string(b->classes[j].first)+")";
				cv::rectangle(frame,r, cvScalar(((j+13)*113) % 255, ((j + 33) * 77) % 255, ((j + 43) * 99) % 255 ), 3);
				putText(frame, s, b->getRect().tl(),
					FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			}
		}

		if (b->history.size() > 0)
		{
			for (int i = 0; i < b->history.size() - 1; i++)
			{
				cv::circle(frame, b->history[i], 3, cv::Scalar(255, 0, 0));
			}
		}
	}


	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 255, 0);
		if (contours[i].size() > 0)
			drawContours(frame, contours, i, color, 1, 8, hierarchy, 0, Point());

	
	}

}


//**********************************************
// CLASS BlobsByFrame
//   - Lista de BLOBS extraidos en cada frame
//**********************************************
//--------------------------------------
//-- Retorna el peso entre BLOBs
//----------------------------------
double weight(Blob* b1, Blob* b)
{
	// compute euclidean distance
	double x0 = b->xPercent;
	double y0 = b->yPercent;

	double x1 = b1->xPercent;
	double y1 = b1->yPercent;

	double dst = sqrt( (x1-x0) * (x1-x0) + (y1 - y0) * (y1 - y0));

	// compute superposed Area
	double as = areaSuperposed(b1->getRect(), b->getRect());
	//cv::Vec2f dir = b1->getDir();
	
	if ((b1->classes.size() > 0) && (b->classes.size() > 0))
	{
		if (b1->classes[0].first != b->classes[0].first)
			return 0;
	}
	return as;
	/*
    // Si no se encuentra activo
	double s0 = b->getRect().area();
	double s1 = b1->getRect().area();
	
	double area = (MIN(s1, s0) / MAX(s1, s0));
	
	// compute color distance
	double colord = 0.0;
	for (int i = 0; i < 17; i++)
	{
		colord += abs(b->colorPalette[i] - b1->colorPalette[i]);
	}
	//std::cout << " dst " << dst << " area:" << area << " coord:" << colord << "\n";
	// function area superposed + distance + area
	return (1 - dst) + as;// +(1 - colord);  // (1 - dst) * 0.5 + (1 - colord) * 0.2 + area * 0.3;
	*/
}


Blob* findSimilarBlob(std::vector<Blob*>& activeblobs, Blob* b)
{
	// Si se solapa con otro objeto

	//vector<Blob*> candidates;
	//unordered_map<double, Blob*> it;
	double best = 0;
	Blob* _b = NULL;
	//double perim0 = arcLength(b->contour, true);
	if (activeblobs.size() > 0)
	{
		for (int j = 0; j < activeblobs.size(); j++)
		{
			if (activeblobs[j]->life < 0) continue;
			
			// Si se solapan los rectangulos
			double as = weight(activeblobs[j], b);
			if (as > best)
			{
				_b =  activeblobs[j];
				best = as;
			}
		}
	}

	
	if (best > 1.0)
	{
		return _b;
	}
	else
	{
		return NULL;
	}
}

int idGenerator = 0;
std::vector<trackingLib::Blob*> findSimilarBlobs(std::vector<trackingLib::Blob*>& activeblobs, std::vector<trackingLib::Blob*>& bbs)
{
	std::vector<trackingLib::Blob*> unassignedB;
	//make a copy
	for (auto bN : bbs)
	{
		
		unassignedB.push_back(bN);
	}

	// now track displacemente back
	for (auto bN : unassignedB)
	{
		Blob* parent = findSimilarBlob(activeblobs, bN);

		if (parent == NULL)
		{
			bN->id = idGenerator;			
			bN->guid = "blob_" + to_string(idGenerator);
			bN->before.push_back(bN);

			bN->parent = NULL;
			
			activeblobs.push_back(bN);
			idGenerator++;
		}
		else
		{
			bN->parent = parent;
			parent->before.push_back(bN);
			parent->history.push_back(bN->getCenter());
		}

	}

	return unassignedB;
}

BlobsHistory::BlobsHistory(int frameR, int frameWidth, int frameHeight, string idSceneDetector)
{
	idGenerator = 0;
	this->storeBlobs = false;
	this->frameRate = frameR;
	this->videoWidth = frameWidth;
	this->videoHeight = frameHeight;
	// Tengo una sola instancia de Blobs History
	if (_bhInstance == NULL)
	{
		_bhInstance = this;
	
	}

}

// Calculo la direccion de movimiento
void BlobsHistory::computeMovement()
{
	for (int j = 0; j < this->activeblobs.size(); j++)
	{
		Blob* parent = activeblobs[j];
		for (int i = 1; i < parent->before.size(); i++)
		{
			Blob* b0 = parent->before[i - 1];
			Blob* b1 = parent->before[i];
			cv::Point v0 = b1->getCenter() - b0->getCenter();

			b0->dir[0] = (float)v0.x;
			b0->dir[1] = (float)v0.y;
			if (i == parent->before.size() - 1)
			{
				b1->dir[0] = (float)v0.x;
				b1->dir[1] = (float)v0.y;
			}
		}
	}

}

void BlobsHistory::updateBlobsState(int frame, std::chrono::time_point<std::chrono::system_clock> now)
{
	std::vector<Blob*> _active;
	for (int j = 0; j < this->activeblobs.size(); j++)
	{
		activeblobs[j]->life--;
		if (activeblobs[j]->life < 0)
		{
			onObjectDead(activeblobs[j], frame, now);
		}
		else
		{
			_active.push_back(activeblobs[j]);
		}
	}

	activeblobs.clear();
	activeblobs.swap(_active);
}

std::vector<Blob*> BlobsHistory::addBlobs(BlobsByFrame* bbF, int frame, std::chrono::time_point<std::chrono::system_clock> now,
	                       cv::Size originalSize, cv::Size processingSize)
{
	

	std::vector<Blob*> newBlobs;
	
	//	int minDistanceCenter = this->videoWidth * 0.010;
	history.push_back(bbF);

	//
	// Asocio los blobs Nuevos con los que tenemos en la histori
	for (auto bb : bbF->trackBlobs)
	{
		if (bb->getRect().area() == 0) continue;

		Blob *b = Blob::getInstance(frame);
		b->name = bb->name;
		cv::Rect2d r = bb->getRect();
		
		b->topLeft = cv::Point(r.x, r.y);
		b->cutImg = bb->cutImg;
		b->updatePos(r,processingSize);
		b->classes.swap(bb->classes);
				
		if (bb->contour.size() > 0)
		{
			b->contour.swap(bb->contour);
		}
		/*FORMATO: RGBA */
		//cv::Scalar meanColor = mean(b->cutImg);
		/*Mat1b mean_img2 = b->cutImg.reshape(1, b->cutImg.rows*b->cutImg.cols);
		reduce(mean_img2, mean_img2, 1, REDUCE_AVG);
		mean_img2 = mean_img2.reshape(1, b->cutImg.rows);*/

		///TODO extraer de la imagen
		//b->color = cv::Vec4b(meanColor);
		//b->color = cv::Vec4b((idGenerator * 31 + 15) % 255, (idGenerator * 71 + 1555) % 255, (idGenerator * 91 + 4560) % 255, 0);		
		//GenerateColors(b->blob);
		computeDate(b, now);
		b->frameNumber = frame;
		b->creationTimePoint = now;
	
   	    Blob* parent = findSimilarBlob(activeblobs, b); //TODO templatematching

		if (parent)
		{
			b->id = (int)parent->before.size(); //El primero de before se descarta porque es la referencia
			b->guid = "child" + to_string(idGenerator);
			parent->life = INITIAL_BLOB_LIFE;
			parent->active = 1;
			parent->updatePos(b->getRect(), processingSize);
			b->parent = parent;		
			parent->before.push_back(b); //Si tiene uno similar, lo inserto al final

			if (parent->classes.size() == 0)
			{
				parent->classes.swap(bb->classes);
			}
			
		}
		else  
		//Si no tiene uno similar, lo agrego como nuevo a activeBlobs
		{
			b->id = idGenerator;
			b->guid = to_string(idGenerator);
			b->parent = NULL;
			b->before.push_back(b);
			activeblobs.push_back(b);
			newBlobs.push_back(b);
			idGenerator++;
		}
	}

	return newBlobs;
}


void BlobsHistory::findAndAddBlobs(cv::Mat& frame, cv::Mat& mask, int nframe)
{
	BlobsByFrame* bbN = new BlobsByFrame();
	bbN->findContours(frame, mask, nframe);

	std::chrono::time_point<std::chrono::system_clock> now;

	this->addBlobs(bbN, nframe, now, cv::Size(frame.cols, frame.rows), cv::Size(frame.cols, frame.rows));
	//	trackBlobs(frame1,bb1, frameNumber, net, skipFrames, templateAlgorithm);

}
BlobsHistory::BlobsHistory()
{
}

BlobsHistory* BlobsHistory::getInstance()
{
	if (_bhInstance == NULL)
	{
		_bhInstance = new BlobsHistory(30, 1024, 768, "e9f50b5f-6ed6-4e5b-be02-b1a1950b48d3");

	}
	return _bhInstance;
}

void BlobsHistory::compactMem()
{
	std::vector<Blob*> _active;

	for (auto b : activeblobs)
	{
		if (b->active)
			_active.push_back(b);
		else
		{
			b->release();
		}
	}

	_active.swap(activeblobs);
}


void BlobsHistory::onObjectDead(Blob* b, int nroFrame, chrono::time_point<chrono::system_clock> now)
{
	if (!b->active) return;

	// Si tienen pocas muestras o son muy chicos, se descartan y no se guardan
	if ((b->before.size() < getAlgorithmParameters()->minBlobHistory) || (b->getMaxArea() < getAlgorithmParameters()->minBlobArea) )
	{
		if ((b->before.size() < getAlgorithmParameters()->minBlobHistory))
		{
			std::cout << " Object DISCARDED  ID " << b->id << " LOW SAMPLES COUNT " << b->before.size() << "\n";

		}
		else
		{
			std::cout << " Object DISCARDED  ID " << b->id << " MIN SIZE NOT REACHED \n";
		}
		b->release();
		return;
	}
	

	std::cout << " Object DEAD  ID " << b->id << " GUID " << b->guid << "\n";
	b->life = -1;
	b->active = 0;
	// Si es invocado de afuera, lockeo los blobs hasta que se liberen
	b->locked = false;

	if (!b->tracker.empty())
	{
		b->tracker->clear();
		b->tracker.release();
		
	}

	auto params = getAlgorithmParameters();
	BlobsDetector* bd = NULL;
	for (int i = 0; i < params->detectors.size(); i++)
	{
		if (params->detectors[i]->id == params->idSceneDetector)
		{
			bd = params->detectors[i];
		}
	}

	if (this->storeBlobs)
	{
		// Este objeto no debe ser removido
		trackedBlobsToSave.push_back(b);
	}
	else
	{
		b->release();
	}
}

int BlobsHistory::GenerateColors(Mat& src)
{
	if (src.empty()) {
		std::cout << "unable to load an input image\n";
		return 1;
	}
	if (src.cols < 250)
		return 1;

	assert(src.type() == CV_8UC3);
	imshow("src image", src);

	cv::Mat reshaped_image = src.reshape(1, src.cols * src.rows);
	assert(reshaped_image.type() == CV_8UC1);

	cv::Mat reshaped_image32f;
	reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
	std::cout << "reshaped image 32f: " << reshaped_image32f.rows << ", " << reshaped_image32f.cols << std::endl;
	assert(reshaped_image32f.type() == CV_32FC1);

	cv::Mat labels;
	int cluster_number = 10;
	cv::TermCriteria criteria{ cv::TermCriteria::COUNT, 100, 1 };
	cv::Mat centers;
	cv::kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);

	show_result(labels, centers, src.rows, src.cols);



	return 0;
}

void BlobsHistory::show_result(const cv::Mat& labels, const cv::Mat& centers, int height, int width)
{
	std::cout << "===\n";
	std::cout << "labels: " << labels.rows << " " << labels.cols << std::endl;
	std::cout << "centers: " << centers.rows << " " << centers.cols << std::endl;
	assert(labels.type() == CV_32SC1);
	assert(centers.type() == CV_32FC1);

	cv::Mat rgb_image(height, width, CV_8UC3);
	cv::MatIterator_<cv::Vec3b> rgb_first = rgb_image.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> rgb_last = rgb_image.end<cv::Vec3b>();
	cv::MatConstIterator_<int> label_first = labels.begin<int>();

	cv::Mat centers_u8;
	centers.convertTo(centers_u8, CV_8UC1, 255.0);
	cv::Mat centers_u8c3 = centers_u8.reshape(3);

	while (rgb_first != rgb_last) {
		const cv::Vec3b& rgb = centers_u8c3.ptr<cv::Vec3b>(*label_first)[0];
		*rgb_first = rgb;
		++rgb_first;
		++label_first;
	}
	cv::imshow("tmp", rgb_image);
	cv::imwrite("C:\\Camaras\\result.jpg", rgb_image);
}

/*Le establece la fecha actual al blob dado por parametros*/
void BlobsHistory::computeDate(Blob* b, std::chrono::time_point<std::chrono::system_clock> tp) const
{
	// Reduce verbosity but let you know what is in what namespace
	b->creationTimePoint = tp;

	time_t tt = std::chrono::system_clock::to_time_t(tp);
	tm utc_tm = *gmtime(&tt);
	
	
	b->year = utc_tm.tm_year + 1900;
	b->month = utc_tm.tm_mon + 1;
	b->day = utc_tm.tm_mday;
	b->hour = utc_tm.tm_hour;
	b->minute = utc_tm.tm_min;
	b->second = utc_tm.tm_sec;
	//b->ms = utc_tm.subseconds().count();

	string ms = to_string(b->ms);
	if (ms.length() == 1)
	{
		ms = "00" + ms;
	}
	else if (ms.length() == 2)
	{
		ms = "0" + ms;
	}
		

	b->blobCreationTime = to_string(b->year) + '-' + to_string(b->month) + '-' + to_string(b->day) +
		                   " " + to_string(b->hour) + ':' + to_string(b->minute) + ':' + to_string(b->second) + '.' +ms;
	
}


void BlobsHistory::saveTo(int minHistory, int minArea, int frameN)
{
	int64 start = cv::getTickCount();
	int totalActiveBlobs = (int)this->activeblobs.size();

	if (totalActiveBlobs > 0)
	{
		for (int j = 0; j < totalActiveBlobs; j++)
		{
			Blob* b = activeblobs[j];

			// Si todavia estan activos, los descarto
			if (b->life > 0) continue;

			auto params = getAlgorithmParameters();
			// Remuevo este BLOB y lo pongo a guardar
			onObjectDead(b, frameN, params->computeTime(frameN, params->frameRate));
			activeblobs[j] = NULL;
		}

		// Me quedo con los blobs activos
		vector<Blob*> temp;
		activeblobs.swap(temp);
		activeblobs.clear();

		for (int j = 0; j < temp.size(); j++)
		{
			if (temp[j] != NULL)
				activeblobs.push_back(temp[j]);

		}
		temp.clear();
	}


}

void BlobsHistory::drawOnFrameFiltered(Mat& frame, int minHistory, int minArea)
{
	RNG rng(12345);

	for (int j = 0; j < this->activeblobs.size(); j++)
	{
		if (activeblobs[j]->lived < minHistory) continue;
		if (activeblobs[j]->getMaxArea() < minArea) continue;
		Scalar color = Scalar(0, 255, 0);
		Blob* b = activeblobs[j];
		Blob* lastB = b->before[b->before.size() - 1];
		putText(frame, std::to_string(b->id), b->topLeft,
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

		for (int k = 0; k < b->before.size(); k++)
		{
			lastB = b->before[k];

			cv::rectangle(frame, lastB->getRect(), cvScalar(250, 100, 250), 1);
		}

	}

}

void BlobsHistory::drawOnFrame(Mat& frame)
{
	RNG rng(12345);
	for (int j = 0; j < this->activeblobs.size(); j++)
	{
				
		Blob* b = activeblobs[j];
		
		if (!b->active) continue;
		if (b->life < 0) continue;
		if (!b->tracker) continue;

		Blob* lastB;
		if (b->before.size() == 0)
		{
			lastB = b;
		}
		else
		{
			lastB = b->before[b->before.size() - 1];
		}
		//putText(frame, IntToString2(b->id), lastB->topLeft,
		//	FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
		//Scalar color = b->color;
		cv::rectangle(frame, lastB->getRect(), cvScalar(250, 100, 250), 1);
		
		std::pair<std::string, float> tC = b->getTopColor();

		for (int i = 1; i < b->history.size(); i++)
		{
			cv::Point p0 = b->history[i - 1];
			cv::Point p1 = b->history[i ];
			//cv::line(frame, p0, p1, cv::Scalar(100, 250, 100), 1);
		}

		if (lastB->contour.size() > 0)
		{
			Scalar color;
			if (lastB->classes.size() > 0)
			{
				color = cv::Scalar(0, 255, 0, 1);
			}
			else
			{
				color = cv::Scalar(50, 200, 50, 1);
			}
		//	polylines(frame, lastB->contour, true, color, 1, 8);
		}
	}

}
