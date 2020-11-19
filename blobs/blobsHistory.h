//opencv

#ifndef BLOBS_HISTORY_H
#define BLOBS_HISTORY_H

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <iterator>
#include <thread>

using namespace cv;
using namespace std;

#define M_MOG  1
#define M_MOG2 2
#define OURS 3
#define VIBE 4
#define OURS_PLUS 5
#define UPDATE_FRAME_RATE 1

#define APP_SOURCE_DLL 1
#define APP_SOURCE_EXE 2

#define ACTIVE_BLOBS 1
#define DEAD_BLOBS 2

#define BLOB_AREA_OVERLAP 0.05
#define EXPORT_IMG_FORMAT_FILE "file"
#define EXPORT_IMG_FORMAT_SHEET "sheet"
#define EXPORT_IMG_FORMAT_BOTH "both"

#define INITIAL_BLOB_LIFE 40

namespace trackingLib
{

	class BlobsDetector;

	class frameReference
	{
	public:
		cv::Mat allocatedFrame;
		int referenceCounter;
		void release();
	};

	class frameReferenceManager
	{
	public:
		std::vector<frameReference*> frameRef;

		static frameReferenceManager* getInstance();
		frameReference* addReference(cv::Mat& matRef);
		static void releaseRef(frameReference* ref);
	};

	class scImage;

	//**********************************************
	// struct trackedBlob  
	//   * Estructura temporal de cada blob
	//**********************************************
	struct trackedBlob {
		int id;
		string name;
		cv::Mat blob;
		//frameReference* blob;
		Rect2f r;
		vector<Point> contour;
		int assigned;
		int objectPixels;
		int probPixels;
		int backgroundPixels;
		std::tm dateTime;
		std::vector<cv::KeyPoint> keypoints;
		std::vector<std::pair<std::string, float>> classes;
	};


	//**********************************************
	// CLASS BLOBS  
	//   * Informacion de cada BLOB
	//**********************************************
	class Blob;

	class blobFeature
	{
	public:
		int id;
		Blob* b;
		string type;

		virtual void drawOnFrame(cv::Mat& frame, cv::Scalar color, int params) = 0;

	};

	class Blob
	{
	protected:
		Rect2d r;
		Blob(int frameN);
	public:
		uint id;
		int assigned;
		string name;
		string guid;
		int active;
		cv::Mat cutImg;
	
		float xPercent;
		float yPercent;
		float widthPercent;
		float heightPercent;
		cv::Scalar color;
		int life;
		int savedBlobIndex;
		int frameNumber;
		float* colorPalette;
		//RGBA
		//Vec4b color;
		Vec2f dir;
		Vec2f speed;
		Vec2f latlong;
		int lived;
		bool saved;
		vector<blobFeature*> features;
		//vector<int> harrisPoints;
		vector<Point2f> history;
		vector<cv::Rect2d> historyR;
		vector<Point> contour;
		vector<Vec4i> hierarchy;
		Point topLeft;
		//float width, height;
		vector<Blob*> before;
		bool longTime = false;//Indica si ya se notific que el objeto supero el tiempo en escena
		int year;
		int month;
		int day;
		string blobCreationDate;

		int hour;
		int minute;
		int second;
		int ms;
		string blobCreationTime;
		int tag;
		int locked;
		Blob* parent;
		std::chrono::time_point<std::chrono::system_clock> timeEnterZone;
		std::chrono::time_point<std::chrono::system_clock> timeExitZone;
		std::chrono::time_point<std::chrono::system_clock> creationTimePoint;
		int64 updateTracker;
		Ptr<Tracker> tracker;
		int trackerType;
		std::vector<cv::KeyPoint> keypoints;
		std::vector<std::pair<float ,std::string >> classes;
		////////// METHODS
		void updatePos(cv::Rect2d r, cv::Size processingSize);
		bool updatePos(cv::Rect2d r, cv::Mat& frame, cv::Mat mask, std::vector<Point>& contour, bool saveCut, int frameN);
		string getFilename();
		string getBlobCreationDate();
		string getBlobCreationTime();
		Blob* getLast();
		cv::Rect2d getRect();
		Point getCenter();
		void release();
		double getMaxArea();
		double getMeanArea();
		cv::Vec2f getDir();
		float centerX();
		float centerY();
		Blob* clone();
		
		static Blob* getInstance(int frameN);
		static int getNumberOfBlobs();
	
		
		std::pair<std::string, float> getTopColor();
	};

	//**********************************************
	// CLASS BlobsByFrame
	//   - Lista de BLOBS extraidos en cada frame
	//**********************************************
	class BlobsByFrame
	{
	public:

		int iLowH = 40;
		int iHighH = 60;

		int iLowS = 150;
		int iHighS = 255;

		int iLowV = 60;
		int iHighV = 255;


		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		std::vector<cv::Vec3b> mainColors;
		std::vector<Blob*> trackBlobs;


		int nFrame;
		int blobsMemSize;
		BlobsByFrame();
		void computeMainColors(Mat &src);
		void findContours(Mat& frame, Mat& mask, int nFrame, int minArea = 100);
		void drawOnFrame(Mat& frame);
		void enqueue(Mat& frame, Mat& mask, string dir, int firstID, int minSize);
		void clear();

	};


	//**********************************************
	// CLASS BlobsHistory
	//   - Lista de todos los BLOBS activos 
	//**********************************************

	class BlobsHistory
	{
	public:
		vector<BlobsByFrame*> history;
		//Lleva todos los TrackedBlob activos
		vector<Blob*> activeblobs;
		bool storeBlobs;
		vector<Blob*> trackedBlobsToSave;
		int frameRate;
		int videoWidth;
		int videoHeight;
		int idGenerator;
		static BlobsHistory* getInstance();
		std::chrono::time_point<std::chrono::system_clock> startDate;
		BlobsHistory();
		BlobsHistory(int frameR, int videoWidth, int videoHeight, string idSceneDetector);
		virtual void findAndAddBlobs(cv::Mat& frame, cv::Mat& mask, int nframe);
		virtual std::vector<Blob*> addBlobs(BlobsByFrame*, int frame, std::chrono::time_point<std::chrono::system_clock> now, cv::Size originalSize, cv::Size processingSize);
		virtual void drawOnFrame(Mat& frame);
		void drawOnFrameFiltered(Mat& frame, int minHistory, int minArea);
		void saveTo(int firstID, int minSize, int frameN);
		void computeMovement();
		int GenerateColors(Mat& blob);
		void show_result(const cv::Mat& labels, const cv::Mat& centers, int height, int width);
		void onObjectDead(Blob* b, int nroFrame, chrono::time_point<chrono::system_clock> now);
		void computeDate(Blob* b, std::chrono::time_point<std::chrono::system_clock> time) const;
		void updateBlobsState(int frame, std::chrono::time_point<std::chrono::system_clock> now);
		void compactMem();

	};


}

//*****************************************
// METODOS VARIOS
//*****************************************
std::chrono::time_point<std::chrono::system_clock> initTM(string filename);
std::string encodeDate(int frameN, int frameR);
std::string encodeDateTime(int frameN, int frameR);
std::string encodeDateTimeStr(int frameN, int frameR);
std::string encodeDateBlob(trackingLib::Blob* b);
int createAlphaMat(Mat &mat, vector<cv::Point>&  contours, string filename, int xorig, int yorig, int objectID);
std::string IntToString2(int value);
trackingLib::Blob* findSimilarBlob(std::vector<trackingLib::Blob*>& activeblobs, trackingLib::Blob* b);
std::vector<trackingLib::Blob*> findSimilarBlobs(std::vector<trackingLib::Blob*>& activeblobs, std::vector<trackingLib::Blob*>& bbs);
double weight(trackingLib::Blob* b1, trackingLib::Blob* b);
double areaSuperposed(Rect r1, Rect r2);

void addSlashAtEnd(std::string& folderToSave);
string addQuotes(std::string s);
bool fileExists(const std::string& name);
#endif