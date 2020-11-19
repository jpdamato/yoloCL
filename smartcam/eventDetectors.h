#ifndef BLOB_EVENT_DETECTORS
#define BLOB_EVENT_DETECTORS

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "../blobs/blobsHistory.h"

using namespace cv;
using namespace std;

#define EVENT_TYPE_CROSS_LINE 0
#define EVENT_TYPE_ENTER_AREA 1
#define EVENT_TYPE_LEAVE_AREA 2
#define EVENT_TYPE_DISSAPEAR_AREA 3
#define EVENT_TYPE_LONGTIME_SCENE 4
#define EVENT_TYPE_LEAVE_SCENE 5
#define EVENT_TYPE_BLOB_STORED 6

#define MIN_OBJECT_HISTORY 5



namespace trackingLib
{
	class BlobsDetector;

	struct Event
	{
	public:
		int id;
		int type;
		string eventUID;
		string trackedObjectId;
		string message;
		Blob* blob;
		string idDetector; //Es el id del trigger, que puede ser linecross, scenemotino, etc
		cv::Point position;
		cv::Rect boundingBox;
		int nroFrame;
		Mat frame;
		std::chrono::time_point<std::chrono::system_clock> datetime;
		std::string path;

		static int getID();
		Event();
		static string addQuotes(std::string s);
		std::string getAsJSON();
		Event(int _id, int _type, string trackedOId, string _mess, Blob* b, int nroframe, cv::Rect boundingbox, string idDetectorAlgorithm, cv::Point pos, std::chrono::time_point<std::chrono::system_clock> now, Mat frame);
		void save(std::string dir, std::string name);
		bool similar(Event e);


	};

	class BlobsDetector
	{
	public:
		std::string id;
		std::vector<Event> detectedEvents;

		virtual std::vector<Event> detect(vector<Blob*>& activeblobs, Mat& frame, int nFrame, std::chrono::time_point<std::chrono::system_clock> now) = 0;
		virtual void drawOn(Mat& frame) = 0;

	};
	class LineCrossDetector :public BlobsDetector
	{
	public:
		cv::Point2f start;
		cv::Point2f end;
		int wasCrossed;
		LineCrossDetector(int _id, cv::Point2f s, cv::Point2f e) { id = _id;  start = s; end = e; wasCrossed = 0; }
		std::vector<Event> detect(vector<Blob*>& activeblobs, Mat& frame, int nFrame, std::chrono::time_point<std::chrono::system_clock> now);
		void drawOn(Mat& frame);
	};

	class AreaDetector :public BlobsDetector
	{
	public:
		std::vector<cv::Point2f> contour;
		std::vector<Blob*> innerblobs;
		int wasCrossed;
		AreaDetector(int _id, std::vector<cv::Point2f> cnt) { id = _id;  cnt.swap(contour); }
		//	void CheckTimeOnScene(int nFrame, std::chrono::time_point<std::chrono::system_clock> now, int i, Blob* b);
		std::vector<Event> detect(vector<Blob*>& activeblobs, Mat& frame, int nFrame, std::chrono::time_point<std::chrono::system_clock> now);
		void drawOn(Mat& frame);
	};

	class SceneDetector :public BlobsDetector
	{
	public:
		int _time;
		SceneDetector(string _id, int time) { id = _id;  _time = time; }
		void CheckTimeOnScene(int nFrame, std::chrono::time_point<std::chrono::system_clock> now, int i, Blob* b, Mat frame);
		std::vector<Event> detect(vector<Blob*>& activeblobs, Mat& frame, int nFrame, std::chrono::time_point<std::chrono::system_clock> now);
		void drawOn(Mat& frame);
	};

	class EventManager
	{
	private:
		EventManager() { lockBlobs = 0; };
		int innerIndex = 0;

	public:
		std::vector<Event> events;

		Event lastEvents[10];
		int lockBlobs;



		static EventManager* getInstance();
		void addNewEvent(Event e);
		void addNewEvents(std::vector<Event> es);
		static void printError(string s);

	};

}
#endif
