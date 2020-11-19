#include "date.h"
#include "trackingLib.h"

#include "../blobs/blobsHistory.h"
#include "../blobs/Export.h"
#include "tinyxml.h"
#include "eventDetectors.h"
#include "tz.h"

using namespace std;
using namespace cv;

// preferable to wrap Vibe.h in a namespace
extern "C"  char* getVibeVersion()
{
#ifdef  DEBUG

	return (char*)"trackingLib 1.06 DEBUG";
#else

	return (char*)"trackingLib 1.06 RELEASE";
#endif //  DEBUG
}

#define STRING_SIZE 512
AlgorithmParameters* _params;
//VideoCapture _cap;
Mat _frame;
Mat frameN;
int _captureMode = 0;
int _stopProcess = 0;
Event _eventRef;
int _hasToStop = 0;
int _Finished = 0;
string FullPathGPU;
string idSceneDetector;

char _stringRef[STRING_SIZE];
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
								 //create Background Subtractor objects
BlobsHistory* bh = NULL;
cv::Mat _matBackRef, _blobMatRef;



BlobsHistory* getBlobsHistoryInstance()
{
	return bh;
}

///Usado para pasaje de información entre DLL y MotionDetector
char* storeStringInGlobalMem(string txt)
{
	for (int i = 0; i < STRING_SIZE; i++) _stringRef[i] = 0;

	for (int i = 0; i < MIN(STRING_SIZE, txt.size()); i++)
		_stringRef[i] = txt[i];
	return _stringRef;
}


float relativePositionX(int x)
{
	if (_params)
	{
		return (x * 100) / (float)(_params->processRes.width);
	}
	else
		return x;
}

float relativePositionY(int y)
{
	if (_params)
	{
		return (y * 100) / (float)(_params->processRes.height);
	}
	else
		return y;
}

AlgorithmParameters* getAlgorithmParameters()
{
	if (!_params)
	{
		_params = new AlgorithmParameters();
	}
	return _params;
}


extern "C" int initializeTrackingLib(char* fullPathGPU)
{
	const size_t last_slash_idx = std::string(fullPathGPU).rfind('\\');
	if (std::string::npos != last_slash_idx)
	{
		FullPathGPU = std::string(fullPathGPU).substr(0, last_slash_idx);
	}	
	string configFile(fullPathGPU);
	std::cout << configFile;
	

	_hasToStop = true;
	if (_params == NULL)
		_params = new AlgorithmParameters();
	else
		_params->clear();


	TiXmlDocument doc(configFile.c_str());
	bool loadOkay = doc.LoadFile();
	if (!loadOkay)
	{
		std::cout << "Could not load test file Config. Error='%s'. Exiting.\n" << doc.ErrorDesc() << "\n";
		return -1;
	}
	//parseParameters(doc, _params);
	return 1;
}

extern "C" int setCaptureMode(int mode)
{
	_captureMode = mode;
	return 1;
}

extern "C" Blob* getEventObject(Event* e)
{
	if (e != NULL)
		return e->blob;
	else
		return NULL;
}

extern "C"  int getEventsCount()
{
	return (int)EventManager::getInstance()->events.size();
}

extern "C"  Event* consumeEvent()
{
	if (EventManager::getInstance()->events.size() == 0)
		return NULL;
	_eventRef = EventManager::getInstance()->events[0];
	EventManager::getInstance()->events.erase(EventManager::getInstance()->events.begin() + 0);
	return &_eventRef;
}
extern "C"  int stopProcess()
{
	_hasToStop = 1;
	return 1;
}

//Indica si los recursos fueron liberados
extern "C" int isFree()
{
	return _Finished;
}
extern "C" char* getLastError()
{
	string stmp = getLibError().c_str();

	storeStringInGlobalMem(stmp);

	return _stringRef;
}



extern "C"  char* getLastMessage()
{
	storeStringInGlobalMem(_params->lastMessage);

	return _stringRef;
}

extern "C"  Mat* getFrame()
{
	_params->cap >> _matBackRef;
	_params->unlock();
	return &_matBackRef;
}

extern "C" char* getFrameInfo()
{
	storeStringInGlobalMem(_params->frameInfo);
	return _stringRef;
}

extern "C" int getNroFrame()
{
	return _params->idFrame;
}


extern "C" char* getActualDate()
{
	///http://www.modernescpp.com/index.php/time-point

	chrono::time_point<chrono::system_clock> now = _params->computeTime(_params->idFrame, _params->frameRate);
	std::time_t tp = std::chrono::system_clock::to_time_t(now);
	std::string sTp = std::asctime(std::gmtime(&tp));
	storeStringInGlobalMem(sTp);
	return _stringRef;
}

extern "C" long getDurationActualDate()
{
	///http://www.modernescpp.com/index.php/time-duration

	chrono::time_point<chrono::system_clock> now = _params->computeTime(_params->idFrame, _params->frameRate);
	auto duration = now.time_since_epoch();
	return duration.count();
}

using namespace std;
using namespace cv;





extern "C"   uint getBlobID(Blob* b)
{
	if (b == NULL)
		return -1;
	else
		return b->id;
}

extern "C"  int getTrackBlobsCount(Blob* b)
{
	return (int)b->before.size();
}

extern "C"  Blob* getTrackBlobAt(Blob* b, int index)
{
	// Es un blob padre
	if (b->parent == NULL)
		return b->before[index];
	else
		return NULL;
}

extern "C" void getSpeedBlob(Blob *b, float* sp)
{
	sp[0] = 0;
	sp[1] = 0;
	sp[0] = b->speed[0];
	sp[1] = b->speed[1];
}

extern "C" void getDirBlob(Blob *b, float* dir)
{
	dir[0] = 0;
	dir[1] = 0;
	dir[0] = b->dir[0];
	dir[1] = b->dir[1];
}

extern "C" char* getBlobCreationDate(Blob *b) {
	string creation = b->getBlobCreationDate();
	storeStringInGlobalMem(creation);
	return _stringRef;
}
extern "C" char* getBlobCreationTime(Blob* b)
{
	string creation = b->getBlobCreationTime();
	storeStringInGlobalMem(creation);
	return _stringRef;

	/*
		strftime(dateTemp, sizeof(dateTemp), "%d-%m-%Y %I:%M:%S", (const tm*)&b->dateTime);

		int ms = (int)b->ms;
		std::string s(dateTemp);
		s = s + "." + IntToString(ms);

		return storeStringInGlobalMem(s);*/
}

//////////////////////////////////////////////////
/// MAT INTERFACE
//////////////////////////////////////////////////


extern "C"  Mat* getBiggerBlobBitmap(Blob* b)
{
	// Es un blob padre
	if (b->parent == NULL)
	{
		int bestBlobIndex = 0;
		for (int i = 0; i < b->before.size(); i++)
		{
			if (b->before[bestBlobIndex]->getRect().area() < b->before[i]->getRect().area())
				bestBlobIndex = i;
		}
		//		_blobMatRef = b->before[bestBlobIndex]->cutImg->allocatedFrame.clone();
		//_blobMatRef = b->before[bestBlobIndex]->cutImg.clone();

		return &_blobMatRef;
	}
	else
	{
		//		_blobMatRef = b->cutImg->allocatedFrame.clone();
		//_blobMatRef = b->cutImg.clone();

		return &_blobMatRef;
	}

}

extern "C"  void startShow()
{
	_params->show = true;
}

extern "C"  void stopShow()
{
	_params->show = false;
}

extern "C"   Mat* getBackgroundFrame()
{
	_matBackRef = _params->getBackFrame();

	return &_matBackRef;
}

extern "C"  int getFrameW(Mat* f)
{
	if (f == nullptr)
		return 0;
	return f->cols;
}
extern "C"  int getFrameH(Mat* f)
{
	if (f == nullptr)
		return 0;
	return f->rows;
}
extern "C"  int getFrameC(Mat* f)
{
	if (f == nullptr)
		return 0;
	return f->channels();
}

extern "C"  uchar* getFrameData(Mat* f)
{
	return f->data;
}

extern "C"  int   getProcessinResolutionWidth(AlgorithmParameters* params) { return params->originalFrameSize.width; }
extern "C" int   getProcessinResolutionHeight(AlgorithmParameters* params) { return params->originalFrameSize.width; }

extern "C"  int   getOriginalResolutionWidth(AlgorithmParameters* params) { return params->originalFrameSize.width; }
extern "C"  int   getOriginalResolutionHeight(AlgorithmParameters* params) { return params->originalFrameSize.height; }
//////////////////////////////////////////////////
/// EVENT INTERFACE
//////////////////////////////////////////////////
extern "C"  int getEventID(Event* e)
{
	return e->id;
}

extern "C" char* getPathSheet(Event* e)
{
	return storeStringInGlobalMem(e->path);
}
extern "C"  char* getEventMessage(Event* e)
{

	return storeStringInGlobalMem(e->message);
}

extern "C"  int getEventType(Event* e)
{
	return e->type;
}

extern "C"  int getEventLocationX(Event* e)
{
	return relativePositionX(e->position.x);
}

extern "C"  int getEventLocationY(Event* e)
{
	return relativePositionY(e->position.y);
}

extern "C"  int getEventBoundingBoxWidth(Event* e)
{
	return relativePositionX(e->boundingBox.width);
}

extern "C"  int getEventNroFrame(Event* e)
{
	return e->nroFrame;
}

extern "C"  long getEventDateTime(Event* e)
{
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(e->datetime);
	auto epoch = now_ms.time_since_epoch();
	auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);

	return value.count();
}


extern "C"  int getEventBoundingBoxHeight(Event* e)
{
	return relativePositionX(e->boundingBox.height);
}

extern "C"  char* getEventDetectorID(Event* e)
{
	return storeStringInGlobalMem(e->idDetector);
}



extern "C"  char* getObjectID(Event* e)
{
	return storeStringInGlobalMem(e->trackedObjectId);
}

//////////////////////////////////////////////////
/// AlgorithmParameters
//////////////////////////////////////////////////

Mat AlgorithmParameters::getBackFrame()
{
	unlock();
	return this->_orig;
}

void AlgorithmParameters::lock()
{
	frameLocked = true;
}
void AlgorithmParameters::unlock()
{
	frameLocked = false;
}

void AlgorithmParameters::setBackFrame(Mat& frame, string info)
{
	this->originalFrameSize = cv::Size(frame.cols, frame.rows);
	if (frameLocked)
	{
		return;
	}
	this->_orig = frame.clone();
	this->frameInfo = info;
}

void AlgorithmParameters::clear()
{
	ID = 0;
	idFrame = 0;
	ommitFrames = 100;
	filesToProcess.clear();
	attemptingToReconnect = 0;
	gpuWasInitialized = 0;
}

AlgorithmParameters::AlgorithmParameters()
{
	appSource = APP_SOURCE_EXE;
	this->imageExtension = ".jpg";
	this->minBlobArea = 1000;
	this->minBlobHistory = 40;
	
}

/*Retorna el datetime actual de reproduccion.
Si se trata de un video, toma los parametros frameN y frameRate para calcular a partir de la fecha startDate, el tiempo actual
Si se trata de un video en vivo, toma el valor de now()*/
std::chrono::time_point<std::chrono::system_clock> AlgorithmParameters::computeTime(int frameN, int frameRate)
{
	std::chrono::time_point<std::chrono::system_clock> dateTime;
	if (this->datasourceType == SOURCE_VIDEO_FILE)
	{
		long ml = (1000)*(static_cast<double>(frameN) / frameRate);
		dateTime = this->startDate;
		dateTime += std::chrono::milliseconds(ml);

	}
	else //stream
	{
		auto t3h = std::chrono::hours(3);
		dateTime = std::chrono::system_clock::now() - t3h;
	}

	return dateTime;

}

void parseParametersBgs(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("bgs") != NULL)
	{
		TiXmlElement* node = root->FirstChildElement("bgs")->FirstChildElement("modelUpdate");
		

		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			string rs(node->IterateChildren(0)->Value());
			params->BGS_modelUpdate = StringToInt(rs);
		}

		node = root->FirstChildElement("bgs")->FirstChildElement("mask");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			string rs(node->IterateChildren(0)->Value());
			std::replace(rs.begin(), rs.end(), '/', '\\');

			params->backgroundMask = imread(rs);
		}
		else
		{
		}

	}
}

void parseParametersPlatform(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("platform") != NULL)
	{
		TiXmlElement* node = root->FirstChildElement("platform")->FirstChildElement("platformProcessingIndex");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			string rs(node->IterateChildren(0)->Value());
			params->platformProcessingIndex = StringToInt(rs);
		}

		node = root->FirstChildElement("platform")->FirstChildElement("deviceProcessingIndex");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			string rs(node->IterateChildren(0)->Value());
			params->deviceProcessingIndex = StringToInt(rs);
		}


	}
	else
	{
		params->platformProcessingIndex = 0;
		params->deviceProcessingIndex = 0;
	}
}

void parseParametersDetectors(AlgorithmParameters* params, TiXmlElement* root)
{
	params->detectors.clear();
	if (root->FirstChildElement("blobDetectors") != NULL)
	{
		TiXmlElement* node = root->FirstChildElement("blobDetectors")->FirstChildElement("LineDetector");
		std::vector<string> st;
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			TiXmlNode* ele2 = 0;
			while ((ele2 = node->IterateChildren(ele2)) != 0)
			{
				string rs(ele2->FirstChild()->Value());
				std::vector<cv::Point2f> sp;
				st = splitString(rs, ',');
				string id = st[0];

				for (int i = 1; i < st.size(); i += 2)
				{
					cv::Point2f p;
					p.x = stof(st[i]) / 100.0f;
					p.y = stof(st[i + 1]) / 100.0f;
					sp.push_back(p);
				}

				BlobsDetector *bd = new LineCrossDetector(params->detectors.size(), sp[0], sp[1]);
				bd->id = id;
				params->detectors.push_back(bd);

			}
		}

		node = root->FirstChildElement("blobDetectors")->FirstChildElement("AreaDetector");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			TiXmlNode* ele2 = 0;
			while ((ele2 = node->IterateChildren(ele2)) != 0)
			{
				string rs(ele2->FirstChild()->Value());
				std::vector<cv::Point2f> sp;
				st = splitString(rs, ',');
				string id = st[0];
				for (int i = 1; i < st.size(); i += 2)
				{
					cv::Point2f p;
					p.x = stof(st[i]) / 100.0f;
					p.y = stof(st[i + 1]) / 100.0f;
					sp.push_back(p);
				}
				BlobsDetector *bd = new AreaDetector(params->detectors.size(), sp);
				bd->id = id;
				params->detectors.push_back(bd);
			}


		}
		
		node = root->FirstChildElement("blobDetectors")->FirstChildElement("SceneDetector");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			TiXmlNode* ele2 = 0;
			while ((ele2 = node->IterateChildren(ele2)) != 0)
			{
				string rs(ele2->FirstChild()->Value());
				std::vector<cv::Point2f> sp;
				st = splitString(rs, ',');
				string id = st[0];

				std::string::size_type sz;   // alias of size_t
				int time = std::stoi(st[1], &sz);


				BlobsDetector *bd = new SceneDetector(id, time);
				bd->id = id;
				idSceneDetector = id;
				params->detectors.push_back(bd);
				params->idSceneDetector = id;
			}


		}
	}
}

void parseParametersBlobs(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("blobs") != NULL)
	{
		TiXmlElement* node = root->FirstChildElement("blobs")->FirstChildElement("minBlobArea");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			string rs(node->IterateChildren(0)->Value());
			params->minBlobArea = StringToInt(rs);
		}

		node = root->FirstChildElement("blobs")->FirstChildElement("minBlobHistory");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			string rs(node->IterateChildren(0)->Value());
			params->minBlobHistory = StringToInt(rs);
		}
	}
}

void parseParametersOmmitFrames(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("ommitFrames") != NULL && (root->FirstChildElement("ommitFrames")->IterateChildren(0) != NULL))
	{
		string rs(root->FirstChildElement("ommitFrames")->IterateChildren(0)->Value());
		params->ommitFrames = StringToInt(rs);

	}
	else
	{
		params->ommitFrames = 100;
	}
}

void parseParametersStep(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("step") != NULL && (root->FirstChildElement("step")->IterateChildren(0) != NULL))
	{
		string rs(root->FirstChildElement("step")->IterateChildren(0)->Value());
		params->step = StringToInt(rs);

	}
	else
	{
		params->step = 1;
	}
}

void parseParametersSleepTime(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("sleepTime") != NULL && (root->FirstChildElement("sleepTime")->IterateChildren(0) != NULL))
	{
		string rs(root->FirstChildElement("sleepTime")->IterateChildren(0)->Value());
		params->sleepTime = StringToInt(rs);

	}
	else
	{
		params->sleepTime = 0;
	}
}

void parseParametersAppSource(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("appSource") != NULL && (root->FirstChildElement("appSource")->IterateChildren(0) != NULL))
	{
		string rs(root->FirstChildElement("appSource")->IterateChildren(0)->Value());
		params->appSource = StringToInt(rs);

	}
	else
	{
		params->appSource = 0;
	}
}

void parseParametersImageFormat(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("imageFormat") != NULL && (root->FirstChildElement("imageFormat")->IterateChildren(0) != NULL))
	{
		string rs(root->FirstChildElement("imageFormat")->IterateChildren(0)->Value());
		if (rs == "file")
	     	params->imageFormat = EXPORT_IMG_FORMAT_FILE;
		else if (rs == "both")
			params->imageFormat = EXPORT_IMG_FORMAT_BOTH;
		else 
			params->imageFormat = EXPORT_IMG_FORMAT_SHEET;
	}
	else
	{
		params->imageFormat = EXPORT_IMG_FORMAT_FILE;
	}
}

void parseParametersImageExtension(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("imageExtension") != NULL && (root->FirstChildElement("imageExtension")->IterateChildren(0) != NULL))
	{
		string rs(root->FirstChildElement("imageExtension")->IterateChildren(0)->Value());
		params->imageExtension = rs;
	}
	else
	{
		params->imageExtension = ".png";
	}
}

void parseParametersMediaPath(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("mediaPath") != NULL && (root->FirstChildElement("mediaPath")->IterateChildren(0) != NULL))
	{
		params->mediaPath = root->FirstChildElement("mediaPath")->IterateChildren(0)->Value();
		std::replace(params->mediaPath.begin(), params->mediaPath.end(), '/', '\\');
	}
	else
	{
		params->mediaPath = "";
	}
}

void parseParametersOutputDir(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("outputDir") != NULL && (root->FirstChildElement("outputDir")->IterateChildren(0) != NULL))
	{
		params->outputDir = root->FirstChildElement("outputDir")->IterateChildren(0)->Value();
		std::replace(params->outputDir.begin(), params->outputDir.end(), '/', '\\');
	}
	else
	{
		params->outputDir = "";
	}
}

void parseParametersShow(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("show") != NULL)
	{
		TiXmlElement* node = root->FirstChildElement("show");
		if (node != NULL && (node->IterateChildren(0) != NULL))
		{
			string rs(node->IterateChildren(0)->Value());
			params->show = (rs == "true");
		}
		else
		{
			params->show = false;
		}
	}
}

void parseParametersProcessingResolution(AlgorithmParameters* params, TiXmlElement* root)
{
	if (root->FirstChildElement("processingResolution") != NULL)
	{
		string rs(root->FirstChildElement("processingResolution")->IterateChildren(0)->Value());
		if (rs == "320x180")	params->processRes = cv::Point(320, 180);
		else if (rs == "640x360")	params->processRes = cv::Point(640, 360);
		else if (rs == "1280x720")	params->processRes = cv::Point(1280, 720);
		else if (rs == "1920x1080")	params->processRes = cv::Point(1920, 1080);
		else 	params->processRes = cv::Point(-1, -1);
	}
	else
	{
		params->processRes = cv::Point(-1, -1);
	}
}

void parseParametersFiles(AlgorithmParameters* params, TiXmlElement* root)
{
	TiXmlNode* ele = 0;
	if (root->FirstChildElement("files") != NULL && (root->FirstChildElement("files")->IterateChildren(0) != NULL))
	{
		while ((ele = root->FirstChildElement("files")->IterateChildren(ele)) != 0)
		{
			string st = ele->FirstChild()->Value();
			if (st.find("rtsp") != std::string::npos)
			{
				params->datasourceType = SOURCE_STREAMING;
			}
			else
			{
				replace(st.begin(), st.end(), '/', '\\');
				params->datasourceType = SOURCE_VIDEO_FILE;
			}
			params->filesToProcess.push_back(st);
			cout << " reading parameters" << st << "\n";
		}
	}
}


bool fileExist(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

bool initModelBGS(Mat imgGray, int& result)
{
	if (!_params->gpuWasInitialized)
	{
		omp_set_num_threads(16);
		//_params->ocl_bgs.updateStep = _params->BGS_modelUpdate;


		if (endsWith(FullPathGPU, "\\") == false)
			FullPathGPU.append("\\");
		std::string bgocl = "D:\\Proyects\\smartCAM\\pcBox\\scMotionDetector\\bin\\x64\\Release\\gpuVibe\\bgsocl.cl";
		//string bgsocl("C:\\FFMPEGexe\\gpu\\bgsocl.cl");		
		if (!fileExist(bgocl)) {
			logMessage(string("No se encontró el archivo ").append(bgocl), MESSAGE_ERROR);
			result = EXIT_FAILURE;
			return false;
		}
		int result = 0; // _params->ocl_bgs.init(bgocl, 256, _params->platformProcessingIndex, _params->deviceProcessingIndex);
		if (result != EXIT_SUCCESS)
		{
			logMessage(string("Failed to load GPU program on path ").append(bgocl), MESSAGE_ERROR);
			result = result;
			return false;
		}
		_params->gpuWasInitialized = 1;
	//	_params->ocl_bgs.initialize(imgGray);
		pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
	}
	result = EXIT_SUCCESS;
	return true;
}

//Inicia la mascara para el background y retorna el frame con la mascara aplicada en imgGray, listo para ser usado por initModelBGS
void InitMask(Mat& frame, Mat& imgGray)
{
	if (!_params->backgroundMask.empty())
	{
		cv::resize(_params->backgroundMask, _params->backgroundMask, _params->processRes);
	}
	imgGray = Mat::zeros(frame.size(), CV_8UC1);
	if (!_params->backgroundMask.empty())
	{
		cv::bitwise_and(frame, _params->backgroundMask, frame);
	}
	if (frame.channels() == 3)
		cv::cvtColor(frame, imgGray, CV_RGB2GRAY);
}

int initialize(AlgorithmParameters* params, Mat &frame, int sourceApp)
{
	EventManager::getInstance()->lockBlobs = (sourceApp == APP_SOURCE_DLL);

	if (params->processRes.width < 0)
		params->processRes = frame.size();
	if (frame.rows == 0 || frame.cols == 0) {
		logMessage(string("El frame llegó vacio y no se pudo utilizar"), MESSAGE_ERROR);
		return EXIT_FAILURE;
	}
	if (params->processRes.width == 0)
	{
		params->processRes.width = frame.cols;
		params->processRes.height = frame.rows;

	}
	cv::resize(frame, frame, params->processRes);

	params->appSource = sourceApp;
	_params = params;
	Mat imgGray;
	InitMask(frame, imgGray);

	// Inicializo el modelo
	int result;
	if (!initModelBGS(imgGray, result)) return result;

	bh = new BlobsHistory(params->frameRate, frame.cols, frame.rows,idSceneDetector);
	bh->startDate = params->startDate;


	return EXIT_SUCCESS;

}

cv::Mat  imgGray, bf, fgmask, fgMaskMOG2, bgimg, fgimg, renderFrame;
cv::Mat colorImage, black;

/*
* Case Sensitive Implementation of endsWith()
* It checks if the string 'mainStr' ends with given string 'toMatch'
*/
bool endsWith(const std::string &mainStr, const std::string &toMatch)
{
	if (mainStr.size() >= toMatch.size() &&
		mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
		return true;
	else
		return false;
}

extern "C" int closeCapture()
{
	if (_params == NULL) return 1;
	_params->cap.release();
	cv::destroyAllWindows();
	_Finished = 1;
	return 1;

}
extern "C" Mat* getFrameN(int index)
{
	if (_params->filesToProcess.empty()) return NULL;
	if (_params->filesToProcess.size() <= 0) return NULL;
	string f = _params->filesToProcess[0];
	_params->cap.open(f);
	if (!_params->cap.isOpened())
	{
		logMessage("Failed to open Camera o video File " + _params->filesToProcess[0], MESSAGE_ERROR);
		return NULL;
	}
	Mat tmp;
	for (int i = 0; i <= index; i++)
	{
		_params->cap >> tmp;
	}

	frameN = tmp.clone();
	_params->cap.release();

	return &frameN;
}

extern "C" int openCapture(int fileIndex)
{
	if (_params->filesToProcess.empty()) return EXIT_FAILURE;
	if (_params->filesToProcess.size() <= fileIndex) return EXIT_FAILURE;
	string f = _params->filesToProcess[fileIndex];
	_params->cap.open(f);
	if (!_params->cap.isOpened())
	{
		logMessage("Failed to open Camera o video File " + _params->filesToProcess[0], MESSAGE_ERROR);
		return EXIT_FAILURE;
	}
	_params->cap >> _frame;
	_hasToStop = false;
	_Finished = 0;
	int res = initialize(_params, _frame, APP_SOURCE_DLL);
	if (res != EXIT_SUCCESS)
		return res;
	_params->setBackFrame(_frame, "0");
	std::cout << "---------------------------------------" << "\n";
	std::cout << "---------------------------------------" << "\n";
	std::cout << " .... STARTING NEW VIDEO FILE ......." << "\n";
	std::cout << _params->filesToProcess[fileIndex] << "\n";
	std::cout << "---------------------------------------" << "\n";
	std::cout << "---------------------------------------" << "\n";

	
	return EXIT_SUCCESS;
}
