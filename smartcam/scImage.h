#pragma once
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "../blobs/blobsHistory.h" 


namespace trackingLib
{
	// *******************************
	// Single Image class for sheet
	// *******************************
	class scImage
	{
	public:
		scImage(trackingLib::Blob* b, std::string name, size_t tx, size_t ty, size_t tw, size_t th, size_t r, cv::Size frameSize);
		~scImage();

		size_t getTx();
		size_t getTy();
		size_t getTw();
		size_t getTh();
		size_t getR();
		trackingLib::Blob* blob;
		std::string name;
		std::string uuid;
		std::string date;
		cv::Size propSize;
		cv::Point2f propPos;

	private:
		size_t tx;
		size_t ty;
		size_t tw;
		size_t th;
		size_t r;
	};

	// *******************************
	//  Image class for sheet
	// *******************************
	class PackedImage
	{
	public:
		std::vector<scImage> images; // xml data of the images


		cv::Size size;
		cv::Mat rend; // texture to render the sprite sheet

		PackedImage(std::string name, int maxW, int maxH);
		static void buildImageFromFiles(std::vector<std::string> listAll, std::string dir, std::string imgformat);
		void GenerateDocumentTBlob(std::string dir, std::string imgformat, std::string outFormat, trackingLib::Blob* Tblob, string quality, string resolution);
		double buildImage(std::vector<cv::Mat> imgTex, std::vector<trackingLib::Blob*>& blobs, std::string imgformat);
		void save(std::string dir, std::string imgformat, trackingLib::Blob* Tblob, string quality, string resolution);
		string getJSONExtraData(const string trackedBlobGuid, const vector<trackingLib::Blob*>& blobs) const;
		void saveExtraData(const string dir, const string trackedBlobGuid, const vector<trackingLib::Blob*>& blobs) const;
		void release();
		void clear();

	};

	
}

std::string getXMLSheet(std::vector < trackingLib::scImage > scImages, std::string name, string quality, string resolution);
std::string getJSON(std::vector<trackingLib::scImage> scImages, std::string name, string folderToSave, string extensionSheetscImage, string quality, string resolution);

