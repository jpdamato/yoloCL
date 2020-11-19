#include "scImage.h"

#include <iostream>

#include <fstream>
#include <sstream>

#include "MaxRectsBinPack.h"
#include <filesystem>

using namespace trackingLib;

template <typename T> string tostr(const T& t) {
	ostringstream os;
	os << t;
	return os.str();
}

scImage::scImage(Blob* b, std::string name, size_t tx, size_t ty, size_t tw, size_t th, size_t r, cv::Size frameSize) 
{	
	this->name = name;
	this->tx = tx;
	this->ty = ty;
	this->tw = tw;
	this->th = th;
	this->r = r;
	
	this->blob = b;
	this->propSize.width = 1.0 *  tw / frameSize.width;
	this->propSize.height = 1.0 *  th / frameSize.height;

	this->propPos.x = 1.0 *  tx / frameSize.width;
	this->propPos.y = 1.0 *  ty / frameSize.height;

}

scImage::~scImage() {
}


///Posición X en el sheet
size_t scImage::getTx() {
	return tx;
}

///Posición Y en el sheet
size_t scImage::getTy() {
	return ty;
}

///Ancho en el sheet
size_t scImage::getTw() {
	return tw;
}

///Alto en el sheet
size_t scImage::getTh() {
	return th;
}

///Rotación en el sheet
size_t scImage::getR() {
	return r;
}

string addQuotes(std::string s)
{
	return  "\"" + s + "\"";
}

// *******************************
//  scImage class for sheet
// *******************************

rbp::MaxRectsBinPack::FreeRectChoiceHeuristic chooseBestHeuristic(std::vector<cv::Mat> *rects, size_t texWidth, size_t texHeight) {
	rbp::MaxRectsBinPack pack;
	std::vector<rbp::MaxRectsBinPack::FreeRectChoiceHeuristic> listHeuristics;
	listHeuristics.push_back(rbp::MaxRectsBinPack::RectBestAreaFit); //M0
	//listHeuristics.push_back(rbp::MaxRectsBinPack::RectBestLongSideFit);
	//listHeuristics.push_back(rbp::MaxRectsBinPack::RectBestShortSideFit);
//	listHeuristics.push_back(rbp::MaxRectsBinPack::RectBottomLeftRule);  //M2
//	listHeuristics.push_back(rbp::MaxRectsBinPack::RectContactPointRule);// M1

	rbp::MaxRectsBinPack::FreeRectChoiceHeuristic res;
	float max = 0;

	for (auto& heu : listHeuristics)
	{
		pack.Init(texWidth, texHeight);

		for (size_t j = 0; j < rects->size(); j++)
		{
			pack.Insert(rects->at(j).size().width, rects->at(j).size().height, heu);
		}

		if (pack.Occupancy() > max) {
			max = pack.Occupancy();
			res = heu;
		}
	}
	return res;
}


PackedImage::PackedImage(std::string name, int maxW, int maxH)
{	
	size = cv::Size(maxW, maxH); // size of the sprite sheet
}
double PackedImage::buildImage(std::vector<cv::Mat> imgSpriteSheet, std::vector<Blob*>& blobs, std::string imgformat)
{
	if (rend.cols == 0)
	{
		if (imgformat == ".png")
			rend.create(cv::Size(size.width, size.height), CV_8UC4);
		else
		rend.create(cv::Size(size.width, size.height), CV_8UC3);
	}
	rbp::MaxRectsBinPack pack(size.width, size.height); //pack of scImage
	float rotation = 0;

	
	// choose the best heuristic
	const rbp::MaxRectsBinPack::FreeRectChoiceHeuristic best1 =  chooseBestHeuristic(&imgSpriteSheet, size.width, size.height);

	for (size_t i = 0; i < imgSpriteSheet.size(); i++)
	{
		if (imgSpriteSheet[i].size().width == 0) continue;
		if (imgSpriteSheet[i].size().height == 0) continue;

		// insert the scImage into the pack
		rbp::Rect packedRect = pack.Insert(imgSpriteSheet[i].size().width, imgSpriteSheet[i].size().height, best1);

		if (packedRect.height <= 0)
		{
			std::cout << "Error: The pack is full\n";
			break;
		}

		// if the scImage is rotated
		if (imgSpriteSheet[i].size().width == packedRect.height && packedRect.width != packedRect.height)
		{
			rotation = 90; // set the rotation for the xml data
			cv::rotate(imgSpriteSheet[i], imgSpriteSheet[i], cv::ROTATE_90_CLOCKWISE);

			//imgTex[i].copyTo(rend.submat(0, 30, 0, 30))
			for (int x = 0; x < imgSpriteSheet[i].cols; x++)
				for (int y = 0; y < imgSpriteSheet[i].rows; y++)
					for (int c = 0; c < imgSpriteSheet[i].channels(); c++)
					{
						int indDst = (packedRect.y + y) * rend.cols * rend.channels() + (packedRect.x + x) * rend.channels() + c;
						rend.data[indDst] = imgSpriteSheet[i].data[y * imgSpriteSheet[i].cols * imgSpriteSheet[i].channels() + x *imgSpriteSheet[i].channels() + c];
						//cv::Vec4b pix = imgTex[i].at<cv::Vec4b>(y, x);
					    //rend.at<cv::Vec4b>(packedRect.y + y, packedRect.x + x) = pix;
				}
			// rotate the sprite to draw
			//		size_t oldHeight = spr.getTextureRect().height;
			//	spr.setPosition((float)packedRect.x, (float)packedRect.y);
			//	spr.rotate(rotation);
			//	spr.setPosition(spr.getPosition().x + oldHeight, spr.getPosition().y);
		}
		else
		//La scImagen no está rotada
		{
			rotation = 0;
			//imgTex[i].copyTo(rend.submat(0, 30, 0, 30))
			for (int x = 0; x < imgSpriteSheet[i].cols; x++)
				for (int y = 0; y < imgSpriteSheet[i].rows; y++)
					for (int c = 0; c < imgSpriteSheet[i].channels();c++)
				{
					int indDst = (packedRect.y + y) * rend.cols * rend.channels() + (packedRect.x + x) * rend.channels() + c;
					rend.data[indDst] = imgSpriteSheet[i].data[y * imgSpriteSheet[i].cols * imgSpriteSheet[i].channels() + x *imgSpriteSheet[i].channels() + c];
					//cv::Vec4b pix = imgTex[i].at<cv::Vec4b>(y, x);
					//rend.at<cv::Vec4b>(packedRect.y + y, packedRect.x + x) = pix;
				}
		}
		// draw the sprite on the sprite sheet
		// save data of the scImage for the xml file
		scImage im = scImage(blobs[i], blobs[i]->name,	packedRect.x, packedRect.y, packedRect.width, packedRect.height, rotation,	cv::Size(1000, 1000));
		images.push_back(im);
	}


	return pack.Occupancy();

}

void PackedImage::release()
{
	rend.release();
	images.clear();
}


void PackedImage::clear()
{
	rend.setTo(cv::Scalar(0, 0, 0));
}


void PackedImage::buildImageFromFiles(std::vector<std::string> listAll, std::string dir, std::string imgformat)
{
	std::vector<cv::Mat> imgTex;
	std::vector<std::string> imgTexID; // name of the scImages
	// load all the scImages
	for (auto& img : listAll)
	{
		cv::Mat texP = cv::imread(dir + img);
		imgTex.push_back(texP);
		imgTexID.push_back(img.substr(0, listAll.size() - 4));
	}

	//buildscImage(imgTex, imgTexID, imgformat);

	// free the memory of the scImages
	for (auto& tex : imgTex) {
		tex.release();
	}
}

void PackedImage::GenerateDocumentTBlob(std::string dir, std::string imgformat, std::string outFormat, Blob* Tblob, string quality, string resolution)
{
	// generate the xml document
	std::string ssheet = "";
	if (outFormat == ".xml")
	{
		ssheet = getXMLSheet(images, Tblob->guid + imgformat,quality,resolution);
	}
	else
	{
		ssheet = getJSON(images, Tblob->guid, dir, imgformat, quality, resolution);
	}

	// save the xml document
	std::ofstream xml_file;
	std::string outFileName = dir +"_" + Tblob->guid + "_" + quality + "_" + resolution + outFormat;
	replace(outFileName.begin(), outFileName.end(), '-', '_');

	xml_file.open(outFileName);
	xml_file << ssheet;
	xml_file.close();
}

///El dir llega de la forma MediaPath\idSmartcam\idCamera\fecha\events\sheets
void PackedImage::save(std::string dir, std::string imgformat, Blob* Tblob, string quality, string resolution)
{
	namespace fs = std::experimental::filesystem;
	if (!fs::exists(dir)) { // Check if src folder exists			
		fs::create_directories(dir); // create src folder
	}
	addSlashAtEnd(dir);
	//Creo la scImagen del SpriteSheet	
	cv::imwrite(dir + Tblob->guid + "_" + quality + "_" + resolution + imgformat, rend);

	
}
