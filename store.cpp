
#include <filesystem>
#include <unordered_map>
#include <queue>
#include <stdio.h>
#include <chrono>
#include <cmath>        // std::abs

#include "blobs/colorClassifier.h"
#include "blobs/blobsHistory.h" 
#include "blobs/Export.h"
#include "smartcam/trackingLib.h" 
#include "smartcam/contourDetection.h"
#include "smartcam/imageStorage.h"
#include "smartcam/MaxRectsBinPack.h"
#include "smartcam/scImage.h"
#include "u_ProcessTime.h"
#include "store.h"


using namespace trackingLib;

// Variables globales
std::thread storeThread;


bool _stopStore = false;

void stopStoring()
{
	_stopStore = true;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

void addTransparentChannel(cv::Mat& cropped, std::vector<Point>& contour, int offsetX, int offsetY, std::string imgformat)
{
	if (contour.size() == 0) return;

	if ((imgformat == ".png") && (cropped.channels() == 3))
	{
		cv::cvtColor(cropped, cropped, CV_BGR2BGRA);
	}

	for (int y = 0; y < cropped.rows; y++)
	{
		for (int x = 0; x < cropped.cols; x++)
		{

			double d = pointPolygonTest(contour, Point2f(x + offsetX, y + offsetY), true);
			if (imgformat == ".png")
			{
				if (d < -0.1)
				{
					cropped.data[y*cropped.cols * 4 + x * 4 + 3] = 0;
				}
				else
					cropped.data[y*cropped.cols * 4 + x * 4 + 3] = 255;

			}
			else
			{
				// Asigno el canal transparente
				if (d < 0)
				{
					cropped.data[y*cropped.cols * 3 + x * 3 + 0] = 255;
					cropped.data[y*cropped.cols * 3 + x * 3 + 1] = 0;
					cropped.data[y*cropped.cols * 3 + x * 3 + 2] = 255;
				}
			}

		}
	}
}

//---------------------------------------
// CREA UNA IMAGEN A PARTIR DE UN BLOB
int createAlphaMat(Mat &mat, vector<cv::Point>&  contours, string filename, int xorig, int yorig, int objectID)
{
	if (contours.size() == 0) return 0;
	if (mat.empty()) return 0;

	addTransparentChannel(mat, contours, xorig, yorig, ".jpg");

	if (fileExists(filename))
	{
		return 1;
	}
	else
		if (!imageStorage::writeToFile(mat, filename, objectID, 1))
		{
			logMessage("Failed to write file BLOB " + filename, MESSAGE_ERROR);
			return 0;
		}
		else
			return 1;


}


int ApplyTransparency(std::vector<cv::Mat> &imgSpriteSheet, std::vector<std::string> &imgTexID, std::vector<std::string> imgInfo, AlgorithmParameters* params, Blob* Tblob)
{
	int totalSize = 0;
	for (int bi = 0; bi < Tblob->before.size(); bi++)
	{
		Blob* lastB = Tblob->before[bi];

		if (lastB == NULL) continue;
		// Si estan activos
		Scalar color = Scalar(0, 255, 0);
		std::string name = lastB->getFilename();

		//Guardo la información del trackedblob que se esta guardando
		std::string row = IntToString2(Tblob->id) + ";" + IntToString2(lastB->id) + ";" + IntToString2(lastB->frameNumber) + ";" + IntToString2(lastB->getRect().x) + ";" +
			IntToString2(lastB->getRect().y) + ";" + IntToString2(lastB->getRect().width) + ";" + IntToString2(lastB->getRect().height) + ";" + name;
		imgInfo.push_back(row);

		// Copio la seccion de la imagen en el spritesheet
		//addTransparentChannel(lastB->cutImg, lastB->contour, lastB->getRect().x, lastB->getRect().y, params->imageExtension);

		//imgSpriteSheet.push_back(lastB->cutImg);
		imgTexID.push_back(lastB->guid);



		totalSize += lastB->getRect().area();
	}
	return totalSize;
}


string addQuotess(std::string s)
{
	return  "\"" + s + "\"";
}

void generateBlobs(Blob* images, std::string& jsonString)
{
	//Inicio el array de blobs
	jsonString = jsonString + "\t" + addQuotess("blobs") + ":[" + "\n";

	//Para todos los blobs
	for (int i = 0; i < images->before.size(); i++)
	{
		Blob* b = images->before[i];

		if (b->frameNumber == 0)
		{
			// this Blobs has not been well assigned
			continue;
		}

		std::chrono::time_point<std::chrono::system_clock> sclock = getAlgorithmParameters()->computeTime(b->frameNumber, 30);
		///////////////////////////////////////////////////
		std::time_t t = std::chrono::system_clock::to_time_t(sclock);

		char buf[80];
		strftime(buf, 80, "%d.%m.%Y %H:%M:%S.%MS", localtime(&t));
		std::string creationTime(buf);


		jsonString = jsonString +
			"\t\t" + "{" + "\n" +
			"\t\t\t" + addQuotess("id") + ": " + std::to_string(i) + "," + "\n" +
			"\t\t\t" + addQuotess("centroid") + ": " + addQuotess("{\\" + addQuotess("x\\") + ":" + std::to_string(b->xPercent) + "," + "\\" + addQuotess("y\\") + ":" + std::to_string(b->yPercent) + "}") + "," + "\n" +
			"\t\t\t" + addQuotess("width") + ": " + std::to_string(b->widthPercent) + "," + "\n" +
			"\t\t\t" + addQuotess("height") + ": " + std::to_string(b->heightPercent) + "," + "\n" +
			"\t\t\t" + addQuotess("dir") + ": " + addQuotess(std::to_string(b->dir[0]) + ',' + std::to_string(b->dir[1])) + "," + "\n" +
			"\t\t\t" + addQuotess("speed") + ": " + addQuotess(std::to_string(b->speed[0]) + ',' + std::to_string(b->speed[1])) + "," + "\n" +
			"\t\t\t" + addQuotess("frame") + ": " + std::to_string(b->frameNumber) + "," + "\n" +
			"\t\t\t" + addQuotess("time") + ": " + creationTime + "\n" +
			"\t\t" + "} ," + "\n";
	}

	//Elimino la ultima coma y cierro la estructura de blobs
	jsonString = jsonString.substr(0, jsonString.length() - 2) + "\n" + "]";
}

void export_image_file(std::vector<cv::Mat> imgSpriteSheet, std::vector<std::string> imgTexID, string path, string imgformat, Blob* idTblob)
{
	if (endsWith(path, "\\") == false)
		path.append("\\");

	path = path + idTblob->guid;
	if (!experimental::filesystem::exists(path)) { // Check if src folder exists			
		experimental::filesystem::create_directories(path); // create src folder
	}

	for (int i = 0; i < imgSpriteSheet.size(); i++)
	{
		//cv::imwrite(path + "\\" + std::to_string(i) + "_" + imgTexID[i] + imgformat, imgSpriteSheet[i]);
		//  Export::ExportToCSV(path + "videoAnalytics.csv", imgInfo[i], true);
		imgSpriteSheet[i].release();
	}

	Blob* lastB = idTblob->before[idTblob->before.size() - 1];

	//Inicio la estructura general
	std::string jsonString =
		"{ \t" + addQuotess("tracked_blob_id") + ":" + addQuotess(idTblob->guid) + "," + "\n" +
		"\t" + addQuotess("init") + ":" + addQuotess(idTblob->getBlobCreationDate() + " " + idTblob->getBlobCreationTime()) + "," + "\n" +
		"\t" + addQuotess("finish") + ":" + addQuotess(lastB->getBlobCreationDate() + " " + lastB->getBlobCreationTime()) + "," + "\n";

	generateBlobs(idTblob, jsonString);
	//Cierro la estructura general
	jsonString = jsonString + "\n" + " }";

	// save the xml document
	std::ofstream xml_file;
	xml_file.open(path + "\\" + idTblob->guid + ".json");
	xml_file << jsonString;
	xml_file.close();
}


std::string generateSprites(std::vector<trackingLib::scImage> scImages, std::string jsonString)
{
	for (int i = 0; i < scImages.size(); i++)
	{
		if (scImages[i].blob->guid == scImages[i].blob->guid)
		{
			std::string filename = scImages[i].blob->getFilename();
			jsonString = jsonString +
				"\t\t" + "{" +
				"\t\t" + addQuotes("blob_id") + ": " + addQuotes(scImages[i].blob->guid) + "," + "\n" +
				"\t\t" + addQuotes("x_in_sheet") + ": " + std::to_string(scImages[i].getTx()) + "," + "\n" +
				"\t\t" + addQuotes("y_in_sheet") + ": " + std::to_string(scImages[i].getTy()) + "," + "\n" +
				"\t\t" + addQuotes("width") + ": " + std::to_string(scImages[i].getTw()) + "," + "\n" +
				"\t\t" + addQuotes("height") + ": " + std::to_string(scImages[i].getTh()) + "," + "\n" +
				"\t\t" + addQuotes("rotation") + ": " + std::to_string(scImages[i].getR()) + "\n" +
				"} ," + "\n";
		}
	}
	return jsonString;
}

std::string generateBlobs(std::vector<trackingLib::scImage> scImages, std::string jsonString)
{
	//Inicio el array de blobs
	jsonString = jsonString + "\t" + addQuotes("blobs") + ":[" + "\n";

	int index = 0;
	//Para todos los blobs
	for (auto& i : scImages)
	{
		if (i.blob->frameNumber == 0)
		{
			// this Blobs has not been well assigned
			continue;
		}
		Blob* b = i.blob;

		std::chrono::time_point<std::chrono::system_clock> sclock = getAlgorithmParameters()->computeTime(b->frameNumber, 30);
		///////////////////////////////////////////////////
		std::time_t t = std::chrono::system_clock::to_time_t(sclock);

		char buf[80];
		strftime(buf, 80, "%d.%m.%Y %H:%M:%S.%MS", localtime(&t));
		std::string creationTime(buf);


		jsonString = jsonString +
			"\t\t" + "{" + "\n" +
			"\t\t\t" + addQuotess("id") + ": " + std::to_string(index) + "," + "\n" +
			"\t\t\t" + addQuotess("centroid") + ": " + addQuotess("{\\" + addQuotess("x\\") + ":" + std::to_string(b->xPercent) + "," + "\\" + addQuotess("y\\") + ":" + std::to_string(b->yPercent) + "}") + "," + "\n" +
			"\t\t\t" + addQuotess("width") + ": " + std::to_string(b->widthPercent) + "," + "\n" +
			"\t\t\t" + addQuotess("height") + ": " + std::to_string(b->heightPercent) + "," + "\n" +
			"\t\t\t" + addQuotess("dir") + ": " + addQuotess(std::to_string(b->dir[0]) + ',' + std::to_string(b->dir[1])) + "," + "\n" +
			"\t\t\t" + addQuotess("speed") + ": " + addQuotess(std::to_string(b->speed[0]) + ',' + std::to_string(b->speed[1])) + "," + "\n" +
			"\t\t\t" + addQuotess("frame") + ": " + std::to_string(b->frameNumber) + "," + "\n" +
			"\t\t\t" + addQuotess("time") + ": " + creationTime + "\n" +
			"\t\t" + "} ," + "\n";

		index++;
	}

	//Elimino la ultima coma y cierro la estructura de blobs
	jsonString = jsonString.substr(0, jsonString.length() - 2) + "\n" + "]";

	return jsonString;
}

void addSlashAtEnd(std::string& folderToSave)
{
	if (folderToSave.substr(folderToSave.length() - 2, 1) != "\\")
		folderToSave = folderToSave + "\\";
}

std::string generateSpriteSheets(std::vector<trackingLib::scImage> scImages, std::string jsonString, std::string folderToSave, std::string extensionSheetscImage, string quality, string resolution)
{
	//Inicio el array de spritesheets
	jsonString = jsonString + "\t" + addQuotes("spritesheets") + ":[" + "\n";

	addSlashAtEnd(folderToSave);

	//Para cada spritesheet
	for (int j = 0; j < 1; j++)
	{
		string urlscImage = folderToSave + scImages[0].blob->guid + "_" + quality + "_" + resolution + extensionSheetscImage;
		auto it = std::find(urlscImage.begin(), urlscImage.end(), '\\');
		while (it != urlscImage.end()) {
			auto it2 = urlscImage.insert(it, '\\');

			// skip over the slashes we just inserted
			it = std::find(it2 + 2, urlscImage.end(), '\\');
		}

		jsonString = jsonString + "{ \t" + addQuotes("tracked_blob_id") + ":" + addQuotes(scImages[0].blob->guid) + "," + "\n" +
			"\t" + addQuotes("quality") + ":" + addQuotes(quality) + "," + "\n" +
			"\t" + addQuotes("fps") + ":" + "15" + "," + "\n" +
			"\t" + addQuotes("resolution") + ":" + addQuotes(resolution) + "," + "\n" +
			"\t" + addQuotes("scImage_url") + ":" + addQuotes(urlscImage) + "," + "\n";
		//Inicio el array de sprites
		jsonString = jsonString + "\t" + addQuotes("sprites") + ":[" + "\n";

		//Para todos los sprites
		jsonString = generateSprites(scImages, jsonString);
		//Elimino la ultima coma y cierro la estructura de sprites
		jsonString = jsonString.substr(0, jsonString.length() - 2) + "] } ,";
	}

	//Elimino la ultima coma y cierro la estructura de spritesheets
	jsonString = jsonString.substr(0, jsonString.length() - 2) + "\n" + "]" + "\n";

	return jsonString;
}

std::string getJSON(std::vector<trackingLib::scImage> scImages, std::string name, string folderToSave, string extensionSheetscImage, string quality, string resolution)
{
	if (scImages.size() == 0) return "";
	Blob* firstBlob = scImages[0].blob;
	Blob* lastBlob = scImages[scImages.size() - 1].blob;
	//Inicio la estructura general
	std::string jsonString =
		"{ \t" + addQuotes("tracked_blob_id") + ":" + addQuotes(firstBlob->guid) + "," + "\n" +
		"\t" + addQuotes("init") + ":" + addQuotes(firstBlob->getBlobCreationDate() + " " + firstBlob->getBlobCreationTime()) + "," + "\n" +
		"\t" + addQuotes("finish") + ":" + addQuotes(lastBlob->getBlobCreationDate() + " " + lastBlob->getBlobCreationTime()) + "," + "\n";

	//Agrego los blobs
	jsonString = generateBlobs(scImages, jsonString);
	jsonString = jsonString + " ," + "\n";
	//Agrego los SpriteSheets
	jsonString = generateSpriteSheets(scImages, jsonString, folderToSave, extensionSheetscImage, quality, resolution);

	//Cierro la estructura general
	jsonString = jsonString + "\n" + " }";
	return jsonString;
}

std::string getXMLSheet(std::vector<trackingLib::scImage> scImages, std::string name, string quality, string resolution)
{
	std::string xml_as_string = "<t f=" + addQuotes(name) + ">" + "\n";

	for (auto& i : scImages)
	{
		xml_as_string = xml_as_string + "\t" + "<i " + "n=" + addQuotes(i.name) +
			" x=" + addQuotes(std::to_string(i.getTx())) +
			" y=" + addQuotes(std::to_string(i.getTy())) +
			" w=" + addQuotes(std::to_string(i.getTw())) +
			" h=" + addQuotes(std::to_string(i.getTh())) +
			" r=" + addQuotes(std::to_string(i.getR())) + "/>" + "\n";

	}
	xml_as_string = xml_as_string + "</t>";
	return xml_as_string;
}

void export_image_sheet(PackedImage* packI, AlgorithmParameters* params, std::vector<cv::Mat> imgSpriteSheet, Blob* Tblob, string path, string quality, string resolution)
{

	startProcess("gen_sprite_sheet");
	double occupancy = packI->buildImage(imgSpriteSheet, Tblob->before, params->imageExtension);
	packI->save(path, params->imageExtension, Tblob, quality, resolution);
	packI->GenerateDocumentTBlob(path, params->imageExtension, ".json", Tblob, quality, resolution);
	packI->clear();
	packI->release();
	double time = endProcess("gen_sprite_sheet");
	// see the occupancy of the packing
	std::cout << "pack1 : " << occupancy << "%\n";
	std::cout << "Time to export" << time << "\n";
}

void RemoveAndSaveBlobs(PackedImage* pack3600x3600, PackedImage* pack2500x2500, PackedImage* pack1200x1200, PackedImage* pack600x600, PackedImage* packI, AlgorithmParameters* params, string path)
{
	int totalBlobsToSave = (int)BlobsHistory::getInstance()->trackedBlobsToSave.size();

	if (totalBlobsToSave > 0)
	{
		// Me quedo con los blobs activos
		vector<Blob*> temp;
		BlobsHistory::getInstance()->trackedBlobsToSave.swap(temp);
		BlobsHistory::getInstance()->trackedBlobsToSave.clear();

		std::vector<cv::Mat> imgSpriteSheet;
		std::vector<std::string> imgTexID;
		std::vector<std::string> imgInfo;

		bool makeSpriteSheet = true;

		for (int j = 0; j < totalBlobsToSave; j++)
		{
			Blob* Tblob = temp[j];
			imgSpriteSheet.clear();
			imgTexID.clear();

			int totalSizeOfSprite = ApplyTransparency(imgSpriteSheet, imgTexID, imgInfo, params, Tblob);
			//Aplico las transparencias				


			string quality = "high";
			string resolution = "1280x720";

			//Guardo a Archivo
			if (params->imageFormat == EXPORT_IMG_FORMAT_FILE)
			{
				export_image_file(imgSpriteSheet, imgTexID, path, params->imageExtension, Tblob);
			}
			else {
				//Preparo la hoja del sheet
				if (totalSizeOfSprite < 600 * 600 * 0.7) { packI = pack600x600; }
				if (totalSizeOfSprite < 1200 * 1200 * 0.7) { packI = pack1200x1200; }
				else if (totalSizeOfSprite < 2500 * 2500 * 0.7) { packI = pack2500x2500; }
				else { packI = pack3600x3600; }

				if (params->imageFormat == EXPORT_IMG_FORMAT_SHEET)
				{
					export_image_sheet(packI, params, imgSpriteSheet, Tblob, path, quality, resolution);
				}
				else /*EXPORT_IMG_FORMAT_BOTH*/
				{
					export_image_file(imgSpriteSheet, imgTexID, path, params->imageExtension, Tblob);
					export_image_sheet(packI, params, imgSpriteSheet, Tblob, path, quality, resolution);
				}
			}

			// Lanzo un evento para avisar que ya lo guarde
			/*Event e = Event(0, EVENT_TYPE_BLOB_STORED, Tblob->guid, path + "\\" + "sheet.json", Tblob->clone(), 0, Tblob->r, idSceneDetector, Tblob->getCenter(), std::chrono::system_clock::now());

				EventManager::getInstance()->addNewEvent(e);*/
			Tblob->release();
		}
	}
}


void asyncrchonousSTORE(string idSceneDetector)
{
	
	std::cout << " Thread STORE starting" << "\n";

	PackedImage* pack3600x3600 = new PackedImage("sheet", 3600, 3600);
	PackedImage* pack2500x2500 = new PackedImage("sheet", 2500, 2500);
	PackedImage* pack1200x1200 = new PackedImage("sheet", 1200, 1200);
	PackedImage* pack600x600 = new PackedImage("sheet", 600, 600);

	PackedImage* packI = NULL;
	vector<cv::Mat> originalFrames;
	auto params = getAlgorithmParameters();

	//bh->startDate,  fecha para el backName
	string mediaPath = params->mediaPath;  /*  por ej, viene de la forma: C:\Temporal  */
	if ((endsWith(mediaPath, "\\") == false) && (mediaPath != ""))
		mediaPath.append("\\");
	string folder = params->outputDir;  /*    viene de la forma: idSmartcam\idCamera\  */
	if (endsWith(folder, "\\") == false)
		folder.append("\\");

	while (true)
	{
		if (_stopStore) break;
		string fecha = encodeDate(params->idFrame, params->frameRate);

		string urlBackground = mediaPath + folder + "backgrounds\\" + fecha;
		if (!experimental::filesystem::exists(urlBackground)) { // Check if src folder exists			
			experimental::filesystem::create_directories(urlBackground); // create src folder
		}

		string urlAlerts = mediaPath + folder + "events\\alerts\\" + fecha;
		if (!experimental::filesystem::exists(urlAlerts)) { // Check if src folder exists			
			experimental::filesystem::create_directories(urlAlerts); // create src folder
		}

		string pathSheets = mediaPath + folder + "events\\sheets\\" + fecha;
		if (!experimental::filesystem::exists(pathSheets)) { // Check if src folder exists			
			experimental::filesystem::create_directories(pathSheets); // create src folder
		}

		namespace fs = std::experimental::filesystem;

		/////////////////////////////////////////////
		// Remuevo y guardo todos los blobs
		/////////////////////////////////////////////
		RemoveAndSaveBlobs(pack3600x3600, pack2500x2500, pack1200x1200, pack600x600, packI, params, pathSheets);

		// Si no tengo nada que hacer, 

		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
}

