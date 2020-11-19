
#include <iostream>
#include <fstream>
#include <Windows.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// YOLO Includes
#include "../include/darknet.h"
#include "../src/yolo_layer.h"
#include "../src/softmax_layer.h"
#include "../src/reorg_layer.h"
#include "../src/region_layer.h"
#include "../src/parser.h"
#include "../src/network.h"
#include "../src/option_list.h"

#include "gpuUtils/cnnInstance.h"
#include "gpuUtils/gpu_param.h"
#include "gpuUtils/yoloopenCL.h"
#include "u_ProcessTime.h"


char** __names;

network* buildNet(std::string _dirEXE, std::string cfgfile, std::string weightfile)
{
	network *net = load_network_custom((char*)cfgfile.c_str(), (char*)weightfile.c_str(), 1, 1);
	set_batch_network(net, 1);

	std::string  datacfg = _dirEXE + "cfg/coco.data";
	list *options = read_data_cfg((char*)datacfg.c_str());
	char *name_list = option_find_str(options, (char*) "names", (char*)"data/names.list");

	std::string snameList = _dirEXE + name_list;
	__names = get_labels((char*)snameList.c_str());
	char* _alphabet = NULL; // load_alphabet();

	return net;

}


std::string getExePath()
{
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}


int inner_find_arg(int argc, char* argv[], char *arg)
{
	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg)) {
			return i+1;
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	
	
	std::string dirEXE = getExePath() + "\\"; // "D:\\Sdks\\YoloV4\\darknet\\";


	///////////////////////////////////////
   // INIT GPU
	clUtils::initDevice(0, 0);
	YoloCL::setAllocationScheme(-1, 2048 * 1024 * 1024);

	if (YoloCL::initYoloCL(dirEXE, 256, 0, 0) != CL_SUCCESS)
{
	std::cout << " Failed to init YOLO CL. Now exit" << "\n";
	return -1;
}
	if (inner_find_arg(argc, argv, (char*)"-nogpu"))
	{
		gpu_index = -1;
	}


	////////////////////////////////////////////////////
	// LOAD VIDEO FILE
	std::string videoFN;
	int vIndex = inner_find_arg(argc, argv, (char*)"-v");
	if (vIndex > 0)
	{
		videoFN = argv[vIndex];
	}
	else
	{
		videoFN = "D:\\googleDrive\\Projects\\DataVia\\Videos\\20200717215938.mp4";
	}

	cv::VideoCapture cap;
	cap.open(videoFN);
	if (!cap.isOpened())
	{
		std::cout << "Could not load video File" << "\n";
	}

	///////////////////////////////////////
	// Init CNN
	cnnInstance* cnn = NULL;
	char *cfgfile = (char*)"D:\\Sdks\\YoloV4\\darknet\\cfg\\yolov4.cfg";
	char *weightfile = (char*)"D:\\Sdks\\YoloV4\\darknet\\cfg\\yolov4.weights";

	int cfgIndex = inner_find_arg(argc, argv, (char*)"-c");
	if (cfgIndex > 0)
	{
		cfgfile = argv[cfgIndex];
	}
	
	int weightIndex = inner_find_arg(argc, argv, (char*)"-w");
	if (weightIndex > 0)
	{
		weightfile = argv[weightIndex];
	}


	std::string  datacfg = dirEXE + "cfg\\coco.data";
	cnn = new cnnInstance(buildNet(dirEXE, cfgfile, weightfile), 1);
	cnn->names = __names;
	if (!cnn)
	{
			std::cout << " Failed to load CNN Files. Now exit" << "\n";
			return -1;
	}
	
	cv::Mat frame; //	= cv::imread("D:\\Resources\\Andenes\\17-GRAND BOURG_CAM 10_2020_05_21_12_21_29.jpg");

	cap >> frame ;

	cv::Mat outFrame = frame.clone();
	int nframe = 0;
	////////////////////////////////////////////////
	// force 1 prediction
	while (1)
	{
		cv::Mat iframe;
		//cap >> iframe;
		iframe = cv::imread("D:\\Personal\\googleDrive\\Projects\\Pulpou\\dataset_pulpo\\0065-0204\\J12-275_Anuncio_Vogue_Revitaluxe_Joi-0084.jpg");

		frame = iframe; // (cv::Rect(0, 0, 700, 700));
		cv::Mat outputFrame;
		outputFrame = frame.clone();

		if (nframe % 2 == 0)
		{

			int64 t1 = cv::getTickCount();

			if (cnn)
			{
				cnn->setFrame(frame);
#ifdef DEBUG
				cnn->yoloMode = true;
#else
				cnn->yoloMode = false;
#endif
				cnn->predict(false);
			}
			double msecs = abs(t1 - cv::getTickCount()) / cv::getTickFrequency();

			
			cv::resize(cnn->squared, outputFrame, cv::Size(), 2.0, 2.0);
			//cnn->thresh = 0.25;
			cnn->drawResults(outputFrame);
			
			std::string wname = "Detections.. Gpu " + to_string(gpu_index);
			cv::imshow(wname.c_str(), outputFrame); // cnn->squared);
			std::cout << "fps:" << 1.0 / msecs << "  \n";
			int k = cv::waitKey(5);

			if (k == 'p')
			{
				showProcessTime();
			}

			if (k == '+')
			{
				cnn->thresh += 0.1;
			}

			if (k == '-')
			{
				cnn->thresh -= 0.1;
			}
		}
		nframe++;
		
	}
	return 0;
}