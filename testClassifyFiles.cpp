#include <thread>
#include <mutex>

#pragma warning( disable : 4244)

#include <opencv2/core.hpp>   // Include OpenCV API
#include <opencv2/highgui.hpp>   // Include OpenCV API
#include <opencv2/imgcodecs.hpp>   // Include OpenCV API
#include <opencv2/imgproc.hpp>   // Include OpenCV API
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>

#include <time.h>
#include <winsock.h>
#include <vector>
#include <thread>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "mainApps.h"
#include "Yolo/network.h"
#include "gpuUtils/yoloopenCL.h"
#include "gpuUtils/cl_utils.h"
#include "gpuUtils/cl_MemHandler.h"
#include "gpuUtils/clVibe.h"
#include "Yolo/opencvUtils.h"
#include "u_ProcessTime.h"


void draw_detections(cv::Mat im, std::vector< object_detected>& objs)
{
	for (auto od : objs)
	{
		int width = im.rows * .006;
		box b = od.reg;


		int left = (b.x - b.w / 2.)*im.cols;
		int right = (b.x + b.w / 2.)*im.cols;
		int top = (b.y - b.h / 2.)*im.rows;
		int bot = (b.y + b.h / 2.)*im.rows;
		for (int i = 0; i < od.classes.size(); i++)
		{
			std::string labels(od.classes[i].first);
			labels += std::to_string(od.classes[i].second);

			cv::rectangle(im, cv::Rect(left, top, right - left, bot - top), cv::Scalar(255, 25, 25), 2);
			cv::putText(im, labels, cv::Point(left, top), 3, 1, cv::Scalar(255, 255, 255));
		}
	}

}


int mainClassify(int argc, char **argv)
{

	float nms = .45f;
	float hier_thresh = 0.1f;
	float thresh = 0.1;
	char **names;
	image **alphabet;
	std::string dirEXE = "";

	//test_resize("data/bad.jpg");
	//test_box();
	//test_convolutional_layer();
	if (argc < 2) {
		fprintf(stderr, "usage: %s <function>\n", argv[0]);
		return 0;
	}

	
	gpu_index = find_int_arg(argc, argv, (char*)"-i", 0);

	size_t totalAvailableMem = find_int_arg(argc, argv, (char*)"-MB", 0);

	if (totalAvailableMem > 0)
	{
		
		YoloCL::setAllocationScheme(-1, totalAvailableMem);
	}
	else
	{
		YoloCL::setAllocationScheme(1,0);
	}


	if (find_arg(argc, argv, "-nogpu")) {
		gpu_index = -1;
	}

	if (gpu_index > 0)
	{
		gpuMemHandler::setDeviceHasOwnMem(false);
	}

	dirEXE = getExePath() + "\\";
	clUtils::initDevice(gpu_index, 0);
	YoloCL::initYoloCL(dirEXE , 256, gpu_index, 0);

	std::string  datacfg = dirEXE + "cfg/coco.data";
	char *cfgfile = argv[3];
	char *weightfile = argv[4];

	thresh = find_float_arg(argc, argv, "-thresh", 0.1);
	char *filename = (argc > 4) ? argv[4] : 0;

	drklist *options = read_data_cfg((char*)datacfg.c_str());
	char *name_list = option_find_str(options, "names", "data/names.list");

	std::string snameList = dirEXE + name_list;
	names = get_labels((char*)snameList.c_str());

	alphabet = NULL;// load_alphabet();

	network *net = load_network(cfgfile, weightfile, 0);
	size_t totalMemSize = 0;

	totalMemSize += _msize(net->input);
	totalMemSize += _msize(net->workspace);
	int biggerLayer = 0;
	size_t biggerlayerSize = 0;

	for (int i = 0; i < net->n; i++)
		{
			layer l = net->layers[i];
			size_t layerSize = 0;
			if (l.output) 	layerSize += _msize(l.output);
			if (l.scales) layerSize += _msize(l.scales);
			if (l.biases) layerSize += _msize(l.biases);
			if (l.weights) layerSize += _msize(l.weights);
			if (l.indexes) layerSize += _msize(l.indexes);
			if (l.rolling_variance) layerSize += _msize(l.rolling_variance);
			if (l.rolling_mean) layerSize += _msize(l.rolling_mean);
			//	if (l.delta) free(l.delta); //not used in this

			totalMemSize += layerSize;
			if (biggerlayerSize < layerSize)
			{
				biggerLayer = i;
				biggerlayerSize = layerSize;
			}

		}
	
	set_batch_network(net, 1);
	srand(2222222);

	image im;
	image sized;
	float* imData;
	im.data = NULL;
	int nframe = 0;
	std::string videoFN;
	std::string mediaPath;

	int vIndex = find_arg(argc, argv, "-v");
	if (vIndex > 0)
	{
		videoFN = argv[vIndex];
	}
	else
	{
		videoFN = "D:\\Resources\\video\\orco\\2019-03-11_18-47-54.mp4";
	}


	
	int mIndex = find_arg(argc, argv, "-media");
	if (mIndex > 0)
	{
		mediaPath = argv[mIndex];
	}
	else
	{
		mediaPath = "D:\\temp\\tracker2_0\\";
	}

	cv::VideoCapture cap;
	
	cap.open(videoFN);


	for (int i = 0; i < 4500; i++)
	{
		clearTimers();
		//std::string filename = "D:\\Resources\\Data_Drone\\8ago2019_50mts\\1 (" + std::to_string(i) + ").jpg";
		//cv::Mat mM = cv::imread(filename);

		cv::Mat mM;
		cap >> mM;

		if (mM.empty()) continue;
		//cv::cvtColor(mM, mM, CV_BGR2RGB);
		if (mM.rows > 2000)
		{
			cv::resize(mM, mM, cv::Size(), 0.25, 0.25);
		}
		else
		if (mM.rows > 1080)
		{
			cv::resize(mM, mM, cv::Size(), 0.5, 0.5);
		}

		//Take image data
		if (!im.data)
		{
			im = make_image(mM.cols, mM.rows, mM.channels());
			imData = (float*)calloc(net->w * net->h * net->c, sizeof(float));
			sized = make_image(net->w, net->h, mM.channels());
		}
		cv::Mat squared = GetSquareImage(mM, net->w);
		//Resize for network
		mat_to_image(squared, sized);

		startProcess("CNN");
		auto actualTime = std::chrono::steady_clock::now();
		std::vector<object_detected> detections;
		if (gpu_index < 0)
		{
			forward_network(net);
		}
		else
		{
			YoloCL::predictYoloCL(net, i, 0,false);
			detections = YoloCL::detectOnFrameCNN(net, sized.data, mM.cols, mM.rows, nframe, names);

		}

		endProcess("CNN");
		int nboxes;
		layer l = net->layers[net->n - 1];

		// GET

		if (detections.size())
		{
			draw_detections(mM, detections);

			for (auto od : detections)
			{
				std::vector<std::string> rows;
				rows.push_back(std::to_string(-1));
				rows.push_back(std::to_string(od.reg.x ));
				rows.push_back(std::to_string(od.reg.y ));
				rows.push_back(std::to_string(od.reg.w ));
				rows.push_back(std::to_string(od.reg.h ));
				for (int j = 0; j < od.classes.size(); j++)
				{
					rows.push_back(od.classes[j].first);
					rows.push_back(to_string(od.classes[j].second));

				}

				clUtils::exportCSV(mediaPath + "\\det.txtF", rows, i);

			}
		}
	
		cv::imshow("frame", mM);
		showProcessTime();

		cv::waitKey(1);

	}


	return 0;
}