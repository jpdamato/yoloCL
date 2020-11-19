#pragma once
// =================================================================================================
// Project: 
// AIS - Video detector and Tracking
// 
// File information:
// File for testing different configurations
// Institution.... ais.pladema.net
// Author......... Juan D'Amato
// Changed at..... 2019-11-10
// License........ MIT license
// =================================================================================================


std::string getExePath();

int mainVideoDetector(int argc, char **argv);

int mainClassify(int argc, char **argv);


int mainVibe(int argc, char **argv);

// void run_yolo(int argc, char **argv);



void mainObjectMatching(int argc, char **argv);

void mainOpticalFlow(int argc, char **argv);

int mainTrackAlgs(int argc, char** argv);
