#include "Export.h"
#include <iostream>
#include <string>
#include <string.h>
#include <fstream>

#include <Windows.h>
#include <omp.h>


using namespace std;


int StringToInt(string s)
{
	    return atoi(s.c_str());
	}

float StringToFloat(string s)
{
	return (float)atof(s.c_str());
}

string IntToString(int value)
{
	    char convC[10];
		_itoa_s(value,convC,10);
		std::string convert(convC);
		return convert;
	}

double sft_clock(void)
{
	/* _WIN32: use QueryPerformance (very accurate) */
	LARGE_INTEGER freq, t;
	/* freq is the clock speed of the CPU */
	QueryPerformanceFrequency(&freq);
	/* cout << "freq = " << ((double) freq.QuadPart) << endl; */
	/* t is the high resolution performance counter (see MSDN) */
	QueryPerformanceCounter(&t);
	return (t.QuadPart / (double)freq.QuadPart);
}

template<typename Out>
void splits(const std::string &s, char delim, Out result) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

std::vector<std::string> splitString(const std::string &s, char delim) {
	std::vector<std::string> elems;
	splits(s, delim, std::back_inserter(elems));
	return elems;
}

std::vector<string> errors;

void logMessage(string s, int type)
{
	errors.push_back(s);
	std::cout << s << "\n";
}

string getLibError()
{
	if (errors.size() == 0)
		return "";
	else
	return errors[errors.size()-1];
}


void Export::ExportToCSV(string fileName, string data,bool append)
{
  ofstream myfile;
  if (append)
	 myfile.open(fileName, ios::out | ios::app);
  else
	 myfile.open(fileName);

  if (myfile.is_open())
  {	 	
	  myfile << data << "\n";
	  myfile.close();
  }
  
}

