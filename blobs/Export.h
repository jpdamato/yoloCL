#ifndef HELPER_EXPORT
#define HELPER_EXPORT

#include <iostream>
#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <vector>
#include <iterator>


#define MESSAGE_ERROR 1
#define MESSAGE_LOG 2


using namespace std;


string IntToString(int value);
int StringToInt(string s);
float StringToFloat(string s);
template<typename Out>
void splits(const std::string &s, char delim, Out result);
std::vector<std::string> splitString(const std::string &s, char delim);


double sft_clock(void);

void logMessage(string s, int type);
string getLibError();

class Export
{
public:
  static void ExportToCSV(string fileName, string data, bool append);
  
  
};

#endif