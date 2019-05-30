#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include <sstream>
#include "include/Distribution.h"

#include <opencv2/core/core.hpp>

#include <System.h>


namespace settings {
string distributionSetting = "ORBextractor.distribution";
int distrOffset = 27;
string dplSetting = "ORBextractor.ORBextractor.distributePerLevel";
int dplOffset = 33;
string scoreSetting = "ORBextractor.scoreType";
int scoreOffset = 24;
string nfeatSetting = "ORBextractor.nFeatures";
int nfeatOffset = 24;
string scaleFSetting = "ORBextractor.scaleFactor";
int scaleFOffset = 26;
string FASTiniThSetting = "ORBextractor.iniThFAST";
string FASTminThSetting = "ORBextractor.minThFAST";
int FASTThOffset = 24;
string nLevelsSetting = "ORBextractor.nLevels";
int nLevelsOffset = 22;
string patternSizeSetting = "ORBextractor.patternSize";
int patternSizeOffset = 26;
}
//patternsize 8 fast thresholds: ini 8, min 2

using namespace std;
using namespace settings;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);
string GetDistributionName(Distribution::DistributionMethod d, FASTdetector::ScoreType s);

/**
 * @param argc needs to be 6
 * @param argv ./rgbdwrapper path_to_vocabulary path_to_settings path_to_sequence path_to_association
 * number of iterations
 * @return
 */

int CallORB_SLAM2(char **argv);
void replaceLine(string &path, string &toFind, string set, int offset);
void resetSettings(string settingsPath);


int main(int argc, char **argv)
{
    if (argc != 6)
    {
        cerr << "\nRequired args: ./rgbdwrapper path_to_vocabulary path_to_settings path_to_sequence "
                "path_to_association number_of_iterations(int)\n";
        exit(EXIT_FAILURE);
    }

    int N = stoi(argv[5]);
    string vocPath = string(argv[1]);
    string settingsPath = string(argv[2]);
    string sequencePath = string(argv[3]);
    string associationPath = string(argv[4]);



    string call = "(cd /home/ralph/CLionProjects/ORB_SLAM2/ && exec Examples/RGB-D/rgbd_tum ";
    call += vocPath + " " + settingsPath + " " + sequencePath + " " + associationPath + ")";

    resetSettings(settingsPath);
    int mode = 2;
    for (; mode < 7; ++mode)
    {
        replaceLine(settingsPath, distributionSetting, to_string(mode), distrOffset);
        for (int i = 0; i < N; ++i)
            system(call.c_str());

        replaceLine(settingsPath, scaleFSetting, "1.01", scaleFOffset);
        for (int i = 0; i < N; ++i)
            system(call.c_str());

        resetSettings(settingsPath);
    }
}

void replaceLine(string &path, string &toFind, string set, int offset)
{
    cout << "\nSetting " << toFind << " to " << set << "...\n";
    fstream settingsFile;
    settingsFile.open(path, ios::in);
    fstream tempFile;
    string tempPath = "tempSettings.yaml";
    tempFile.open("tempSettings.yaml", ios::out);
    if (!settingsFile.is_open() || !tempFile.is_open())
    {
        cerr << "\nfailed to modify settings file...\n";
        exit(EXIT_FAILURE);
    }
    string line;
    while(getline(settingsFile, line))
    {
        int k = line.find(toFind);
        if (k != string::npos)
        {
            line.replace(offset, set.size(), set);
        }
        tempFile << line << "\n";
    }
    remove(path.c_str());
    rename(tempPath.c_str(), path.c_str());
    settingsFile.close();
    tempFile.close();
}

void resetSettings(string settingsPath)
{
    cout << "\nResetting settingsfile:\n";
    replaceLine(settingsPath, nfeatSetting, "1000", nfeatOffset);
    replaceLine(settingsPath, scaleFSetting, "1.2", scaleFOffset);
    replaceLine(settingsPath, nLevelsSetting, "8", nLevelsOffset);
    replaceLine(settingsPath, FASTiniThSetting, "20", FASTThOffset);
    replaceLine(settingsPath, FASTminThSetting, "7", FASTThOffset);
    replaceLine(settingsPath, distributionSetting, "2", distrOffset);
    replaceLine(settingsPath, dplSetting, "1", dplOffset);
    replaceLine(settingsPath, scoreSetting, "0", scoreOffset);
    replaceLine(settingsPath, patternSizeSetting, "16", patternSizeOffset);
}

