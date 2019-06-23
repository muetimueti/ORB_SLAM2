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
string softThSetting = "ORBextractor.softSSCThreshold";
int softThOffset = 31;
}
//patternsize 8 fast thresholds: ini 8, min 2

using namespace std;
using namespace settings;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

/**
 * @param argc needs to be 6
 * @param argv ./rgbdwrapper path_to_vocabulary path_to_settings path_to_sequence path_to_association
 * number of iterations
 * @return
 */

void replaceLine(string &path, string &toFind, string set, int offset);
void resetSettings(string settingsPath, string program);


int main(int argc, char **argv)
{
    if (argc != 7)
    {
        cerr << "\nRequired args: ./rgbdwrapper path_to_vocabulary path_to_settings path_to_sequence "
                "path_to_association <'rgbd'/'kitti'/'euroc'> number_of_iterations(int)\n";
        exit(EXIT_FAILURE);
    }

    string vocPath = string(argv[1]);
    string settingsPath = string(argv[2]);
    string sequencePath = string(argv[3]);
    string associationPath = string(argv[4]);
    string program = string(argv[5]);
    int N = stoi(argv[6]);


    string rgbdcall = "(cd /home/ralph/CLionProjects/ORB_SLAM2/ && exec Examples/RGB-D/rgbd_tum ";
    rgbdcall += vocPath + " " + settingsPath + " " + sequencePath + " " + associationPath + ")";

    string kitticall = "(cd /home/ralph/CLionProjects/ORB_SLAM2/ && exec Examples/Stereo/stereo_kitti ";
    kitticall += vocPath + " " + settingsPath + " " + sequencePath + ")";

    string euroccall = "(cd /home/ralph/CLionProjects/ORB_SLAM2/ && exec Examples/Stereo/stereo_euroc ";
    euroccall += vocPath + " " + settingsPath + " " + sequencePath + " " + associationPath + " " + program + ")";

    string call;
    if (program == "rgbd")
        call = rgbdcall;
    else if (program == "kitti")
        call = kitticall;
    else
        call = euroccall;

    resetSettings(settingsPath, program);
    int mode = 1;
    for (; mode < 4; ++mode)
    {
        replaceLine(settingsPath, distributionSetting, to_string(mode), distrOffset);
        replaceLine(settingsPath, nfeatSetting, "800", nfeatOffset);
        replaceLine(settingsPath, scaleFSetting, "1.05", scaleFOffset);
        replaceLine(settingsPath, nLevelsSetting, "5", nLevelsOffset);
        replaceLine(settingsPath, softThSetting, "15", softThOffset);
        for (int i = 0; i < N; ++i)
            system(call.c_str());

        resetSettings(settingsPath, program);
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
            //line.replace(offset, set.size(), set);
            string newline = line.substr(0, offset);
            newline += set;
            line = newline;
        }
        tempFile << line << "\n";
    }
    remove(path.c_str());
    rename(tempPath.c_str(), path.c_str());
    settingsFile.close();
    tempFile.close();
}

void resetSettings(string settingsPath, string program)
{
    cout << "\nResetting settingsfile:\n";
    int nFeatures = program == "kitti" ? 2000 : program == "rgbd" ? 1000 : 1200;
    replaceLine(settingsPath, nfeatSetting, to_string(nFeatures), nfeatOffset);
    replaceLine(settingsPath, scaleFSetting, "1.20", scaleFOffset);
    replaceLine(settingsPath, nLevelsSetting, "8", nLevelsOffset);
    replaceLine(settingsPath, FASTiniThSetting, "20", FASTThOffset);
    replaceLine(settingsPath, FASTminThSetting, "7", FASTThOffset);
    replaceLine(settingsPath, distributionSetting, "2", distrOffset);
    replaceLine(settingsPath, dplSetting, "1", dplOffset);
    replaceLine(settingsPath, scoreSetting, "0", scoreOffset);
    replaceLine(settingsPath, patternSizeSetting, "16", patternSizeOffset);
}

