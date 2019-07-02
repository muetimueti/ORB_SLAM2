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
string GetDistributionName(Distribution::DistributionMethod d);

/**
 * @param argc needs to be 6
 * @param argv ./rgbdwrapper path_to_vocabulary path_to_settings path_to_sequence path_to_association
 * number of iterations
 * @return
 */

void replaceLine(string &path, string &toFind, string set, int offset);
void resetSettings(string settingsPath, string program);

typedef struct ORBSlamSettings
{
    int nFeatures;
    float scaleFactor;
    int nLevels;
    Distribution::DistributionMethod kptDistribution;
    int softSSCThreshold;
} ORBSlamSettings;

void CallORBSlamAndEvaluate(string &call, ORBSlamSettings settings, string program, int iter);

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

    ORBSlamSettings activeSettings;
    activeSettings.nLevels = 4;
    activeSettings.nFeatures = 1500;
    activeSettings.scaleFactor = 1.05;
    activeSettings.softSSCThreshold = 0;

    resetSettings(settingsPath, program);
    int mode = 7;
    for (; mode < 8; ++mode)
    {
        int maxIter = N;
        int i = 0;
        if (mode != 7)
            i = 20;

        activeSettings.kptDistribution = static_cast<Distribution::DistributionMethod>(mode);

        replaceLine(settingsPath, distributionSetting, to_string(mode), distrOffset);
        replaceLine(settingsPath, softThSetting, to_string(activeSettings.softSSCThreshold), softThOffset);
        replaceLine(settingsPath, nfeatSetting, to_string(activeSettings.nFeatures), nfeatOffset);
        replaceLine(settingsPath, nLevelsSetting, to_string(activeSettings.nLevels), nLevelsOffset);
        replaceLine(settingsPath, scaleFSetting, to_string(activeSettings.scaleFactor), scaleFOffset);

        for (i = 3; i < 15; ++i)
            CallORBSlamAndEvaluate(call, activeSettings, program, i);

        activeSettings.softSSCThreshold = 15;

        replaceLine(settingsPath, softThSetting, to_string(activeSettings.softSSCThreshold), softThOffset);

        for (i = 3; i < 15; ++i)
            CallORBSlamAndEvaluate(call, activeSettings, program, i);

        activeSettings.softSSCThreshold = 30;

        replaceLine(settingsPath, softThSetting, to_string(activeSettings.softSSCThreshold), softThOffset);

        for (i = 3; i < 15; ++i)
            CallORBSlamAndEvaluate(call, activeSettings, program, i);

        activeSettings.softSSCThreshold = 50;

        replaceLine(settingsPath, softThSetting, to_string(activeSettings.softSSCThreshold), softThOffset);
        for (i = 3; i < 15; ++i)
            CallORBSlamAndEvaluate(call, activeSettings, program, i);


        exit(EXIT_SUCCESS);
        for (; i < maxIter; ++i)
        {
            CallORBSlamAndEvaluate(call, activeSettings, program, i);
        }

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


void CallORBSlamAndEvaluate(string &call, ORBSlamSettings settings, string program, int iter)
{
    int failure;
    do
    {
        failure = system(call.c_str());
    }
    while (failure);
    std::string changeDir = "(cd /home/ralph/CLionProjects/ORB_SLAM2/trajectories/";
    changeDir += program == "kitti" ? "stereo_kitti/kitti_seq_07/" :
                 program == "rgbd" ? "rgbd_tum/rgbd_dataset_freiburg1_room/" :
                 "stereo_euroc/mh5/";

    string addInfo;
    addInfo = (settings.kptDistribution == Distribution::SSC || settings.kptDistribution == Distribution::RANMS
            || settings.kptDistribution == Distribution::KEEP_ALL) ?
              (to_string(settings.softSSCThreshold) + "Th") : "";

    std::string ev = "&& evo_rpe kitti groundtruth.txt ";
    std::string evtum = "&& evo_rpe tum groundtruth.txt ";
    std::string eveur = "&& evo_rpe euroc groundtruth.csv";
    stringstream ss;
    ss << (program == "kitti"? ev : program == "rgbd"? evtum : eveur) << GetDistributionName(settings.kptDistribution) << "/" << to_string(settings.nFeatures) <<
       "_" << (to_string(settings.nLevels) + "l_") << to_string(settings.scaleFactor) << "_" << addInfo <<
       "_" << to_string(iter + 1) << ".txt -a -s --save_result " << GetDistributionName(settings.kptDistribution) <<
       "/" << GetDistributionName(settings.kptDistribution) << "_" << to_string(settings.nFeatures) << "f_" << addInfo <<
       to_string(iter + 1) << ".zip)";

    string evalCall = ss.str();
    changeDir += evalCall;
    system(changeDir.c_str());
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


string GetDistributionName(Distribution::DistributionMethod d)
{
    using distr = Distribution::DistributionMethod;
    string res;
    switch(d)
    {
        case (distr::KEEP_ALL):
            res.append("all_kept");
            break;
        case (distr::NAIVE):
            res.append("topN");
            break;
        case (distr::RANMS):
            res.append("ranms");
            break;
        case (distr::QUADTREE_ORBSLAMSTYLE):
            res.append("quadtree");
            break;
        case (distr::GRID):
            res.append("bucketing");
            break;
        case (distr::ANMS_KDTREE):
            res.append("KDT-ANMS");
            break;
        case (distr::ANMS_RT):
            res.append("RT-ANMS");
            break;
        case (distr::SSC):
            res.append("SSC");
            break;
        case (distr::SOFT_SSC):
            res.append("SoftSSC");
            break;
        default:
            res.append("unknown");
            break;
    }
    return res;
}
