/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);
string GetDistributionName(Distribution::DistributionMethod d);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    std::chrono::steady_clock::time_point BEGIN = std::chrono::steady_clock::now();

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,false);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // Main loop
    cv::Mat imLeft, imRight;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    Distribution::DistributionMethod d;
    d = SLAM.GetTracker()->GetLeftExtractor()->GetDistribution();
    //int ptnsz = SLAM.GetTracker()->GetLeftExtractor()->GetPatternsize();
    int nlvls = SLAM.GetTracker()->GetLeftExtractor()->GetLevels();
    int nFeatures = SLAM.GetTracker()->GetLeftExtractor()->GetFeaturesNum();
    float scalefac = SLAM.GetTracker()->GetLeftExtractor()->GetScaleFactor();
    string distributionName = GetDistributionName(d);
    string addInfo;
    addInfo = (d == Distribution::SSC || d == Distribution::RANMS) ?
            (to_string(SLAM.GetTracker()->GetLeftExtractor()->GetSoftSSCThreshold())+"Th")
            : "";

    // Stop all threads
    SLAM.Shutdown();

    std::chrono::steady_clock::time_point END = std::chrono::steady_clock::now();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    cout << "total runtime: " << std::chrono::duration_cast<std::chrono::seconds>(END-BEGIN).count() << endl;

    // Save camera trajectory
    string name;
    string delim = "/";
    name = string(argv[3]);
    int n = name.rfind(delim, name.length()-2);
    cout << "\n" << name;
    name = name.substr(n+1, name.length()-n-2);
    name += "/";

    stringstream ssC, ssK;

    struct stat buf{};
    string tem = "trajectories/stereo_kitti/";
    tem += name;
    tem += "/";
    bool dex = (stat(tem.c_str(), &buf) == 0);
    if (!dex) mkdir(tem.c_str(), S_IRWXU);

    tem += distributionName;
    tem += "/";
    dex = (stat(tem.c_str(), &buf) == 0);
    if (!dex) mkdir(tem.c_str(), S_IRWXU);

    for (int i = 1; i < 5000; ++i)
    {
        ssC.str(string());
        ssC << "trajectories/stereo_kitti/" << name << distributionName << "/" << to_string(nFeatures) << "_"
            << (to_string(nlvls)+"l_") << to_string(scalefac) << "_" << addInfo << "_" << to_string(i);
        string sC = ssC.str();
        string sK = sC;
        sC += ".txt";
        bool ex = (stat(sC.c_str(), &buf) == 0);
        if (!ex)
        {
            SLAM.SaveTrajectoryKITTI(sC);
            sK += "_KT.txt";
            SLAM.SaveKeyFrameTrajectoryTUM(sK);
            break;
        }
        if (i == 4999)
            cerr << "\nToo many trajectory files, would rather not write another one\n";
    }

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
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