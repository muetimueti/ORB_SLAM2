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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <sys/stat.h>
#include <sstream>
#include "include/Distribution.h"

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);
string GetDistributionName(Distribution::DistributionMethod d);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);

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
    string tem = "trajectories/rgbd_tum/";
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
        ssK.str(string());
        ssC << "trajectories/rgbd_tum/" << name << distributionName << "/" << to_string(nFeatures) << "_"
        << (to_string(nlvls)+"l_") << to_string(scalefac) << "_" << addInfo << "_" << to_string(i) << ".txt";
        ssK << "trajectories/rgbd_tum/" << name << distributionName << "/" << to_string(nFeatures) << "_"
        << (to_string(nlvls)+"l_") << to_string(scalefac) << "_" << to_string(i) << "_KT.txt";
        string sC = ssC.str();
        string sK = ssK.str();
        bool ex = (stat(sC.c_str(), &buf) == 0);
        if (!ex)
        {
            SLAM.SaveTrajectoryTUM(sC);
            SLAM.SaveKeyFrameTrajectoryTUM(sK);
            break;
        }
        if (i == 4999)
            cerr << "\nToo many trajectory files, would rather not write another one\n";
    }

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
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