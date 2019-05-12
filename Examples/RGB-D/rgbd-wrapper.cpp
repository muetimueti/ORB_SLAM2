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

/**
 * @param argc needs to be 6
 * @param argv ./rgbdwrapper path_to_vocabulary path_to_settings path_to_sequence path_to_association
 * number of iterations
 * @return
 */

int CallORB_SLAM2(char **argv);

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        cerr << "\nRequired args: ./rgbdwrapper path_to_vocabulary path_to_settings path_to_sequence "
                "path_to_association number_of_iterations(int)\n";
        exit(EXIT_FAILURE);
    }

    int N = stoi(argv[5]);

    string settingspath = string(argv[2]);

    ifstream settingsfile;
    //TODO: implement

    CallORB_SLAM2(argv);
}

int CallORB_SLAM2(char **argv)
{

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
    string distributionName = GetDistributionName(d);
    bool dPerLvl = SLAM.GetTracker()->GetLeftExtractor()->AreKptsDistributedPerLevel();

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

    stringstream ssC, ssK;

    struct stat buf{};
    for (int i = 1; i< 5000; ++i)
    {
        ssC.str(string());
        ssK.str(string());
        ssC << "trajectories/" << name << "_" << distributionName << "_" << (dPerLvl? "dpl_" : "npl_") << to_string(i)
            << "-ct.txt";
        ssK << "trajectories/" << name << "_" << distributionName << "_" << (dPerLvl? "dpl_" : "npl_") << to_string(i)
            << "-kt.txt";
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
    typedef Distribution::DistributionMethod distr;
    switch(d)
    {
        case (distr::KEEP_ALL):
            return string("all_kept");
        case (distr::NAIVE):
            return string("topN");
        case (distr::QUADTREE):
            return string("quadtree");
        case (distr::QUADTREE_ORBSLAMSTYLE):
            return string("quadtree_os");
        case (distr::GRID):
            return string("bucketing");
        case (distr::ANMS_KDTREE):
            return string("KDT-ANMS");
        case (distr::ANMS_RT):
            return string("RT-ANMS");
        case (distr::SSC):
            return string("SSC");
        default:
            return string("unknown");
    }
}


