#include "include/FeatureFileInterface.h"
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <assert.h>
#include <sstream>
#include <iterator>

using namespace std;

/** fileformat:
 * x y size angle response octave
 * 1 kpt per line
 */

bool FeatureFileInterface::SaveFeatures(vector<cv::KeyPoint> &kpts)
{
    message_assert("Save path must be set", !path.empty());
    cout << "Saving features to " << path << "...\n";

    string filename = path + "features/" + to_string(saveCounter) + ".orbf";

    if (CheckExistence(filename))
    {
        cerr << "Feature-file already exists!\n";
        return false;
    }

    fstream file;
    file.open(filename, ios::out);
    if (!file.is_open())
    {
        cerr << "Failed to open " << filename << "...\n";
        return false;
    }


    for (auto &kpt : kpts)
    {
        file << kpt.pt.x << " " << kpt.pt.y << " " << kpt.size << " " << kpt.angle << " " << kpt.response << " " <<
             kpt.octave << "\n";
    }
    file.close();
    return true;
}


vector<cv::KeyPoint> FeatureFileInterface::LoadFeatures(std::string &path)
{
    vector<cv::KeyPoint> kpts;
    string fpath = path + "features/" + to_string(loadCounter) + ".orbf";

    fstream file;
    file.open(fpath, ios::in);
    if (!file.is_open())
    {
        cerr << "Failed to open " << fpath << "...\n";
        return kpts;
    }

    string line;
    while (getline(file, line))
    {
        if (!line.empty())
        {
            istringstream iss(line);
            vector<string> vals{istream_iterator<string>{iss}, istream_iterator<string>{}};
            kpts.emplace_back(cv::KeyPoint(stof(vals[0]), stof(vals[1]), stof(vals[2]), stof(vals[3]),
                                              stof(vals[4]), stoi(vals[5])));
        }
    }
    file.close();
    return kpts;
}


bool FeatureFileInterface::SaveDescriptors(cv::Mat &descriptors)
{
    message_assert("Save path must be set", !path.empty());
    cout << "\nSaving descriptors to " << path << "...\n";

    string filename = path + "descriptors/" + to_string(saveCounter++) + ".orbd";

    if (CheckExistence(filename))
    {
        cerr << "Descriptor-file already exists!\n";
        return false;
    }

    fstream file;
    file.open(filename, ios::out);
    if (!file.is_open())
    {
        cerr << "Failed to open " << filename << "...\n";
        return false;
    }

    auto ptr = descriptors.ptr<uchar>(0);
    int step = descriptors.step1();

    for (int i = 0; i < descriptors.rows; ++i)
    {
        for (int j = 0; j < descriptors.cols; ++j)
        {
            file << (int)ptr[i*step + j] << (j == descriptors.cols-1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
    return true;
}


cv::Mat FeatureFileInterface::LoadDescriptors(string &path, cv::Mat &descriptors, int nkpts)
{
    string dpath = path + "descriptors/" + to_string(loadCounter++) + ".orbd";
    fstream file;
    file.open(dpath, ios::in);
    if (!file.is_open())
    {
        cerr << "Failed to open " << dpath << "...\n";
        return descriptors;
    }

    string line;
    auto ptr = descriptors.ptr<uchar>(0);
    int c = 0;
    while (getline(file, line))
    {
        if (!line.empty())
        {
            istringstream iss(line);
            vector<string> vals{istream_iterator<string>{iss}, istream_iterator<string>{}};
            for (auto &v : vals)
            {
                ptr[c++] = (uchar)stoi(v);
            }
        }
    }
    file.close();
    return descriptors;
}


bool FeatureFileInterface::SaveInfo(fileInfo &info)
{
    std::string infoPath = path + "settings.info";
    fstream file;
    file.open(infoPath, ios::out);
    if (!file.is_open())
    {
        cerr << "Failed to open " << infoPath << "...\n";
        return false;
    }
    file << "nFeatures: " << info.nFeatures << "\n" <<
         "nLevels: " << info.nLevels << "\n" <<
         "scaleFactor: " << info.scaleFactor << "\n" <<
         "Distribution: " << GetDistributionName(info.kptDistribution) << "\n" <<
         ((info.kptDistribution == Distribution::RANMS || info.kptDistribution == Distribution::SSC ||
           info.kptDistribution == Distribution::SOFT_SSC)?
          "SSC Threshold: " + to_string(info.SSCThreshold) + "\n" : "");
    file.close();
    return true;
}


string FeatureFileInterface::GetDistributionName(Distribution::DistributionMethod d)
{
    using distr = Distribution::DistributionMethod;
    string res;
    switch(d)
    {
        case (distr::KEEP_ALL):
            res.append("all_kept");
            break;
        case (distr::NAIVE):
            res.append("Top N");
            break;
        case (distr::RANMS):
            res.append("Bucketed Soft SSC");
            break;
        case (distr::QUADTREE_ORBSLAMSTYLE):
            res.append("Quadtree");
            break;
        case (distr::GRID):
            res.append("Bucketing");
            break;
        case (distr::ANMS_KDTREE):
            res.append("KD-Tree-ANMS");
            break;
        case (distr::ANMS_RT):
            res.append("Range-Tree-ANMS");
            break;
        case (distr::SSC):
            res.append("SSC");
            break;
        case (distr::SOFT_SSC):
            res.append("Soft SSC");
            break;
        default:
            res.append("unknown");
            break;
    }
    return res;
}


string FeatureFileInterface::GetFilenameFromPath(string &path)
{
    string name;
    string delim = "/";
    name = string(path);
    int n = name.rfind(delim, name.length()-2);
    return name.substr(n+1, name.length()-n-2);
}

bool FeatureFileInterface::CheckExistence(string &path)
{
    struct stat buf{};
    return (stat(path.c_str(), &buf) == 0);
}