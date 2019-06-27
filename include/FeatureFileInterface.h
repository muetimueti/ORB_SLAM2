#ifndef ORB_SLAM2_FEATUREFILEINTERFACE_H
#define ORB_SLAM2_FEATUREFILEINTERFACE_H


#include <string>
#include <opencv2/core/mat.hpp>
#include "include/Types.h"
#include "Distribution.h"
#include <sys/stat.h>

#define message_assert(expr, msg) assert(( (void)(msg), (expr) ))

class FeatureFileInterface
{
public:
    typedef struct fileInfo
    {
    int nLevels;
    int nFeatures;
    float scaleFactor;
    int SSCThreshold;
    Distribution::DistributionMethod kptDistribution;
    } fileInfo;

    explicit FeatureFileInterface(std::string &_path) : saveCounter(0), loadCounter(0) {SetPath(_path);}
    FeatureFileInterface() : path(), saveCounter(0), loadCounter(0) {}

    bool SaveFeatures(std::vector<cv::KeyPoint> &kpts);

    std::vector<cv::KeyPoint> LoadFeatures(std::string &path);

    bool SaveDescriptors(cv::Mat &descriptors);

    cv::Mat LoadDescriptors(std::string &path, cv::Mat &descriptors, int nkpts);

    bool SaveInfo(fileInfo &info);

    std::string GetDistributionName(Distribution::DistributionMethod d);

    std::string GetFilenameFromPath(std::string &path);

    bool CheckExistence(std::string &path);

    void SetPath(std::string &_path)
    {
        path = _path;
        struct stat buf{};
        bool dex = (stat(path.c_str(), &buf) == 0);
        if (!dex)
        {
            mkdir(path.c_str(), S_IRWXU);
            std::string fpath = path + "features/";
            std::string dpath = path + "descriptors/";
            mkdir(fpath.c_str(), S_IRWXU);
            mkdir(dpath.c_str(), S_IRWXU);
        }
    }

    void inline SetCurrentImage(int n)
    {
        loadCounter = n;
    }

private:
    std::string path;
    int saveCounter;
    int loadCounter;
};


#endif //ORB_SLAM2_FEATUREFILEINTERFACE_H
