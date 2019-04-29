#ifndef ORBEXTRACTOR_ORBEXTRACTOR_H
#define ORBEXTRACTOR_ORBEXTRACTOR_H

#include <vector>
//#include <list>
#include <opencv/cv.h>
#include "include/Distribution.h"


#ifndef NDEBUG
#   define D(x) x
#else
#   define D(x)
#endif


namespace ORB_SLAM2
{

class ORBextractor
{
public:

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor() = default;


    void operator()( cv::InputArray image, cv::InputArray mask,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray descriptors);

    void operator()(cv::InputArray inputImage, cv::InputArray mask,
                    std::vector<cv::KeyPoint> &resultKeypoints, cv::OutputArray outputDescriptors,
                    DistributionMethod distributionMode);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;


protected:

    CV_INLINE int myRound(float value)
    {
#if defined HAVE_LRINT || defined CV_ICC || defined __GNUC__
        return (int)lrint(value);
#else
        // not IEEE754-compliant
      return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
    }

    static float IntensityCentroidAngle(const uchar* pointer, int step);

    void ComputeScalePyramid(cv::Mat &image);

    void DivideAndFAST(std::vector<std::vector<cv::KeyPoint> >& allKeypoints,
                       DistributionMethod mode = DISTRIBUTION_NAIVE, bool divideImage = false, int cellSize = 30);

    void FAST(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, int &threshold, int level = 0);

    int OptimizedCornerScore(const uchar *pointer, const int offset[], int &threshold);

    void ComputeAngles(std::vector<std::vector<cv::KeyPoint>> &allkpts);

    void ComputeDescriptors(std::vector<std::vector<cv::KeyPoint>> &allkpts, cv::Mat &descriptors);


    std::vector<cv::Point> pattern;

    //inline float getScale(int lvl);


    int continuousPixelsRequired;
    int onePointFiveCircles;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;


    uchar threshold_tab_min[512];
    uchar threshold_tab_init[512];
    std::vector<int> pixelOffset;

    std::vector<int> nfeaturesPerLevelVec;


    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};


}

#endif //ORBEXTRACTOR_ORBEXTRACTOR_H