#ifndef ORBEXTRACTOR_ORBEXTRACTOR_H
#define ORBEXTRACTOR_ORBEXTRACTOR_H

#include <vector>
#include <opencv/cv.h>
#include "include/Distribution.h"
#include "include/FAST.h"


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

    void inline SetDistribution(Distribution::DistributionMethod mode)
    {
        kptDistribution = mode;
    }

    /**@overload
     */
    void inline SetDistribution(int mode)
    {
        kptDistribution = static_cast<Distribution::DistributionMethod>(mode);
    }

    Distribution::DistributionMethod inline GetDistribution()
    {
        return kptDistribution;
    }

    bool inline AreKptsDistributedPerLevel()
    {
        return distributePerLevel;
    }

    void inline SetDistributionPerLevel(bool dpL)
    {
        distributePerLevel = dpL;
    }

    void inline SetPatternsize(int n)
    {
        assert (n == 16 || n == 8 || n == 12);
        patternsize = n;
    }

    int inline GetPatternsize()
    {
        return patternsize;
    }

    void SetnFeatures(int n);

    void SetFASTThresholds(int ini, int min);

    void inline SetScoreType(FASTdetector::ScoreType s)
    {
        fast.SetScoreType(s);
    }

    FASTdetector::ScoreType inline GetScoreType()
    {
        return fast.GetScoreType();
    }

    int inline GetnLevels()
    {
        return nlevels;
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

    void DivideAndFAST(std::vector<std::vector<cv::KeyPoint>> &allKeypoints,
                       Distribution::DistributionMethod mode = Distribution::QUADTREE,
                       bool divideImage = true, int cellSize = 30, bool distributePerLevel = false);

    void ComputeAngles(std::vector<std::vector<cv::KeyPoint>> &allkpts);

    void ComputeDescriptors(std::vector<std::vector<cv::KeyPoint>> &allkpts, cv::Mat &descriptors);


    std::vector<cv::Point> pattern;

    //inline float getScale(int lvl);

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;
    int patternsize;

    Distribution::DistributionMethod kptDistribution;

    bool distributePerLevel;

    std::vector<int> pixelOffset;

    std::vector<int> nfeaturesPerLevelVec;


    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    FASTdetector fast;
};


}

#endif //ORBEXTRACTOR_ORBEXTRACTOR_H