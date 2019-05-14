#ifndef ORB_SLAM2_FAST_H
#define ORB_SLAM2_FAST_H

#include <opencv2/core/core.hpp>

const int CIRCLE_SIZE = 16;

const int CIRCLE_OFFSETS[16][2] =
        {{0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
         {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}};

const int PIXELS_TO_CHECK[16] =
        {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15};

class FASTdetector
{
public:
    FASTdetector(int _iniThreshold, int _minThreshold, int _nlevels);

    ~FASTdetector() = default;

    void SetStepVector(std::vector<int> &_steps);

    void SetFASTThresholds(int ini, int min);

    void FAST(cv::Mat img, std::vector<cv::KeyPoint> &keypoints, int threshold, int lvl);

    enum ScoreType
    {
    OPENCV,
    HARRIS,
    SUM,
    EXPERIMENTAL
    };

    void inline SetScoreType(ScoreType t)
    {
        scoreType = t;
    }

protected:

    int iniThreshold;
    int minThreshold;

    int nlevels;

    int continuousPixelsRequired;
    int onePointFiveCircles;

    ScoreType scoreType;

    std::vector<int> pixelOffset;
    std::vector<int> steps;

    uchar threshold_tab_init[512];
    uchar threshold_tab_min[512];

    template <typename scoretype>
    void FAST(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, int threshold, int lvl);

    float CornerScore_Harris(const uchar* ptr, int lvl);

    float CornerScore_Experimental(const uchar* ptr, int lvl);

    float CornerScore_Sum(const uchar* ptr, const int offset[]);

    float CornerScore(const uchar* pointer, const int offset[], int threshold);
};

#endif //ORB_SLAM2_FAST_H
