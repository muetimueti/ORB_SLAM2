#ifndef ORBEXTRACTOR_DISTRIBUTION_H
#define ORBEXTRACTOR_DISTRIBUTION_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <list>

class ExtractorNode
{
public:
    ExtractorNode():leaf(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> nodeKpts;
    cv::Point2i UL, UR, LL, LR;
    std::list<ExtractorNode>::iterator lit;
    bool leaf;
};

class Distribution
{
public:

    enum DistributionMethod
    {
    NAIVE,
    QUADTREE,
    QUADTREE_ORBSLAMSTYLE,
    GRID,
    ANMS_KDTREE,
    ANMS_RT,
    SSC,
    KEEP_ALL
    };

    static void DistributeKeypoints(std::vector<cv::KeyPoint> &kpts, const int &minX, const int &maxX, const int &minY,
                                    const int &maxY, const int &N, DistributionMethod mode);

protected:

    static void DistributeKeypointsNaive(std::vector<cv::KeyPoint> &kpts, const int &N);

    static void DistributeKeypointsQuadTree(std::vector<cv::KeyPoint>& kpts, const int &minX,
                                            const int &maxX, const int &minY, const int &maxY, const int &N);

    static void DistributeKeypointsQuadTree_ORBSLAMSTYLE(std::vector<cv::KeyPoint>& kpts, const int &minX,
                                                         const int &maxX, const int &minY, const int &maxY, const int &N);

    static void DistributeKeypointsGrid(std::vector<cv::KeyPoint>& kpts, const int &minX,
                                        const int &maxX, const int &minY, const int &maxY, const int &N);

};

CV_INLINE  int myRound( float value )
{
#if defined HAVE_LRINT || defined CV_ICC || defined __GNUC__
    return (int)lrint(value);
#else
    // not IEEE754-compliant rounding
      return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}

#endif