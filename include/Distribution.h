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

enum DistributionMethod
{
DISTRIBUTION_NAIVE,
DISTRIBUTION_QUADTREE,
DISTRIBUTION_QUADTREE_ORBSLAMSTYLE,
DISTRIBUTION_GRID,
DISTRIBUTION_ANMS_KDTREE,
DISTRIBUTION_ANMS_RT,
DISTRIBUTION_SSC,
DISTRIBUTION_KEEP_ALL
};



void DistributeKeypoints(std::vector<cv::KeyPoint> &kpts, const int &minX, const int &maxX, const int &minY,
                         const int &maxY, const int &N, DistributionMethod mode);

void DistributeKeypointsNaive(std::vector<cv::KeyPoint> &kpts, const int &N);

void DistributeKeypointsQuadTree(std::vector<cv::KeyPoint>& kpts, const int &minX,
                                 const int &maxX, const int &minY, const int &maxY, const int &N);

void DistributeKeypointsQuadTree_ORBSLAMSTYLE(std::vector<cv::KeyPoint>& kpts, const int &minX,
                                              const int &maxX, const int &minY, const int &maxY, const int &N);

void DistributeKeypointsGrid(std::vector<cv::KeyPoint>& kpts, const int &minX,
                             const int &maxX, const int &minY, const int &maxY, const int &N);

CV_INLINE  int myRound( float value )
{
#if defined HAVE_LRINT || defined CV_ICC || defined __GNUC__
    return (int)lrint(value);
#else
    // not IEEE754-compliant rounding
      return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}