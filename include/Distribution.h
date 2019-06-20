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
    NAIVE = 0,
    RANMS = 1,
    QUADTREE_ORBSLAMSTYLE = 2,
    GRID = 3,
    ANMS_KDTREE = 4,
    ANMS_RT = 5,
    SSC = 6,
    KEEP_ALL = 7,
    SOFT_SSC = 8
    };

    static void DistributeKeypoints(std::vector<cv::KeyPoint> &kpts, int minX, int maxX, int minY,
                                    int maxY, int N, DistributionMethod mode, int softSSCThreshold = 0);

protected:

    static void DistributeKeypointsNaive(std::vector<cv::KeyPoint> &kpts, int N);

    static void DistributeKeypointsQuadTree(std::vector<cv::KeyPoint> &kpts, int minX,
                                            int maxX, int minY, int maxY, int N);

    static void DistributeKeypointsQuadTree_ORBSLAMSTYLE(std::vector<cv::KeyPoint> &kpts, int minX,
                                                         int maxX, int minY, int maxY, int N);

    static void DistributeKeypointsGrid(std::vector<cv::KeyPoint> &kpts, int minX,
                                        int maxX, int minY, int maxY, int N);

    static void DistributeKeypointsKdT_ANMS(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon);

    static void DistributeKeypointsRT_ANMS(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon);

    static void DistributeKeypointsSSC(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon);

    static void DistributeKeypointsRANMS(std::vector<cv::KeyPoint> &kpts, int minX, int maxX, int minY, int maxY, int N,
            float epsilon, int softSSCThreshold);

    static void SSCperCell(std::vector<cv::KeyPoint> &kpts, const int minX, const int maxX,
                           const int minY, const int maxY, int N, float epsilon, float threshold);

    static void DistributeKeypointsSoftSSC(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N,
                                                         float epsilon, float threshold);
};

template <typename T>
struct PointCloud
{
    struct Point
    {
     T  x,y;
    };
    std::vector<Point>  pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); } // Must return the number of data points
    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored i
    // n the class:
    inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
    {
        const T d0=p1[0]-pts[idx_p2].x;
        const T d1=p1[1]-pts[idx_p2].y;
        return d0*d0+d1*d1;
    }
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim==0) return pts[idx].x;
        else if (dim==1) return pts[idx].y;
        return 0;
    }
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided
    //   to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

template <typename T>
void generatePointCloud(PointCloud<T> &point, std::vector<cv::KeyPoint> keyPoints)
{
    point.pts.resize(keyPoints.size());
    for (size_t i=0;i<keyPoints.size();i++)
    {
        point.pts[i].x = keyPoints[i].pt.x;
        point.pts[i].y = keyPoints[i].pt.y;
    }
}

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