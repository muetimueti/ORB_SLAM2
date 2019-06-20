#include "include/Distribution.h"
#include "include/Nanoflann.h"
#include "include/RangeTree.h"

#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>

//TODO:remove include of iostream and chrono after debugging
#include <iostream>
#include <chrono>



static void RetainBestN(std::vector<cv::KeyPoint> &kpts, int N)
{
    if (kpts.size() <= N)
        return;
    std::nth_element(kpts.begin(), kpts.begin()+N, kpts.end(),
                     [](const cv::KeyPoint &k1, const cv::KeyPoint &k2){return k1.response > k2.response;});
    kpts.resize(N);
}


void
Distribution::DistributeKeypoints(std::vector<cv::KeyPoint> &kpts, const int minX, const int maxX, const int minY,
                                  const int maxY, const int N, DistributionMethod mode, int softSSCThreshold)
{
    if (kpts.size() <= N)
        return;
    const float epsilon = 0.1;

    if (mode == ANMS_RT || mode == ANMS_KDTREE || mode == SSC || mode == RANMS || mode == SOFT_SSC)
    {
        std::vector<int> responseVector;
        for (int i = 0; i < kpts.size(); i++)
            responseVector.emplace_back(kpts[i].response);
        std::vector<int> idx(responseVector.size()); std::iota (std::begin(idx), std::end(idx), 0);
        cv::sortIdx(responseVector, idx, CV_SORT_DESCENDING);
        std::vector<cv::KeyPoint> kptsSorted;
        for (int i = 0; i < kpts.size(); i++)
            kptsSorted.emplace_back(kpts[idx[i]]);
        kpts = kptsSorted;
    }
    switch (mode)
    {
        case NAIVE :
        {
            DistributeKeypointsNaive(kpts, N);
            break;
        }
        case RANMS :
        {
            DistributeKeypointsRANMS(kpts, minX, maxX, minY, maxY, N, epsilon, softSSCThreshold);
            break;
        }
        case QUADTREE_ORBSLAMSTYLE :
        {
            DistributeKeypointsQuadTree_ORBSLAMSTYLE(kpts, minX, maxX, minY, maxY, N);
            break;
        }
        case GRID :
        {
            DistributeKeypointsGrid(kpts, minX, maxX, minY, maxY, N);
            break;
        }
        case KEEP_ALL :
        {
            break;
        }
        case ANMS_KDTREE :
        {
            int cols = maxX - minX;
            int rows = maxY - minY;
            DistributeKeypointsKdT_ANMS(kpts, rows, cols, N, epsilon);
            break;
        }
        case ANMS_RT :
        {
            int cols = maxX - minX;
            int rows = maxY - minY;
            DistributeKeypointsRT_ANMS(kpts, rows, cols, N, epsilon);
            break;
        }
        case SSC :
        {
            //TODO: fix 8th distribution not working for some reason
            int cols = maxX - minX;
            int rows = maxY - minY;
            DistributeKeypointsSoftSSC(kpts, rows, cols, N, epsilon, softSSCThreshold);
            break;
        }
        case SOFT_SSC :
        {
            int cols = maxX - minX;
            int rows = maxY - minY;
            DistributeKeypointsSoftSSC(kpts, rows, cols, N, epsilon, 0);
        }
        default:
        {
            DistributeKeypointsQuadTree(kpts, minX, maxX, minY, maxY, N);
            break;
        }
    }
}




void Distribution::DistributeKeypointsNaive(std::vector<cv::KeyPoint> &kpts, const int N)
{
    RetainBestN(kpts, N);
}


void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    int middleX = UL.x + (int)std::ceil((float)(UR.x - UL.x)/2.f);
    int middleY = UL.y + (int)std::ceil((float)(LL.y - UL.y)/2.f);

    cv::Point2i M (middleX, middleY);
    cv::Point2i upperM (middleX, UL.y);
    cv::Point2i lowerM (middleX, LL.y);
    cv::Point2i leftM (UL.x, middleY);
    cv::Point2i rightM (UR.x, middleY);

    n1.UL = UL, n1.UR = upperM, n1.LL = leftM, n1.LR = M;
    n2.UL = upperM, n2.UR = UR, n2.LL = M, n2.LR = rightM;
    n3.UL = leftM, n3.UR = M, n3.LL = LL, n3.LR = lowerM;
    n4.UL = M, n4.UR = rightM, n4.LL = lowerM, n4.LR = LR;

    for (auto &kpt : nodeKpts)
    {
        if (kpt.pt.x < middleX)
        {
            if(kpt.pt.y < middleY)
                n1.nodeKpts.emplace_back(kpt);
            else
                n3.nodeKpts.emplace_back(kpt);

        }
        else
        {
            if (kpt.pt.y < middleY)
                n2.nodeKpts.emplace_back(kpt);
            else
                n4.nodeKpts.emplace_back(kpt);
        }
    }
}


void Distribution::DistributeKeypointsQuadTree(std::vector<cv::KeyPoint>& kpts, const int minX,
                                               const int maxX, const int minY, const int maxY, const int N)
{
    assert(!kpts.empty());

    const int nroots = myRound((float)(maxX-minX)/(float)(maxY-minY));

    int nodeWidth = myRound((float)(maxX - minX) / nroots);

    std::list<ExtractorNode> nodesList;

    std::vector<ExtractorNode*> rootVec;
    rootVec.resize(nroots);

    for (int i = 0; i < nroots; ++i)
    {
        int x0 = minX + nodeWidth * i;
        int x1 = minX + nodeWidth * (i+1);
        int y0 = minY;
        int y1 = maxY;
        ExtractorNode n;
        n.UL = cv::Point2i(x0, y0);
        n.UR = cv::Point2i(x1, y0);
        n.LL = cv::Point2i(x0, y1);
        n.LR = cv::Point2i(x1, y1);
        n.nodeKpts.reserve(kpts.size());

        nodesList.push_back(n);
        rootVec[i] = &nodesList.back();
    }

    for (auto &kpt : kpts)
    {
        rootVec[(int)(kpt.pt.x / nodeWidth)]->nodeKpts.emplace_back(kpt);
    }

    std::list<ExtractorNode>::iterator current;

    bool omegadoom = false;
    int lastSize = 0;
    while (!omegadoom)
    {
        current = nodesList.begin();
        lastSize = nodesList.size();

        while (current != nodesList.end())
        {
            if (current->nodeKpts.empty())
            {
                current = nodesList.erase(current);
            }

            if (current->leaf)
            {
                ++current;
                continue;
            }

            ExtractorNode n1, n2, n3, n4;
            current->DivideNode(n1, n2, n3, n4);
            if (!n1.nodeKpts.empty())
            {
                nodesList.push_front(n1);
                if (n1.nodeKpts.size() == 1)
                    n1.leaf = true;
            }
            if (!n2.nodeKpts.empty())
            {
                nodesList.push_front(n2);
                if (n2.nodeKpts.size() == 1)
                    n2.leaf = true;
            }
            if (!n3.nodeKpts.empty())
            {
                nodesList.push_front(n3);
                if (n3.nodeKpts.size() == 1)
                    n3.leaf = true;
            }
            if (!n4.nodeKpts.empty())
            {
                nodesList.push_front(n4);
                if (n4.nodeKpts.size() == 1)
                    n4.leaf = true;
            }

            current = nodesList.erase(current);

            if (nodesList.size() == lastSize)
            {
                omegadoom = true;
                break;
            }

            if (nodesList.size() >= N)
            {
                omegadoom = true;
                break;
            }
        }
    }

    std::vector<cv::KeyPoint> resKpts;
    resKpts.reserve(N);

    auto iter = nodesList.begin();
    for (; iter != nodesList.end(); ++iter)
    {
        std::vector<cv::KeyPoint> &nodekpts = iter->nodeKpts;
        cv::KeyPoint* kpt = &nodekpts[0];
        if (iter->leaf)
        {
            resKpts.emplace_back(*kpt);
            continue;
        }

        float maxScore = kpt->response;
        for (auto &k : nodekpts)
        {
            if (k.response > maxScore)
            {
                kpt = &k;
                maxScore = k.response;
            }

        }
        resKpts.emplace_back(*kpt);
    }
    kpts = resKpts;
}


void Distribution::DistributeKeypointsQuadTree_ORBSLAMSTYLE(std::vector<cv::KeyPoint>& kpts, const int minX,
                                                            const int maxX, const int minY, const int maxY, const int N)
{
    //TODO: fix so results equal orbslam's results
    // (seems it literally cannot be done, as orbslams implementation is not deterministic)

    assert(!kpts.empty());

    const int nroots = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float nodeWidth = static_cast<float>(maxX - minX) / nroots;

    std::list<ExtractorNode> nodesList;

    std::vector<ExtractorNode*> rootVec;
    rootVec.resize(nroots);


    for (int i = 0; i < nroots; ++i)
    {
        int x0 = nodeWidth * (float)i;
        int x1 = nodeWidth * (float)(i+1);
        int y0 = 0;
        int y1 = maxY-minY;
        ExtractorNode n;
        n.UL = cv::Point2i(x0, y0);
        n.UR = cv::Point2i(x1, y0);
        n.LL = cv::Point2i(x0, y1);
        n.LR = cv::Point2i(x1, y1);
        n.nodeKpts.reserve(kpts.size());

        nodesList.push_back(n);
        rootVec[i] = &nodesList.back();
    }


    for (auto &kpt : kpts)
    {
        rootVec[(int)(kpt.pt.x / nodeWidth)]->nodeKpts.emplace_back(kpt);
    }

    std::list<ExtractorNode>::iterator current;
    current = nodesList.begin();

    while (current != nodesList.end())
    {
        if (current->nodeKpts.size() == 1)
        {
            current->leaf = true;
            ++current;
        }
        else if (current->nodeKpts.empty())
        {
            current = nodesList.erase(current);
        }
        else
            ++current;
    }

    std::vector<ExtractorNode*> nodesToExpand;
    nodesToExpand.reserve(nodesList.size()*4);

    bool omegadoom = false;
    int lastSize = 0;
    while (!omegadoom)
    {
        current = nodesList.begin();
        lastSize = nodesList.size();

        nodesToExpand.clear();
        int nToExpand = 0;

        while (current != nodesList.end())
        {

            /*
            //TODO:remove
            if (current->nodeKpts[0].pt.x > 0 && current->nodeKpts[0].pt.y > 50 &&
                current->nodeKpts[0].pt.x < 40 && current->nodeKpts[0].pt.y < 90)
            {
                std::cout << "kpt vec of node:\n";
                for (auto &kpt : current->nodeKpts)
                {
                    std::cout << kpt.pt << "\n";
                }
            }
            */
            //////////////

            if (current->leaf)
            {
                ++current;
                continue;
            }

            ExtractorNode n1, n2, n3, n4;
            current->DivideNode(n1, n2, n3, n4);
            if (!n1.nodeKpts.empty())
            {
                nodesList.push_front(n1);
                if (n1.nodeKpts.size() == 1)
                    n1.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }
            if (!n2.nodeKpts.empty())
            {
                nodesList.push_front(n2);
                if (n2.nodeKpts.size() == 1)
                    n2.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }
            if (!n3.nodeKpts.empty())
            {
                nodesList.push_front(n3);
                if (n3.nodeKpts.size() == 1)
                    n3.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }
            if (!n4.nodeKpts.empty())
            {
                nodesList.push_front(n4);
                if (n4.nodeKpts.size() == 1)
                    n4.leaf = true;
                else
                {
                    ++nToExpand;
                    nodesToExpand.emplace_back(&nodesList.front());
                    nodesList.front().lit = nodesList.begin();
                }
            }

            current = nodesList.erase(current);

        }
        if ((int)nodesList.size() >= N || (int)nodesList.size()==lastSize)
        {
            omegadoom = true;
        }

        else if ((int)nodesList.size() + nToExpand*3 > N)
        {
            while(!omegadoom)
            {
                lastSize = nodesList.size();
                std::vector<ExtractorNode*> prevNodes = nodesToExpand;

                nodesToExpand.clear();

                std::sort(prevNodes.begin(), prevNodes.end(),
                          [](const ExtractorNode *n1, const ExtractorNode *n2)
                          {return n1->nodeKpts.size() > n2->nodeKpts.size();});

                for (auto &node : prevNodes)
                {
                    ExtractorNode n1, n2, n3, n4;
                    node->DivideNode(n1, n2, n3, n4);

                    if (!n1.nodeKpts.empty())
                    {
                        nodesList.push_front(n1);
                        if (n1.nodeKpts.size() == 1)
                            n1.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    if (!n2.nodeKpts.empty())
                    {
                        nodesList.push_front(n2);
                        if (n2.nodeKpts.size() == 1)
                            n2.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    if (!n3.nodeKpts.empty())
                    {
                        nodesList.push_front(n3);
                        if (n3.nodeKpts.size() == 1)
                            n3.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    if (!n4.nodeKpts.empty())
                    {
                        nodesList.push_front(n4);
                        if (n4.nodeKpts.size() == 1)
                            n4.leaf = true;
                        else
                        {
                            nodesToExpand.emplace_back(&nodesList.front());
                            nodesList.front().lit = nodesList.begin();
                        }

                    }
                    nodesList.erase(node->lit);

                    if ((int)nodesList.size() >= N)
                        break;
                }
                if ((int)nodesList.size() >= N || (int)nodesList.size() == lastSize)
                    omegadoom = true;


            }
        }
    }


    std::vector<cv::KeyPoint> resKpts;
    resKpts.reserve(N*2);
    auto iter = nodesList.begin();
    for (; iter != nodesList.end(); ++iter)
    {
        std::vector<cv::KeyPoint> &nodekpts = iter->nodeKpts;
        cv::KeyPoint* kpt = &nodekpts[0];
        if (iter->leaf)
        {
            resKpts.emplace_back(*kpt);
            continue;
        }

        float maxScore = kpt->response;
        for (auto &k : nodekpts)
        {
            if (k.response > maxScore)
            {
                kpt = &k;
                maxScore = k.response;
            }

        }
        resKpts.emplace_back(*kpt);
    }

    kpts = resKpts;
}

/**
 *
 * @param kpts : keypoints to distribute
 * @param minX, maxX, minY, maxY : relevant image dimensions
 * @param N : number of keypoints to retain
 */
void Distribution::DistributeKeypointsGrid(std::vector<cv::KeyPoint>& kpts, const int minX, const int maxX,
                                           const int minY, const int maxY, const int N)
{
    //std::sort(kpts.begin(), kpts.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b){return (a.pt.x < b.pt.x ||
    //        (a.pt.x == b.pt.x && a.pt.y < b.pt.y));});

    const float width = maxX - minX;
    const float height = maxY - minY;

    int cellSize = (int)std::min(80.f, std::min(width, height));

    const int npatchesInX = width / cellSize;
    const int npatchesInY = height / cellSize;
    const int patchWidth = ceil(width / npatchesInX);
    const int patchHeight = ceil(height / npatchesInY);

    int nCells = npatchesInX * npatchesInY;
    std::vector<std::vector<cv::KeyPoint>> cellkpts(nCells);
    int nPerCell = (float)N / nCells;


    for (auto &kpt : kpts)
    {
        int idx = (int)(kpt.pt.y/patchHeight) * npatchesInX + (int)(kpt.pt.x/patchWidth);
        if (idx >= nCells)
            idx = nCells-1;
        cellkpts[idx].emplace_back(kpt);
    }

    kpts.clear();
    kpts.reserve(N*2);

    for (auto &kptVec : cellkpts)
    {
        RetainBestN(kptVec, nPerCell);
        kpts.insert(kpts.end(), kptVec.begin(), kptVec.end());
    }
}

/*
 * PointCloud and KdTree taken from BAILOOL/ANMS-Codes
 */

void Distribution::DistributeKeypointsKdT_ANMS(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon)
{
    int numerator1 = rows + cols + 2*N;
    long long discriminant = (long long)4*cols + (long long)4*N + (long long)4*rows*N +
                             (long long)rows*rows + (long long)cols*cols - (long long)2*cols*rows + (long long)4*cols*rows*N;

    double denominator = 2*(N-1);

    double sol1 = (numerator1 - sqrt(discriminant))/denominator;
    double sol2 = (numerator1 + sqrt(discriminant))/denominator;

    int high = (sol1>sol2)? sol1 : sol2; //binary search range initialization with positive solution
    int low = floor(sqrt((double)kpts.size()/N));

    PointCloud<int> cloud;
    generatePointCloud(cloud, kpts);
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<int, PointCloud<int>>, PointCloud<int>, 2>
            a_kd_tree;
    a_kd_tree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(25));
    tree.buildIndex();

    bool done = false;
    int kMax = myRound(N + N*epsilon), kMin = myRound(N - N*epsilon);
    std::vector<int> resultIndices;
    int radius, prevradius = 1;

    std::vector<int> tempResult;
    tempResult.reserve(kpts.size());
    while (!done)
    {
        std::vector<bool> selected(kpts.size(), true);
        radius = low + (high-low)/2;
        if (radius == prevradius || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();

        for (int i = 0; i < kpts.size(); ++i)
        {
            if (selected[i])
            {
                selected[i] = false;
                tempResult.emplace_back(i);
                int searchRadius = static_cast<int>(radius*radius);
                std::vector<std::pair<size_t, int>> retMatches;
                nanoflann::SearchParams params;
                int querypt[2] = {(int)kpts[i].pt.x, (int)kpts[i].pt.y};
                size_t nMatches = tree.radiusSearch(&querypt[0], searchRadius, retMatches, params);

                for (size_t idx = 0; idx < nMatches; ++idx)
                {
                    if(selected[retMatches[idx].first])
                        selected[retMatches[idx].first] = false;
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = radius - 1;
        else
            low = radius + 1;

        prevradius = radius;
    }
    std::vector<cv::KeyPoint> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(kpts[resultIndices[i]]);
    }
    kpts = reskpts;
}


void Distribution::DistributeKeypointsRT_ANMS(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon)
{
    int numerator1 = rows + cols + 2*N;
    long long discriminant = (long long)4*cols + (long long)4*N + (long long)4*rows*N +
                             (long long)rows*rows + (long long)cols*cols - (long long)2*cols*rows + (long long)4*cols*rows*N;

    double denominator = 2*(N-1);

    double sol1 = (numerator1 - sqrt(discriminant))/denominator;
    double sol2 = (numerator1 + sqrt(discriminant))/denominator;

    int high = (sol1>sol2)? sol1 : sol2; //binary search range initialization with positive solution
    int low = floor(sqrt((double)kpts.size()/N));

    RangeTree<u16, u16> tree(kpts.size(), kpts.size());
    for (int i = 0; i < kpts.size(); ++i)
    {
        tree.add(kpts[i].pt.x, kpts[i].pt.y, (u16 *)(intptr_t)i);
    }
    tree.finalize();

    bool done = false;
    int kMin = myRound(N - N*epsilon), kMax = myRound(N + N*epsilon);
    std::vector<int> resultIndices;
    int width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(kpts.size());
    while (!done)
    {
        std::vector<bool> selected(kpts.size(), true);
        width = low + (high-low)/2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();

        for (int i = 0; i < kpts.size(); ++i)
        {
            if (selected[i])
            {
                selected[i] = false;
                tempResult.emplace_back(i);
                int minX = static_cast<int>(kpts[i].pt.x - width);
                int maxX = static_cast<int>(kpts[i].pt.x + width);
                int minY = static_cast<int>(kpts[i].pt.y - width);
                int maxY = static_cast<int>(kpts[i].pt.y + width);

                if (minX < 0)
                    minX = 0;
                if (minY < 0)
                    minY = 0;

                std::vector<u16*> *he = tree.search(minX, maxX, minY, maxY);
                for (int j = 0; j < he->size(); ++j)
                {
                    if (selected[(u64)(*he)[j]])
                        selected[(u64)(*he)[j]] = false;
                }
                delete he;
                he = nullptr;
            }

        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<cv::KeyPoint> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(kpts[resultIndices[i]]);
    }
    kpts = reskpts;
}


void Distribution::DistributeKeypointsSSC(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon)
{
    //for (auto &kpt : kpts)
    //    std::cout << "\nkpt.response=" << kpt.response;
    //return;
    int numerator1 = rows + cols + 2*N;
    long long discriminant = (long long)4*cols + (long long)4*N + (long long)4*rows*N +
                             (long long)rows*rows + (long long)cols*cols - (long long)2*cols*rows + (long long)4*cols*rows*N;

    double denominator = 2*(N-1);

    double sol1 = (numerator1 - sqrt(discriminant))/denominator;
    double sol2 = (numerator1 + sqrt(discriminant))/denominator;

    int high = (sol1>sol2)? sol1 : sol2;
    int low = floor(sqrt((double)kpts.size()/N));

    bool done = false;
    int kMin = myRound(N - N*epsilon), kMax = myRound(N + N*epsilon);
    std::vector<int> resultIndices;
    int width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(kpts.size());

    while(!done)
    {
        width = low + (high-low)/2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();
        double c = (double)width/2.0;
        int cellCols = std::floor(cols/c);
        int cellRows = std::floor(rows/c);
        std::vector<std::vector<bool>> covered(cellRows+1, std::vector<bool>(cellCols+1, false));


        for (int i = 0; i < kpts.size(); ++i)
        {
            int row = (int)(kpts[i].pt.y/c);
            int col = (int)(kpts[i].pt.x/c);
            if (covered[row][col] == false)
            {
                tempResult.emplace_back(i);
                int rowMin = row - (int)(width/c) >= 0 ? (row - (int)(width/c)) : 0;
                int rowMax = row + (int)(width/c) <= cellRows ? (row + (int)(width/c)) : cellRows;
                int colMin = col - (int)(width/c) >= 0 ? (col - (int)(width/c)) : 0;
                int colMax = col + (int)(width/c) <= cellCols ? (col + (int)(width/c)) : cellCols;

                for (int dy = rowMin; dy <= rowMax; ++dy)
                {
                    for (int dx = colMin; dx <= colMax; ++dx)
                    {
                        if (!covered[dy][dx])
                            covered[dy][dx] = true;
                    }
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<cv::KeyPoint> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(kpts[resultIndices[i]]);
    }
    kpts = reskpts;
}


void Distribution::DistributeKeypointsRANMS(std::vector<cv::KeyPoint> &kpts, int minX, int maxX, int minY, int maxY,
        int N, float epsilon, int softSSCThreshold)
{
    const float width = maxX - minX;
    const float height = maxY - minY;
    int cellSize = (int)std::min((float)80, std::min(width, height));

    const int npatchesInX = width / cellSize;
    const int npatchesInY = height / cellSize;
    const int patchWidth = ceil(width / npatchesInX);
    const int patchHeight = ceil(height / npatchesInY);

    int nCells = npatchesInX * npatchesInY;
    std::vector<std::vector<cv::KeyPoint>> cellkpts(nCells);
    int nPerCell = (float)N / nCells;


    for (auto &kpt : kpts)
    {
        int idx = (int)(kpt.pt.y/patchHeight) * npatchesInX + (int)(kpt.pt.x/patchWidth);
        if (idx >= nCells)
            idx = nCells-1;
        cellkpts[idx].emplace_back(kpt);
    }

    kpts.clear();
    kpts.reserve(N*2);

    for (int i = 0; i < nCells; ++i)
    {
        int cellMinX = (i%npatchesInX) * patchWidth;
        int cellMaxX = cellMinX + patchWidth;
        int cellMinY = i/npatchesInX * patchHeight;
        int cellMaxY = cellMinY + patchHeight;
        if (nPerCell < cellkpts[i].size())
            SSCperCell(cellkpts[i], cellMinX, cellMaxX, cellMinY, cellMaxY, nPerCell, epsilon,
                                       softSSCThreshold);
        kpts.insert(kpts.end(), cellkpts[i].begin(), cellkpts[i].end());
    }
}


void Distribution::SSCperCell(std::vector<cv::KeyPoint> &kpts, const int minX, const int maxX,
        const int minY, const int maxY, int N, float epsilon, float threshold)
{
    int cols = maxX - minX;
    int rows = maxY - minY;
    int numerator1 = rows + cols + 2*N;
    long long discriminant = (long long)4*cols + (long long)4*N + (long long)4*rows*N +
                             (long long)rows*rows + (long long)cols*cols - (long long)2*cols*rows + (long long)4*cols*rows*N;

    double denominator = 2*(N-1);

    double sol1 = (numerator1 - sqrt(discriminant))/denominator;
    double sol2 = (numerator1 + sqrt(discriminant))/denominator;

    int high = (sol1>sol2)? sol1 : sol2;
    int low = floor(sqrt((double)kpts.size()/N));

    bool done = false;
    int kMin = myRound(N - N*epsilon), kMax = myRound(N + N*epsilon);
    std::vector<int> resultIndices;
    int width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(kpts.size());

    while(!done)
    {
        width = low + (high-low)/2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();
        double c = (double)width/2.0;
        int cellCols = std::floor(cols/c);
        int cellRows = std::floor(rows/c);
        std::vector<std::vector<float>> covered(cellRows+1, std::vector<float>(cellCols+1, -1));

        for (int i = 0; i < kpts.size(); ++i)
        {
            int row = (int)((kpts[i].pt.y-minY)/c);
            int col = (int)((kpts[i].pt.x-minX)/c);
            int score = kpts[i].response;

            if (covered[row][col] < score - threshold)
            {
                tempResult.emplace_back(i);
                int rowMin = row - (int)(width/c) >= 0 ? (row - (int)(width/c)) : 0;
                int rowMax = row + (int)(width/c) <= cellRows ? (row + (int)(width/c)) : cellRows;
                int colMin = col - (int)(width/c) >= 0 ? (col - (int)(width/c)) : 0;
                int colMax = col + (int)(width/c) <= cellCols ? (col + (int)(width/c)) : cellCols;

                for (int dy = rowMin; dy <= rowMax; ++dy)
                {
                    for (int dx = colMin; dx <= colMax; ++dx)
                    {
                        if (covered[dy][dx] < score)
                            covered[dy][dx] = score;
                    }
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<cv::KeyPoint> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(kpts[resultIndices[i]]);
    }
    kpts = reskpts;
}


void Distribution::DistributeKeypointsSoftSSC(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N,
                                              float epsilon, float threshold)
{
    int numerator1 = rows + cols + 2*N;
    long long discriminant = (long long)4*cols + (long long)4*N + (long long)4*rows*N +
                             (long long)rows*rows + (long long)cols*cols - (long long)2*cols*rows + (long long)4*cols*rows*N;

    double denominator = 2*(N-1);

    double sol1 = (numerator1 - sqrt(discriminant))/denominator;
    double sol2 = (numerator1 + sqrt(discriminant))/denominator;

    int high = (sol1>sol2)? sol1 : sol2;
    int low = floor(sqrt((double)kpts.size()/N));

    bool done = false;
    int kMin = myRound(N - N*epsilon), kMax = myRound(N + N*epsilon);
    std::vector<int> resultIndices;
    int width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(kpts.size());

    while(!done)
    {
        width = low + (high-low)/2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();
        double c = (double)width/2.0;
        int cellCols = std::floor(cols/c);
        int cellRows = std::floor(rows/c);
        std::vector<std::vector<float>> covered(cellRows+1, std::vector<float>(cellCols+1, -1));

        for (int i = 0; i < kpts.size(); ++i)
        {
            int row = (int)(kpts[i].pt.y/c);
            int col = (int)(kpts[i].pt.x/c);
            int score = kpts[i].response;

            if (covered[row][col] < score - threshold)
            {
                tempResult.emplace_back(i);
                int rowMin = row - (int)(width/c) >= 0 ? (row - (int)(width/c)) : 0;
                int rowMax = row + (int)(width/c) <= cellRows ? (row + (int)(width/c)) : cellRows;
                int colMin = col - (int)(width/c) >= 0 ? (col - (int)(width/c)) : 0;
                int colMax = col + (int)(width/c) <= cellCols ? (col + (int)(width/c)) : cellCols;

                for (int dy = rowMin; dy <= rowMax; ++dy)
                {
                    for (int dx = colMin; dx <= colMax; ++dx)
                    {
                        if (covered[dy][dx] < score)
                            covered[dy][dx] = score;
                    }
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<cv::KeyPoint> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(kpts[resultIndices[i]]);
    }
    kpts = reskpts;
}