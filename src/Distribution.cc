#include "include/Distribution.h"

#include <vector>
#include <opencv2/core/core.hpp>
#include <iterator>
#include <algorithm>

//TODO:remove include after debugging
#include <iostream>


static bool CompareVectorSize(const ExtractorNode *n1, const ExtractorNode *n2)
{
    return (n1->nodeKpts.size() > n2->nodeKpts.size());
}
struct CompareVecSz
{
bool operator()(const ExtractorNode *n1, const ExtractorNode *n2) const
{
    return (n1->nodeKpts.size() > n2->nodeKpts.size());
}
};

static bool ResponseComparison(const cv::KeyPoint &k1, const cv::KeyPoint &k2)
{
    return (k1.response > k2.response);
}
struct CompareResponse {
bool operator()(const cv::KeyPoint &k1, const cv::KeyPoint &k2) const
{
    return k1.response > k2.response;
}
};
static void RetainBestN(std::vector<cv::KeyPoint> &kpts, int N)
{
    std::sort(kpts.begin(), kpts.end(), CompareResponse());
    kpts.resize(N);
}


void
DistributeKeypoints(std::vector<cv::KeyPoint> &kpts, const int &minX, const int &maxX, const int &minY,
                    const int &maxY, const int &N, DistributionMethod mode)
{
    switch (mode)
    {
        case DISTRIBUTION_NAIVE :
        {
            DistributeKeypointsNaive(kpts, N);
            break;
        }
        case DISTRIBUTION_QUADTREE :
        {
            DistributeKeypointsQuadTree(kpts, minX, maxX, minY, maxY, N);
            break;
        }
        case DISTRIBUTION_QUADTREE_ORBSLAMSTYLE :
        {
            DistributeKeypointsQuadTree_ORBSLAMSTYLE(kpts, minX, maxX, minY, maxY, N);
            break;
        }
        case DISTRIBUTION_GRID :
        {
            DistributeKeypointsGrid(kpts, minX, maxX, minY, maxY, N, 30);
            break;
        }
        case DISTRIBUTION_KEEP_ALL :
        {
            break;
        }
        default:
        {
            DistributeKeypointsNaive(kpts, N);
            break;
        }
    }

}




void DistributeKeypointsNaive(std::vector<cv::KeyPoint> &kpts, const int &N)
{
    RetainBestN(kpts, N);
}


void DistributeKeypointsQuadTree(std::vector<cv::KeyPoint>& kpts, const int &minX,
                                 const int &maxX, const int &minY, const int &maxY, const int &N)
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
    for (int i = 0; i < N&& iter != nodesList.end(); ++i, ++iter)
    {
        if (!iter->leaf)
            RetainBestN(iter->nodeKpts, 1);

        resKpts.emplace_back(iter->nodeKpts[0]);
    }

    kpts = resKpts;
}


void DistributeKeypointsQuadTree_ORBSLAMSTYLE(std::vector<cv::KeyPoint>& kpts, const int &minX,
                                              const int &maxX, const int &minY, const int &maxY, const int &N)
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

                std::sort(prevNodes.begin(), prevNodes.end(), CompareVecSz());

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
 * @param cellSize : must be larger than 16, lesser than min(width, height)
 */
void DistributeKeypointsGrid(std::vector<cv::KeyPoint>& kpts, const int &minX, const int &maxX, const int &minY,
                             const int &maxY, const int &N, const int &cellSize)
{
    //TODO: implement
    const int width = maxX - minX;
    const int height = maxY - minY;

    int c = std::min(width, height);
    assert(cellSize < c && cellSize > 16);

    const int npatchesInX = width / cellSize;
    const int npatchesInY = height / cellSize;
    const int patchWidth = std::ceil(width / npatchesInX);
    const int patchHeight = std::ceil(height / npatchesInY);


    for (int py = 0; py < npatchesInY; ++py)
    {
        int startY = minY + py * patchHeight;
        int endY = startY + patchHeight + 6;
        if (startY >= maxY)
            continue;
        if (endY > maxY)
            endY = maxY;
        for (int px = 0; px < npatchesInX; ++px)
        {
            int startX = minX + px * patchWidth;
            int endX = startX + patchWidth + 6;
            if (startX >= maxX)
                continue;
            if (endX > maxX)
                endX = maxX;

            //TODO: sort kpts into cells, calculate N and retain N per cell
        }
    }
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