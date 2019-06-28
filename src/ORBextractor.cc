#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <unistd.h>
#include "include/ORBextractor.h"
#include "include/ORBconstants.h"

#define MYFAST 1
#define ENABLE_MAX_DURATION 1



namespace ORB_SLAM2
{

float ORBextractor::IntensityCentroidAngle(const uchar* pointer, int step)
{
    //m10 ~ x^1y^0, m01 ~ x^0y^1
    int x, y, m01 = 0, m10 = 0;

    int half_patch = PATCH_SIZE / 2;

    for (x = -half_patch; x <= half_patch; ++x)
    {
        m10 += x * pointer[x];
    }

    for (y = 1; y <= half_patch; ++y)
    {
        int cols = CIRCULAR_ROWS[y];
        int sumY = 0;
        for (x = -cols; x <= cols; ++x)
        {
            int uptown = pointer[x + y*step];
            int downtown = pointer[x - y*step];
            sumY += uptown - downtown;
            m10 += x * (uptown + downtown);
        }
        m01 += y * sumY;
    }

    return cv::fastAtan2((float)m01, (float)m10);
}


ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST):
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), iniThFAST(_iniThFAST),
        minThFAST(_minThFAST), patternsize(16), kptDistribution(Distribution::DistributionMethod::SSC),
        distributePerLevel(true), softSSCThreshold(0), fast(_iniThFAST, _minThFAST, _nlevels),
        fileInterface(), saveFeatures(false), usePrecomputedFeatures(false)
{
    mvScaleFactor.resize(nlevels);
    mvInvScaleFactor.resize(nlevels);
    mvImagePyramid.resize(nlevels);
    nfeaturesPerLevelVec.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);

    SetFASTThresholds(_iniThFAST, _minThFAST);

    mvScaleFactor[0] = 1.f;
    mvInvScaleFactor[0] = 1.f;


    for (int i = 1; i < nlevels; ++i) {
        mvScaleFactor[i] = scaleFactor * mvScaleFactor[i - 1];
        mvInvScaleFactor[i] = 1 / mvScaleFactor[i];

        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.f / mvLevelSigma2[i];
    }

    SetnFeatures(nfeatures);

    const int nPoints = 512;
    const auto tempPattern = (const cv::Point*) bit_pattern_31_;
    std::copy(tempPattern, tempPattern+nPoints, std::back_inserter(pattern));

}

void ORBextractor::SetnFeatures(int n)
{
    //reject unreasonable values
    if (n < 1 || n > 10000)
        return;

    nfeatures = n;

    float fac = 1.f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.f - fac) / (1.f - (float) pow((double) fac, (double) nlevels));

    int sumFeatures = 0;
    for (int i = 0; i < nlevels - 1; ++i)
    {
        nfeaturesPerLevelVec[i] = myRound(nDesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevelVec[i];
        nDesiredFeaturesPerScale *= fac;
    }
    nfeaturesPerLevelVec[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
}

void ORBextractor::SetFASTThresholds(int ini, int min)
{
    if ((ini == iniThFAST && min == minThFAST))
        return;

    iniThFAST = std::min(255, std::max(1, ini));
    minThFAST = std::min(iniThFAST, std::max(1, min));

    fast.SetFASTThresholds(ini, min);
}


/**
 * @param inputImage single channel img-matrix
 * @param mask ignored
 * @param resultKeypoints keypoint vector in which results will be stored
 * @param outputDescriptors matrix in which descriptors will be stored
 * @param distributePerLevel true->distribute kpts per octave, false->distribute kpts per image
 */

void ORBextractor::operator()(cv::InputArray inputImage, cv::InputArray mask,
                              std::vector<cv::KeyPoint> &resultKeypoints, cv::OutputArray outputDescriptors)
{
    std::chrono::high_resolution_clock::time_point funcEntry = std::chrono::high_resolution_clock::now();

    if (usePrecomputedFeatures)
    {
        resultKeypoints = fileInterface.LoadFeatures(loadPath);

        cv::Mat BRIEFdescriptors;
        int nkpts = resultKeypoints.size();
        if (nkpts <= 0)
        {
            outputDescriptors.release();
        }
        else
        {
            outputDescriptors.create(nkpts, 32, CV_8U);
            BRIEFdescriptors = outputDescriptors.getMat();
        }

        fileInterface.LoadDescriptors(loadPath, BRIEFdescriptors, nkpts);

        if (inputImage.empty())
            return;

        cv::Mat image = inputImage.getMat();
        assert(image.type() == CV_8UC1);
        ComputeScalePyramid(image);

        return;
    }



    if (inputImage.empty())
        return;

    cv::Mat image = inputImage.getMat();
    assert(image.type() == CV_8UC1);

    ComputeScalePyramid(image);

    std::vector<int> steps(nlevels);
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        steps[lvl] = (int)mvImagePyramid[lvl].step1();
    }
    fast.SetStepVector(steps);

    std::vector<std::vector<cv::KeyPoint>> allkpts;

    //using namespace std::chrono;
    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    DivideAndFAST(allkpts, kptDistribution, true, 30, distributePerLevel);

    if (!distributePerLevel)
    {
        ComputeAngles(allkpts);

        int lvl;
        int nkpts = 0;
        for (lvl = 0; lvl < nlevels; ++lvl)
        {
            nkpts += allkpts[lvl].size();

            float size = PATCH_SIZE * mvScaleFactor[lvl];
            float scale = mvScaleFactor[lvl];
            for (auto &kpt : allkpts[lvl])
            {
                kpt.size = size;
                if (lvl)
                    kpt.pt *= scale;
            }
        }

        auto temp = allkpts[0];
        for (lvl = 1; lvl < nlevels; ++lvl)
        {
            temp.insert(temp.end(), allkpts[lvl].begin(), allkpts[lvl].end());
        }
        Distribution::DistributeKeypoints(temp, 0, mvImagePyramid[0].cols, 0, mvImagePyramid[0].rows,
                                          nfeatures, kptDistribution, softSSCThreshold);

        for (lvl = 0; lvl < nlevels; ++lvl)
            allkpts[lvl].clear();

        for (auto &kpt : temp)
        {
            allkpts[kpt.octave].emplace_back(kpt);
        }
    }

    if (distributePerLevel)
        ComputeAngles(allkpts);

    //high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //auto d = duration_cast<microseconds>(t2-t1).count();
    //std::cout << "\nmy comp time for FAST + distr: " << d << "\n";

    cv::Mat BRIEFdescriptors;
    int nkpts = 0;
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        nkpts += (int)allkpts[lvl].size();
    }
    if (nkpts <= 0)
    {
        outputDescriptors.release();
    }
    else
    {
        outputDescriptors.create(nkpts, 32, CV_8U);
        BRIEFdescriptors = outputDescriptors.getMat();
    }

    resultKeypoints.clear();
    resultKeypoints.reserve(nkpts);

    ComputeDescriptors(allkpts, BRIEFdescriptors);

    if (distributePerLevel)
    {
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            float size = PATCH_SIZE * mvScaleFactor[lvl];
            float scale = mvScaleFactor[lvl];
            for (auto &kpt : allkpts[lvl])
            {
                kpt.size = size;
                if (lvl)
                    kpt.pt *= scale;
            }
        }
    }

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        resultKeypoints.insert(resultKeypoints.end(), allkpts[lvl].begin(), allkpts[lvl].end());
    }
#if ENABLE_MAX_DURATION
    //ensure feature detection always takes x ms
    unsigned long maxDuration = 20000;
    std::chrono::high_resolution_clock::time_point funcExit = std::chrono::high_resolution_clock::now();
    auto funcDuration = std::chrono::duration_cast<std::chrono::microseconds>(funcExit-funcEntry).count();
    //assert(funcDuration <= maxDuration);
    if (funcDuration < maxDuration)
    {
        auto sleeptime = maxDuration - funcDuration;
        usleep(sleeptime);
    }
#endif
}


void ORBextractor::ComputeAngles(std::vector<std::vector<cv::KeyPoint>> &allkpts)
{
#pragma omp parallel for
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (auto &kpt : allkpts[lvl])
        {
            kpt.angle = IntensityCentroidAngle(&mvImagePyramid[lvl].at<uchar>(myRound(kpt.pt.y), myRound(kpt.pt.x)),
                                               mvImagePyramid[lvl].step1());
        }
    }
}


void ORBextractor::ComputeDescriptors(std::vector<std::vector<cv::KeyPoint>> &allkpts, cv::Mat &descriptors)
{
    const auto degToRadFactor = (float)(CV_PI/180.f);
    const cv::Point* p = &pattern[0];

    int current = 0;

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        cv::Mat lvlClone = mvImagePyramid[lvl].clone();
        cv::GaussianBlur(lvlClone, lvlClone, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        const int step = (int)lvlClone.step;


        int i = 0, nkpts = allkpts[lvl].size();
        for (int k = 0; k < nkpts; ++k, ++current)
        {
            const cv::KeyPoint &kpt = allkpts[lvl][k];
            auto descPointer = descriptors.ptr<uchar>(current);        //ptr to beginning of current descriptor
            const uchar* pixelPointer = &lvlClone.at<uchar>(myRound(kpt.pt.y), myRound(kpt.pt.x));  //ptr to kpt in img

            float angleRad = kpt.angle * degToRadFactor;
            auto a = (float)cos(angleRad), b = (float)sin(angleRad);

            int byte = 0, v0, v1, idx0, idx1;
            for (i = 0; i <= 512; i+=2)
            {
                if (i > 0 && i%16 == 0) //working byte full
                {
                    descPointer[i/16 - 1] = (uchar)byte;  //write current byte to descriptor-mat
                    byte = 0;      //reset working byte
                    if (i == 512)  //break out after writing very last byte, so oob indices aren't accessed
                        break;
                }

                idx0 = myRound(p[i].x*a - p[i].y*b) + myRound(p[i].x*b + p[i].y*a)*step;
                idx1 = myRound(p[i+1].x*a - p[i+1].y*b) + myRound(p[i+1].x*b + p[i+1].y*a)*step;

                v0 = pixelPointer[idx0];
                v1 = pixelPointer[idx1];

                byte |= (v0 < v1) << ((i%16)/2); //write comparison bit to current byte
            }
        }
    }
}


/**
 * @param allkpts KeyPoint vector in which the result will be stored
 * @param mode decides which method to call for keypoint distribution over image, see Distribution.h
 * @param divideImage  true-->divide image into cellSize x cellSize cells, run FAST per cell
 * @param cellSize must be greater than 16 and lesser than min(rows, cols) of smallest image in pyramid
 */
void ORBextractor::DivideAndFAST(std::vector<std::vector<cv::KeyPoint>> &allkpts,
                                 Distribution::DistributionMethod mode, bool divideImage, int cellSize, bool distributePerLevel)
{
    allkpts.resize(nlevels);

    const int minimumX = EDGE_THRESHOLD - 3, minimumY = minimumX;

    if (!divideImage)
    {
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            std::vector<cv::KeyPoint> levelKpts;
            levelKpts.clear();
            levelKpts.reserve(nfeatures*10);

            const int maximumX = mvImagePyramid[lvl].cols - EDGE_THRESHOLD + 3;
            const int maximumY = mvImagePyramid[lvl].rows - EDGE_THRESHOLD + 3;


#if MYFAST
            fast.FAST(mvImagePyramid[lvl].rowRange(minimumY, maximumY).colRange(minimumX, maximumX),
                      levelKpts, iniThFAST, lvl);

            if (levelKpts.empty())
            {
                fast.FAST(mvImagePyramid[lvl].rowRange(minimumY, maximumY).colRange(minimumX, maximumX),
                          levelKpts, minThFAST, lvl);
            }
#else
            switch (patternsize)
            {
                case (12):
                {
                    cv::FAST(mvImagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                             levelKpts, iniThFAST, true, cv::FastFeatureDetector::TYPE_7_12);
                    if (levelKpts.empty())
                    {
                        cv::FAST(mvImagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                                 levelKpts, minThFAST, true, cv::FastFeatureDetector::TYPE_7_12);
                    }
                    break;
                }
                case (8):
                {
                    cv::FAST(mvImagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                             levelKpts, iniThFAST, true, cv::FastFeatureDetector::TYPE_5_8);
                    if (levelKpts.empty())
                    {
                        cv::FAST(mvImagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                                 levelKpts, minThFAST, true, cv::FastFeatureDetector::TYPE_5_8);
                    }
                    break;
                }
                default:
                {
                    fast.FAST(mvImagePyramid[lvl].rowRange(minimumY, maximumY).colRange(minimumX, maximumX),
                              levelKpts, iniThFAST, lvl);

                    if (levelKpts.empty())
                    {
                        fast.FAST(mvImagePyramid[lvl].rowRange(minimumY, maximumY).colRange(minimumX, maximumX),
                                  levelKpts, minThFAST, lvl);
                    }
                }
            }

#endif


            if(levelKpts.empty())
                continue;

            allkpts[lvl].reserve(nfeaturesPerLevelVec[lvl]);

            if (distributePerLevel)
                Distribution::DistributeKeypoints(levelKpts, minimumX, maximumX, minimumY, maximumY,
                                                  nfeaturesPerLevelVec[lvl], mode, softSSCThreshold);


            allkpts[lvl] = levelKpts;

            for (auto &kpt : allkpts[lvl])
            {
                kpt.pt.y += minimumY;
                kpt.pt.x += minimumX;
                kpt.octave = lvl;
                //kpt.angle = IntensityCentroidAngle(&mvImagePyramid[lvl].at<uchar>(
                //        myRound(kpt.pt.x), myRound(kpt.pt.y)), mvImagePyramid[lvl].step1());
            }
        }
    }
    else
    {
        int c = std::min(mvImagePyramid[nlevels-1].rows, mvImagePyramid[nlevels-1].cols);
        assert(cellSize < c && cellSize > 16);
#pragma omp parallel for
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            std::vector<cv::KeyPoint> levelKpts;
            levelKpts.clear();
            levelKpts.reserve(nfeatures*10);

            const int maximumX = mvImagePyramid[lvl].cols - EDGE_THRESHOLD + 3;
            const int maximumY = mvImagePyramid[lvl].rows - EDGE_THRESHOLD + 3;
            const float width = maximumX - minimumX;
            const float height = maximumY - minimumY;

            const int npatchesInX = width / cellSize;
            const int npatchesInY = height / cellSize;
            const int patchWidth = ceil(width / npatchesInX);
            const int patchHeight = ceil(height / npatchesInY);

            for (int py = 0; py < npatchesInY; ++py)
            {
                float startY = minimumY + py * patchHeight;
                float endY = startY + patchHeight + 6;

                if (startY >= maximumY-3)
                    continue;

                if (endY > maximumY)
                    endY = maximumY;


                for (int px = 0; px < npatchesInX; ++px)
                {
                    float startX = minimumX + px * patchWidth;
                    float endX = startX + patchWidth + 6;

                    if (startX >= maximumX-6)
                        continue;

                    if (endX > maximumX)
                        endX = maximumX;

                    std::vector<cv::KeyPoint> patchKpts;


                    std::chrono::high_resolution_clock::time_point FASTEntry =
                            std::chrono::high_resolution_clock::now();

#if MYFAST
                    fast.FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                              patchKpts, iniThFAST, lvl);
                    if (patchKpts.empty())
                    {
                        fast.FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                  patchKpts, minThFAST, lvl);
                    }
#else
                    switch (patternsize)
                    {
                        case (12):
                        {
                            cv::FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                     patchKpts, iniThFAST, true, cv::FastFeatureDetector::TYPE_7_12);
                            if (patchKpts.empty())
                            {
                                cv::FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                         patchKpts, minThFAST, true, cv::FastFeatureDetector::TYPE_7_12);
                            }
                            break;
                        }
                        case (8):
                        {
                            cv::FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                     patchKpts, iniThFAST, true, cv::FastFeatureDetector::TYPE_5_8);
                            if (patchKpts.empty())
                            {
                                cv::FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                         patchKpts, minThFAST, true, cv::FastFeatureDetector::TYPE_5_8);
                            }
                            break;
                        }
                        default:
                        {
                            fast.FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                      patchKpts, iniThFAST, lvl);
                            if (patchKpts.empty())
                            {
                                fast.FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                          patchKpts, minThFAST, lvl);
                            }
                            break;
                        }
                    }
#endif
                    if(patchKpts.empty())
                        continue;

                    for (auto &kpt : patchKpts)
                    {
                        kpt.pt.y += py * patchHeight;
                        kpt.pt.x += px * patchWidth;
                        levelKpts.emplace_back(kpt);
                    }
                }
            }

            allkpts[lvl].reserve(nfeatures);

            if (distributePerLevel)
                Distribution::DistributeKeypoints(levelKpts, minimumX, maximumX, minimumY, maximumY,
                                                  nfeaturesPerLevelVec[lvl], mode, softSSCThreshold);


            allkpts[lvl] = levelKpts;



            for (auto &kpt : allkpts[lvl])
            {
                kpt.pt.y += minimumY;
                kpt.pt.x += minimumX;
                kpt.octave = lvl;
                //kpt.angle = IntensityCentroidAngle(&mvImagePyramid[lvl].at<uchar>(
                //        myRound(kpt.pt.x), myRound(kpt.pt.y)), mvImagePyramid[lvl].step1());
            }
        }
    }
}


void ORBextractor::ComputeScalePyramid(cv::Mat &image)
{
    for (int lvl = 0; lvl < nlevels; ++ lvl)
    {
        int width = (int)myRound(image.cols * mvInvScaleFactor[lvl]); // 1.f / getScale(lvl));
        int height = (int)myRound(image.rows * mvInvScaleFactor[lvl]); // 1.f / getScale(lvl));
        int doubleEdge = EDGE_THRESHOLD * 2;
        int borderedWidth = width + doubleEdge;
        int borderedHeight = height + doubleEdge;

        cv::Mat borderedImg(borderedHeight, borderedWidth, image.type());
        cv::Range rowRange(EDGE_THRESHOLD, height + EDGE_THRESHOLD);
        cv::Range colRange(EDGE_THRESHOLD, width + EDGE_THRESHOLD);

        mvImagePyramid[lvl] = borderedImg(rowRange, colRange);


        if (lvl)
        {
            cv::resize(mvImagePyramid[lvl-1], mvImagePyramid[lvl], cv::Size(width, height), 0, 0, CV_INTER_LINEAR);

            cv::copyMakeBorder(mvImagePyramid[lvl], borderedImg, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               EDGE_THRESHOLD, cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
        }
        else
        {
            cv::copyMakeBorder(image, borderedImg, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101);
        }
    }
}

}


