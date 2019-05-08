#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <unistd.h>
#include "include/ORBextractor.h"
#include "include/ORBconstants.h"



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



ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                           int _iniThFAST, int _minThFAST):
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        iniThFAST(_iniThFAST), minThFAST(_minThFAST), threshold_tab_min{}, threshold_tab_init{}, pixelOffset{}

{

    mvScaleFactor.resize(nlevels);
    mvInvScaleFactor.resize(nlevels);
    mvImagePyramid.resize(nlevels);
    nfeaturesPerLevelVec.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    pixelOffset.resize(nlevels * CIRCLE_SIZE);

    continuousPixelsRequired = CIRCLE_SIZE / 2;
    onePointFiveCircles = CIRCLE_SIZE + continuousPixelsRequired + 1;

    iniThFAST = std::min(255, std::max(0, iniThFAST));
    minThFAST = std::min(iniThFAST, std::max(0, minThFAST));

    //initialize threshold tabs for init and min threshold
    int i;
    for (i = 0; i < 512; ++i)
    {
        int v = i - 255;
        if (v < -iniThFAST)
        {
            threshold_tab_init[i] = (uchar)1;
            threshold_tab_min[i] = (uchar)1;
        } else if (v > iniThFAST)
        {
            threshold_tab_init[i] = (uchar)2;
            threshold_tab_min[i] = (uchar)2;

        } else
        {
            threshold_tab_init[i] = (uchar)0;
            if (v < -minThFAST)
            {
                threshold_tab_min[i] = (uchar)1;
            } else if (v > minThFAST)
            {
                threshold_tab_min[i] = (uchar)2;
            } else
                threshold_tab_min[i] = (uchar)0;
        }
    }

    mvScaleFactor[0] = 1.f;
    mvInvScaleFactor[0] = 1.f;

    for (i = 1; i < nlevels; ++i) {
        mvScaleFactor[i] = scaleFactor * mvScaleFactor[i - 1];
        mvInvScaleFactor[i] = 1 / mvScaleFactor[i];

        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.f / mvLevelSigma2[i];
    }


    float fac = 1.f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.f - fac) / (1.f - (float) pow((double) fac, (double) nlevels));

    int sumFeatures = 0;
    for (i = 0; i < nlevels - 1; ++i)
    {
        nfeaturesPerLevelVec[i] = myRound(nDesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevelVec[i];
        nDesiredFeaturesPerScale *= fac;
    }
    nfeaturesPerLevelVec[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    const int nPoints = 512;
    const auto tempPattern = (const cv::Point*) bit_pattern_31_;
    std::copy(tempPattern, tempPattern+nPoints, std::back_inserter(pattern));
}


void ORBextractor::operator()(cv::InputArray inputImage, cv::InputArray mask,
                              std::vector<cv::KeyPoint> &resultKeypoints, cv::OutputArray outputDescriptors)
{
    /// CHANGE DISTRIBUTION METHOD WITH LAST TWO ARGS!
    this->operator()(inputImage, mask, resultKeypoints, outputDescriptors, Distribution::QUADTREE_ORBSLAMSTYLE, false);
}

/** @overload
 * @param inputImage single channel img-matrix
 * @param mask ignored
 * @param resultKeypoints keypoint vector in which results will be stored
 * @param outputDescriptors matrix in which descriptors will be stored
 * @param distributionMode decides the method to call for kpt-distribution, see Distribution.h
 * @param distributePerLevel true->distribute kpts per octave, false->distribute kpts per image
 */

void ORBextractor::operator()(cv::InputArray inputImage, cv::InputArray mask,
                              std::vector<cv::KeyPoint> &resultKeypoints, cv::OutputArray outputDescriptors,
                              Distribution::DistributionMethod distributionMode, bool distributePerLevel)
{
    std::chrono::high_resolution_clock::time_point funcEntry = std::chrono::high_resolution_clock::now();

    if (inputImage.empty())
        return;

    cv::Mat image = inputImage.getMat();
    assert(image.type() == CV_8UC1);

    ComputeScalePyramid(image);


    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (int i = 0; i < CIRCLE_SIZE; ++i)
        {
            pixelOffset[lvl*CIRCLE_SIZE + i] =
                    CIRCLE_OFFSETS[i][0] + CIRCLE_OFFSETS[i][1] * (int)mvImagePyramid[lvl].step1();
        }
    }


    std::vector<std::vector<cv::KeyPoint>> allkpts;

    //using namespace std::chrono;
    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    DivideAndFAST(allkpts, distributionMode, true, 30, distributePerLevel);


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

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        resultKeypoints.insert(resultKeypoints.end(), allkpts[lvl].begin(), allkpts[lvl].end());
    }

    //TODO: de-/activate fixed duration
    //ensure feature detection always takes 50ms
    unsigned long maxDuration = 50000;
    std::chrono::high_resolution_clock::time_point funcExit = std::chrono::high_resolution_clock::now();
    auto funcDuration = std::chrono::duration_cast<std::chrono::microseconds>(funcExit-funcEntry).count();
    //assert(funcDuration <= maxDuration);
    if (funcDuration < maxDuration)
    {
        auto sleeptime = maxDuration - funcDuration;
        usleep(sleeptime);
    }
}


void ORBextractor::ComputeAngles(std::vector<std::vector<cv::KeyPoint>> &allkpts)
{
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
 * @param allKeypoints KeyPoint vector in which the result will be stored
 * @param mode decides which method to call for keypoint distribution over image, see Distribution.h
 * @param divideImage  true-->divide image into cellSize x cellSize cells, run FAST per cell
 * @param cellSize must be greater than 16 and lesser than min(rows, cols) of smallest image in pyramid
 * @param distributePerLevel distribute keypoints per level or per image after levels are merged
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


            this->FAST(mvImagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                       levelKpts, iniThFAST, lvl);

            if (levelKpts.empty())
                this->FAST(mvImagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                           levelKpts, minThFAST, lvl);


            if(levelKpts.empty())
                continue;

            allkpts[lvl].reserve(nfeaturesPerLevelVec[lvl]);

            if (distributePerLevel)
                Distribution::DistributeKeypoints(levelKpts, minimumX, maximumX, minimumY, maximumY,
                                                  nfeaturesPerLevelVec[lvl], mode);

            allkpts[lvl] = levelKpts;

            for (auto &kpt : allkpts[lvl])
            {
                kpt.pt.y += minimumY;
                kpt.pt.x += minimumX;
                kpt.octave = lvl;
                //kpt.angle = IntensityCentroidAngle(&imagePyramid[lvl].at<uchar>(
                //        myRound(kpt.pt.x), myRound(kpt.pt.y)), imagePyramid[lvl].step1());
            }
        }
    }
    else
    {
        int c = std::min(mvImagePyramid[nlevels-1].rows, mvImagePyramid[nlevels-1].cols);
        assert(cellSize < c && cellSize > 16);
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

                    this->FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                            patchKpts, iniThFAST, lvl);
                    //cv::FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                    //         patchKpts, iniThFAST, true);

                    if (patchKpts.empty())

                        this->FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                               patchKpts, minThFAST, lvl);
                        //cv::FAST(mvImagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                        //         patchKpts, minThFAST, true);



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
                                                  nfeaturesPerLevelVec[lvl], mode);


            allkpts[lvl] = levelKpts;



            for (auto &kpt : allkpts[lvl])
            {
                kpt.pt.y += minimumY;
                kpt.pt.x += minimumX;
                kpt.octave = lvl;
                //kpt.angle = IntensityCentroidAngle(&imagePyramid[lvl].at<uchar>(
                //        myRound(kpt.pt.x), myRound(kpt.pt.y)), imagePyramid[lvl].step1());
            }
        }
    }
    if (!distributePerLevel)
    {
        int nkpts = 0;
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            nkpts += allkpts[lvl].size();
        }
        std::vector<cv::KeyPoint> temp(nkpts);
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            temp.insert(temp.end(), allkpts[lvl].begin(), allkpts[lvl].end());
        }
        Distribution::DistributeKeypoints(temp, 0, mvImagePyramid[0].cols, 0, mvImagePyramid[0].rows, nfeatures, mode);

        for (int lvl = 0; lvl < nlevels; ++lvl)
            allkpts[lvl].clear();

        for (auto &kpt : temp)
        {
            allkpts[kpt.octave].emplace_back(kpt);
        }
    }
}


//move to separate file?
void ORBextractor::FAST(cv::Mat img, std::vector<cv::KeyPoint> &keypoints, int &threshold, int level)
{
    keypoints.clear();

    int offset[CIRCLE_SIZE];
    for (int i = 0; i < CIRCLE_SIZE; ++i)
    {
        offset[i] = pixelOffset[level*CIRCLE_SIZE + i];
    }

    assert(threshold == minThFAST || threshold == iniThFAST); //only initial or min threshold should be passed


    uchar *threshold_tab;
    if (threshold == iniThFAST)
        threshold_tab = threshold_tab_init;
    else
        threshold_tab = threshold_tab_min;


    uchar cornerScores[img.cols*3];
    int cornerPos[img.cols*3];

    memset(cornerScores, 0, img.cols*3);
    memset(cornerPos, 0, img.cols*3);

    uchar* currRowScores = &cornerScores[0];
    uchar* prevRowScores = &cornerScores[img.cols];
    uchar* pprevRowScores = &cornerScores[img.cols*2];

    int* currRowPos = &cornerPos[0];
    int* prevRowPos = &cornerPos[img.cols];
    int* pprevRowPos = &cornerPos[img.cols*2];



    int i, j, k, ncandidates = 0, ncandidatesprev = 0;

    for (i = 3; i < img.rows - 2; ++i)
    {
        const uchar* pointer = img.ptr<uchar>(i) + 3;

        ncandidatesprev = ncandidates;
        ncandidates = 0;

        int* tempPos = pprevRowPos;
        uchar* tempScores = pprevRowScores;

        pprevRowPos = prevRowPos;
        pprevRowScores = prevRowScores;
        prevRowPos = currRowPos;
        prevRowScores = currRowScores;

        currRowPos = tempPos;
        currRowScores = tempScores;

        memset(currRowPos, 0, img.cols);
        memset(currRowScores, 0, img.cols);

        if (i < img.rows - 3) // skip last row
        {
            for (j = 3; j < img.cols-3; ++j, ++pointer)
            {
                int val = pointer[0];                           //value of central pixel
                const uchar *tab = &threshold_tab[255] - val;       //shift threshold tab by val

                int discard = tab[pointer[offset[PIXELS_TO_CHECK[0]]]]
                              | tab[pointer[offset[PIXELS_TO_CHECK[1]]]];

                if (discard == 0)
                    continue;

                bool gotoNextCol = false;
                for (k = 2; k < 16; k+=2)
                {
                    discard &= tab[pointer[offset[PIXELS_TO_CHECK[k]]]]
                               | tab[pointer[offset[PIXELS_TO_CHECK[k+1]]]];
                    if (k == 6 && discard == 0)
                    {
                        gotoNextCol = true;
                        break;
                    }
                    if (k == 14 && discard == 0)
                    {
                        gotoNextCol = true;
                    }

                }
                if (gotoNextCol) // initial FAST-check failed
                    continue;


                if (discard & 1) // check for continuous circle of pixels darker than threshold
                {
                    int compare = val - threshold;
                    int contPixels = 0;

                    for (k = 0; k < onePointFiveCircles; ++k)
                    {
                        int a = pointer[offset[k%CIRCLE_SIZE]];
                        if (a < compare)
                        {
                            ++contPixels;
                            if (contPixels > continuousPixelsRequired)
                            {
                                currRowPos[ncandidates++] = j;

                                currRowScores[j] = OptimizedCornerScore(pointer, offset, threshold);
                                break;
                            }
                        } else
                            contPixels = 0;
                    }
                }

                if (discard & 2) // check for continuous circle of pixels brighter than threshold
                {
                    int compare = val + threshold;
                    int contPixels = 0;

                    for (k = 0; k < onePointFiveCircles; ++k)
                    {
                        int a = pointer[offset[k%CIRCLE_SIZE]];
                        if (a > compare)
                        {
                            ++contPixels;
                            if (contPixels > continuousPixelsRequired)
                            {
                                currRowPos[ncandidates++] = j;

                                currRowScores[j] = OptimizedCornerScore(pointer, offset, threshold);
                                break;
                            }
                        } else
                            contPixels = 0;
                    }
                }
            }
        }


        if (i == 3)   //skip first row
            continue;

        for (k = 0; k < ncandidatesprev; ++k)
        {
            int pos = prevRowPos[k];
            int score = prevRowScores[pos];

            if (score > pprevRowScores[pos-1] && score > pprevRowScores[pos] && score > pprevRowScores[pos+1] &&
                score > prevRowScores[pos+1] && score > prevRowScores[pos-1] &&
                score > currRowScores[pos-1] && score > currRowScores[pos] && score > currRowScores[pos+1])
            {
                keypoints.emplace_back(cv::KeyPoint((float)pos, (float)(i-1),
                                                    7.f, -1, (float)score, level));
            }
        }
    }
}


int ORBextractor::OptimizedCornerScore(const uchar* pointer, const int offset[], int &threshold)
{
    int val = pointer[0];
    int i;
    int diff[onePointFiveCircles];
    for (i = 0; i < CIRCLE_SIZE; ++i)
    {
        diff[i] = (val - pointer[offset[i]]);
    }
    for ( ; i < onePointFiveCircles; ++i)
    {
        diff[i] = diff[i-CIRCLE_SIZE];
    }

    int a0 = threshold;
    for (i = 0; i < CIRCLE_SIZE; i += 2)
    {
        int a;
        if (diff[i+1] < diff[i+2])
            a = diff[i+1];
        else
            a = diff[i+2];

        if (diff[i+3] < a)
            a = diff[i+3];
        if (a0 > a)
            continue;

        if (diff[i+4] < a)
            a = diff[i+4];
        if (diff[i+5] < a)
            a = diff[i+5];
        if (diff[i+6] < a)
            a = diff[i+6];
        if (diff[i+7] < a)
            a = diff[i+7];
        if (diff[i+8] < a)
            a = diff[i+8];

        int c;
        if (a < diff[i])
            c = a;
        else
            c = diff[i];

        if (c > a0)
            a0 = c;
        if (diff[i+9] < a)
            a = diff[i+9];
        if (a > a0)
            a0 = a;
    }

    int b0 = -a0;
    for (i = 0; i < CIRCLE_SIZE; i += 2)
    {
        int b;
        if (diff[i+1] > diff[i+2])
            b = diff[i+1];
        else
            b = diff[i+2];

        if (diff[i+3] > b)
            b = diff[i+3];
        if (diff[i+4] > b)
            b = diff[i+4];
        if (diff[i+5] > b)
            b = diff[i+5];

        if (b0 < b)
            continue;

        if (diff[i+6] > b)
            b = diff[i+6];
        if (diff[i+7] > b)
            b = diff[i+7];
        if (diff[i+8] > b)
            b = diff[i+8];

        int c;
        if (diff[i] > b)
            c = diff[i];
        else
            c = b;

        if (c < b0)
            b0 = c;
        if (diff[i+9] > b)
            b = diff[i+9];
        if (b < b0)
            b0 = b;
    }
    return -b0 - 1;
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


