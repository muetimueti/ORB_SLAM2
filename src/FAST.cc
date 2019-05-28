#include "include/FAST.h"


FASTdetector::FASTdetector(int _iniThreshold, int _minThreshold, int _nlevels) :
        iniThreshold(0), minThreshold(0), nlevels(_nlevels), scoreType(OPENCV), pixelOffset{},
        threshold_tab_init{}, threshold_tab_min{}
{
    pixelOffset.resize(nlevels * CIRCLE_SIZE);
    SetFASTThresholds(_iniThreshold, _minThreshold);

    continuousPixelsRequired = CIRCLE_SIZE / 2;
    onePointFiveCircles = CIRCLE_SIZE + continuousPixelsRequired + 1;
}

void FASTdetector::SetFASTThresholds(int ini, int min)
{
    if ((ini == iniThreshold && min == minThreshold))
        return;

    iniThreshold = std::min(255, std::max(1, ini));
    minThreshold = std::min(iniThreshold, std::max(1, min));

    //initialize threshold tabs for init and min threshold
    int i;
    for (i = 0; i < 512; ++i)
    {
        int v = i - 255;
        if (v < -iniThreshold)
        {
            threshold_tab_init[i] = (uchar)1;
            threshold_tab_min[i] = (uchar)1;
        } else if (v > iniThreshold)
        {
            threshold_tab_init[i] = (uchar)2;
            threshold_tab_min[i] = (uchar)2;

        } else
        {
            threshold_tab_init[i] = (uchar)0;
            if (v < -minThreshold)
            {
                threshold_tab_min[i] = (uchar)1;
            } else if (v > minThreshold)
            {
                threshold_tab_min[i] = (uchar)2;
            } else
                threshold_tab_min[i] = (uchar)0;
        }
    }
}


void FASTdetector::SetStepVector(std::vector<int> &_steps)
{
    steps = _steps;
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (int i = 0; i < CIRCLE_SIZE; ++i)
        {
            pixelOffset[lvl*CIRCLE_SIZE + i] = CIRCLE_OFFSETS[i][0] + CIRCLE_OFFSETS[i][1] * steps[lvl];
        }
    }
}

void FASTdetector::FAST(cv::Mat img, std::vector<cv::KeyPoint> &keypoints, int threshold, int lvl)
{
    switch (scoreType)
    {
        case (OPENCV):
        {
            this->FAST<uchar>(img, keypoints, threshold, lvl);
            break;
        }
        case (SUM):
        {
            this->FAST<int>(img, keypoints, threshold, lvl);
            break;
        }
        case (HARRIS):
        {
            this->FAST<float>(img, keypoints, threshold, lvl);
            break;
        }
        case (EXPERIMENTAL):
        {
            this->FAST<float>(img, keypoints, threshold, lvl);
            break;
        }
        default:
        {
            this->FAST<uchar>(img, keypoints, threshold, lvl);
            break;
        }
    }
}


template <typename scoretype>
void FASTdetector::FAST(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, int threshold, int lvl)
{
    keypoints.clear();

    assert(!steps.empty());

    int offset[CIRCLE_SIZE];
    for (int i = 0; i < CIRCLE_SIZE; ++i)
    {
        offset[i] = pixelOffset[lvl*CIRCLE_SIZE + i];
    }

    assert(threshold == minThreshold || threshold == iniThreshold); //only initial or min threshold should be passed

    uchar *threshold_tab;
    if (threshold == iniThreshold)
        threshold_tab = threshold_tab_init;
    else
        threshold_tab = threshold_tab_min;


    scoretype cornerScores[img.cols*3];
    int cornerPos[img.cols*3];

    memset(cornerScores, 0, img.cols*3);
    memset(cornerPos, 0, img.cols*3);

    scoretype* currRowScores = &cornerScores[0];
    scoretype* prevRowScores = &cornerScores[img.cols];
    scoretype* pprevRowScores = &cornerScores[img.cols*2];

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
        scoretype* tempScores = pprevRowScores;

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

                                if (scoreType == OPENCV)
                                    currRowScores[j] = CornerScore(pointer, offset, threshold);
                                else if (scoreType == HARRIS)
                                    currRowScores[j] = CornerScore_Harris(pointer, steps[lvl]);
                                else if (scoreType == SUM)
                                    currRowScores[j] = CornerScore_Sum(pointer, offset);
                                else if (scoreType == EXPERIMENTAL)
                                    currRowScores[j] = CornerScore_Experimental(pointer, steps[lvl]);
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

                                if (scoreType == OPENCV)
                                    currRowScores[j] = CornerScore(pointer, offset, threshold);
                                else if (scoreType == HARRIS)
                                    currRowScores[j] = CornerScore_Harris(pointer, steps[lvl]);
                                else if (scoreType == SUM)
                                    currRowScores[j] = CornerScore_Sum(pointer, offset);
                                else if (scoreType == EXPERIMENTAL)
                                    currRowScores[j] = CornerScore_Experimental(pointer, steps[lvl]);
                                break;
                            }
                        } else
                            contPixels = 0;
                    }
                }
            }
        }


        if (i == 3)
            continue;

        for (k = 0; k < ncandidatesprev; ++k)
        {
            int pos = prevRowPos[k];
            float score = prevRowScores[pos];


            if (score > pprevRowScores[pos-1] && score > pprevRowScores[pos] && score > pprevRowScores[pos+1] &&
                score > prevRowScores[pos+1] && score > prevRowScores[pos-1] &&
                score > currRowScores[pos-1] && score > currRowScores[pos] && score > currRowScores[pos+1])
            {
                keypoints.emplace_back(cv::KeyPoint((float)pos, (float)(i-1), 7.f, -1, (float)score, lvl));
            }
        }
    }
}


float FASTdetector::CornerScore_Harris(const uchar* pointer, int step)
{
    float k = 0.04f;
    int sz = 2;
    int sz2 = sz*sz;
    int offset[sz2];
    for (int i = 0; i < sz; ++i)
        for (int j = 0; j < sz; ++j)
        {
            offset[i*sz + j] = i*step + j;
        }

    int Ixx = 0, Iyy = 0, Ixy = 0;
    for (int i = 0; i < sz2; ++i)
    {
        const uchar *ptr = pointer + offset[i];
        int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
        int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
        Ixx += Ix*Ix;
        Iyy += Iy*Iy;
        Ixy += Ix*Iy;
    }
    return Ixx*Iyy - Ixy*Ixy - k*(Ixx+Iyy)*(Ixx+Iyy);
}


float FASTdetector::CornerScore_Experimental(const uchar* ptr, int step)
{
    //float a = ptr[-1] - ptr[1] + ptr[-2] - ptr[2];
    //float b = ptr[-step] - ptr[step] + ptr[-2*step] - ptr[2*step];
    //float c = ptr[-step-1] - ptr[step+1] + ptr[-step+1] - ptr[step-1];

    //return a * b;
    return (ptr[-step] - ptr[step] + ptr[-1] - ptr[1]);
}


float FASTdetector::CornerScore_Sum(const uchar* ptr, const int offset[])
{
    int v = ptr[0];
    int diff = 0;
    for (int i = 0; i < CIRCLE_SIZE; ++i)
    {
        diff += v - ptr[offset[i]];
    }
    return (float)diff;
}



float FASTdetector::CornerScore(const uchar* pointer, const int offset[], int threshold)
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