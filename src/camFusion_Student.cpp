
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, 
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Loop over matches
    int count = 0;
    double mean_dist = 0.0;
    std::vector<double> kpt_dist;
    for (auto match : kptMatches)
    {
        // Check box in curr. image
        cv::KeyPoint curr_kpt = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(curr_kpt.pt))
        {
            cv::KeyPoint prev_kpt = kptsPrev[match.queryIdx];
            double distance = cv::norm(curr_kpt.pt - prev_kpt.pt);
            kpt_dist.push_back(distance);

            mean_dist += distance;
            count++;
        }
    }
    mean_dist /= count;

    // Remove outliers
    count = 0;
    for (auto match : kptMatches)
    {
        cv::KeyPoint curr_kpt = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(curr_kpt.pt))
        {
            double distance = kpt_dist[count];
            count++;

            if (std::abs(distance - mean_dist) < 10.)
                boundingBox.kptMatches.push_back(match);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    assert(!distRatios.empty());
    double median_dist_ratio;
    if (distRatios.size()%2 == 0) // even
    {
        const auto median_itr1 = distRatios.begin() + distRatios.size()/2 - 1;
        const auto median_itr2 = distRatios.begin() + distRatios.size()/2 + 1;

        nth_element(distRatios.begin(), median_itr1, distRatios.end());
        double median1 = *median_itr1;

        nth_element(distRatios.begin(), median_itr2, distRatios.end());
        double median2 = *median_itr2;

        median_dist_ratio = (median1 + median2) / 2;
    }
    else // odo
    {
        const auto median_itr = distRatios.begin() + distRatios.size()/2;
        nth_element(distRatios.begin(), median_itr, distRatios.end());

        median_dist_ratio = *median_itr;
    }
    double dT = 1 / frameRate;
    TTC = -dT / (1 - median_dist_ratio);
}

// TODO: Compute the time-to-collision using only Lidar measurements
// A comparison operator to sort the vecotr by increasing order of x
bool comparison(LidarPoint a, LidarPoint b)
{
    return (a.x < b.x);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // Calculate the median value in x coordinate & remove the outliers which are too far away form the median
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), comparison);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), comparison);

    // Calculate the median values of both lidar points
    double prev_med, curr_med;
    size_t pn = lidarPointsPrev.size();
    if (pn != 0)
    {
        if (pn % 2 == 0)
            prev_med = (lidarPointsPrev[pn/2 -1].x + lidarPointsPrev[pn/2].x) / 2;
        else
            prev_med = lidarPointsPrev[pn/2].x;
    }

    size_t cn = lidarPointsCurr.size();
    if (cn != 0)
    {
        if (cn % 2 == 0)
            curr_med = (lidarPointsCurr[cn/2 -1].x + lidarPointsCurr[cn/2].x) / 2;
        else
            curr_med = lidarPointsCurr[cn/2].x;
    }
    // Ignore the outliers & Finding closest distance to Lidar points
    double thresh = 0.3; // determing whether the point is a outlier or not
    double min_prev = 1e9, min_curr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
    {
        if ((prev_med - it->x) > thresh)
            continue;
        
        min_prev = min_prev > it->x ? it->x : min_prev;
    }
    
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
    {
        if ((curr_med - it->x) > thresh)
            continue;

        min_curr = min_curr > it->x ? it->x : min_curr;
    }

    // Compute the TTC from both measurements
    TTC = min_curr / (min_prev - min_curr) / frameRate;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    vector<Box_Pair> box_pairs;
    for (auto match : matches)
    {
        // Check box in prev. image
        cv::KeyPoint prev_kpt = prevFrame.keypoints[match.queryIdx];
        vector<vector<BoundingBox>::iterator> prev_boxes;
        for (auto pb_itr = prevFrame.boundingBoxes.begin(); pb_itr != prevFrame.boundingBoxes.end(); pb_itr++)
        {
            if (pb_itr->roi.contains(prev_kpt.pt))
                prev_boxes.push_back(pb_itr);
        }

        // Check box in curr. image
        cv::KeyPoint curr_kpt = currFrame.keypoints[match.trainIdx];
        vector<vector<BoundingBox>::iterator> curr_boxes;
        for (auto cb_itr = currFrame.boundingBoxes.begin(); cb_itr != currFrame.boundingBoxes.end(); cb_itr++)
        {
            if (cb_itr->roi.contains(curr_kpt.pt))
                curr_boxes.push_back(cb_itr);
        }

        // Store the matched boxes & counting
        if (prev_boxes.size() == 1 && curr_boxes.size() == 1)
        {
            bool matched = false;
            for (auto it : box_pairs)
            {
                if (prev_boxes[0]==it.prev_box && curr_boxes[0]==it.curr_box)
                {
                    it.match_count++;
                    matched = true;
                }
            }
            if (!matched)
            {
                Box_Pair box_pair;
                box_pair.prev_box = prev_boxes[0];
                box_pair.curr_box = curr_boxes[0];
                box_pair.match_count++;
                box_pairs.push_back(box_pair);                
            }
        }
    }

    // Finding the match with the highest number of occurences
    for (auto pb_itr = prevFrame.boundingBoxes.begin(); pb_itr != prevFrame.boundingBoxes.end(); pb_itr++)
    {
        bool matched = false;
        vector<BoundingBox>::iterator enclos_box;
        int max_count = 0;
        for (auto bit : box_pairs)
        {
            if (bit.prev_box == pb_itr && bit.match_count > max_count)
            {
                enclos_box = bit.curr_box;
                max_count = bit.match_count;
                matched = true;
            }
        }

        // Store result in output
        if (matched)
            bbBestMatches.insert(make_pair(pb_itr->boxID, enclos_box->boxID));
    }
}
