//
// Created by zhenyaf on 6/13/19.
//

#ifndef SSLAM_VISUAL_ODOMETRY_HPP
#define SSLAM_VISUAL_ODOMETRY_HPP


#include <opencv/cv.hpp>
#include "sslam/frame.hpp"
#include "sslam/config.hpp"
#include "sslam/camera.hpp"

namespace sslam {

class VisualOdometry {
public:
    VisualOdometry(Config const& config);
    // process one @frame, optionally draw visualizations on @canvas
    void Process(Frame::Ptr frame, Mat* canvas = nullptr);

private:
    enum {kInit, kTracking, kLost} state_;


    Frame::Ptr ref_;                         // reference frame
    std::vector<cv::Point3f> ref_pts_3d_;    // reference 3D points in ref frame coordinates
    Mat ref_descriptors_;


    struct OrbResult {
        std::vector<cv::KeyPoint> key_points;
        Mat descriptors;
    };
    cv::Ptr<cv::ORB> orb_;
    std::vector<cv::KeyPoint> ExtractKeyPoints(const Mat &color);
    Mat ComputeDescriptors(std::vector<cv::KeyPoint> &keypoints, const Mat &color);

    void SetRef3dPoints(const std::vector<cv::KeyPoint> &keypoints, Mat const &descriptors);


    SE3 T_c_r_estimated_;   // Tcw = Tcr * Trw, where Trw is T of ref frame, Tcw is T of current frame.
                            // Tcr is T from ref to current frame`

    // parameters read from config
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;      // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers

    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
};

}

#endif //SSLAM_VISUAL_ODOMETRY_HPP
