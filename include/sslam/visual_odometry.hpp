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
    int num_lost_;


    Frame::Ptr ref_frame_;                   // reference frame
    std::vector<cv::Point3f> ref_pts_3d_;    // reference 3D points in ref frame coordinates
    std::vector<cv::KeyPoint> ref_keypoints_;
    Mat ref_descriptors_;                    // descriptors of those ref_pts_3d_. index-by-index correspondence


    struct OrbResult {
        std::vector<cv::KeyPoint> keypoints;
        Mat descriptors;
    };
    cv::Ptr<cv::ORB> orb_;
    std::vector<cv::KeyPoint> ExtractKeyPoints(const Mat &color);
    Mat ComputeDescriptors(std::vector<cv::KeyPoint> &keypoints, const Mat &color);
    void SetRef3dPoints(const std::vector<cv::KeyPoint> &keypoints, Mat const &descriptors);

    std::vector<cv::DMatch> MatchWithReferenceFrame(Mat const &descriptors);

    struct PoseEstimationResult {
        // Tcw = T_c_r * Trw, where Trw is T of ref frame, Tcw is T of current frame.
        // T_c_r is T from ref to current frame`
        SE3 T_c_r; // pose relative to reference frame
        double score;
        bool is_significant_changed;
    };
    PoseEstimationResult
    EstimatePosePnp(std::vector<cv::KeyPoint> const &keypoints, std::vector<cv::DMatch> const &matches);

    // parameters read from config
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;      // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers

    double key_frame_min_rot_;   // minimal rotation of two key-frames
    double key_frame_min_trans_; // minimal translation of two key-frames
};

}

#endif //SSLAM_VISUAL_ODOMETRY_HPP
