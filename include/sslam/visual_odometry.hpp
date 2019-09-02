//
// Created by zhenyaf on 6/13/19.
//

#ifndef SSLAM_VISUAL_ODOMETRY_HPP
#define SSLAM_VISUAL_ODOMETRY_HPP

#include <opencv/cv.hpp>
#include "sslam/frame.hpp"
#include "sslam/map.hpp"
#include "sslam/config.hpp"
#include "sslam/camera.hpp"
#include "ui.hpp"

namespace sslam {

class VisualOdometry {
public:
    VisualOdometry(Config const& config, std::shared_ptr<Map> map, std::shared_ptr<Ui> ui);
    void Process(Frame::Ptr frame);

private:
    enum {kInit, kTracking, kLost} state_;
    int num_lost_;
    std::shared_ptr<Map> map_;
    std::shared_ptr<Ui> ui_;


    Camera::Ptr camera_;

    struct PoseEstimationResult {
        SE3 T;
        double score;
        bool is_significant_changed;
    };
    PoseEstimationResult
    EstimatePosePnp(std::vector<cv::KeyPoint> const &keypoints, std::vector<cv::DMatch> const &matches);
    SE3 OptimizePoseBundleAdjustment(const std::vector<cv::Point3d> &ref_pts_3d,
                                     const std::vector<cv::Point2d> &cur_pts_2d, Mat const &inliers, SE3 const&T) const;

    // parameters read from config
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers

    double key_frame_min_rot_;   // minimal rotation of two key-frames
    double key_frame_min_trans_; // minimal translation of two key-frames
};

}

#endif //SSLAM_VISUAL_ODOMETRY_HPP
