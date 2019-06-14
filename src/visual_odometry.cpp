//
// Created by zhenyaf on 6/13/19.
//

#include <opencv/cv.hpp>
#include "sslam/visual_odometry.hpp"

using namespace std;

sslam::VisualOdometry::VisualOdometry(const sslam::Config &config)
        : state_(kInit) {
    num_of_features_    = config.get<int> ( "number_of_features" );
    scale_factor_       = config.get<double> ( "scale_factor" );
    level_pyramid_      = config.get<int> ( "level_pyramid" );
    match_ratio_        = config.get<float> ( "match_ratio" );
    max_num_lost_       = config.get<int> ( "max_num_lost" );
    min_inliers_        = config.get<int> ( "min_inliers" );
    key_frame_min_rot   = config.get<double> ( "keyframe_rotation" );
    key_frame_min_trans = config.get<double> ( "keyframe_translation" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

void sslam::VisualOdometry::Process(Frame::Ptr frame, Mat* canvas) {
    switch (state_) {
        case kInit: {
            ref_ = frame;
            // map_->InsertKeyFrame
            auto kpts = ExtractKeyPoints(frame->color_);
            if (canvas) {
                cv::drawKeypoints(*canvas, kpts, *canvas);
            }
            auto descs = ComputeDescriptors(kpts, frame->color_);
            SetRef3dPoints(kpts, descs);
            state_ = kTracking;
            break;
        }
        case kTracking: {
            break;
        }
        default:
            state_ = kInit;
    }
}

std::vector<cv::KeyPoint> sslam::VisualOdometry::ExtractKeyPoints(const Mat &color) {
    std::vector<cv::KeyPoint> ret;
    orb_->detect(color, ret);
    return ret;
}

Mat sslam::VisualOdometry::ComputeDescriptors(std::vector<cv::KeyPoint> &keypoints, const Mat &color) {
    Mat ret;
    orb_->compute(color, keypoints, ret);
    return ret;
}

void sslam::VisualOdometry::SetRef3dPoints(const vector <cv::KeyPoint> &keypoints, Mat const &descriptors) {
    ref_pts_3d_.clear();
    ref_descriptors_ = Mat();
    int cnt = 0;
    for (size_t i = 0; i < keypoints.size(); i++) {
        auto& p = keypoints[i];
        double d = ref_->GetDepth(p);
        if (d > 0) {
            auto p_cam = ref_->camera_->pixel2camera(Vector2d{p.pt.x, p.pt.y}, d);
            ref_pts_3d_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
            ref_descriptors_.push_back(descriptors.row(i));
            cnt++;
        }
    }
    cout << "set " << cnt << " ref points" << endl;
}
