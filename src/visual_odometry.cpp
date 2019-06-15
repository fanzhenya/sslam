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
    key_frame_min_rot_   = config.get<double> ( "keyframe_rotation" );
    key_frame_min_trans_ = config.get<double> ( "keyframe_translation" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

void sslam::VisualOdometry::Process(Frame::Ptr frame, Mat* canvas) {
    switch (state_) {
        case kInit: {
            ref_frame_ = frame;
            // map_->InsertKeyFrame
            auto kpts = ExtractKeyPoints(frame->color_);
            auto descs = ComputeDescriptors(kpts, frame->color_);
            SetRef3dPoints(kpts, descs);

            if (canvas) {
                cv::drawKeypoints(*canvas, kpts, *canvas);
            }
            state_ = kTracking;
            break;
        }
        case kTracking: {
            auto kpts = ExtractKeyPoints(frame->color_);
            auto descs = ComputeDescriptors(kpts, frame->color_);
            auto matches = MatchWithReferenceFrame(descs);
            if (canvas) {
                cv::drawMatches(ref_frame_->color_, ref_keypoints_, frame->color_, kpts, matches, *canvas);
            }

            auto estimation = EstimatePosePnp(kpts, matches);
            if (estimation.score > 0) {
                frame->T_c_w_ = estimation.T_c_r * ref_frame_->T_c_w_;
                ref_frame_ = frame;
                SetRef3dPoints(kpts, descs);
                num_lost_ = 0;
            } else {
                num_lost_++;
                if (num_lost_ > max_num_lost_) {
                    cerr << "============ lost. Re-init =============" << endl;
                    state_ = kInit;
                }
            }

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
    ref_keypoints_.clear();
    ref_descriptors_ = Mat();

    int cnt = 0;
    for (size_t i = 0; i < keypoints.size(); i++) {
        auto& p = keypoints[i];
        double d = ref_frame_->GetDepth(p);
        if (d > 0) {
            auto p_cam = ref_frame_->camera_->pixel2camera(Vector2d{p.pt.x, p.pt.y}, d);
            ref_pts_3d_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
            ref_keypoints_.push_back(p);
            ref_descriptors_.push_back(descriptors.row(i));
            cnt++;
        }
    }
    cout << "set " << cnt << " ref points" << endl;
}

std::vector<cv::DMatch> sslam::VisualOdometry::MatchWithReferenceFrame(Mat const &descriptors) {
    vector<cv::DMatch> matches;
    //auto matcher = cv::FlannBasedMatcher();
    auto matcher = cv::BFMatcher(cv::NORM_HAMMING);
    matcher.match(ref_descriptors_, descriptors, matches);

    // filter
    vector<cv::DMatch> ret;
    auto min_dis = min_element(matches.begin(), matches.end())->distance;
    for (auto& m : matches) {
        if (m.distance < max(min_dis * match_ratio_, 30.f)) {
            ret.push_back(m);
        }
    }
    cout << "good matches: " << ret.size() << endl;
    return ret;
}

sslam::VisualOdometry::PoseEstimationResult
sslam::VisualOdometry::EstimatePosePnp(std::vector<cv::KeyPoint> const &keypoints, std::vector<cv::DMatch> const &matches) {
    vector<cv::Point3f> ref;
    vector<cv::Point2d> cur;
    for (auto& m : matches) {
        ref.push_back(ref_pts_3d_[m.queryIdx]);
        cur.push_back(keypoints[m.trainIdx].pt);
    }

    Mat r_vec, t_vec, inliers;
    cv::solvePnPRansac(ref, cur, ref_frame_->camera_->GetK(), Mat{}, r_vec, t_vec, false, 100, 4.0, 0.99, inliers);

    SE3 T(
            // rot_x, rot_y, rot_z
            SO3(r_vec.at<double>(0, 0), r_vec.at<double>(1, 0),  r_vec.at<double>(2, 0)),
            Vector3d(t_vec.at<double>(0, 0), t_vec.at<double>(1, 0),  t_vec.at<double>(2, 0))
    );

    cout << "PnpRansac inlisers: " << inliers.rows << endl;
    double score = inliers.rows < min_inliers_ ? 0 : // too few inliners
            T.log().norm() > 5.0 ? 0 : 1; // motion is too large

    bool is_significantly_changed = (cv::norm(r_vec) > key_frame_min_rot_) && (cv::norm(t_vec) > key_frame_min_trans_);

    return {T, score, is_significantly_changed};
}
