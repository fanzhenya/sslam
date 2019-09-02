//
// Created by zhenyaf on 6/13/19.
//

#include "sslam/frame.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

using namespace std;

namespace sslam {

std::atomic<uint64_t> Frame::factory_id_{0};

Frame::Ptr Frame::FromAssociateRecord(const sslam::Config &config, const std::string &line, const std::string &dataset_dir, Camera::Ptr camera) {
    using namespace std;
    stringstream ss(line);
    double rgb_ts, depth_ts;
    string rgb_fname, depth_fname;
    ss >> rgb_ts >> rgb_fname >> depth_ts >> depth_fname;

    auto ret = Ptr(new Frame(config, factory_id_++));
    ret->time_stamp_ = rgb_ts;
    ret->color_ = cv::imread(dataset_dir + "/" + rgb_fname);
    ret->depth_ = cv::imread(dataset_dir + "/" + depth_fname, -1);
    ret->camera_ = camera;

    // compute kpts and descriptors
    auto num_of_features    = config.get<int> ( "number_of_features" );
    auto scale_factor       = config.get<double> ( "scale_factor" );
    auto level_pyramid      = config.get<int> ( "level_pyramid" );
    auto orb = cv::ORB::create(num_of_features, scale_factor, level_pyramid);
    orb->detect(ret->color_, ret->kpts_);
    orb->compute(ret->color_, ret->kpts_, ret->descriptors_);

    return ret;
}

std::ostream &operator<<(std::ostream &os, const Frame &frame) {
    os << "id_: " << frame.id_ << " time_stamp_: " << uint64_t(frame.time_stamp_)
       << " color_: " << frame.color_.size << " depth_: " << frame.depth_.size;
    return os;
}

std::vector<cv::DMatch> Frame::MatchWith(cv::Mat const& descriptors) {
    vector<cv::DMatch> matches;
    auto matcher_ = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5,10,2));
    matcher_.match(descriptors, descriptors_, matches);

    // filter
    vector<cv::DMatch> ret;
    auto match_ratio = config_.get<float>("match_ratio");
    auto min_dis = min_element(matches.begin(), matches.end())->distance;
    for (auto& m : matches) {
        if (m.distance < max(min_dis * match_ratio, 30.f)) {
            ret.push_back(m);
        }
    }
    cout << "good matches: " << ret.size() << endl;
    return ret;
}

double Frame::GetDepth(cv::KeyPoint const &p) const {
    vector<int> dx{0, 1, 0, -1, 0}, dy{0, 0, 1, 0, -1};
    for (int k = 0; k < dx.size(); k++) {
        int i = cvRound(p.pt.x) + dx[k], j = cvRound(p.pt.y) + dy[k];
        auto d = depth_.at<ushort>(j, i);
        if (d != 0)
            return double(d) / camera_->depth_scale_;
    }
    return -1.0;
}

}
