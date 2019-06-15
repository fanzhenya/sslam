//
// Created by zhenyaf on 6/13/19.
//

#include "sslam/frame.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

using namespace std;

namespace sslam {

std::atomic<uint64_t> Frame::factory_id_{0};

Frame::Ptr Frame::FromAssociateRecord(const std::string &line, const std::string &dataset_dir, Camera::Ptr camera) {
    using namespace std;
    stringstream ss(line);
    double rgb_ts, depth_ts;
    string rgb_fname, depth_fname;
    ss >> rgb_ts >> rgb_fname >> depth_ts >> depth_fname;

    auto ret = Ptr(new Frame(factory_id_++));
    ret->time_stamp_ = rgb_ts;
    ret->color_ = cv::imread(dataset_dir + "/" + rgb_fname);
    ret->depth_ = cv::imread(dataset_dir + "/" + depth_fname, -1);
    ret->camera_ = camera;
    return ret;
}

std::ostream &operator<<(std::ostream &os, const Frame &frame) {
    os << "id_: " << frame.id_ << " time_stamp_: " << uint64_t(frame.time_stamp_)
       << " color_: " << frame.color_.size << " depth_: " << frame.depth_.size;
    return os;
}

double Frame::GetDepth(cv::KeyPoint const &p) {
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
