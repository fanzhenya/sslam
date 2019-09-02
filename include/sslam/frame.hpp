//
// Created by zhenyaf on 6/13/19.
//

#ifndef SSLAM_FRAME_HPP
#define SSLAM_FRAME_HPP

#include <memory>
#include <atomic>

#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

using cv::Mat;

#include <sophus/se3.h>
#include <sophus/so3.h>
using Sophus::SE3;
using Sophus::SO3;

#include <ostream>
#include "sslam/camera.hpp"

namespace sslam {

class Frame {
public:
    typedef std::shared_ptr<Frame> Ptr;

    Camera::Ptr camera_; // camera model that generated this frame

    uint64_t id_;
    double time_stamp_;
    SE3 T_c_w_;
    Mat color_;
    Mat depth_;

    Config const& config_;
    std::vector<cv::KeyPoint> kpts_;
    cv::Mat descriptors_;

    // rgb_timestamp, rgb_file_path, depth_timestamp, depth_file_path
    static std::shared_ptr<Frame>
    FromAssociateRecord(const Config &config, const std::string &line, const std::string &dataset_dir = "/", Camera::Ptr camera = {});

    double GetDepthAt(cv::KeyPoint const& p) const;

    friend std::ostream &operator<<(std::ostream &os, const Frame &frame);

private:
    static std::atomic<uint64_t> factory_id_;
    Frame(Config const& config, uint64_t id, double time_stamp=0, SE3 T_c_w=SE3(), /*Camera::Ptr camera=nullptr,*/ Mat color=Mat(), Mat depth=Mat() )
            : config_(config), id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), /*camera_(camera),*/ color_(color), depth_(depth) {
    }

};

}

#endif //SSLAM_FRAME_HPP
