//
// Created by zhenyaf on 6/14/19.
//

#ifndef SSLAM_CAMERA_HPP
#define SSLAM_CAMERA_HPP

#include <Eigen/Core>
using Eigen::Vector2d;
using Eigen::Vector3d;

#include <memory>

#include <sophus/se3.h>
#include <unordered_map>

using Sophus::SE3;

#include "sslam/config.hpp"

namespace sslam {

class Camera {

public:
    typedef std::shared_ptr<Camera> Ptr;
    float   fx_, fy_, cx_, cy_, depth_scale_;  // Camera intrinsics

    Camera ( float fx, float fy, float cx, float cy, float depth_scale=0 ) :
            fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), depth_scale_ ( depth_scale ) {}

    static Ptr FromConfig(Config const& config) {
        if (!config.IsValid()) return {};
        return Ptr(new Camera(config.get<float>("camera.fx"), config.get<float>("camera.fy"),
                          config.get<float>("camera.cx"), config.get<float>("camera.cy"),
                          config.get<float>("camera.depth_scale")));
    }


    // return camera intrinsics in 3x3 matrix
    cv::Mat GetK() const {
        return (cv::Mat_<double>(3,3) <<
                fx_, 0, cx_,
                0, fy_, cy_,
                0, 0,   1
        );
    }

    // coordinate transform: world, camera, pixel
    // Vector3d world2camera( const Vector3d& p_w, const SE3& T_c_w ) {return T_c_w * p_w};
    // Vector3d camera2world( const Vector3d& p_c, const SE3& T_c_w );


    // normalized camera coordinate => pixel coordinate
    Vector2d camera2pixel( const Vector3d& p_c ) {
        return {
            fx_ * p_c(0, 0) + cx_,
            fy_ * p_c(1, 0) + cy_
        };
    }

    // pixel coordinate => camera coordinate
    Vector3d pixel2camera( const Vector2d& p_p, double depth=1 ) {
        return {
                (p_p(0, 0) - cx_) * depth / fx_,
                (p_p(1, 0) - cy_) * depth / fy_,
                depth
        };
    }

    Vector3d pixel2world ( const Vector2d& p_p, const SE3& T_c_w, double depth=1 ) {
        return T_c_w.inverse() * pixel2camera(p_p, depth);
    }

    Vector2d world2pixel ( const Vector3d& p_w, const SE3& T_c_w ) {
        return camera2pixel(T_c_w * p_w);
    }

private:
    //static std::unordered_map<std::string, Ptr> camera_flyweights_;
};

}

#endif //SSLAM_CAMERA_HPP
