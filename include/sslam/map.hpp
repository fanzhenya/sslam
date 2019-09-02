#pragma once

#include <Eigen/Core>
#include "opencv/cv.hpp"
#include "sslam/config.hpp"
#include "sslam/frame.hpp"

namespace sslam
{

class Map {
public:
  struct Point
  {
    cv::Point3d xyz;
    cv::Mat descriptor;
    cv::Point2d uv;
  };

  Map(Config const& config) : config_(config){};
  std::vector<Point> points_;

  void AddObservation(Frame const& frame) {
    // simple map of points, which only live for one frame
    points_.clear();
    auto keypoints = frame.kpts_;
    auto descriptors = frame.descriptors_;
    for (size_t i = 0; i < keypoints.size(); i++) {
        auto& p = keypoints[i];
        double d = frame.GetDepthAt(p);
        if (d <= 0) continue;
        auto p_cam = frame.camera_->pixel2camera(Vector2d{p.pt.x, p.pt.y}, d);
        points_.push_back({cv::Point3d(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)),
                           descriptors.row(i),
                           p.pt});
    }
  }

  std::vector<cv::DMatch> MatchWith(Frame const& frame) {
    using namespace std;
    vector<cv::DMatch> matches;
    auto matcher =
        cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5, 10, 2));
    matcher.match(GetAllDescriptors(), frame.descriptors_, matches);

    // filter
    vector<cv::DMatch> ret;
    auto match_ratio = config_.get<float>("match_ratio");
    auto min_dis = min_element(matches.begin(), matches.end())->distance;
    for (auto& m : matches) {
      if (m.distance < max(min_dis * match_ratio, 30.f)) {
        ret.push_back(m);
      }
    }
    return ret;
  }

  cv::Mat GetAllDescriptors() {
    cv::Mat ret;
    for (auto const& p : points_) {
      ret.push_back(p.descriptor);
    }
    return ret;
  }

  std::vector<cv::Point2d> GetAllUVs() {
    std::vector<cv::Point2d> ret;
    for (auto const& p : points_)
      ret.push_back(p.uv);
    return ret;
  }

private:
  Config const& config_; 
};
  
} // namespace nam sslam
