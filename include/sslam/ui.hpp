#pragma once

#include <opencv/cv.hpp>
#include "sslam/frame.hpp"
#include "sslam/map.hpp"
#include "sslam/config.hpp"

namespace sslam {

class Ui {
public:
  Ui(Config const& config);

  void NewCanvas(cv::Mat canvas) {
    canvas_ = canvas;
  }

  void DrawMatches(std::vector<cv::DMatch> const &matches,
                   std::vector<cv::Point2d> const& ref, std::vector<cv::Point2d> const& cur) {
    for (auto const& m : matches) {
      auto last = ref[m.queryIdx];
      auto now = cur[m.trainIdx];
      cv::line(canvas_, last, now, {0, 255, 0});
      cv::circle(canvas_, now, 2, {0, 255, 0});
    }
    cv::imshow("d2d", canvas_);
    // cv::drawMatches(ref_frame_->color_, ref_keypoints_, frame->color_, kpts,
    //             matches, *canvas, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
    //             {}, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  }

  // poses: [Twc]
  void Draw(std::vector<Sophus::SE3> const &poses, std::vector<Map::Point> const &points);

 private:

  cv::Mat canvas_;
};
}