#include <iostream>
#include "sslam/ui.hpp"

#include <pangolin/pangolin.h>

pangolin::View d_cam_;
pangolin::OpenGlRenderState s_cam_;

sslam::Ui::Ui(Config const &config) {
  cv::namedWindow("d2d");
  {
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    d_cam_ = pangolin::CreateDisplay()
                 .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0,
                            -1024.0f / 768.0f)
                 .SetHandler(new pangolin::Handler3D(s_cam_));
  }
}
void sslam::Ui::Draw(std::vector<Sophus::SE3> const &poses,
                     std::vector<Map::Point> const &points) {
  // if (poses.empty() || points.empty()) {
  //   cerr << "parameter is empty!" << endl;
  //   return;
  // }
  float fx = 277.34;
  float fy = 291.402;
  float cx = 312.234;
  float cy = 239.777;

  // while (pangolin::ShouldQuit() == false) {
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam_.Activate(s_cam_);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // draw poses
    float sz = 0.1;
    int width = 640, height = 480;
    for (auto const &p : poses) {
      glPushMatrix();
      Sophus::Matrix4f m = p.matrix().cast<float>();
      glMultMatrixf((GLfloat *)m.data());
      glColor3f(1, 0, 0);
      glLineWidth(2);
      glBegin(GL_LINES);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
      glEnd();
      glPopMatrix();
    }

    // points
    glPointSize(2);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < points.size(); i++) {
      auto color = cv::normalize(cv::Vec3d(points[i].color));
      glColor3f(color[0], color[1], color[2]);
      glVertex3d(points[i].xyz.x, points[i].xyz.y, points[i].xyz.z);
    }
    glEnd();

    pangolin::FinishFrame();
    // std::this_thread::sleep_for(std::chrono::milliseconds(5));  // sleep 5 ms
  }
}