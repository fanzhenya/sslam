//
// Created by zhenyaf on 6/13/19.
//

#include <iostream>
#include <fstream>
#include "sslam/frame.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <sslam/config.hpp>
#include <sslam/visual_odometry.hpp>

using namespace std;
using namespace sslam;

int main(int argc, char** argv) {
    Config config(argc == 2 ? argv[1] : "../config/default.yaml");
    if (!config.IsValid()) {
        cerr << "cannot load config file. Usage: " <<
                std::string(argv[0]) + " path/to/config.yml" << endl;
        return 1;
    }

    auto dataset_dir = config.get<std::string>("dataset_dir");
    ifstream associate_file(dataset_dir + "/associate.txt");
    if (!associate_file) {
        cerr << "cannot find associate file in dataset dir" << endl;
        return 1;
    }

    VisualOdometry vo(config);
    Camera::Ptr camera = Camera::FromConfig(config);

    // visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    // cv::Point3d cam_pos( 3, 3, 3 ), cam_focal_point(0,0,0), cam_y_dir(1,0,0);
    // cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    // vis.setViewerPose( cam_pose );

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );

    for (string each; getline(associate_file, each); ) {
        Frame::Ptr frame = Frame::FromAssociateRecord(each, dataset_dir, camera);
        cout << *frame << endl;

        auto canvas = frame->color_.clone();
        vo.Process(frame, &canvas);

        //auto M = vis.getWidgetPose("Camera");
        //cout << M.matrix << endl;
        //auto M2 = M.rotate(cv::Affine3d::Mat3(0.707, -0.707, 0,
        //                                      0.707, 0.707,  0,
        //                                      0,     0,      1.0 ));
        //auto M2 = M.translate(cv::Affine3d::Vec3(0.01,0.01,0.01));
        //vis.setWidgetPose("Camera", M2);

        cv::imshow("color", frame->color_);
        //cv::imshow("canvas", canvas);
        cv::waitKey(1);
        vis.spinOnce(1, false);
    }
    return 0;

}
