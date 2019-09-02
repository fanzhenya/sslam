//
// Created by zhenyaf on 6/13/19.
//

#include <opencv/cv.hpp>
#include "sslam/visual_odometry.hpp"

#include "g2o_types.hpp"
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;

sslam::VisualOdometry::VisualOdometry(Config const& config, std::shared_ptr<Map> map, std::shared_ptr<Ui> ui)
        : state_(kInit), map_(map), ui_(ui) {
    max_num_lost_       = config.get<int> ( "max_num_lost" );
    min_inliers_        = config.get<int> ( "min_inliers" );
    key_frame_min_rot_   = config.get<double> ( "keyframe_rotation" );
    key_frame_min_trans_ = config.get<double> ( "keyframe_translation" );
}

void sslam::VisualOdometry::Process(Frame::Ptr frame) {
    switch (state_) {
        case kInit: {
            ref_frame_ = frame;
            map_->AddObservation(*frame);
            state_ = kTracking;
            break;
        }
        case kTracking: {
            auto matches = map_->MatchWith(*frame);
            cout << "good matches: " << matches.size() << endl;

            if (ui_)
                ui_->DrawMatches(matches, map_->GetAllUVs(), 
                [&](){
                    vector<cv::Point2d> uvs;
                    for (auto const& kp : frame->kpts_) {
                      uvs.push_back(kp.pt);
                    }
                    return uvs;
                }());

            auto estimation = EstimatePosePnp(frame->kpts_, matches);
            if (estimation.score > 0) {
                frame->T_c_w_ = estimation.T_c_r * ref_frame_->T_c_w_;
                ref_frame_ = frame;
                map_->AddObservation(*frame);
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
    cout << "map has " << map_->points_.size() << " points" << endl;
}

sslam::VisualOdometry::PoseEstimationResult
sslam::VisualOdometry::EstimatePosePnp(std::vector<cv::KeyPoint> const &keypoints, std::vector<cv::DMatch> const &matches) {
    vector<cv::Point3d> ref;
    vector<cv::Point2d> cur;
    for (auto& m : matches) {
        ref.push_back(map_->points_[m.queryIdx].xyz);
        cur.push_back(keypoints[m.trainIdx].pt);
    }

    Mat r_vec, t_vec, inliers;
    cv::solvePnPRansac(ref, cur, ref_frame_->camera_->GetK(), Mat{}, r_vec, t_vec, false, 100, 4.0, 0.99, inliers);
    cout << "PnpRansac inlisers: " << inliers.rows << endl;

    SE3 T(
            // rot_x, rot_y, rot_z
            SO3(r_vec.at<double>(0, 0), r_vec.at<double>(1, 0),  r_vec.at<double>(2, 0)),
            Vector3d(t_vec.at<double>(0, 0), t_vec.at<double>(1, 0),  t_vec.at<double>(2, 0))
    );

    cout << "before BA " << T << endl;
    T = OptimizePoseBundleAdjustment(ref, cur, inliers, T);
    cout << "after  BA " << T << endl;

    double score = inliers.rows < min_inliers_ ? 0 : // too few inliners
            T.log().norm() > 5.0 ? 0 : 1; // motion is too large

    bool is_significantly_changed = (cv::norm(r_vec) > key_frame_min_rot_) && (cv::norm(t_vec) > key_frame_min_trans_);

    return {T, score, is_significantly_changed};
}

SE3 sslam::VisualOdometry::OptimizePoseBundleAdjustment(const vector<cv::Point3d> &ref_pts_3d,
                                                        const vector<cv::Point2d> &cur_pts_2d,
                                                        Mat const &inliers, SE3 const &T) const {
    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
            T.rotation_matrix(),
            T.translation()
    ) );
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ ) {
        int index = inliers.at<int>(i,0);
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->camera_ = ref_frame_->camera_.get();
        edge->point_ = Vector3d( ref_pts_3d[index].x, ref_pts_3d[index].y, ref_pts_3d[index].z );
        edge->setMeasurement( Vector2d(cur_pts_2d[index].x, cur_pts_2d[index].y) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        optimizer.addEdge( edge );
    }

    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(10);

    return SE3 (
            pose->estimate().rotation(),
            pose->estimate().translation()
    );
}
