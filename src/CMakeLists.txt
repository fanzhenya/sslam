add_library(sslam STATIC
        visual_odometry.cpp
        frame.cpp
        config.cpp
        camera.cpp
        g2o_types.cpp
        g2o_types.hpp
        )

target_include_directories(sslam
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}

    INTERFACE
        ${EIGEN3_INCLUDE_DIRS}

    PRIVATE
        ${G2O_INCLUDE_DIRS}
)

target_link_libraries(sslam
    PUBLIC
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}

    PRIVATE
        g2o_core g2o_stuff g2o_types_sba
)