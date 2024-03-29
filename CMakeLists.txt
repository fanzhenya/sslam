cmake_minimum_required(VERSION 3.2)
project (sslam)


set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
# Eigen
find_package(Eigen3 REQUIRED)
include_directories( "/usr/include/eigen3" )
message("Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")

# OpenCV
find_package( OpenCV 3.1 REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

# Sophus 
find_package( Sophus REQUIRED )
#include_directories( ${Sophus_INCLUDE_DIRS} )

# G2O
find_package( G2O REQUIRED )
#include_directories( ${G2O_INCLUDE_DIRS} )

# Pangolinf
find_package( Pangolin )
# include_directories( ${Pangolin_INCLUDE_DIRS})

message(${Pangolin_INCLUDE_DIRS})
message(${Pangolin_LIBRARIES})

# set( THIRD_PARTY_LIBS 
#     ${OpenCV_LIBS}
#     ${Sophus_LIBRARIES}
#     g2o_core g2o_stuff g2o_types_sba
#     ${Pangolin_LIBRARIES}
# )

############### source and test ######################
#include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
add_subdirectory( test )