cmake_minimum_required(VERSION 2.8.3)
project(momonga_navigation)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_geometry
  image_transport
  image_view
  tf
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  ImageDetection.srv
)

catkin_package(
)



###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
add_executable          (draw_frames                        src/draw_frames.cpp)
#add_executable          (trafficlight_detector              src/trafficlight_detector.cpp)

## ================================================================
## dependencies to build msgs and srvs
## ================================================================
#add_dependencies(waypoint_generator_node ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries   (draw_frames                        ${catkin_LIBRARIES})
#target_link_libraries   (trafficlight_detector              ${catkin_LIBRARIES})



## C++ 11でコンパイルするためにCMakefile.txtに以下を追加．
add_definitions(-std=c++11)
