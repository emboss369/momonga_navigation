cmake_minimum_required(VERSION 2.8.3)
project(momonga_navigation)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_geometry
  image_transport
  image_view
  tf
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
add_executable          (trafficlight_detector              src/waypoint_navigator.cpp)

target_link_libraries   (draw_frames                        ${catkin_LIBRARIES})
target_link_libraries   (trafficlight_detector              ${catkin_LIBRARIES})



## C++ 11でコンパイルするためにCMakefile.txtに以下を追加．
add_definitions(-std=c++11)
