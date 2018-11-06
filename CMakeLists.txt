cmake_minimum_required(VERSION 2.8.3)
project(momonga_navigation)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_geometry
  image_transport
  image_view
  tf
  message_generation
  sensor_msgs
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  TrafficLightDetect.srv
)
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
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
add_executable          (image_server                        src/image_server.cpp)
add_executable          (image_client                        src/image_client.cpp)
#add_executable          (trafficlight_detector              src/trafficlight_detector.cpp)

## ================================================================
## dependencies to build msgs and srvs
## ================================================================
add_dependencies(draw_frames ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(image_server ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(image_client ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries   (draw_frames                        ${catkin_LIBRARIES})
target_link_libraries   (image_server                        ${catkin_LIBRARIES})
target_link_libraries   (image_client                        ${catkin_LIBRARIES})
#target_link_libraries   (trafficlight_detector              ${catkin_LIBRARIES})



## C++ 11でコンパイルするためにCMakefile.txtに以下を追加．
add_definitions(-std=c++11)
