cmake_minimum_required(VERSION 2.8.3)
project(stereo_depth)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_filters
  sensor_msgs
  cv_bridge
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/disparity.cfg
)

catkin_package()

find_package(OpenCV REQUIRED)

include_directories(include ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})


# make sure configure headers are built before any node using them

add_executable(stereo_depth src/stereo_depth.cpp)
target_link_libraries(stereo_depth ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_dependencies(stereo_depth ${PROJECT_NAME}_gencfg)