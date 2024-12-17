cmake_minimum_required(VERSION 3.5.0)
project(lomo VERSION 0.1.0 LANGUAGES C CXX)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(lomo main.cpp)
target_link_libraries( lomo ${OpenCV_LIBS} )