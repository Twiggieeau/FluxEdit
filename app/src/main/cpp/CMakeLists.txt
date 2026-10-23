cmake_minimum_required(VERSION 3.22.1)

project("flux_engine")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DANDROID_STL=c++_shared)

add_library(flux_engine SHARED
    jni_bridge.cpp
    flux_engine.cpp
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(flux_engine
    ${log-lib}
    ${android-lib}
    m
)
