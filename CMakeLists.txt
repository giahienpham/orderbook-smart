cmake_minimum_required(VERSION 3.20)
project(orderbook_smart LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

add_subdirectory(engine)

