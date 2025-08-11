
project(orderbook_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.0.2
)
FetchContent_MakeAvailable(fmt)

add_library(orderbook_engine STATIC
  src/top_of_book.cpp
)
target_include_directories(orderbook_engine PUBLIC include)
target_link_libraries(orderbook_engine PUBLIC fmt::fmt)

include(CTest)
enable_testing()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(engine_smoke_test tests/engine_smoke_test.cpp)
target_link_libraries(engine_smoke_test PRIVATE orderbook_engine GTest::gtest_main)
add_test(NAME engine_smoke_test COMMAND engine_smoke_test)

