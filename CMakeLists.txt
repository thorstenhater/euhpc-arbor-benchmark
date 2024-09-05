cmake_minimum_required(VERSION 3.19)

project(busyring)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(busyring ring.cpp)

find_package(arbor 0.10.0 CONFIG REQUIRED)

# Hack around dependency issues...
set(UNITS_ENABLE_TESTS OFF CACHE INTERNAL "")
set(UNITS_BUILD_STATIC_LIBRARY ON CACHE INTERNAL "")
set(UNITS_BUILD_SHARED_LIBRARY OFF CACHE INTERNAL "")
set(UNITS_BUILD_CONVERTER_APP OFF CACHE INTERNAL "")
set(UNITS_BUILD_WEBSERVER OFF CACHE INTERNAL "")
set(UNITS_INSTALL ON CACHE INTERNAL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/ext/units" "${PROJECT_BINARY_DIR}/ext/units")

set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(ext/json)

target_link_libraries(busyring PRIVATE arbor::arbor arbor::arborio arbor::arborenv nlohmann_json::nlohmann_json units::units)
