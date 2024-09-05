cmake_minimum_required(VERSION 3.19)

project(busyring)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(busyring ring.cpp)

find_package(arbor 0.10.0 CONFIG REQUIRED)

CPMAddPackage(NAME json
              GITHUB_REPOSITORY nlohmann/json
              VERSION 3.11.2
              OPTIONS "CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON")

target_link_libraries(busyring PRIVATE arbor::arbor arbor::arborio arbor::arborenv nlohmann_json::nlohmann_json)
