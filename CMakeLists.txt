cmake_minimum_required(VERSION 3.16)

project(FAST_LIO)

find_package(nav_msgs QUIET)
find_package(sensor_msgs QUIET)
find_package(pcl_conversions QUIET)
find_package(livox_ros_driver2 QUIET)
find_package(PCL QUIET)
find_package(OpenMP REQUIRED)
find_package(Sophus REQUIRED) 

add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
add_link_options(-fsanitize=address)

add_compile_options(
    -Wno-reorder
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-sign-compare
    -Wno-parentheses
    -Wno-unused-but-set-variable
)

add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)

add_library(${PROJECT_NAME} SHARED
  src/nodes.cpp
  src/preprocess.cpp
  include/ikd-Tree/ikd_Tree.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE fins_sdk)
target_include_directories(${PROJECT_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ikd-Tree>
    ${PCL_INCLUDE_DIRS}
)

if(TARGET livox_ros_driver2::livox_ros_driver2)
    target_link_libraries(${PROJECT_NAME} PRIVATE livox_ros_driver2::livox_ros_driver2)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${livox_ros_driver2_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${livox_ros_driver2_LIBRARIES})
endif()

if(TARGET sensor_msgs::sensor_msgs)
    target_link_libraries(${PROJECT_NAME} PRIVATE sensor_msgs::sensor_msgs)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${sensor_msgs_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${sensor_msgs_LIBRARIES})
endif()

if(TARGET nav_msgs::nav_msgs)
    target_link_libraries(${PROJECT_NAME} PRIVATE nav_msgs::nav_msgs)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${nav_msgs_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${nav_msgs_LIBRARIES})
endif()

if(TARGET pcl_conversions::pcl_conversions)
    target_link_libraries(${PROJECT_NAME} PRIVATE pcl_conversions::pcl_conversions)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${pcl_conversions_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${pcl_conversions_LIBRARIES})
endif()


target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenMP::OpenMP_CXX
    Sophus::Sophus
    ${PCL_LIBRARIES} 
    pcl_surface
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    include(ProcessorCount)
    ProcessorCount(N)
    message("Processor number: ${N}")
    
    if(N GREATER 16)
        target_compile_definitions(${PROJECT_NAME} PRIVATE MP_EN MP_PROC_NUM=15)
        message("core for MP: 15")
    elseif(N GREATER 8)
        target_compile_definitions(${PROJECT_NAME} PRIVATE MP_EN MP_PROC_NUM=7)
        message("core for MP: 7")
    elseif(N GREATER 4)
        target_compile_definitions(${PROJECT_NAME} PRIVATE MP_EN MP_PROC_NUM=3)
        message("core for MP: 3")
    elseif(N GREATER 3)
        target_compile_definitions(${PROJECT_NAME} PRIVATE MP_EN MP_PROC_NUM=2)
        message("core for MP: 2")
    else()
        target_compile_definitions(${PROJECT_NAME} PRIVATE MP_PROC_NUM=1)
    endif()
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE MP_PROC_NUM=1)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(NOT MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE)
endif()