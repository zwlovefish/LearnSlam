ADD_EXECUTABLE(main main.cpp)

#增加pcl库的依赖
find_package(PCL REQUIRED COMPONENTS common io visualization filters)

#use this in Ubuntu16.04
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
#增加opencv的依赖
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#增加头文件和库文件
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})

ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp )
TARGET_LINK_LIBRARIES( generate_pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_library(slambase slamBase.cpp)
target_link_libraries(slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(detectFeatures detectFeatures.cpp)
target_link_libraries(detectFeatures slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(joinPointCloud jointPointCloud.cpp)
target_link_libraries(joinPointCloud slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(visualOdometry visualOdometry.cpp)
target_link_libraries(visualOdometry slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})