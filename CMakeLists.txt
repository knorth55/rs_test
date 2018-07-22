cmake_minimum_required(VERSION 2.8.3)
project(rs_test)
find_package(catkin REQUIRED robosherlock)
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
################################################################################
## Constants for project                                                      ##
################################################################################
set(NAMESPACE rs_test)
set(TYPESYSTEM_CPP_PATH ${PROJECT_SOURCE_DIR}/include/rs_test/types)
set(TYPESYSTEM_XML_PATH ${PROJECT_SOURCE_DIR}/descriptors/typesystem)
set(ANNOTATOR_PATH      ${PROJECT_SOURCE_DIR}/descriptors/annotators)
set(ENGINE_PATH         ${PROJECT_SOURCE_DIR}/descriptors/analysis_engines)
set(CMAKE_MODULE_PATH   ${PROJECT_SOURCE_DIR}/cmake)
################################################################################
## Configure project configuration files                                      ##
################################################################################
configure_file(${RS_PROJECT_CONFIG} ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}_config.cmake @ONLY)
################################################################################
## Find all include directories                                               ##
################################################################################
find_include_dirs(RS_INCLUDE_DIRS_LIST)
catkin_package(
   CFG_EXTRAS ${PROJECT_NAME}_config.cmake
   )
################################################################################
## Package dependencies                                                       ##
################################################################################
#Add required catkin packages here
#find_package(catkin REQUIRED COMPONENTS xxx)
#Add required system packages here
#find_package(xxx)

catkin_python_setup()

rs_add_library(rs_exampleGSOCAnnotator src/ExampleGSOCAnnotator.cpp)
target_link_libraries(rs_exampleGSOCAnnotator ${CATKIN_LIBRARIES})
rs_add_library(rs_pythonTestAnnotator src/PythonTestAnnotator.cpp)
target_link_libraries(rs_pythonTestAnnotator ${CATKIN_LIBRARIES} boost_numpy)
rs_add_library(rs_fasterRCNNAnnotator src/FasterRCNNAnnotator.cpp)
target_link_libraries(rs_fasterRCNNAnnotator ${CATKIN_LIBRARIES} ${OpenCV_LIBRARIES} boost_numpy)
rs_add_library(rs_ssdAnnotator src/SSDAnnotator.cpp)
target_link_libraries(rs_ssdAnnotator ${CATKIN_LIBRARIES} ${OpenCV_LIBRARIES} boost_numpy)
rs_add_library(rs_fcisAnnotator src/FCISAnnotator.cpp)
target_link_libraries(rs_fcisAnnotator ${CATKIN_LIBRARIES} ${OpenCV_LIBRARIES} boost_numpy)
rs_add_library(rs_maskRCNNAnnotator src/MaskRCNNAnnotator.cpp)
target_link_libraries(rs_maskRCNNAnnotator ${CATKIN_LIBRARIES} ${OpenCV_LIBRARIES} boost_numpy)

# ------------------------------------------------------------------------------------
# Download
# ------------------------------------------------------------------------------------

# download and install trained data
add_custom_target(install_trained_data ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_trained_data.py)

################################################################################
## Build                                                                      ##
################################################################################
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${RS_INCLUDE_DIRS_LIST}
        ${catkin_INCLUDE_DIRS}
        )
################################################################################
## Update analysis engines, typesystem and include all relevant files         ##
################################################################################
##Update xml list of annotators inside analysis engines
update_analysis_engines(robosherlock)
## generate classes from the typesystem xml files
generate_type_system(robosherlock)
#find all relevant files
find_additional_files()
################################################################################
## Sub Projects                                                               ##
################################################################################
#If you want to divide your projects into subprojects include the subdirectories
#each containing a CMakeLists.txt here
#add_subdirectory(src/xxx)
