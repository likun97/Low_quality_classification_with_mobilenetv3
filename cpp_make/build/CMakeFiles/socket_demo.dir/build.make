# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /search/odin/songminghui/Softwares/bin/cmake

# The command to remove a file.
RM = /search/odin/songminghui/Softwares/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /search/odin/songminghui/Documents/TensorRTcpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /search/odin/songminghui/Documents/TensorRTcpp/build

# Include any dependencies generated for this target.
include CMakeFiles/socket_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/socket_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/socket_demo.dir/flags.make

CMakeFiles/socket_demo.dir/scripts/post.cpp.o: CMakeFiles/socket_demo.dir/flags.make
CMakeFiles/socket_demo.dir/scripts/post.cpp.o: ../scripts/post.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/search/odin/songminghui/Documents/TensorRTcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/socket_demo.dir/scripts/post.cpp.o"
	/usr/bin/g++  -std=gnu++0x  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/socket_demo.dir/scripts/post.cpp.o -c /search/odin/songminghui/Documents/TensorRTcpp/scripts/post.cpp

CMakeFiles/socket_demo.dir/scripts/post.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/socket_demo.dir/scripts/post.cpp.i"
	/usr/bin/g++  -std=gnu++0x $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /search/odin/songminghui/Documents/TensorRTcpp/scripts/post.cpp > CMakeFiles/socket_demo.dir/scripts/post.cpp.i

CMakeFiles/socket_demo.dir/scripts/post.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/socket_demo.dir/scripts/post.cpp.s"
	/usr/bin/g++  -std=gnu++0x $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /search/odin/songminghui/Documents/TensorRTcpp/scripts/post.cpp -o CMakeFiles/socket_demo.dir/scripts/post.cpp.s

# Object files for target socket_demo
socket_demo_OBJECTS = \
"CMakeFiles/socket_demo.dir/scripts/post.cpp.o"

# External object files for target socket_demo
socket_demo_EXTERNAL_OBJECTS =

socket_demo: CMakeFiles/socket_demo.dir/scripts/post.cpp.o
socket_demo: CMakeFiles/socket_demo.dir/build.make
socket_demo: /usr/lib64/libopencv_calib3d.so
socket_demo: /usr/lib64/libopencv_contrib.so
socket_demo: /usr/lib64/libopencv_core.so
socket_demo: /usr/lib64/libopencv_features2d.so
socket_demo: /usr/lib64/libopencv_flann.so
socket_demo: /usr/lib64/libopencv_highgui.so
socket_demo: /usr/lib64/libopencv_imgproc.so
socket_demo: /usr/lib64/libopencv_legacy.so
socket_demo: /usr/lib64/libopencv_ml.so
socket_demo: /usr/lib64/libopencv_objdetect.so
socket_demo: /usr/lib64/libopencv_photo.so
socket_demo: /usr/lib64/libopencv_stitching.so
socket_demo: /usr/lib64/libopencv_superres.so
socket_demo: /usr/lib64/libopencv_ts.so
socket_demo: /usr/lib64/libopencv_video.so
socket_demo: /usr/lib64/libopencv_videostab.so
socket_demo: CMakeFiles/socket_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/search/odin/songminghui/Documents/TensorRTcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable socket_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/socket_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/socket_demo.dir/build: socket_demo

.PHONY : CMakeFiles/socket_demo.dir/build

CMakeFiles/socket_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/socket_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/socket_demo.dir/clean

CMakeFiles/socket_demo.dir/depend:
	cd /search/odin/songminghui/Documents/TensorRTcpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /search/odin/songminghui/Documents/TensorRTcpp /search/odin/songminghui/Documents/TensorRTcpp /search/odin/songminghui/Documents/TensorRTcpp/build /search/odin/songminghui/Documents/TensorRTcpp/build /search/odin/songminghui/Documents/TensorRTcpp/build/CMakeFiles/socket_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/socket_demo.dir/depend

