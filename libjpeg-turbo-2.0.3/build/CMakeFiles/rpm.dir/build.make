# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3/build

# Utility rule file for rpm.

# Include the progress variables for this target.
include CMakeFiles/rpm.dir/progress.make

CMakeFiles/rpm:
	sh pkgscripts/makerpm

rpm: CMakeFiles/rpm
rpm: CMakeFiles/rpm.dir/build.make

.PHONY : rpm

# Rule to build all files generated by this target.
CMakeFiles/rpm.dir/build: rpm

.PHONY : CMakeFiles/rpm.dir/build

CMakeFiles/rpm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rpm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rpm.dir/clean

CMakeFiles/rpm.dir/depend:
	cd /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3 /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3 /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3/build /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3/build /experiment/data/william/share/semi-seg/libjpeg-turbo-2.0.3/build/CMakeFiles/rpm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rpm.dir/depend

