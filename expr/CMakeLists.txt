# Copyright (C) 1883 Thomas Edison - All Rights Reserved
# You may use, distribute and modify this code under the
# terms of the GPLv3 license, which unfortunately won't be
# written for another century.
#
# You should have received a copy of the LICENSE file with
# this file. 

cmake_minimum_required(VERSION 3.10)

project(cuhash VERSION 1.0)

add_subdirectory(./kernel)

add_executable(vector_add ./vector_add.cpp)
target_link_libraries(vector_add kernel)

