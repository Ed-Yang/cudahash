# Copyright (C) 1883 Thomas Edison - All Rights Reserved
# You may use, distribute and modify this code under the
# terms of the GPLv3 license, which unfortunately won't be
# written for another century.
#
# You should have received a copy of the LICENSE file with
# this file. 

cmake_minimum_required(VERSION 3.10)

project(cudahash VERSION 1.0)

find_package(CUDA REQUIRED)

set(ETHASH_PATH ./ethash)
include_directories(${ETHASH_PATH})
include_directories(${ETHASH_PATH}/include)

include_directories(./)
include_directories(./libcuda)

add_subdirectory(./getopt)
add_subdirectory(./libcuda)

link_directories(${ETHASH_PATH}/build/getopt)
link_directories(${ETHASH_PATH}/build/lib/ethash)
link_directories(${ETHASH_PATH}/build/lib/global_context)
link_directories(${ETHASH_PATH}/build/lib/keccak)

set(SOURCES ./test/main.cpp  ./test/eth_tester.cpp ./test/nv_dev.cpp)

add_executable(cudahash ${SOURCES})
target_include_directories(cudahash PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(cudahash libethash.a libethash-global-context.a libkeccak.a cuda)
