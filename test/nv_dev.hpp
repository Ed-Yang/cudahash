#pragma once

#include <string>
#include <iostream>
#include <mutex>
#include <chrono>
#include <vector>

#include <ethash/ethash.hpp>

#include "ethash_cuda_miner_kernel.h"
#include "epoch_ctx.hpp"

using namespace std;

#define MAX_STREAMS 4

typedef struct test_result {
    int devId;
    // settings
    int epoch;
    int streams;
    int block_multiple;
    // return results
    int streamIdx;
    uint64_t nonce;
    ethash::hash256 mix_hash;
    ethash::hash256 final_hash;
    float duration;
} test_result_t;

typedef struct {
    int cuDeviceIndex;
    string uniqueId;
    string boardName;
    size_t totalMemory;
    size_t cuComputeMajor;
    size_t cuComputeMinor;
    string cuCompute;
    size_t maxThreadsPerBlock;
    size_t computeMode;
    size_t cuBlockSize;
    size_t cuStreamSize;

    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    size_t regsPerBlock;
    size_t memoryBusWidth;

    size_t multiProcessorCount;
    size_t maxThreadsPerMultiProcessor;
    size_t maxBlocksPerMultiProcessor;
} NvDevInfo;

class NvDev 
{
public:
    NvDev() {}
    NvDev(int devId) ;
    ~NvDev() {}
    int getNumDevices();
    int enumDevices(NvDevInfo nvInfo[]);
    bool getDevicInfo(int devId, NvDevInfo& nvInfo);
    int getDeviceId() {
        return m_devId;
    }
    bool set_epoch_ctx(struct EpochContexts ctx);
    bool gen_dag();
    void set_search_params(int streams, int block_multiple) {
        cuStreamSize = streams;
        m_block_multiple = block_multiple;
    }
    std::vector<test_result_t> search(void *header, uint64_t target, uint64_t start_nonce);
    void stop() {
        m_stop = true;
    }
    bool shouldStop() {
        return m_stop;
    }
protected:
    int m_devId = -1;
    struct EpochContexts m_ctx = {};
    Search_results* m_search_buf[MAX_STREAMS];
    cudaStream_t m_streams[MAX_STREAMS];
    unsigned cuStreamSize = 2;
    unsigned m_block_multiple = 1000;
    unsigned cuBlockSize = 128;

    uint64_t m_current_target;
    
    volatile bool m_done = true;
    // std::mutex m_doneMutex;

    volatile bool m_stop = false;
};

