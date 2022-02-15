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
#define CU_BLOCK_SIZE 128

typedef struct test_result {
    int devId;
    // settings
    int epoch;
    int streams;
    int block_multiple;
    int block_size;
    // return results
    int streamIdx;
    uint64_t nonce;
    ethash::hash256 mix_hash;
    ethash::hash256 final_hash;
    float duration;
    uint32_t total_hashs;
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
    std::vector<NvDevInfo> enumDevices();
    bool getDevicInfo(int devId, NvDevInfo& nvInfo);
    int getDeviceId() {
        return m_devId;
    }
    bool set_epoch_ctx(struct EpochContexts ctx);
    bool gen_dag();
    void set_search_params(int streams, uint32_t block_multiple, uint32_t block_size) {
        cuStreamSize = streams;
        m_block_multiple = block_multiple;
        cuBlockSize = block_size;
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
    unsigned cuBlockSize = CU_BLOCK_SIZE;

    uint64_t m_current_target;
    
    volatile bool m_done = true;

#if 0 // FIXME: error: use of deleted function ... mutex& operator=(const mutex&) = delete
    std::mutex m_doneMutex;
#endif

    volatile bool m_stop = false;
};

