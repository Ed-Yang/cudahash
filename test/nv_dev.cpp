#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#include "nv_dev.hpp"

#define HostToDevice(dst, src, siz) CUDA_CALL(cudaMemcpy(dst, src, siz, cudaMemcpyHostToDevice))
#define DeviceToHost(dst, src, siz) CUDA_CALL(cudaMemcpy(dst, src, siz, cudaMemcpyDeviceToHost))

using namespace std;
using namespace chrono;


NvDev::NvDev(int devId) 
{
    m_devId = devId;
    cuStreamSize = 2;   
}

int NvDev::getNumDevices() 
{
    int deviceCount;
    cudaError_t err(cudaGetDeviceCount(&deviceCount));
    if (err == cudaSuccess)
        return deviceCount;

    if (err == cudaErrorInsufficientDriver) {
        int driverVersion(0);
        cudaDriverGetVersion(&driverVersion);
        if (driverVersion == 0)
            std::cout << "No CUDA driver found";
        else
            std::cout << "Insufficient CUDA driver " << to_string(driverVersion);
    } else
        std::cout << "CUDA Error : " << cudaGetErrorString(err);

    return 0;
} 

bool NvDev::getDevicInfo(int devId, NvDevInfo& nvInfo)
{
    string uniqueId;
    ostringstream s;
    cudaDeviceProp props;
    int i = devId;

    size_t freeMem, totalMem;
    CUDA_CALL(cudaGetDeviceProperties(&props, i));
    // CUDA_CALL(cudaSetDevice(i));
    // CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));

    // info
    s << setw(4) << setfill('0') << hex << props.pciDomainID << ':' << setw(2) << props.pciBusID << ':'
        << setw(2) << props.pciDeviceID << ".0";
    uniqueId = s.str();
    nvInfo.boardName = string(props.name);
    nvInfo.uniqueId = uniqueId;
    nvInfo.cuDeviceIndex = i;
    
    // cap
    nvInfo.cuComputeMajor = props.major;
    nvInfo.cuComputeMinor = props.minor;
    nvInfo.cuCompute = (to_string(props.major) + "." + to_string(props.minor));
    nvInfo.maxThreadsPerBlock = props.maxThreadsPerBlock;
    nvInfo.computeMode = props.computeMode;
    nvInfo.cuBlockSize = CU_BLOCK_SIZE;
    nvInfo.cuStreamSize = 2;
    
    // mem
    nvInfo.totalGlobalMem = props.totalGlobalMem;
    nvInfo.sharedMemPerBlock = props.sharedMemPerBlock;
    nvInfo.regsPerBlock = props.regsPerBlock;
    nvInfo.memoryBusWidth = props.memoryBusWidth;

    // multi processor
    nvInfo.multiProcessorCount = props.multiProcessorCount;
    nvInfo.maxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    nvInfo.maxBlocksPerMultiProcessor = props.maxBlocksPerMultiProcessor;

    return true;
}

std::vector<NvDevInfo> NvDev::enumDevices() 
{
    int count=0;
    std::vector<NvDevInfo> devices={};

    int numDevices(getNumDevices());

    for (int i = 0; i < numDevices; i++) {
        NvDevInfo devInfo;
        if (getDevicInfo(i, devInfo))
            devices.push_back(devInfo);
        else
            break;
    }

    return devices;
}

bool NvDev::set_epoch_ctx(struct EpochContexts ctx)
{
    m_ctx = ctx;

    std::cout << "set context for epoch " << m_ctx.epochNumber << std::endl;

    return true;
}

bool NvDev::gen_dag()
{
    hash128_t* dag;
    hash64_t* light;

    m_current_target = 0;

    std::cout << "device: [" << m_devId << "] dag generated for epoch " << m_ctx.epochNumber << std::endl;

    CUDA_CALL(cudaDeviceReset());
    CUDA_CALL(cudaSetDevice(m_devId));
    CUDA_CALL(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    try {
        CUDA_CALL(cudaMalloc((void**)&light, m_ctx.lightSize));
    } catch (...) {
        return false; // This will prevent to exit the thread and
    }

    try {
        CUDA_CALL(cudaMalloc((void**)&dag, m_ctx.dagSize));
    } catch (...) {
        return false; // This will prevent to exit the thread and
    }

    for (unsigned i = 0; i < cuStreamSize; ++i) {
        try {
            CUDA_CALL(cudaMalloc(&m_search_buf[i], sizeof(Search_results)));
        } catch (...) {
            return false; // This will prevent to exit the thread and
        }
        CUDA_CALL(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
    }

    HostToDevice(light, m_ctx.lightCache, m_ctx.lightSize);

    set_constants(dag, m_ctx.dagNumItems, (hash64_t *)light, m_ctx.lightNumItems); // in ethash_cuda_miner_kernel.cu

#if 1  // seperate search, dag params
    ethash_generate_dag(m_ctx.dagSize, 1000, 128, m_streams[0]);
#else
    ethash_generate_dag(m_ctx.dagSize, m_block_multiple, cuBlockSize, m_streams[0]);
#endif

    std::cout << "dag generated." << std::endl;

    return true; 
}

static const uint32_t zero3[3] = {0, 0, 0}; // zero the result count

std::vector<test_result_t> NvDev::search(void *header, uint64_t target, uint64_t start_nonce)
{
    std::vector<test_result_t> test_results;

    std::cout << "dev: start nonce = " << start_nonce << std::endl;
    std::cout << "dev: streams = " << cuStreamSize << std::endl;
    std::cout << "dev: m_block_multiple = " << m_block_multiple << std::endl;
    std::cout << "dev: block_size = " << cuBlockSize << std::endl;
    // CUDA_CALL(cudaSetDevice(m_devId));

    auto t_start = high_resolution_clock::now();

    set_header(*((const hash32_t*)header));
    if (m_current_target != target) {
        printf("search target: 0x%016lx\n", target);
        set_target(target);
        m_current_target = target;
    }
    uint32_t batch_blocks(m_block_multiple * cuBlockSize);
    uint32_t stream_blocks(batch_blocks * cuStreamSize);

#if 0 // FIXME: error: use of deleted function ... mutex& operator=(const mutex&) = delete
    m_doneMutex.lock();
#endif

    // prime each stream, clear search result buffers and start the search
    for (uint32_t streamIdx = 0; streamIdx < cuStreamSize;
         streamIdx++, start_nonce += batch_blocks) {
        HostToDevice(m_search_buf[streamIdx], zero3, sizeof(zero3));
        // m_hung_miner.store(false);
        run_ethash_search(m_block_multiple, cuBlockSize, m_streams[streamIdx],
                          m_search_buf[streamIdx], start_nonce);
    }
    m_done = false;


#if 0 // FIXME: error: use of deleted function ... mutex& operator=(const mutex&) = delete
    m_doneMutex.unlock();
#endif

    uint32_t streams_bsy((1 << cuStreamSize) - 1);

    // process stream batches until we get new work.

    uint32_t batchCount(0);

    while (streams_bsy) {
        // if (paused()) {
        //     unique_lock<mutex> l(m_doneMutex);
        //     m_done = true;
        // }


        // This inner loop will process each cuda stream individually
        for (uint32_t streamIdx = 0; streamIdx < cuStreamSize;
             streamIdx++, start_nonce += batch_blocks) {
            uint32_t stream_mask(1 << streamIdx);
            if (!(streams_bsy & stream_mask))
                continue;

            cudaStream_t stream(m_streams[streamIdx]);
            uint8_t* buffer((uint8_t*)m_search_buf[streamIdx]);

            // Wait for the stream complete
            CUDA_CALL(cudaStreamSynchronize(stream));

            Search_results r;

            DeviceToHost(&r, buffer, sizeof(r));

            // clear solution count, hash count and done
            HostToDevice(buffer, zero3, sizeof(zero3));

#if 0 // NEW: move down re-search
            if (m_done)
                streams_bsy &= ~stream_mask;
            else {
                // m_hung_miner.store(false);
                run_ethash_search(m_block_multiple, cuBlockSize, stream, (Search_results*)buffer,
                                  start_nonce);
            }
#endif

            if (r.solCount > MAX_SEARCH_RESULTS)
                r.solCount = MAX_SEARCH_RESULTS;
            batchCount += r.hashCount;

            for (uint32_t i = 0; i < r.solCount; i++) {
                uint64_t nonce(start_nonce - stream_blocks + r.gid[i]);
                // Farm::f().submitProof(Solution{nonce, h256(), w, chrono::steady_clock::now(), m_index});
                // ReportSolution(w.header, nonce);
                test_result_t res ;
                res.streams = cuStreamSize;
                res.block_multiple = m_block_multiple;
                res.epoch = m_ctx.epochNumber;
                res.nonce = nonce;
                res.streamIdx = streamIdx;
                res.hashCount = batchCount * cuBlockSize; // hashCount only count first thread
                auto t_end = high_resolution_clock::now();
                res.duration = double(duration_cast<microseconds>(t_end - t_start).count())/1000;
                test_results.push_back(res);
                printf("search: nonce = start_nonce(%lu) - stream_blocks(%u) + r.gid[i](%u)\n", 
                    start_nonce, stream_blocks, r.gid[i]);
                printf("search: found nonce = %lu, took %6.2f ms, calc-nonces=%u\n", 
                    nonce, res.duration, res.hashCount);
                m_done = true;
            }

            if (shouldStop()) {
#if 0 // FIXME: error: use of deleted function ... mutex& operator=(const mutex&) = delete
                unique_lock<mutex> l(m_doneMutex);
#endif
                std::cout << "stop." << std::endl;
                m_done = true;
            }

#if 1 // NEW: move down re-search
            if (m_done)
                streams_bsy &= ~stream_mask;
            else {
                // m_hung_miner.store(false);
                run_ethash_search(m_block_multiple, cuBlockSize, stream, (Search_results*)buffer,
                                  start_nonce);
            }
#endif
        }
    }

    if (test_results.size() <=0) {
        printf("search: no sol found, calc-nonces=%u\n", batchCount);
    }

    return test_results;
}