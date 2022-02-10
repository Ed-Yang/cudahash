#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include <ethash/ethash.hpp>
#include <ethash/global_context.hpp>
#include <test/unittests/helpers.hpp>

#include "getopt/getopt.h"

// #include "libcuda/ethash_cuda_miner_kernel.h"
#include "nv_dev.hpp"
#include "eth_tester.hpp"
#include "eth_utils.hpp"

using namespace std;
using namespace chrono;

#define BOUNDARY (const uint8_t *)"00000000ffff0000"
#define NONCE   0xff4136b6b6a244ec

#if 0 // INFO: epoch 41 data
const uint8_t *hhash_str = (const uint8_t *)"f5afa3074287b2b33e975468ae613e023e478112530bc19d4187693c13943445";
uint64_t nonce = 0xff4136b6b6a244ec; // result nonce
const uint8_t *mix_str = (const uint8_t *)"47da5e47804594550791c24331163c1f1fde5bc622170e83515843b2b13dbe14";
const uint8_t *final_str = (const uint8_t *)"0000000000095d18875acd4a2c2a5ff476c9acf283b4975d7af8d6c33d119c74";
uint64_t target = 0x0000000fffffffff;
#endif

std::vector<test_result_t> run_test(EthTester& tester, int devId, uint64_t start_nonce, 
    int num_streams, uint32_t block_multiple, uint32_t block_size)
{
    int epoch = 41;
    const uint8_t *hhash_str = (const uint8_t *)"f5afa3074287b2b33e975468ae613e023e478112530bc19d4187693c13943445";
    uint64_t target = 0x0000000fffffffff;

    ethash::result result;
    ethash::hash256 hhash;

    printf(">>> Run dev: %d, start-nonce: %lu, streams: %d, multiple: %u, block-size=%u\n", 
        devId, start_nonce, num_streams, block_multiple, block_size);

    vector<NvDev>::iterator p_dev = tester.get_nv_dev(devId);
    p_dev->set_search_params(num_streams, block_multiple, block_size);

    bool rv = tester.gen_dag(devId);

    hex2bin(hhash_str, hhash.bytes);
    std::vector<test_result_t> results = tester.search(devId, hhash.bytes, target, start_nonce);

    return results;
}

void usage(char **argv)
{
    printf("%s\n", argv[0]);
    printf("    -d <device-id>: <device-id> start from 0\n");
    printf("    -s <cuda-streams>: <cuda-streams> in range of 2-4\n");
    printf("    -m <mblks> : <mblks> that is grid size, default is 1000\n");
    printf("    -b <blk-size> : <blk-size> block size, default is 128 (must be multiple of 8)\n");
    printf("    -n <start-nonce>: <start-nonce> default is 18393042511399634156\n");
    printf("    -a: run all <cuda-streams> and <blks> combinations\n");
}

int main(int argc, char **argv)
{
    int epoch=41;
    bool debug = true;
    int c;
    const uint8_t *boundary_str = BOUNDARY;
    int devId = 0;
    uint64_t start_nonce = NONCE ;
    int num_streams = 2;
    uint32_t block_multiple = 1000;
    uint32_t block_size = CU_BLOCK_SIZE;
    bool run_all = false;

    opterr = 0;
    start_nonce -= (128 * 1024 * 1024);

    while ((c = getopt(argc, argv, "ad:s:n:m:b:h?")) != -1)
    {
        switch (c)
        {
        case 'a':
            run_all = true;
            break;   
        case 'd':
            devId = atoi(optarg);
            break;            
        case 'n':
            char* end;
            start_nonce = strtoull(optarg, &end, 10);
            break;
        case 's':
            num_streams = atoi(optarg);
            break;
        case 'm': // grid size
            block_multiple = atoi(optarg);
            break;
        case 'b': // block size
            block_size = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv);
            return -1;
        }
    }

    // check parameters
    if (num_streams < 1 || num_streams > 4) {
        printf("number of streams must in 1-4 !!!\n");
        return -1;
    }

    if (block_size < 8 || (block_size % 8 != 0)) {
        printf("block size must be multiple of 8 !!!\n");
        return -1;
    }

    NvDev nv_dev;
    std::vector<NvDevInfo> devInfos = nv_dev.enumDevices();
    std::cout << "\nNumber of Cuda devices founds: " << devInfos.size() << std::endl;
    for (auto d: devInfos) {
        printf("  Device ID: %02d - %s\n", d.cuDeviceIndex, d.boardName.c_str());
    }
    printf("\n");

    // EthTester tester;
    EthTester tester(epoch);
    tester.add_device(devId);

    std::vector<test_result_t> results;
    if (run_all) {
        std::vector<int> streams = {1, 2, 3, 4};
        std::vector<int> mblksizes = {128, 512, 1024, 2048};

        for (auto st : streams) {
            for (auto mblk : mblksizes) {
                auto r = run_test(tester, devId, start_nonce, st, mblk, block_size);
                results.insert( results.end(), r.begin(), r.end() );
            }
        }
    }
    else {
        results = run_test(tester, devId, start_nonce, num_streams, block_multiple, block_size);
    }

    std::sort(results.begin(), results.end(), 
        [](const test_result_t & a, const test_result_t & b) -> bool
    { 
        return a.duration > b.duration; 
    });

    printf("dev st m_blks idx found nonce          time (ms)\n");
    printf("=== == ====== === ==================== ========\n");

    for (auto it : results) {
        printf("%3d %2d %6d %3d %20lu %8.2f\n",
            it.devId, it.streams, it.block_multiple, it.streamIdx, it.nonce, it.duration);
    }

    return 0;
}