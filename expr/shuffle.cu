#include <stdio.h>
#include <cuda.h>

#define DEV_INLINE __device__ __forceinline__

#define THREADS_PER_HASH    8
#define _PARALLEL_HASH      4

#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))

DEV_INLINE uint32_t fnv_reduce(uint32_t mix_p)
{
    return mix_p;
}

__global__
static void test_sync_1()
{
    uint2 state[12] = {0};

    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH) {
        uint32_t mix[_PARALLEL_HASH];
        for (int p = 0; p < _PARALLEL_HASH; p++) {
            mix[p] = i * 100 + threadIdx.x * 10 + p;
        }

        // printf("%02d i=%d mix %u:%u:%u:%u\n", threadIdx.x, i, 
        //     mix[0], mix[1], mix[2], mix[3]);

        for (int p = 0; p < _PARALLEL_HASH; p++) {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv_reduce(mix[p]);

            // printf("%02d i=%d thread_mix %u\n", threadIdx.x, i, 
            //     thread_mix);

            // update mix across threads
            shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);

            if ((i + p) == thread_id) 
            {
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }
    printf("%02d %u:%u %u:%u %u:%u %u:%u\n",
        threadIdx.x, 
        state[8].x, state[8].y,
        state[9].x, state[9].y,
        state[10].x, state[10].y,
        state[11].x, state[11].y
        );
}

__global__
static void test_sync_2()
{
    uint2 state[12+4] = {0};

    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH) {
        uint32_t mix[_PARALLEL_HASH];
        for (int p = 0; p < _PARALLEL_HASH; p++) {
            mix[p] = i * 100 + threadIdx.x * 10 + p;
        }

        // printf("%02d i=%d mix %u:%u:%u:%u\n", threadIdx.x, i, 
        //     mix[0], mix[1], mix[2], mix[3]);

        for (int p = 0; p < _PARALLEL_HASH; p++) {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv_reduce(mix[p]);

            // printf("%02d i=%d thread_mix %u\n", threadIdx.x, i, 
            //     thread_mix);

            // update mix across threads
            shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);

#if 1 // no conditional
            uint32_t idx = 8 + ((bool)((i + p) ^ thread_id) << 2);
#else
            uint32_t v = (i + p) ^ thread_id;
            uint32_t idx = 8 + (v >> (32 - __clz(v)) - 1) * 4;
#endif
            // printf("%02d %u %u %u\n", thread_id, i, p, idx);
            state[idx+0] = shuffle[0];
            state[idx+1] = shuffle[1];
            state[idx+2] = shuffle[2];
            state[idx+3] = shuffle[3];
        }
    }
    printf("%02d %u:%u %u:%u %u:%u %u:%u\n",
        threadIdx.x, 
        state[8].x, state[8].y,
        state[9].x, state[9].y,
        state[10].x, state[10].y,
        state[11].x, state[11].y
        );
}

__global__
static void test_sync_3(unsigned int mask)
{
    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
    uint32_t value = 0xffff;
    unsigned int active = __activemask();

    value = __shfl_sync(mask, thread_id, 0, THREADS_PER_HASH);

    printf("%02d mask = 0x%x value = 0x%x (active = 0x%x)\n",thread_id, mask, value, active);
}

// /usr/local/cuda/bin/nvcc -arch=sm_61 -o shuffle shuffle.cu
// /usr/local/cuda/bin/nvcc -arch=sm_61 -ptx -o shuffle.ptx shuffle.cu
// /usr/local/cuda/bin/cuda-memcheck ./shuffle
int main()
{
    printf("test_sync_1:\n");
    printf("===============================\n");
    test_sync_1<<<1,8>>>();
    cudaDeviceSynchronize();

    printf("test_sync_2:\n");
    printf("===============================\n");
    test_sync_2<<<1,8>>>();
    cudaDeviceSynchronize();

    printf("test_sync_3 (mask = 0x01):\n");
    printf("===============================\n");
    test_sync_3<<<1,8>>>(0x1);
    cudaDeviceSynchronize();

    printf("test_sync_3 (mask = 0xff):\n");
    test_sync_3<<<1,8>>>(0xff);
    cudaDeviceSynchronize();

    return 0;
}
