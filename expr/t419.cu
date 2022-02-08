#include <stdio.h>

// https://stackoverflow.com/questions/46345811/cuda-9-shfl-vs-shfl-sync

__global__
static void shflTest(int lid){
    int tid = threadIdx.x;
    float value = tid + 0.1f;
    int* ivalue = reinterpret_cast<int*>(&value);

    //use the integer shfl
    int ix = __shfl(ivalue[0],5,32);
    // int ix = __shfl_sync(0xFFFFFFFF,ivalue[0],5,32);
    int iy = __shfl_sync(0xFFFFFFFF, ivalue[0],5,32);

    float x = reinterpret_cast<float*>(&ix)[0];
    float y = reinterpret_cast<float*>(&iy)[0];

    // if(tid == lid){
        printf("tid = %d shfl tmp %d %d\n",tid, ix,iy);
        printf("tid = %d shfl final %f %f\n",tid, x,y);
    // }
}
// /usr/local/cuda/bin/nvcc -arch=sm_61 -o t419 t419.cu
// /usr/local/cuda/bin/cuda-memcheck ./t419
int main()
{
    shflTest<<<1,32>>>(0);
    cudaDeviceSynchronize();
    return 0;
}
