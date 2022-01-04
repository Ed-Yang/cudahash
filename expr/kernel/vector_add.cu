
// #include <cuda.h>
#include <stdio.h>

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
    // printf("out = %f\n", out[0]);
}

void run_vector_add(float *out, float *a, float *b, int n)
{
    float *d_a, *d_b, *d_out; 

    cudaMalloc((void**)&d_a, sizeof(float) * n);
    cudaMalloc((void**)&d_b, sizeof(float) * n);
    cudaMalloc((void**)&d_out, sizeof(float) * n);
    
    cudaMemcpy(d_a, a, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * n, cudaMemcpyHostToDevice);

    vector_add<<<1,1>>>(d_out, d_a, d_b, n);

    cudaMemcpy(out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);

    printf(">>> n = %d\n", n);
    for (int i=0; i < n; i++)
        printf(">>> i=%d, out = %f\n", i, out[i]);

}
