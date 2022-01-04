#include <stdlib.h>

// #define N 10000000
#define N 10

void run_vector_add(float *out, float *a, float *b, int n);

int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = i * 1.0f; b[i] = i * 2.0f;
    }
    // Main function
    run_vector_add(out, a, b, N);
}
