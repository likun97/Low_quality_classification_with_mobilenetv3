#ifndef __GPLUGINGPU_H_
#define __GPLUGINGPU_H_
#include <iostream>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

#define CUDA_NUM_THREADS 512
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
cudaError_t ScaleSE_Forward_gpu(const int batchSize, const int mNbInputChannels, const int mNbInners, const float* bottom_data1, const float* bottom_data2,
                           float* top_data, const int div_factor, cudaStream_t stream, cudnnHandle_t mCudnn, cublasHandle_t mCublas);

#endif
