#include "pluginLayerGPU.hpp"

#define CUDA_KERNEL_LOOP(i, n) \
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
             i < (n); \
             i += blockDim.x * gridDim.x)

__global__ void ScaleSE(const int batchSize, const int mNbInputChannels, const int mNbInners,
                        const float* bottom_data1, const float* bottom_data2, float* top_data, const int div_factor, cudaStream_t stream, cudnnHandle_t mCudnn, cublasHandle_t mCublas) {

        CUDA_KERNEL_LOOP(index,  batchSize * mNbInputChannels * mNbInners) {
                top_data[index]=bottom_data1[index]*bottom_data2[index / mNbInners];
        }
}


cudaError_t ScaleSE_Forward_gpu(const int batchSize, const int mNbInputChannels, const int mNbInners, const float* bottom_data1, const float* bottom_data2,
                                float* top_data, const int div_factor, cudaStream_t stream, cudnnHandle_t mCudnn, cublasHandle_t mCublas){
        ScaleSE<<<GET_BLOCKS(batchSize * mNbInputChannels * mNbInners), CUDA_NUM_THREADS>>>(batchSize, mNbInputChannels, mNbInners, bottom_data1, bottom_data2, top_data, div_factor, stream, mCudnn, mCublas);
        return cudaPeekAtLastError();
}
