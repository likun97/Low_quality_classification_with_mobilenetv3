/**
 * \file lqclassify.hpp
 * \author smh
 * \date 2020.04.23
 * \copyright Copyright (c) 2020 sogou-inc
 */

#ifndef LQ_CLASSIFY_H__
#define LQ_CLASSIFY_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdexcept>
#include <fstream>
#include <thread>
#include <cassert>
#include <iterator>
#include <iostream>
#include <cstdio>
#include <cmath>

#include "logger.h"
#include "NvInfer.h"
#include "common.h"

#include "pluginLayer.hpp"
#include "utils.hpp"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        cerr << file << " CUDA runtime API error at " << line << " : " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

namespace lq
{
class LqClassify
{
public:
    LqClassify(int gid=0);
    ~LqClassify();

    void init(const std::string& porn_plan, const std::string& lq_plan, int bs=1, int cls_porn=4, int cls_lq=7);
    void init(const std::string& porn_plan, const std::string& lq_plan, std::vector<int> data_shapes, int bs=1, int cls_porn=4, int cls_lq=7);
    void do_classify(const std::vector<cv::Mat>& in_img, std::vector<float>& pred_porn, std::vector<float>& pred_lq);
    void do_classify(const std::vector<cv::Mat>& in_img, std::vector<float>& preds);

private:
    void exec_classify(const cv::Mat& in_img, std::vector<float> &scores);
    void load_images(const std::vector<cv::Mat>& in_img);
    void load_engine(const std::string& plan_porn, const std::string& plan_lq);

    int gpu_id_;
    float *img_host224_{nullptr}, *img_host299_{nullptr};
    float *pred_porn_{nullptr}, *pred_lq_{nullptr};
    void *buffer_porn_[2];
    void *buffer_lq_[2];

    int bs_{1};                // batch size
    int cls_nums_porn_{0}, cls_nums_lq_{0};
    std::vector<int> data_shapes_;

    cudaError_t cudaState_;
    cudaStream_t stream0_, stream1_;              
    cudaEvent_t event0_;
    static cv::Vec3f mean_rgb_;
    static cv::Vec3f std_rgb_;

    ICudaEngine *engine_porn_{nullptr}, *engine_lq_{nullptr};
    nvinfer1::IExecutionContext *porn_context_{nullptr}, *lq_context_{nullptr};
    int porn_in_idx_{0}, porn_out_idx_{1};
    int lq_in_idx_{0}, lq_out_idx_{1};
    PluginFactory pluginFactory_;
};          // class LqClassify
}           // namespace lq

#endif
