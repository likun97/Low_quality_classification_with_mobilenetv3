/**
 * \file lqclassify.hpp
 * \author smh
 * \date 2021.04.23
 * \copyright Copyright (c) 2020 sogou-inc
 */

#include "lqclassify.hpp"

namespace lq
{
LqClassify::LqClassify(int gid): gpu_id_(gid)
{
    cudaState_ = cudaSuccess;

    int gpu_count=0;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_count <= 0)
        throw std::runtime_error("The target host contains no cuda!");
    if (gid >= gpu_count)
        throw std::runtime_error("The input gpu id is exceed the gpu counts! use default (0) as gpu id.");
    checkCudaErrors(cudaSetDevice(gid));
    checkCudaErrors(cudaStreamCreate(&stream0_));
    checkCudaErrors(cudaStreamCreate(&stream1_));
}



cv::Vec3f LqClassify::mean_rgb_{123.675, 116.28, 103.53};
cv::Vec3f LqClassify::std_rgb_{58.395, 57.12, 57.375};

// cv::Vec3f LqClassify::mean_rgb_{123.675/255.0, 116.28/255.0, 103.53/255.0};
// cv::Vec3f LqClassify::std_rgb_{58.395/255.0, 57.12/255.0, 57.375/255.0};
  
LqClassify::~LqClassify()
{
    if(buffer_porn_[porn_in_idx_])
        checkCudaErrors(cudaFree(buffer_porn_[porn_in_idx_]));
    if(buffer_porn_[porn_out_idx_])
        checkCudaErrors(cudaFree(buffer_porn_[porn_out_idx_]));
    if(buffer_lq_[lq_in_idx_])
        checkCudaErrors(cudaFree(buffer_lq_[lq_in_idx_]));
    if(buffer_lq_[lq_out_idx_])
        checkCudaErrors(cudaFree(buffer_lq_[lq_out_idx_]));

    if(img_host224_)
    {
        checkCudaErrors(cudaFreeHost(img_host224_));
        img_host224_ = nullptr;
    }
    if(img_host299_)
    {
        checkCudaErrors(cudaFreeHost(img_host299_));
        img_host299_ = nullptr;
    }
    if(pred_porn_)
        checkCudaErrors(cudaFreeHost(pred_porn_));
    if(pred_lq_)
        checkCudaErrors(cudaFreeHost(pred_lq_));

    checkCudaErrors(cudaStreamDestroy(stream0_));
    checkCudaErrors(cudaStreamDestroy(stream1_));

    if (porn_context_)
        porn_context_->destroy();
    if (lq_context_)
        lq_context_->destroy();
    if (engine_porn_)
        engine_porn_->destroy();
    if (engine_lq_)
        engine_lq_->destroy();
    pluginFactory_.destroyPlugin();
}

 



void LqClassify::init(const std::string& porn_plan, const std::string& lq_plan, std::vector<int> data_shapes, int bs, int cls_porn, int cls_lq)
{
    bs_ = bs;
    cls_nums_porn_ = cls_porn;
    cls_nums_lq_ = cls_lq;
    data_shapes_.assign(data_shapes.begin(), data_shapes.end());

    // load the plan
    if (!is_file_exists(porn_plan))
        throw std::runtime_error("Input plan file not exists: " + porn_plan);
    if (!is_file_exists(lq_plan))
        throw std::runtime_error("Input plan file not exists: " + lq_plan);

    load_engine(porn_plan, lq_plan);

    porn_context_ = engine_porn_->createExecutionContext();
    assert(porn_context_ != nullptr);
    lq_context_ = engine_lq_->createExecutionContext();
    assert(lq_context_ != nullptr);

    porn_in_idx_ = engine_porn_->getBindingIndex("data");
    porn_out_idx_ = engine_porn_->getBindingIndex("prob");
    for(int i = 0; i < engine_lq_->getNbBindings(); ++i)
    {
        if (engine_lq_->bindingIsInput(i))
            lq_in_idx_ = i;
        else
            lq_out_idx_ = i;
    }

    // allocate the memory on device & host page-locked memory
    checkCudaErrors(cudaMalloc(&buffer_porn_[porn_in_idx_], sizeof(float) * bs_ * 3 * 224 * 224));
    checkCudaErrors(cudaMalloc(&buffer_porn_[porn_out_idx_], sizeof(float) * bs_ * cls_porn));
    checkCudaErrors(cudaMalloc(&buffer_lq_[lq_in_idx_], sizeof(float) * bs_ * 3 * 224 * 224));
    checkCudaErrors(cudaMalloc(&buffer_lq_[lq_out_idx_], sizeof(float) * bs_ * cls_lq));
    checkCudaErrors(cudaHostAlloc(&img_host224_, sizeof(float) * bs_ * 3 * 224 * 224, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&img_host299_, sizeof(float) * bs_ * 3 * 224 * 224, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&pred_porn_, sizeof(float) * bs_ * cls_porn, cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&pred_lq_, sizeof(float) * bs_ * cls_lq, cudaHostAllocDefault));
}







void LqClassify::init(const std::string& porn_plan, const std::string& lq_plan, int bs, int cls_porn, int cls_lq)
{
    std::vector<int> data_shapes{224, 224};
    init(porn_plan, lq_plan, data_shapes, bs, cls_porn, cls_lq);
}

void LqClassify::do_classify(const std::vector<cv::Mat>& in_img, std::vector<float>& pred_porn, std::vector<float>& pred_lq)
{
    // for debug
    std::cout << in_img[0].cols << " * " << in_img[0].rows << std::endl;
    // step1: preprocess the input images
    load_images(in_img);

    // copy data from host to device
    checkCudaErrors(cudaMemcpyAsync(buffer_porn_[porn_in_idx_], img_host299_, sizeof(float) * bs_ * data_shapes_[0] * data_shapes_[0] * 3, cudaMemcpyHostToDevice, stream0_));
    checkCudaErrors(cudaMemcpyAsync(buffer_lq_[lq_in_idx_], img_host224_, sizeof(float) * bs_ * data_shapes_[1] * data_shapes_[1] * 3, cudaMemcpyHostToDevice, stream1_));

    // do the inference 
    porn_context_->enqueue(bs_, buffer_porn_, stream0_, nullptr);
    lq_context_->enqueue(bs_, buffer_lq_, stream1_, nullptr);

    // copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(pred_porn_, buffer_porn_[porn_out_idx_], sizeof(float) * bs_ * cls_nums_porn_, cudaMemcpyDeviceToHost, stream0_));
    checkCudaErrors(cudaMemcpyAsync(pred_lq_, buffer_lq_[lq_out_idx_], sizeof(float) * bs_ * cls_nums_lq_, cudaMemcpyDeviceToHost, stream1_));

    checkCudaErrors(cudaStreamSynchronize(stream0_));
    checkCudaErrors(cudaStreamSynchronize(stream1_));

    cudaState_ = cudaGetLastError();
    if (cudaState_ != cudaSuccess)
    {
        std::string err_log(cudaGetErrorString(cudaState_));
        cudaState_ = cudaSuccess;
        throw std::runtime_error(cudaGetErrorString(cudaState_));
    }

    pred_porn.assign(pred_porn_, pred_porn_ + cls_nums_porn_);
    pred_lq.assign(pred_lq_, pred_lq_ + cls_nums_lq_);
}

void LqClassify::do_classify(const std::vector<cv::Mat>& in_img, std::vector<float>& preds)
{
    // for debug
    std::cout << in_img[0].cols << " * " << in_img[0].rows << std::endl;
    // step1: preprocess the input images
    load_images(in_img);

    // copy data from host to device
    checkCudaErrors(cudaMemcpyAsync(buffer_porn_[porn_in_idx_], img_host299_, sizeof(float) * bs_ * data_shapes_[0] * data_shapes_[0] * 3, cudaMemcpyHostToDevice, stream0_));
    checkCudaErrors(cudaMemcpyAsync(buffer_lq_[lq_in_idx_], img_host224_, sizeof(float) * bs_ * data_shapes_[1] * data_shapes_[1] * 3, cudaMemcpyHostToDevice, stream1_));

    // do the inference 
    porn_context_->enqueue(bs_, buffer_porn_, stream0_, nullptr);
    lq_context_->enqueueV2(buffer_lq_, stream1_, nullptr);

    // copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(pred_porn_, buffer_porn_[porn_out_idx_], sizeof(float) * bs_ * cls_nums_porn_, cudaMemcpyDeviceToHost, stream0_));
    checkCudaErrors(cudaMemcpyAsync(pred_lq_, buffer_lq_[lq_out_idx_], sizeof(float) * bs_ * cls_nums_lq_, cudaMemcpyDeviceToHost, stream1_));

    checkCudaErrors(cudaStreamSynchronize(stream0_));
    checkCudaErrors(cudaStreamSynchronize(stream1_));

    cudaState_ = cudaGetLastError();
    if (cudaState_ != cudaSuccess)
    {
        std::string err_log(cudaGetErrorString(cudaState_));
        cudaState_ = cudaSuccess;
        throw std::runtime_error(cudaGetErrorString(cudaState_));
    }

    preds.clear();
    preds.assign(pred_porn_, pred_porn_ + cls_nums_porn_);
    preds.insert(preds.end(), pred_lq_, pred_lq_ + cls_nums_lq_);
}

void LqClassify::load_images(const std::vector<cv::Mat>& in_img)
{
    assert(in_img.size() == bs_);
    std::vector<cv::Mat> img_resized_porn(in_img.size());
    std::vector<cv::Mat> img_resized_lq(in_img.size());

    cv::Mat im_resized;
    #pragma omp parallel for
    for(int i = 0; i < bs_; ++i)
    {
        in_img[i].convertTo(im_resized, CV_32FC3, 1.0);
        cv::resize(im_resized, im_resized, cv::Size(data_shapes_[0], data_shapes_[0]));
        img_resized_porn[i] = im_resized.clone();
    }

    #pragma omp parallel for
    for(int i = 0; i < bs_; ++i)
    {
        in_img[i].convertTo(im_resized, CV_32FC3, 1.0);
        cv::resize(im_resized, im_resized, cv::Size(data_shapes_[1], data_shapes_[1]));
        // cv::cvtColor(im_resized, im_resized, cv::COLOR_BGR2RGB);
        // im_resized.convertTo(im_resized, CV_32FC3, 1.0);
        img_resized_lq[i] = im_resized.clone();
    }

    #pragma omp parallel for
    for(int num=0; num<bs_; ++num)
    {
        int data_dim = data_shapes_[0];
        float* data_b = img_host299_ + num * data_dim * data_dim * 3;
        float* data_g = img_host299_ + num * data_dim * data_dim * 3 + data_dim * data_dim;
        float* data_r = img_host299_ + num * data_dim * data_dim * 3 + data_dim * data_dim * 2;
        for (int row_i = 0; row_i < data_dim; ++row_i)
        {
            float *m_data = img_resized_porn[num].ptr<float>(row_i);
            for(int col_i = 0; col_i < data_dim; ++col_i)
            {
                *data_b++ = *m_data++ - std::ceil(mean_rgb_[2]);            // 103.53 -> 104
                *data_g++ = *m_data++ - std::ceil(mean_rgb_[1]);            // 116.28 -> 117
                *data_r++ = *m_data++ - std::floor(mean_rgb_[0]);           // 123.675 -> 123
            }
        }
    }

    #pragma omp parallel for
    for(int num=0; num<bs_; ++num)
    {
        int data_dim = data_shapes_[1];
        float* data_r = img_host224_ + num * data_dim * data_dim * 3;
        float* data_g = img_host224_ + num * data_dim * data_dim * 3 + data_dim * data_dim;
        float* data_b = img_host224_ + num * data_dim * data_dim * 3 + data_dim * data_dim * 2;
        for (int row_i = 0; row_i < data_dim; ++row_i)
        {
            float *m_data = img_resized_lq[num].ptr<float>(row_i);
            for(int col_i = 0; col_i < data_dim; ++col_i)
            {
                *data_b++ = (*m_data++ - mean_rgb_[2]) / std_rgb_[2];
                *data_g++ = (*m_data++ - mean_rgb_[1]) / std_rgb_[1];
                *data_r++ = (*m_data++ - mean_rgb_[0]) / std_rgb_[0];
            }
        }
    }
}

void LqClassify::load_engine(const std::string& plan_porn, const std::string& plan_lq)
{
    // step1: create IRuntime
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    
    // step2: create Engine using IRuntime
    std::fstream inf(plan_porn.c_str(), std::ios::in | std::ios::binary);
    assert(inf.is_open());
    string buffer;
    inf >> noskipws;
    std::copy(std::istreambuf_iterator<char>(inf), std::istreambuf_iterator<char>(), back_inserter(buffer));
    inf.close();
    engine_porn_ = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), &pluginFactory_);
    assert(engine_porn_!= nullptr);

    buffer.clear();
    inf.open(plan_lq.c_str(), std::ios::in | std::ios::binary);
    assert(inf.is_open());
    inf >> noskipws;
    std::copy(std::istreambuf_iterator<char>(inf), std::istreambuf_iterator<char>(), back_inserter(buffer));
    inf.close();
    engine_lq_ = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    assert(engine_lq_!= nullptr);

    // step3: release the runtime
    runtime->destroy();
}

}  // namespace lq


// src/lqclassify.cpp   27 mean var 【 func：initial---load---classify-- 】-
