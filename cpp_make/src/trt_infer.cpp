/**
 * \author smher
 * \date 2020.04.10
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <iterator>
#include <chrono>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "logger.h"
#include "NvInfer.h"
#include "common.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "pluginLayer.hpp"

using namespace std;
using namespace nvinfer1;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        cerr << file << " CUDA runtime API error at " << line << " : " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

static const int IN_HEIGHT = 300;           // height of input data
static const int IN_WIDTH = 300;            // width of input data 

static const int OUTPUT_SIZE = 9;           // class nums: norm, chart, qr, link, pad, ad, nausea, blood

 

void load_img_bin(string src_file, float *in_data)
{
    fstream srcf(src_file, std::ios::in | std::ios::binary);
    srcf.read((char *)in_data,  3 * 224 * 224 * sizeof(float));
}
 
void write_img_bin(string out_file, const float *out_data)
{
    fstream outf(out_file, std::ios::out | std::ios::binary);
    outf.write((char *)out_data, 3 * 224 * 224 * sizeof(float));
}

 


cv::Mat preprocess_img(cv::Mat& img)
{
    cv::Mat img_tmp;
    /// caffe
    /**
    img.convertTo(img_tmp, CV_32FC3, 1.0);
    cv::resize(img_tmp, img_tmp, cv::Size(IN_WIDTH, IN_HEIGHT));
    cv::Mat mean_mat = cv::Mat(img_tmp.rows, img_tmp.cols, CV_32FC3, cv::Scalar(104.0, 117.0, 123.0));
    img_tmp = img_tmp - mean_mat;
    */

    // mxnet
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0/255.);
    // img_tmp.convertTo(img_tmp, CV_32FC3, 1.0/255.);
    cv::resize(img, img_tmp, cv::Size(IN_WIDTH, IN_HEIGHT));
    
    cv::Mat mean_mat = cv::Mat(img_tmp.rows, img_tmp.cols, CV_32FC3, cv::Scalar(123.675/255.0, 116.28/255.0, 103.53/255.0));
    cv::Mat std_mat = cv::Mat(img_tmp.rows, img_tmp.cols, CV_32FC3, cv::Scalar(58.395/255.0, 57.12/255.0, 57.375/255.0));
    // cv::Mat mean_mat = cv::Mat(img_tmp.rows, img_tmp.cols, CV_32FC3, cv::Scalar(123.675, 116.28, 103.53));
    // cv::Mat std_mat = cv::Mat(img_tmp.rows, img_tmp.cols, CV_32FC3, cv::Scalar(58.395, 57.12, 57.375));

    img_tmp = (img_tmp - mean_mat) / std_mat;


    cout << "preprocess_img already " << endl;
    return img_tmp;
}



cv::Mat preprocess_img1(cv::Mat& img)
{
    cv::Mat img_tmp;
    /// caffe
    /**
    img.convertTo(img_tmp, CV_32FC3, 1.0);
    cv::resize(img_tmp, img_tmp, cv::Size(IN_WIDTH, IN_HEIGHT));
    cv::Mat mean_mat = cv::Mat(img_tmp.rows, img_tmp.cols, CV_32FC3, cv::Scalar(104.0, 117.0, 123.0));
    img_tmp = img_tmp - mean_mat;
    */

    // mxnet
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // img.convertTo(img, CV_32FC3, 1.0);         
    cv::resize(img, img_tmp, cv::Size(IN_WIDTH, IN_HEIGHT), (0.0), (0.0), cv::INTER_LINEAR);
    // img_tmp.convertTo(img_tmp, CV_32FC3, 1.0/255);
 
    return img_tmp;
}


 

ICudaEngine* load_engine(const string& plan_file)
{
    // step1: create IRuntime
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    
    // step2: create Engine using IRuntime
    fstream inf(plan_file.c_str(), ios::in | ios::binary);
    assert(inf.is_open());
    string buffer;
    inf >> noskipws;
    // copy(istream_iterator(inf), istream_iterator(), back_inserter(buffer));
    copy(istreambuf_iterator<char>(inf), istreambuf_iterator<char>(), back_inserter(buffer));
    inf.close();
    PluginFactory pluginFactory;
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), &pluginFactory);
    assert(engine != nullptr);

    // step3: release the runtime
    runtime->destroy();

    return engine;
}

 


int do_inference(IExecutionContext& context, float *data, float *output, int batchsize)
{
    // create cuda stream to implement the asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const ICudaEngine& engine = context.getEngine();
    // get binding infos, just for debug
    // int bn = engine.getNbBindings();
    // cout << "Binding nums: " << bn << endl;
    // for (int i = 0; i < bn; ++i)
    // {
    //     the output names just same as the names of mxnet json files of each operator !!!
    //     cout << engine.getBindingName(i) << endl;
    // }
    assert(engine.getNbBindings() == 2);            // just one input, one output
    void* buffer[2];                                // the order should same as Binding orders
    // get the input & output index to assure correct order
    int inputIndex{}, outputIndex{};
    for (int i = 0; i < engine.getNbBindings(); ++i)
    {
        if (engine.bindingIsInput(i))
            inputIndex = i;
        else
            outputIndex = i;
    }
    cout << "inputIdx: "    << inputIndex  << endl;
    cout << "outputIndex: " << outputIndex << endl;

    // prepare the input & output memory space on GPU
    checkCudaErrors(cudaMalloc(&buffer[inputIndex], batchsize * 3 * IN_HEIGHT * IN_WIDTH * sizeof(float)));
    checkCudaErrors(cudaMalloc(&buffer[outputIndex], batchsize * OUTPUT_SIZE * sizeof(float)));

    // data from host to device
    // checkCudaErrors(cudaMemcpyAsync(buffer[inputIndex], data, batchsize * 3 * IN_HEIGHT * IN_WIDTH * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpy(buffer[inputIndex], data, batchsize * 3 * IN_HEIGHT * IN_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    cudaStreamSynchronize(stream);

    // do execution
    context.enqueue(batchsize, buffer, stream, nullptr);      // now event is nullptr

    // data from device to host
    // checkCudaErrors(cudaMemcpyAsync(output, buffer[outputIndex], batchsize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpy(output, buffer[outputIndex], batchsize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // synchronous
    checkCudaErrors(cudaStreamSynchronize(stream));

    // release the resources
    checkCudaErrors(cudaFree(buffer[inputIndex]));
    checkCudaErrors(cudaFree(buffer[outputIndex]));
    checkCudaErrors(cudaStreamDestroy(stream));

    return 0;
}

 
int main(int argc, char **argv)
{
    // get the machine gpu infos
    int gpu_nums{0};
    checkCudaErrors(cudaGetDeviceCount(&gpu_nums));
    cout << "Total gpu nums: " << gpu_nums << endl;
    int gpu_id{-1};
    checkCudaErrors(cudaGetDevice(&gpu_id));
    cout << "Default gpu id: " << gpu_id << endl;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    cout << "Current GPU prop: " << endl;
    cout << "Name: " << prop.name << endl;
    cout << "warpSize: " << prop.warpSize << endl;
    cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << endl;
    // set the gpu id
    cudaSetDevice(1);
    checkCudaErrors(cudaGetDevice(&gpu_id));
    cout << "New gpu id: " << gpu_id << endl;

    // prepare data in the CHW order 
    // string img_file = "WechatIMG2637.jpeg";
    string img_file;
    if (argc > 1)
        img_file = string(argv[1]);
    else
        img_file = "WechatIMG2640.jpeg"; 
     // img_file = "1005.jpg_02.jpg"; 
        cout << "read already " << endl;
 

    cv::Mat img = cv::imread(img_file); 
    cout << "Shape of input img: " << img.rows << " * " << img.cols << endl;

    cv::Mat img_tmp = preprocess_img(img); 
    cout << "Shape of input img_tmp: " << img_tmp.rows << " * " << img_tmp.cols << endl;
 
    // float in_data[224 * 224 * 3];
    float in_data[300 * 300 * 3];

    for (int c = 0; c < 3; ++c)                         // channel
    {
        for (int i = 0; i < img_tmp.rows; ++i)          // height
        {
            for(int j = 0; j < img_tmp.cols; ++j)       // width
            {
                int idx = c * (img_tmp.cols * img_tmp.rows) + i * img_tmp.cols + j;
                in_data[idx] = (float)img_tmp.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
 
 
    // string plan_file = "./aa.trt";
    string plan_file = "./model_large_2_v2_softmax_dim1_V2_dropout2_300.plan"; 

    ICudaEngine* engine = load_engine(plan_file);


    // do inference
    float output[OUTPUT_SIZE];
    for(int i = 0; i < OUTPUT_SIZE; ++i)
        output[i] = -1;
    IExecutionContext* context = engine->createExecutionContext();

    // for (int i = 0; i < 10; ++i)                    
    for (int i = 0; i < 5; ++i)
        do_inference(*context, in_data, output, 1);     // warmup

    // start timing
    auto start = std::chrono::steady_clock::now();
    // for (int i = 0; i  < 100; ++i)                    
    for (int i = 0; i  < 2; ++i)
        do_inference(*context, in_data, output, 1);     // running

        
    auto end = std::chrono::steady_clock::now();
    chrono::duration<double> diff = end - start;

    cout << "total time: " << diff.count() * 1000.0 << endl;

    // output the results
    cout << "Predicted score: " << endl;
    for(int i = 0; i < OUTPUT_SIZE; ++i)
        cout << output[i] << endl;
    
    // destroy the resources
    context->destroy();
    engine->destroy();

    cout << "Done." << endl;
}
 
// src/trt_infer.cpp     
// 49   nums of category   
// 96   mean var  
// 228  given test    
// 252  given .plan 
