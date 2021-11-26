/**
 * \date 2020.04.10
 * \brief
 */

#include "logger.h"
#include "NvInfer.h"
#include "common.h"
#include "NvCaffeParser.h"

#include "pluginLayer.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

 
 
// plugins
// #include "greater.hpp"

using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int height = 224;
static const int width = 224;
static const int channel = 3;

// bool caffe2trt(const string& deployFile, const string& modelFile, const vector<string>& outputs, int maxBatchSize)
// {
//     // step1: create builder & network definition
//     IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
//     assert(builder != nullptr);
//     INetworkDefinition* network = builder->createNetwork();
//     // const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//     // INetworkDefinition* network = builder->createNetworkV2(explicitBatch);


//     // step2: create caffe parser & load plugin
//     auto parser = createCaffeParser();
//     PluginFactory pluginFactory;
//     parser->setPluginFactory(&pluginFactory);

//     // step data type
//     bool mEnableFp16=false;
//     bool mEnableInt8=false;
//     mEnableFp16 = builder->platformHasFastFp16();
//     mEnableInt8 = builder->platformHasFastInt8();
//     DataType modelDataType = mEnableFp16 ? DataType::kHALF : DataType::kFLOAT;

//     // parse to get the network
//     const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(),
//                                                                 modelFile.c_str(),
//                                                                 *network,
//                                                                 modelDataType);

//     assert(blobNameToTensor != nullptr);
//     for (int i = 0, n = network->getNbInputs(); i < n; i++)
//     {
//         Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
//         std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" <<
//                 dims.d[2] << std::endl;
//     }

//     for (auto& s : outputs) network->markOutput(*blobNameToTensor->find(s.c_str()));

//     builder->setMaxBatchSize(maxBatchSize);
//     builder->setMaxWorkspaceSize(1 << 30);

//     // set up the network for paired-fp16 format
//     if(mEnableFp16) builder->setHalf2Mode(true);

//     ICudaEngine* engine = builder->buildCudaEngine(*network);
//     assert(engine);

//     network->destroy();
//     parser->destroy();

//     IHostMemory* gieModelStream = engine->serialize();
//     engine->destroy();
//     builder->destroy();
//     pluginFactory.destroyPlugin();

//     // std::ofstream ofs("pornTRTtmp.plan", std::ios::out | std::ios::binary);
//     std::ofstream ofs("pornTRT.plan", std::ios::out | std::ios::binary);
//     ofs.write((char*)(gieModelStream->data()), gieModelStream->size());
//     ofs.close();
//     gieModelStream->destroy();
//     shutdownProtobufLibrary();
// }




bool onnx2trt(const string& onnx_file, int maxBatchSize=1)
{
    // step1: create builder & network definition
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    // INetworkDefinition* network = builder->createNetwork();
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // step2: create onnx parser
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    // step3: parse the input onnx files
    if(!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        // gLogError << "Failuer while parsing ONNX file" << std::endl;
        // return false;
    }

    // tmp: get the network info
    cout << "Network Name: " << network->getName() << endl;
    int layer_nums = network->getNbLayers();
    cout << "Total layer nums: " << layer_nums << endl;
    ILayer* layer = network->getLayer(0);
    cout << "First Layer: " << layer->getName() << endl;
    ITensor* in_data = network->getInput(0);
    cout << "Input name: " << in_data->getName() << endl;
    int out_nums = network->getNbOutputs();
    cout << "Total output nums: " << out_nums << endl;
    ILayer* layerlast = network->getLayer(layer_nums - 1);
    cout << "last layer: " << layerlast->getName() << endl;
    cout << "last layer: " << layerlast->getNbOutputs() << endl;
    for(int i = 0; i < layer_nums; ++i)
    {
        cout << "i: " << i << network->getLayer(i)->getName() << endl;
    }

    // step4: get the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1<<30);
    
    if (builder->platformHasFastFp16())         // not support fp16
    {
        cout << "** Using FP16 **" << endl;
        builder->setFp16Mode(true);
    }
    
    if (builder->platformHasFastInt8())         // do support int8
    {
        cout << "** INT8 Supported **" << endl;
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);

    // step5: serialize the engine
    IHostMemory *serializeModel = engine->serialize();
    assert(serializeModel != nullptr);
    size_t model_sz = serializeModel->size();
    cout << "Sizeof serialized model: " << model_sz << endl;
    // cout << "Dataype of serializeModel: " << serializeModel->type() << endl;

    // fstream outf("lk_model_large_2_v2_softmax_dim1.plan", ios::binary | ios::out); 

    // fstream outf("lk_model_large_2_v2_softmax_dim1_V2_dropout.plan", ios::binary | ios::out); 
    fstream outf("model_large_2_v2_softmax_dim1_V2_dropout2_300.plan", ios::binary | ios::out); 
    
    
    assert(outf.is_open());

    void *data = serializeModel->data();
    outf.write(static_cast<char*>(data), model_sz);

    // step6: destroy the resource.
    parser->destroy();
    engine->destroy();
    network->destroy();
    builder->destroy();

    serializeModel->destroy();

    // step7: done.  step 
} 

 
## another one 

bool onnx2trt_yxh(const string& onnx_file)

// void TensorNet::OnnxNetToTRT(std::string cache_path, const char* onnx_file)
{
    std::cout << "cache model not found, begin parser and serilize model" << std::endl;
    // step one: create tensorRT builder and network
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

    // IBuilder *builder = createInferBuilder(this->gLogger);
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());

    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    // step two: create tensorRT parser

    // auto parser = nvonnxparser::createParser(*network, this->gLogger);
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    // DataType modelDataType = DataType::kHALF;





    // step three: use parser to parse model


    // if(!parser->parseFromFile(onnx_file, verbosity))
    if(!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        std::cout << "failed to parase onnx file" << std::endl;
        // this->gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
 

    // to build an engine in tensorRT, there are two steps
    // step one: build the engine using the builder object
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(36 << 20);
//    if(!strcmp(this->inference_mode.c_str(), "INT8"))
//    {
//        assert(!builder->platformHasFastInt8());
//        int inputSize = this->kINPUT_C * this->kINPUT_H * this->kINPUT_W;
//        Int8EntropyCalibrator calibrator(1, this->calibInfo.calibImages, this->calibInfo.calibImagesPath, this->calibInfo.calibTableFilePath,
//                                         inputSize, this->kINPUT_H, this->kINPUT_W);
//        builder->setInt8Mode(true);
//        builder->setInt8Calibrator(&calibrator);
//    }
//    else
//        builder->setFp16Mode(true);
    builder->setFp16Mode(true);

    ICudaEngine* t_engine = builder->buildCudaEngine(*network);
    assert(t_engine);

    // step two: dispense of the network, build and parser
    network->destroy();
    parser->destroy();

    // serialize the engine
    IHostMemory* trtModelStream = t_engine->serialize();
    std::cout << "parser and serilize done" << std::endl;

    // save serialize result
    std::cout << "save serilize result" << std::endl;
    std::ofstream serialize_output_stream;

    std::string serialize_str;
    serialize_str.resize((trtModelStream->size()));
    memcpy((void *)serialize_str.data(), trtModelStream->data(), trtModelStream->size());

    // cache_path =

    serialize_output_stream.open("./model_large_2_v2_softmax_dim1_V2_dropout2_300.trt");   //write to .file


    serialize_output_stream << serialize_str;
    serialize_output_stream.close();
    std::cout << "save serilize result done" << std::endl;

    t_engine->destroy();
    builder->destroy();
}




 
int main(int argc, char **argv)
{
    // only used in caffe 2 trt
    /// TODO: add opt parse function
    /**
    if (argc < 4)
    {
        cout << "Usage: \n" 
                "./trt_build deployfile modelfile maxbatchsize" << endl;
        return -1;
    }
    string deployFile(argv[1]);
    string modelFile(argv[2]);
    int maxBatchSize = stoi(argv[3]);
    vector<string> outputs{"prob"};
    caffe2trt(deployFile, modelFile, outputs, maxBatchSize);
    */

    /// only used in mxnet
    // string onnx_file = "../onnx_models/resnet34v1b_lq_nausea.onnx"; 
    // string onnx_file = "../onnx_models/model_large_2_v2_softmax_dim1_V2_dropout2.onnx";  
 

    string onnx_file = "../onnx_models/model_large_2_v2_softmax_dim1_V2_dropout2_300.onnx";  

    onnx2trt(onnx_file); 
 // onnx2trt_yxh(onnx_file); 

    cout << "Done." << endl;
}



//  src/trt_build.cpp  
// 	159  onnx2trt           save to /build 
// 	208  main onnx2trt()    given dir of onnx ../onnx_models/




