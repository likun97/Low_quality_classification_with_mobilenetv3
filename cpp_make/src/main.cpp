


#include "iostream"
#include "vector"
#include "logger.h"

#include "lqclassify.hpp"

using namespace std;

int main(int argc, char **argv)
{
    // for debug
    string porn_plan("pornTRT.plan");
    // string lq_plan("serialized_model_resnet34v1b.plan");

    // string lq_plan("lk_model_large_2_v2.plan"); 
    string lq_plan("lk_model_large_2_v2_softmax.plan"); 

    vector<int> data_shapes{224, 224};
    int bs = 1;
    int cls_porn = 4;
    // int cls_lq = 7;

    int cls_lq = 9;

    lq::LqClassify lqclassify(1);

    cv::Mat img = cv::imread("3.jpeg");
    vector<float> pred_porn;
    vector<float> pred_lq;

    lqclassify.init(porn_plan, lq_plan, data_shapes, bs, cls_porn, cls_lq);
    lqclassify.do_classify({img}, pred_porn, pred_lq);

    for (auto & elem : pred_porn)
        cout << elem << endl;
    cout << "---------" << endl;
    for (auto & elem : pred_lq)
        cout << elem << endl;

    cout << "hello world." << endl;
}

// src/main.cpp   19行 plan.model   26 nums of category

// void TensorNet::Preprocess(cv::Mat &im1)
// {
//     // need resize image ...
//     cv::cvtColor(im1, im1, cv::COLOR_BGR2RGB);
//     cv::resize(im1, this->im1, cv::Size(this->kINPUT_W, this->kINPUT_H), (0.0), (0.0), cv::INTER_LINEAR);

//     int offset_g = this->kINPUT_H * this->kINPUT_W;
//     int offset_b = this->kINPUT_H * this->kINPUT_W * 2;
//     unsigned char *line1 = NULL;

//     for(int i = 0; i < this->kINPUT_H; i++)
//     {
//         line1 = this->im1.ptr< unsigned char >(i);

//         int line_offset = i * this->kINPUT_W;

//         for(int j = 0; j < this->kINPUT_W; j++)
//         {
//             this->inputData[line_offset + j]            = ((float)(line1[j * 3] / 255. - 0.485) / 0.229);
//             this->inputData[offset_g + line_offset + j] = ((float)(line1[j * 3 + 1] / 255. - 0.456) / 0.224);
//             this->inputData[offset_b + line_offset + j] = ((float)(line1[j * 3 + 2] / 255. - 0.406) / 0.225);
//         }
//     }
// }

 
