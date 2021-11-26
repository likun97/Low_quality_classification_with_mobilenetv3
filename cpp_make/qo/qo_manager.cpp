/*************************************************************************
    > File Name: request_manager.cpp
    > Author:  
    > Created Time: Thu 22 Mar 2018 04:22:56 PM CST
 ************************************************************************/

#include "qo_manager.hpp"
#include "reply_manager.hpp"
// #include "code/porn_predict_caffe.hpp"
//#include "code/porn_predict_mxnet.hpp"
#include <Platform/log.h>


int qo_manager::open(client_configuration *conf)
{
    active = true;
    device_num = conf->device_num;
    lq_thread_num = conf->lq_thread_num;
    if(lq_thread_num > MAX_THREAD_NUM)
    {
        SS_DEBUG((LM_ERROR, "too many models!!\n"));
        return -1;
    }
    cls_nums_porn = conf->cls_nums_porn;
    cls_nums_lq = conf->cls_nums_lq;

    // mxnet_json = conf->mxnet_json;
    // mxnet_para = conf->mxnet_para;
    //routine_info *info = new routine_info[mxnet_num + caffe_num];
    //info.type = big_pic;
    //info.data = this;
    porn_engine = conf->porn_engine;
    lq_engine = conf->lq_engine;

    int i;
    /*
    for(i = 0; i < mxnet_num; i++)
    {
        info[i].type = small_pic;
        info[i].data = this;
        info[i].device_id = i % device_num;
        if(pthread_create(&thread_id[i], NULL, routine, &info[i]) != 0)
        {
            SS_DEBUG((LM_ERROR, "error creating mxnet routine.\n"));
            return -1;
        }
        timeval time;
        time.tv_sec = 2;
        time.tv_usec = 0;
        select(1, NULL, NULL, NULL, &time);
    }
    */

    for(i = 0; i < lq_thread_num; i++)
    {
        info[i].type = big_pic;
        info[i].data = this;
        info[i].device_id = i % device_num;
        if(pthread_create(&thread_id[i], NULL, routine, &info[i]) != 0)
        {
            SS_DEBUG((LM_ERROR, "error creating caffe routine.\n"));
            return -1;
        }
    }

    qo_base::open();
    return 0;
}

int qo_manager::close()
{
    active = false;
    // for(int i = 0; i < mxnet_num; i++)
    // {
    //     pthread_join(thread_id[i], NULL);
    // }
    for(int i = 0; i < lq_thread_num; i++)
    {
        pthread_join(thread_id[i], NULL);
    }
    //delete[] info;
    qo_base::close();
    return 0;
}

void *qo_manager::routine(void *arg)
{
    routine_info *info = (routine_info *)arg;
    int type = info->type;
    int device_id = info->device_id;
    qo_manager *qo = (qo_manager *)info->data;
    if(type == big_pic)
    {
        qo->big_routine(device_id);
    }
    else
    {
        qo->small_routine(device_id);
    }
    return (void*)0;
}

int qo_manager::big_routine(int device_id)
{
    fprintf(stderr, "in big routine.\n");
    lq::LqClassify *lq_net = new lq::LqClassify(device_id);
    lq_net->init(porn_engine, lq_engine, 1, cls_nums_porn, cls_nums_lq);        // TODO: change batch size here
    int label = 0;
    const int cls_nums = cls_nums_lq + cls_nums_porn;
    const int valid_cls_nums = cls_nums - 1;

    //PornPredictMxnet *mxnet = new PornPredictMxnet;
    //mxnet->initModel(mxnet_json, mxnet_para, 0, 10);
    while(active)
    {
        // the big_list schedule the qo request
        // is set by request_manager, where it call the qo_manager::put_big(), this func is inherited from qo_base class!
        qo_request *r = (qo_request *)big_list.get_from_head();
        if(r != NULL)
        {
            r->mark_time(QO_BEGIN);
            if(r->type == big_pic && lq_net)
            {
                // lq_net->predict(r->mats, r->predict_probs);
                lq_net->do_classify(r->mats, r->predict_probs);
                for(auto &elem : r->predict_probs)
                    fprintf(stderr, "predicted score: %f\n", elem);

                // if(r->predict_probs.size() < 4 * r->mats.size())
                // TODO: change cls_nums_porn + cls_nums_lq = 11
                // if(r->predict_probs.size() < 11 * r->mats.size())
                if(r->predict_probs.size() < cls_nums * r->mats.size())
                {
                    fprintf(stderr, "qo request %x process error.\n", r->request_id);
                    for(int i = r->predict_probs.size(); i < cls_nums * r->mats.size(); i++)
                    {
                        r->predict_probs.push_back(0.0);
                    }
                }

                // 0:norm 1:porn 2:adult 3:sexy ->0:norm 1:porn 2:sexy
                // 0:norm_porn, 1:porn, 2:adult, 3:sexy, 4:norm_lq, 5:chart, 6:qr, 7:link, 8:blood, 9:nausea, 10:terrorism
                // ->
                // 0:norm_porn+adult, 1:porn, 2:sexy, 3:norm_lq, 4:chart, 5:qr, 6:link, 7:blood, 8:nausea, 9:terrorism
                std::vector<float> tmp = r->predict_probs;
                r->predict_probs.clear();
                // TODO: change 10 -> new_cls_nums
                fprintf(stderr, "Img nums: %d\n", r->mats.size());
                fprintf(stderr, "Size of tmp: %d \n", tmp.size());
                // r->predict_probs.reserve(10 * r->mats.size());
                // r->predict_probs.resize(10 * r->mats.size());
                r->predict_probs.resize(valid_cls_nums * r->mats.size());
                fprintf(stderr, "Size of predict_probs: %d \n", r->predict_probs.size());
                for(int i = 0; i < r->mats.size(); i++)
                {
                    /// step 1: merge the adult -> norm_porn
                    /**
                    r->predict_probs[10 * i + 0] = tmp[11 * i + 0] + tmp[11 * i + 2];
                    r->predict_probs[10 * i + 1] = tmp[11 * i + 1];
                    r->predict_probs[10 * i + 2] = tmp[11 * i + 3];
                    for (int j = 3; j < 10; ++j)
                        r->predict_probs[10 * i + j] = tmp[11 * i + j + 1];
                    for(auto &elem : r->predict_probs)
                        fprintf(stderr, "predicted score: %f\n", elem);

                    auto pred_probs_beg = r->predict_probs.begin() + i * 10;
                    auto pred_probs_end = r->predict_probs.begin() + (i + 1) * 10;
                    */
                    r->predict_probs[valid_cls_nums * i + 0] = tmp[cls_nums * i + 0] + tmp[cls_nums * i + 2];
                    r->predict_probs[valid_cls_nums * i + 1] = tmp[cls_nums * i + 1];
                    r->predict_probs[valid_cls_nums * i + 2] = tmp[cls_nums * i + 3];
                    for (int j = 3; j < valid_cls_nums; ++j)
                        r->predict_probs[valid_cls_nums * i + j] = tmp[cls_nums * i + j + 1];
                    for(auto &elem : r->predict_probs)
                        fprintf(stderr, "predicted score: %f\n", elem);

                    auto pred_probs_beg = r->predict_probs.begin() + i * valid_cls_nums;
                    auto pred_probs_end = r->predict_probs.begin() + (i + 1) * valid_cls_nums;

                    label = (int)std::distance(pred_probs_beg, std::max_element(pred_probs_beg, pred_probs_beg+3));
                    // int label = 0;
                    // float score = r->predict_probs[0];
                    // for (int j = 1; j < 3; ++j)
                    // {
                    //     if (r->predict_probs[j] > score)
                    //         label = j;
                    // }
                    if (label == 1 || label == 2)
                    {
                        // check the prob threshold
                        if (label == 1)
                        {
                            if (*(pred_probs_beg + label) < 0.85)
                            {
                                label = 0;
                            }
                            else
                            {
                                /**
                                for (int j = 3; j < 10; ++j)
                                    r->predict_probs[10 * i + j] = 0.0;
                                */
                                for (int j = 3; j < valid_cls_nums; ++j)
                                    r->predict_probs[valid_cls_nums * i + j] = 0.0;
                                for(auto &elem : r->predict_probs)
                                    fprintf(stderr, "predicted score: %f\n", elem);
                                r->predict_results.emplace_back(label);
                                continue;
                            }
                        }
                        else
                        {
                            if (*(pred_probs_beg + label) < 0.8)
                            {
                                label = 0;
                            }
                            else
                            {
                                /**
                                for (int j = 3; j < 10; ++j)
                                    r->predict_probs[10 * i + j] = 0.0;
                                */
                                for (int j = 3; j < valid_cls_nums; ++j)
                                    r->predict_probs[valid_cls_nums * i + j] = 0.0;
                                for(auto &elem : r->predict_probs)
                                    fprintf(stderr, "predicted score: %f\n", elem);
                                r->predict_results.emplace_back(label);
                                continue;
                            }
                        }
                    }

                    label = (int)std::distance(pred_probs_beg, std::max_element(pred_probs_beg+3, pred_probs_end));
                    // score = r->predict_probs[3];
                    // label = 3;
                    // for(int j = 4; j < 10; ++j)
                    // {
                    //     if (r->predict_probs[j] > score)
                    //         label = j;
                    // }
                    if (*(pred_probs_beg + label) < 0.8)
                        label = 3;

                    r->predict_probs[0] = r->predict_probs[3];
                    /**
                    for (int j = 4; j < 10; ++j)
                        r->predict_probs[j-1] = r->predict_probs[j];
                    */
                    for (int j = 4; j < valid_cls_nums; ++j)
                        r->predict_probs[j-1] = r->predict_probs[j];
                    if (label == 3)
                    {
                        label = 0;
                    }
                    else if(label > 3)
                    {
                        --label;
                    }
                    r->predict_results.emplace_back(label);
                }
            }
            else
            {
                fprintf(stderr, "qo process img error.\n");
            }
            r->mark_time(QO_END);
            // r: qo_request *
            reply_manager::Instance()->put(r);
        }
    }
    //delete mxnet;
    delete lq_net;
    return 0;
}

int qo_manager::small_routine(int device_id)
{
    /*
    fprintf(stderr, "in small routine.\n");
    //PornPredictCaffe *caffe = new PornPredictCaffe;
    //caffe->initModel(caffe_proto, caffe_binary, 0, 10);
    PornPredictMxnet *mxnet = new PornPredictMxnet;
    //mxnet->initModel(mxnet_json, mxnet_para, device_id, 10);
    mxnet->initModel(device_id, 10);
    while(active)
    {
        qo_request *r = (qo_request *)small_list.get_from_head();
        if(r != NULL)
        {
            r->mark_time(QO_BEGIN);
            if(r->type == small_pic && mxnet)
            {
                mxnet->predict(r->mats, r->predict_results, r->predict_probs);
            }
            else
            {
                fprintf(stderr, "qo process img error.\n");
            }
            r->mark_time(QO_END);
            reply_manager::Instance()->put(r);
        }
    }
    delete mxnet;
    //delete caffe;
    */
    return 0;
}
