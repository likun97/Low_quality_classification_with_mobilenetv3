/*************************************************************************
    > File Name: request_manager.cpp
    > Author:  
    > Created Time: Thu 22 Mar 2018 04:22:56 PM CST
 ************************************************************************/

#include "reply_manager.hpp"
#include "request.hpp"
#include <Platform/log.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>

// int reply_manager::svc()
// {
//     //char buff[32];
//     while(active)
//     {
//         qo_request *_request = (qo_request *)_list.get_from_head();
//         if(_request != NULL)
//         {
//             _request->mark_time(REPLY_BEGIN);
//             
//             for(int i = 0; i < _request->num; i++)
//             {
//                 server_result result;
//                 result.request_id = _request->request_ids[i];
//                 result.result = _request->predict_results[i];
//                 result.prob = (_request->predict_probs[i * 10 + result.result] * 100);
// 
//                 int l = send(_request->fds[i], (void *)&result, sizeof(server_result), 0);
//                 
//                 if(l <= 0)
//                 {
//                     fprintf(stderr, "reply request %x error.\n", result.request_id);
//                 }
//             }
// 
//             _request->mark_time(REPLY_END);
// 
//             SS_DEBUG((LM_ERROR, "[Sogou-Observer, reqid %x, total time %dms, queue time:%dms, qo time:%dms, reply time:%dms, queue to qo:%dms, qo to reply:%dms]\n", _request->request_id, _request->time_eval(QUEUE_BEGIN, QO_END), _request->time_eval(QUEUE_BEGIN, QUEUE_END),  _request->time_eval(QO_BEGIN, QO_END), _request->time_eval(REPLY_BEGIN, REPLY_END), _request->time_eval(QUEUE_END, QO_BEGIN), _request->time_eval(QO_END, REPLY_BEGIN)));
// 
//             delete _request;
//         }
//     }
//     return 0;
// }

// std::vector<std::string> reply_manager::synsets_{"norm", "porn", "sexy", "chart", "qr", "link", "blood", "nausea", "blood"};
std::vector<std::string> reply_manager::synsets_{"norm", "porn", "sexy", "chart", "qr", "link", "pad", "ad", "nausea", "blood"};

int reply_manager::svc()
{
    //char buff[32];
    Json::FastWriter fastWriter;

    while(active)
    {
        Json::Value root;
        /// pad to at least 500 elements
        for(int i = 0; i < 500; ++i)
        {
            root["pad"].append(0.0);
        }
        Json::Value prob;
        qo_request *_request = (qo_request *)_list.get_from_head();
        if(_request != NULL)
        {
            _request->mark_time(REPLY_BEGIN);
            
            for(int i = 0; i < _request->num; i++)
            {
                root["request_id"] = _request->request_ids[i];
                root["cid"] = synsets_[_request->predict_results[i]];
                // for (int j = 0; j < j; ++j)
                //     prob.append(static_cast<int>(_request->predict_probs[i * 10 + j] * 100));
                char buf[32];
                /**
                for (int j = 0; j < 9; ++j)
                {
                    // prob[synsets_[j]] = static_cast<int>(_request->predict_probs[i * 10 + j] * 100);
                    // prob[synsets_[j]] = _request->predict_probs[i * 10 + j];
                    sprintf(buf, "%.4f", _request->predict_probs[i * 10 + j]);
                    prob[synsets_[j]] = buf;
                }
                */
                for (int j = 0; j < 10; ++j)
                {
                    // prob[synsets_[j]] = static_cast<int>(_request->predict_probs[i * 10 + j] * 100);
                    // prob[synsets_[j]] = _request->predict_probs[i * 10 + j];
                    sprintf(buf, "%.4f", _request->predict_probs[i * 11 + j]);
                    prob[synsets_[j]] = buf;
                }
                root["prob"] = prob;
                std::string json_res = fastWriter.write(root);

                // return the head first
                uint32_t res_size = 2 * sizeof(uint32_t) + json_res.size();
                env_header res_head = {0, 0};
                res_head.magic_num = uint32_t(_request->request_ids[i]);
                res_head.len = res_size;
                int lh = send(_request->fds[i], (void *)&res_head, sizeof(env_header), 0);
                // fprintf(stderr, "total send res_head len: %d\n", lh);
                if(lh <= 0)
                {
                    fprintf(stderr, "reply request head %x error.\n", root["request_id"].asInt());
                }

                LowQualityFeatureRequestResult * res = (LowQualityFeatureRequestResult*)malloc(res_size);
                if (res == NULL)
                {
                    fprintf(stderr, "malloc LowQualityFeatureRequestResult error.\n");
                }
                else
                {
                    res->req_id = res_head.magic_num;
                    res->req_len = (uint32_t)json_res.size();
                    memcpy((void*)res->req_cont, (const void*)json_res.c_str(), json_res.size());
                    // send back the feature vectors
                    // int l = send(_request->fds[i], (void*)json_res.c_str(), json_res.size(), 0);
                    int l = send(_request->fds[i], (void*)res, res_size, 0);
                    if(l <= 0)
                    {
                        fprintf(stderr, "reply request %x error.\n", root["request_id"].asInt());
                    }
                    free(res);
                }
            }

            _request->mark_time(REPLY_END);

            // SS_DEBUG((LM_ERROR, "[Sogou-Observer, reqid %x, total time %dms, queue time:%dms, qo time:%dms, reply time:%dms, queue to qo:%dms, qo to reply:%dms]\n", _request->request_id, _request->time_eval(QUEUE_BEGIN, QO_END), _request->time_eval(QUEUE_BEGIN, QUEUE_END),  _request->time_eval(QO_BEGIN, QO_END), _request->time_eval(REPLY_BEGIN, REPLY_END), _request->time_eval(QUEUE_END, QO_BEGIN), _request->time_eval(QO_END, REPLY_BEGIN)));

            delete _request;
        }
    }
    return 0;
}


