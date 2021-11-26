/*************************************************************************
    > File Name: request_manager.cpp
    > Author:  
    > Created Time: Thu 22 Mar 2018 04:22:56 PM CST
 ************************************************************************/

#include "time_manager.hpp"
#include "qo_manager.hpp"
#include "request.hpp"
#include <Platform/log.h>

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

int time_manager::svc()
{

    while(active)
    {
        timeval timeout = {0, 0};
        timeout.tv_sec = 0;     //time_out from common.h,should bigger than 1000!!!
        timeout.tv_usec = 50 * 1000;
        ::select(1, NULL, NULL, NULL, &timeout);
        
        timeval tv;
        gettimeofday(&tv, NULL);
        pthread_mutex_lock(&_request_manager->qo_mutex);
        // process each request_manager
        // each requst is stored in request_manager::_qo_request as a list! now fetch this request and call qo_request::makeup_request()
        for(std::list<void *>::iterator i = _request_manager->_qo_request.begin(); i != _request_manager->_qo_request.end();)
        {
            qo_request *qo = (qo_request *)*i;
            if(((tv.tv_sec - qo->time[0].tv_sec) * 1000 + (tv.tv_usec - qo->time[0].tv_usec) / 1000) > _timeout)
            {
                i = _request_manager->_qo_request.erase(i);
                qo->makeup_request();
                //time out !!!
                //for(int i = 0; i < qo->num; i++)
                //{
                //    qo->predict_results.push_back(-1);
                //    qo->predict_probs.push_back(1);
                //}
                SS_DEBUG((LM_ERROR, "query time out, reqid %x, type %d, request num %d\n", qo->request_id, qo->type, qo->num));
                // if do the request at the recived time, then the qo_manager::put_big() is called in request_manager, and the qo_requst is ereased there exactly
                if(qo->type == big_pic)
                    qo_manager::Instance()->put_big(qo);
                else
                    qo_manager::Instance()->put_small(qo);
            }
            else
            {
                i++;
            }
        }
        pthread_mutex_unlock(&_request_manager->qo_mutex);
    }
    return 0;
}

