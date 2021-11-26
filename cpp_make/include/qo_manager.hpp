#ifndef QO_MANAGET_HPP
#define QO_MANAGET_HPP
#define _GLIBCXX_USE_CXX11_ABI 0

#include "qo_base.hpp"
#include "request.hpp"
#include "configuration.hpp"
// #include "code/porn_predict_caffe.hpp"
// #include "code/porn_predict_mxnet.hpp"
#include "lqclassify.hpp"

#include "algorithm"

struct routine_info
{
    int type;
    int device_id;
    void *data;
};

class qo_manager : public qo_base
{
    public:

        static qo_manager *Instance()
        {
            static qo_manager q;
            return &q;
        }

        int open(client_configuration *conf);

        int close();

        qo_manager()
        {
        }

        ~qo_manager()
        {
        }

        static void *routine(void *arg);

    private:
        // std::string caffe_proto;
        // std::string caffe_binary;
        // std::string caffe_engine;
        // std::string mxnet_json;
        // std::string mxnet_para;

        std::string porn_engine;
        std::string lq_engine;

        int cls_nums_porn;
        int cls_nums_lq;
        int lq_thread_num;
        int device_num;
        routine_info info[MAX_THREAD_NUM];
        pthread_t thread_id[MAX_THREAD_NUM];
    private:
        int big_routine(int);
        int small_routine(int);
};

#endif
