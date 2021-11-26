#ifndef REQUEST_HPP         
#define REQUEST_HPP         

#include "qo_common.h"
#include <stdio.h>
#include <sys/time.h>
#include <vector>
//#include <opencv/cv.h>
#include<opencv2/opencv.hpp>

// struct env_header{
//     int magic_num;
//     int len;
// };
struct env_header{
    int len;
    int magic_num;
};

struct request_header{
    unsigned int request_id;
    int img_len;
};

struct server_request : public request_header
{
    int fd;
    int type;
    int w;
    int h;
    //float porn_rate;
    unsigned char *data;
};

enum time_point
{
    QUEUE_BEGIN = 0,
    QUEUE_END,
    QO_BEGIN,
    QO_END,
    REPLY_BEGIN,
    REPLY_END,
    POINT_NUM
};

struct server_result{
    unsigned int request_id;
    int result;
    int prob;
};

struct LowQualityFeatureRequest
{
    uint32_t req_id;
    uint32_t req_len;
    char req_cont[0];
};

struct LowQualityFeatureRequestResult
{
    uint32_t req_id;
    uint32_t req_len;
    char req_cont[0];
};

struct qo_request
{
    unsigned int request_id; 
    int type;
    int num;
    //timeval time;
    //CvSize size[max_pic_batch_num];
    timeval time[POINT_NUM];
    std::vector<int> fds;               // the socket fds, see the reply_manager::send() for verification
    std::vector<int> request_ids;
    std::vector<cv::Mat> mats;
    std::vector<int> predict_results;
    std::vector<float> predict_probs;
    inline int time_eval(int begin, int end)
    {
        if(begin < 0 || end >= POINT_NUM || begin > end)
            return 0;
        return (time[end].tv_sec - time[begin].tv_sec) * 1000 + (time[end].tv_usec - time[begin].tv_usec) / 1000;
    }
    void mark_time(int point)
    {
        if(point <= 0 || point >= POINT_NUM)
            return;
        gettimeofday(&time[point], 0);
    }
    qo_request(server_request *r):num(0), type(r->type), request_id(r->request_id)
    {
        //mats.reserve(10);      //max pic to qo!!!!
        memset(time, 0, sizeof(time));
        gettimeofday(&time[QUEUE_BEGIN], 0);
    }

    int insert_request(server_request *r, cv::Mat &pic)
    {
        fds.push_back(r->fd);
        cv::Mat m;
        pic.copyTo(m);
        request_ids.push_back(r->request_id);
        mats.push_back(m); //do we have to clone?
        num++;
        return num - 1;
    }

    void makeup_request()
    {
        cv::Mat pic = mats[num - 1];
        int i;
        for(i = num; i < batch_num[type]; i++)
        {
            cv::Mat m;
            pic.copyTo(m);
            mats.push_back(m);
        }
        mark_time(QUEUE_END);
        fprintf(stderr, "make up to %d pics, num %d\n", i, num);
    }
};


#endif
