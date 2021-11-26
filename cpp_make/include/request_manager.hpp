#ifndef REQUEST_MANAGET_HPP
#define REQUEST_MANAGET_HPP

#include "base_task.hpp"
#include "request.hpp"
#include "configuration.hpp"
#include "wait_list.hpp"
#include <list>
#include <vector>
#include <FreeImage.h>
class request_manager : public base_task
{
    public:
        
        static request_manager *Instance()
        {
            static request_manager r;
            return &r;
        }

        int open(client_configuration *conf)
        {
            int num = conf->reqmng_num;
            pthread_mutex_init(&qo_mutex, NULL);
            FreeImage_Initialise();
            
            return base_task::open(num);
        }

        int close()
        {
            active = false;
            pthread_mutex_destroy(&qo_mutex);
            base_task::close();

            FreeImage_DeInitialise();
            return 0;
        }

        ~request_manager()
        {
        }

        //void insert_request(void *r);
    public:
        std::list<void *> _qo_request;  //to qo
        pthread_mutex_t qo_mutex;

    private:
        int svc();
        cv::Mat bitMap2Mat(FIBITMAP* fiBmp, const FREE_IMAGE_FORMAT &fif);
        int gif2Mat(char* data, size_t dataSize, cv::Mat& singleImg);


};

#endif
