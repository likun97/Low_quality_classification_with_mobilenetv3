#ifndef REPLY_MANAGET_HPP
#define REPLY_MANAGET_HPP

#include "base_task.hpp"
#include <vector>
#include "request.hpp"
#include "configuration.hpp"
#include "wait_list.hpp"
#include "jsoncpp/json/json.h"
//#include <pthread.h>
//#include <queue>
//#include <list>
class reply_manager : public base_task
{
    public:
        
        static reply_manager *Instance()
        {
            static reply_manager r;
            return &r;
        }

        int open(client_configuration *conf)
        {
            int num = conf->reply_num;
            return base_task::open(num);
        }

        int close()
        {
            base_task::close();
        }

        ~reply_manager()
        {
        }

        //void insert_request(void *r);
    private:
        int svc();
        static std::vector<std::string> synsets_;
};

#endif
