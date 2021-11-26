#ifndef TIME_MANAGET_HPP
#define TIME_MANAGET_HPP

#include "base_task.hpp"
#include "request.hpp"
#include "request_manager.hpp"
class time_manager : public base_task
{
    public:
        
        static time_manager *Instance()
        {
            static time_manager m;
            return &m;
        }

        int open(request_manager *r, int num = 1)
        {
            _request_manager = r;
            return base_task::open(num);
        }

        void set_timeout(int time_out)
        {
            _timeout = time_out;
        }

        ~time_manager()
        {
        }

        //void insert_request(void *r);
    private:
        int svc();
        request_manager *_request_manager;
        int _timeout;
};

#endif
