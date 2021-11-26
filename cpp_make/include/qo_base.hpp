#ifndef QO_BASE_TASK_HPP
#define QO_BASE_TASK_HPP

#include <pthread.h>
#include "wait_list.hpp"
#define MAX_THREAD_NUM 32

class qo_base
{
    public:
        
        virtual ~qo_base()
        {
        }
        int open();
        int close();
        void put_big(void *);
        void put_small(void *);
    private:
        //int big_routine();
        //int small_routine();
    public:
        bool active;
    protected:
        wait_list_t big_list;
        wait_list_t small_list;
    private:
        //int thread_num;
        //pthread_t id[MAX_THREAD_NUM]; 
};

#endif
