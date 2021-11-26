#ifndef BASE_TASK_HPP
#define BASE_TASK_HPP

#include <pthread.h>
#include "wait_list.hpp"
#define MAX_THREAD_NUM 32

class base_task
{
    public:
        
        virtual ~base_task()
        {
        }
        int open(int threadnum);
        int close();
        void put(void *);
    private:
        static void *routine(void *arg);
        virtual int svc() = 0;
    public:
        bool active;
    protected:
        wait_list_t _list;
    private:
        int thread_num;
        pthread_t id[MAX_THREAD_NUM]; 
};

#endif
