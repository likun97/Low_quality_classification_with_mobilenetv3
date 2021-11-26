/*************************************************************************
    > File Name: base_task.cpp
    > Author:  
    > Created Time: Thu 22 Mar 2018 03:58:57 PM CST
 ************************************************************************/
#include <stdio.h>
#include "base_task.hpp"


int base_task::open(int threadnum)
{
    int i;
    thread_num = threadnum > MAX_THREAD_NUM ? MAX_THREAD_NUM : threadnum;
    active = true;
    for(i = 0; i < thread_num; i++)
        if(pthread_create(id + i, NULL, routine, this) != 0)
        {
            break;
        }
    if(i != thread_num)
    {
        fprintf(stderr, "%d thread created.\n", i);
        return -1;
    }

    return 0;
}

void *base_task::routine(void *arg)
{
    base_task *b = (base_task *)arg;
    return (void *)b->svc();
}

void base_task::put(void *node)
{
    _list.put(node);
}

int base_task::close()
{
    _list.flush();
    active = false;
    for(int i = 0; i < thread_num; i++)
        pthread_join(id[i], NULL);
    return 0;
}
