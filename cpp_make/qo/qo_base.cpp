/*************************************************************************
    > File Name: base_task.cpp
    > Author:  
    > Created Time: Thu 22 Mar 2018 03:58:57 PM CST
 ************************************************************************/
#include <stdio.h>
#include "qo_base.hpp"


int qo_base::open()
{
    return 0;
}

/*
void *base_task::routine(void *arg)
{
    base_task *b = (base_task *)arg;
    return (void *)b->svc();
}
*/

void qo_base::put_big(void *node)
{
    big_list.put(node);
}

void qo_base::put_small(void *node)
{
    small_list.put(node);
}

int qo_base::close()
{
    small_list.flush();
    big_list.flush();
    return 0;
}
