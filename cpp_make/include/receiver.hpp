#ifndef RECEIVER_HPP
#define RECEIVER_HPP

#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include  <sys/socket.h>
#include  <netinet/in.h>
#include  <arpa/inet.h>
#include <fcntl.h>
#include  <unistd.h>
#include <sys/epoll.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

#include "request_manager.hpp"
//#include "result_manager.hpp"

#define MAXEPOLLSIZE 1000
#define MAXEPOLLNUM 30

class receiver
{
    public:
    private:
        bool receive_active;
        pthread_t receiver_thread[MAXEPOLLNUM];
        int listen_fd;
        int fd_num, curfds;
        int port;
        int thread_num;
        //struct epoll_event events[MAXEPOLLSIZE];

    public:
        static receiver *Instance()
        {
            static receiver r;
            return &r;
        }

        int open(client_configuration *conf)
        {
            port = conf->listen_port;
            thread_num = conf->receiver_num > MAXEPOLLSIZE ? MAXEPOLLSIZE : conf->receiver_num;
            init(port, thread_num);
            return 0;
        }

        int close()
        {
            receive_active = false;
            for(int i = 0; i < thread_num; i++)
            {
                pthread_join(receiver_thread[i], NULL);
            }
            return 0;
        }
        /*
        receiver(int port = 9025, int thread_num = 1)
        {
            init(port, thread_num);
        }
        ~receiver()
        {
            receive_active = false;
            pthread_join(receiver_thread, NULL);
        }
        */

    private:
        int setbuff(int fd)
        {
            int options = 3 * 1024 * 1024;
            setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &options, sizeof(int));
            setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &options, sizeof(int));
            return 0;
        }
        int setnonblock(int fd)
        {
            if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFD, 0)|O_NONBLOCK) == -1) {
                fprintf(stderr, "setnonblock error!\n");
                return -1;
            }
            return 0;
        }
        int init(int port, int threadnum)
        {
            fprintf(stderr, "init receiver...\n");

            struct sockaddr_in servaddr;
            bzero(&servaddr, sizeof(sockaddr_in));
            servaddr.sin_family = AF_INET;
            servaddr.sin_addr.s_addr = htonl (INADDR_ANY);
            servaddr.sin_port = htons (port);

            if((listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
            {
                fprintf(stderr, "create listen fd fail!\n");
                return -1;
            }

            int opt = 1;
            setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            //setsockopt(listen_fd, SOL_SOCKET, , &opt, sizeof(opt));

            if (setnonblock(listen_fd) < 0)
            {
                fprintf(stderr, "setnonblock error!\n");
                return -1;
            }

            if (::bind(listen_fd, (struct sockaddr *) &servaddr, sizeof(struct sockaddr)) == -1)
            {
                fprintf(stderr, "sock bind error!\n");
                return -1;
            }

            if (listen(listen_fd, 1024) == -1)
            {
                fprintf(stderr, "sock listen error!\n");
                return -1;
            }

            curfds = 1;

            receive_active = true;
            for(int i = 0; i < threadnum; i++)
            {
                if (pthread_create(&receiver_thread[i], NULL, receiver_routine, this)) 
                return -1;
            }
            return 0;
        }
    private:
        static void* receiver_routine(void *arg)
        {
            return (void *)((receiver *)arg)->receive();
        }

        int receive()
        {
            int nfds;
            int epoll_fd = epoll_create(MAXEPOLLSIZE);
			struct epoll_event ev;
            ev.events = EPOLLIN;// | EPOLLET;
            ev.data.fd = listen_fd;
            if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev) < 0)
            {
                fprintf(stderr, "epoll_ctl error!\n");
                return -1;
            }
            struct epoll_event events[MAXEPOLLSIZE];
            fprintf(stderr, "begin loop\n");
            while(receive_active)
            {

                nfds = epoll_wait(epoll_fd, events, MAXEPOLLSIZE, -1);
                if (nfds == -1)
                {
                    fprintf(stderr, "epoll_wait error!\n");
                    continue;
                }
                for(int i = 0; i < nfds; i++)
                {
                    if(events[i].data.fd == listen_fd && (events[i].events & EPOLLIN))
                    {
                        socklen_t sock_len = sizeof(struct sockaddr_in);
                        int connfd = accept(listen_fd, NULL, &sock_len);
                        if(connfd < 0)
                        {
                            fprintf(stderr, "accept error.\n");
                            continue;
                        }
                        fprintf(stderr, "a new connection.\n");
                        //setnonblock(connfd);    //conflict with MSG_WAITALL
                        setbuff(connfd);
                        //struct timeval timeout;
                        //timeout.tv_sec = 0;
                        //timeout.tv_usec = 100 * 1000; //recv timeout!!!
                        //do we have to do this???
                        //setsockopt(connfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
                        //setsockopt(connfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
						
                        struct epoll_event ev;
                        ev.events = EPOLLIN;// | EPOLLET;
                        ev.data.fd = connfd;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, connfd, &ev) < 0)
                        {
                            fprintf(stderr, "epoll_ctl error.\n");
                            continue;
                        }
                        curfds++;
                    }
                    else if(events[i].events & EPOLLIN)
                    {
                        env_header header = {0, 0};
                        
                        int fd = events[i].data.fd;
                        int recv_len = 0;

                        if((recv_len = recvfrom(fd, (void *)(&header), sizeof(header), MSG_WAITALL, 0, 0)) <= 0)
                        {
                            if(recv_len == 0 && errno == EAGAIN)
                            {
                                continue;
                            }
                            fprintf(stderr, "recv env_header from sock %d error\n", fd);
							struct epoll_event ev;
							 ev.events = EPOLLIN;
							 ev.data.fd = fd;
							 epoll_ctl(epoll_fd,EPOLL_CTL_DEL,fd,&ev);
                             ::close(fd);
							 continue;
                        }
                        
                        //max pic len!!!!

                        int max_buff_size = 20 * 1024 * 1024;

                        if(header.len > max_buff_size || header.len <= 0)
                        {
							fprintf(stderr, "receive request error len %d, max len %d.\n", header.len, max_buff_size);
							struct epoll_event ev;
							 ev.events = EPOLLIN;
							 ev.data.fd = fd;
							 epoll_ctl(epoll_fd,EPOLL_CTL_DEL,fd,&ev);
                             ::close(fd);
                            continue;
                        }

                        // unsigned char *buff = (unsigned char *)malloc(header.len);
                        // if(buff == NULL)
                        //     continue;

                        LowQualityFeatureRequest * lqfr = (LowQualityFeatureRequest *) malloc(header.len);
                        if(lqfr == NULL)
                            continue;
						
                        // if((recv_len = recvfrom(fd, (void *)(buff), header.len, MSG_WAITALL, 0, 0)) <= 0 || recv_len != header.len)
                        // {
							// fprintf(stderr, "receive request error, recv_len %d.\n", recv_len);
							// struct epoll_event ev;
							// ev.events = EPOLLIN;
							// ev.data.fd = fd;
							// epoll_ctl(epoll_fd,EPOLL_CTL_DEL,fd,&ev);
                            // ::close(fd);
							// free(buff);
							// continue;
                        // }

                        if((recv_len = recvfrom(fd, (void *)(lqfr), header.len, MSG_WAITALL, 0, 0)) <= 0 || recv_len != header.len)
                        {
							fprintf(stderr, "receive request error, recv_len %d.\n", recv_len);
							struct epoll_event ev;
							ev.events = EPOLLIN;
							ev.data.fd = fd;
							epoll_ctl(epoll_fd,EPOLL_CTL_DEL,fd,&ev);
                            ::close(fd);
							free(lqfr);
							continue;
                        }
                        
                        server_request *r = (server_request *)malloc(sizeof(server_request));//(server_request *)buff;
                        // r->request_id = header.magic_num;
                        // r->fd = fd;
                        // r->img_len = header.len;
                        // r->data = buff;
                        r->request_id = lqfr->req_id;
                        r->fd = fd;
                        r->img_len = lqfr->req_len;
                        r->data = (unsigned char *)lqfr;
                        fprintf(stderr, "req_id %x, img_len %d\n", header.magic_num, header.len);

                        if((r->data) && (r->img_len > 0))
                        {
                            //memcpy(r->data, buff, r->img_len);
                            request_manager::Instance()->put(r);
                        }
                        else
                        {
                            if (r->data)
                                free(r->data);              // delete the malloced LowQualityFeatureRequest
                            fprintf(stderr, "mem alloc error or received img len error.\n");
                            free(r);
                        }
						//fprintf(stderr, "receive %d bytes, img len %d.\n", recv_len, r->img_len);
                    }
                }
            }
            return 0;
        }
};

#endif
