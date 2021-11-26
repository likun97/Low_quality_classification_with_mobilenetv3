#ifndef QUERY_CORE_CLIENT_CONFIGURATION_HPP
#define QUERY_CORE_CLIENT_CONFIGURATION_HPP

#include <string>
#include <vector>

class client_configuration {
	public:
        static client_configuration* instance()
        {
            static client_configuration config;
            return &config;
        }
		int open(const char * filename, const char * key = "");
		int readString(char const * section, char const * config,std::string &str);
		//unsigned int get_max_docs(){return max_docs;}
	protected:
		int readParameter();
		int readThreadNumber();
		int readDataBase();
		int readDataBaseOptions();
		int readTestParameter();
		int readRankOptions();
		int load_dev_conf(const char* filename);

	public:

        unsigned int listen_port;
        unsigned int device_num;
        unsigned int cls_nums_porn;
        unsigned int cls_nums_lq;
        unsigned int reply_num;
        unsigned int reqmng_num;
        unsigned int receiver_num;
        unsigned int lq_thread_num;
        std::string porn_engine;
        std::string lq_engine;

        unsigned int qo_thread_num;
        unsigned int time_out;

        // unsigned int listen_port;
        // unsigned int device_num;
        // unsigned int mxnet_num;
        // unsigned int caffe_num;
        // unsigned int reply_num;
        // unsigned int reqmng_num;
        // unsigned int receiver_num;
        // std::string mxnet_json;
        // std::string mxnet_para;
        // std::string caffe_binary;
        // std::string caffe_proto;
        // std::string caffe_engine;
        // unsigned int qo_thread_num;
        // unsigned int time_out;

	protected:
		std::string m_filename;
		std::string m_keyname;

};

#endif//QUERY_CORE_CLIENT_CONFIGURATION_HPP
