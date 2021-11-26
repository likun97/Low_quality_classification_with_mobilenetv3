#include "configuration.hpp"
#include <string.h>
#include <sstream>
#include <stdio.h>
#include "config_map.hpp"
#include "sys.hpp"
//#include <service_log.hpp>

//static const char * const SummaryTypeName[] = { "Normal", "Index1", "Instant", "Fast", "Biz", "SohuBlog", "Offline","Meta" };
//static const char * const QueryTypeName[] = { "Local", "Index1", "Instant", "Fast", "Biz", "SohuBlog","Meta" };

int client_configuration::open(const char * filename, const char * key)
{
	m_filename = filename;
	m_keyname = key;
	if (readParameter())
    {
        fprintf(stderr, "[readParameter]\n");
        return -1;
    }
    /*
	if (readThreadNumber())
    {
        fprintf(stderr, "[readThreadNumber]\n");
        return -1;
    }
	if (readDataBase())
    {
        fprintf(stderr, "[readDataBase]\n");
        return -1;
    }
	if (readDataBaseOptions())
    {
        fprintf(stderr, "[readDataBaseOptions]\n");
        return -1;
    }
    readTestParameter();
	readRankOptions();
    */
	return 0;
}

int client_configuration::readString( char const * sectionname, char const * configname,std::string &str)
{
    /*
	config_map config;
	const char * value;
	std::string filename = m_filename;
	if(strcmp(sectionname,"/Rank")==0 || strcmp(sectionname,"/DataBase")==0)
		filename = m_configure_dev;

	if (config.import(filename.c_str()))
		_ERROR_RETURN(-1, "[config_map::import(%s)]", m_filename.c_str());

	std::string full_key(std::string(m_keyname.c_str()) + sectionname );
	config.set_section(full_key.c_str());

	if (config.get_value(configname, value))
		_ERROR_RETURN(-1, "load %s fail", configname);

	str = value;
    */
	return 0;
}

int client_configuration::readParameter()
{
	config_map config;
	if (config.import(m_filename.c_str()))
    {
        fprintf(stderr, "[config_map::import(%s)]\n", m_filename.c_str());
        return -1;
    }

	std::string full_key(std::string(m_keyname.c_str()) + "/Parameter");
	//_INFO("[readParameter]key:%s",full_key.c_str());
	config.set_section(full_key.c_str());
	const char *value;

	if (config.get_value("ListenPort", value))
    {
        fprintf(stderr, "[config_map::ListenPort para error!]\n");
        return -1;
    }
	if ((listen_port = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import ListenPort error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, ListenPort=%d\n", listen_port);
    }

    if (config.get_value("DeviceNum", value))
    {
        fprintf(stderr, "[config_map::DeviceNum para error!]\n");
        return -1;
    }
	if ((device_num = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import device_num error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, device_num=%d\n", device_num);
    }

    if (config.get_value("LqThreadNum", value))
    {
        fprintf(stderr, "[config_map::ThreadNum para error!]\n");
        return -1;
    }
	if ((lq_thread_num = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import lq_thread_num error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, lq_thread_num=%d\n", lq_thread_num);
    }

    if (config.get_value("ClsNumsPorn", value))
    {
        fprintf(stderr, "[config_map::ClsNumsPorn para error!]\n");
        return -1;
    }
	if ((cls_nums_porn = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import cls_nums_porn error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, cls_nums_porn=%d\n", cls_nums_porn);
    }

    if (config.get_value("ClsNumsLq", value))
    {
        fprintf(stderr, "[config_map::ClsNumsLq para error!]\n");
        return -1;
    }
	if ((cls_nums_lq = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import cls_nums_lq error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, cls_nums_lq=%d\n", cls_nums_lq);
    }

	if (config.get_value("ReceiverNum", value))
    {
        fprintf(stderr, "[config_map::ReceiverNum para error!]\n");
        return -1;
    }
	if ((receiver_num = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import receiver_num error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, receiver_num=%d\n", receiver_num);
    }

	if (config.get_value("ReqMngNum", value))
    {
        fprintf(stderr, "[config_map::ReqMngNum para error!]\n");
        return -1;
    }
	if ((reqmng_num = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import reqmng_num error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, reqmng_num=%d\n", reqmng_num);
    }

	if (config.get_value("ReplyNum", value))
    {
        fprintf(stderr, "[config_map::ReplyNum para error!]\n");
        return -1;
    }
	if ((reply_num = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import reply_num error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, reply_num=%d\n", reply_num);
    }

	if (config.get_value("TimeOut", value))
    {
        fprintf(stderr, "[config_map::TimeOut para error!]\n");
        return -1;
    }
	if ((time_out = atoi(value)) > (unsigned short)-1)
    {
        fprintf(stderr, "[config_map::import time_out error]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map, time_out=%d\n", time_out);
    }

	if (config.get_value("PornEngine", value))
    {
        fprintf(stderr, "[config_map::PornEngine para error!]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map:get PornEngine: %s\n", value);
    }
    porn_engine = value;

	if (config.get_value("LqEngine", value))
    {
        fprintf(stderr, "[config_map::LqEngine para error!]\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "config_map:get LqEngine: %s\n", value);
    }
    lq_engine = value;

    /*
	if (config.get_value("ThreadStackSize", value))
    {
        fprintf(stderr, "[config_map::import(%s)]\n", m_filename.c_str());
        return -1;
    }
	thread_stack_size = atoi(value) * (1 << 10);
*/
	return 0;
}

int client_configuration::readThreadNumber()
{
    /*
	config_map config;
	if (config.import(m_filename.c_str()))
		_ERROR_RETURN(-1, "[config_map::import(%s)]", m_filename.c_str());

	std::string full_key(std::string(m_keyname.c_str()) + "/ThreadNumber");
	_INFO("[readThreadNumber]key:%s",full_key.c_str());
	config.set_section(full_key.c_str());
	const char *value;

	//if (config.get_value("ClientPreprocess", value))
	//	_ERROR_RETURN(-1,"load ClientPreprocess fail");
	//client_preprocess_num = atoi(value);

	if (config.get_value("ClientSearch", value))
		_ERROR_RETURN(-1,"load ClientSearch fail");
	client_search_num = atoi(value);

	if (config.get_value("ClientIO", value))
		_ERROR_RETURN(-1,"load ClientIO fail");
	client_io_num = atoi(value);
    */

	return 0;
}

int client_configuration::readDataBase()
{
    /*
	config_map config;
	if (config.import(m_filename.c_str()))
		_ERROR_RETURN(-1, "[config_map::import(%s)]", m_filename.c_str());

	std::string full_key(std::string(m_keyname.c_str()) + "/DataBase");
	_INFO("[readDataBase]key:%s",full_key.c_str());
	config.set_section(full_key.c_str());
	const char *value;

	if(config.get_value("QueryDevConf",value))
		_ERROR_RETURN(-1,"load QueryDevConf fail");
	m_configure_dev = value;
	if(load_dev_conf(value) != 0)
		_ERROR_RETURN(-1,"load_dev_conf() fail");

	if (config.get_value("DocInfoMemID", value))
		_ERROR_RETURN(-1,"load DocInfoMemID fail");
	docinfo_memid = atoi(value);

	if (config.get_value("CacheSize", value))
		_ERROR_RETURN(-1,"load CacheSize fail");
	cache_size = atoi(value);

	instidxshmkey = 0;
	if(config.get_value("InstIdxShmKey",value))
		_ERROR("load InstIdxShmKey fail,will be useless");
	else
		instidxshmkey = atoi(value);

	// added for new instant
	shmblocksize = 256;
	if(config.get_value("ShmBlockSize",value))
		_ERROR("load ShmBlockSize fail,will use default value 256M");
	else
		shmblocksize = atoi(value);

	instidxdocshmbase = 0;
	if(config.get_value("InstIdxDocShmBase",value))
		_ERROR("load InstIdxDocShmBase fail,will be useless");
	else
		instidxdocshmbase = atoi(value);

	instidxdocshmnum = 0;
	if(config.get_value("InstIdxDocShmNum",value))
		_ERROR("load InstIdxDocShmNum fail,will be useless");
	else
		instidxdocshmnum = atoi(value);

	instidxoccshmbase = 0;
	if(config.get_value("InstIdxOccShmBase",value))
		_ERROR("load InstIdxOccShmBase fail,will be useless");
	else
		instidxoccshmbase = atoi(value);

	instidxoccshmnum = 0;
	if(config.get_value("InstIdxOccShmNum",value))
		_ERROR("load InstIdxOccShmNum fail,will be useless");
	else
		instidxoccshmnum = atoi(value);

	instforwardshmid = 0;
	if(config.get_value("InstForwardShmKey",value))
		_ERROR("load InstForwardShmKey fail,will be useless");
	else
		instforwardshmid = atoi(value);
	// end for new instant

	if (config.get_value("TmpfsSize", value))
		_ERROR_RETURN(-1,"load TmpfsSize fail");
	tmpfs_size = atoi(value);

	if (config.get_value("Index", value))
		_ERROR_RETURN(-1,"load Index fail");
	index = value;

	if (config.get_value("DocInfo", value))
		_ERROR_RETURN(-1,"load DocInfo fail");
	docinfo = value;

	if (config.get_value("RemoteIndexServerPort", value)) {
		remote_index_server_port = 0;
	} else {
		remote_index_server_port = atoi(value);
	}

	userrank = "NONE";
	if(config.get_value("Userrank",value))
		_INFO("no configure item:Userrank,use default (NONE)");
	else
		userrank = value;

	normal_update = "";
	if(config.get_value("NormalUpdate",value))
		_INFO("no configure item:NormalUpdate,use default ()");
	else
		normal_update = value;

	pornfile = "NONE";
	if(config.get_value("porn_domain",value))
		_INFO("no configure item:porn_domain,use default (NONE)");
	else
		pornfile = value;

	forwardindexfile = "";
	if(config.get_value("ForwardIndexFile",value))
		_ERROR("no configure item:ForwardIndexFile, will load nothing");
	forwardindexfile = value;


	if_docinfo_load_disk = false;
	if (config.get_value("DocInfo_LoadType",value) == 0 )
		if(!strcmp(value,"disk")){
			if_docinfo_load_disk = true;
			fprintf(stderr,"DocInfo_LoadType = disk\n");
		}

	FIDX_loadtype = 0;
	if(config.get_value("ForwardIndex_LoadType",value) == 0 && !strcmp(value,"disk"))
		FIDX_loadtype = 1;
	_INFO("ForwardIndexFile load type is %s\n",FIDX_loadtype?"disk":"memory");

    */
	return 0;
}

int client_configuration::readDataBaseOptions()
{
    /*
	config_map config;
	if (config.import(m_filename.c_str()))
		_ERROR_RETURN(-1, "[config_map::import(%s)]", m_filename.c_str());

	std::string full_key(std::string(m_keyname.c_str()) + "/DataBaseOptions");
	_INFO("[readDataBaseOptions]key:%s",full_key.c_str());
	config.set_section(full_key.c_str());
	const char *value;
	unsigned int i;

	if (config.get_value("SummaryType", value))
		_ERROR_RETURN(-1,"load SummaryType fail");
	for (i=0; i<sizeof(SummaryTypeName)/sizeof(char *); ++i)
		if (strcasecmp(value, SummaryTypeName[i]) == 0)
			break;
	summary_type = i;

	if (config.get_value("QueryType", value))
		_ERROR_RETURN(-1,"load QueryType fail");
	for (i=0; i<sizeof(QueryTypeName)/sizeof(char *); ++i)
		if (strcasecmp(value, QueryTypeName[i]) == 0)
			break;
	query_type = i;
	//inst
	if (query_type == 2) {
		if (config.get_value("DBPrefix", value)) {
			_ERROR_RETURN(-1,"load DBPrefix fail");
		}
		db_prefix.assign(value);

		if (config.get_value("DocInfoPrefix", value)) {
			_ERROR_RETURN(-1,"load DocInfoPrefix fail");
		}
		docinfo_prefix.assign(value);

		if(config.get_value("ForwardPrefix", value)){
			_ERROR_RETURN(-1,"load ForwardPrefix fail");
		}
		forward_prefix.assign(value);

		if (config.get_value("InstTermList", value)) {
			_ERROR_RETURN(-1,"load InstTermList fail");
		}
		termlist_file.assign(value);

	}
	else {
		db_prefix.assign("");
		docinfo_prefix.assign("");
		termlist_file.assign("");
	}

	if (config.get_value("FixBegin", value))
		_ERROR_RETURN(-1,"load FixBegin fail");
	fix_begin = atoi(value);

	if (config.get_value("FixEnd", value))
		_ERROR_RETURN(-1,"load FixEnd fail");
	fix_end = atoi(value);

	bm25_n = 2000000000;

	if (config.get_value("RankFieldWeight", value))
		_ERROR_RETURN(-1,"load RankFieldWeight fail");
	std::istringstream stream(value);
	for (i=0; !stream.eof(); ++i) {
		rank_field_weight.push_back(0);
		stream >> rank_field_weight[i];
	}

    */

	return 0;
}

int client_configuration::readTestParameter() {
    /*
    config_map config;
    if (config.import(m_filename.c_str()))
        _ERROR_RETURN(-1, "[config_map::import(%s)]", m_filename.c_str());

    std::string full_key(std::string(m_keyname.c_str()) + "/Test");
    _INFO("[readDataBase]key:%s",full_key.c_str());
    config.set_section(full_key.c_str());
    const char *value;
    is_test = false;
    test_type = -1;

    if (config.get_value("TestDataFile", value)) {
        _ERROR("no config for TestDataFile, not test");
    }
    else {
        test_data_file = std::string(value);
        is_test = true;
        test_type = 0;
    }

    if (config.get_value("ResDataFile", value)) {
        _ERROR("no config for  ResDataFile, no test");
    }
    else {
        res_data_file = std::string(value);
        is_test = true;
        test_type = 1;
    }

    if (config.get_value("CallBackFile", value)) {
        _ERROR("no config for  CallBackFile, not callback test");
    }
    else {
        callback_file = std::string(value);
        is_test = true;
        test_type = 2;
    }

	if (config.get_value("TestPreloadNum", value)) {
		_ERROR("no config for TestPreloadNum, no test");
	}
	else {
		test_preload_cnt = atoi(value);
	}
    */
    return 0;
}

int client_configuration::readRankOptions()
{
    /*
	config_map config;
	if (config.import(m_filename.c_str()))
		_ERROR_RETURN(-1, "[config_map::import(%s)]", m_filename.c_str());

	std::string full_key(std::string(m_keyname.c_str()) + "/Rank");
	_INFO("[readRank]key:%s",full_key.c_str());
	config.set_section(full_key.c_str());
	const char *value;
	if (config.get_value("rank_mode", value)) {
		 _ERROR("no config for rank_mode");
	}
	else
	{
		rank_mode = std::string(value);
		 _INFO("[rank_mode:%s]",rank_mode.c_str());
	}
	*/
	return 0;
}

int client_configuration::load_dev_conf(const char* filename)
{
    /*
	config_map config;
	if(config.import(filename))
		_ERROR_RETURN(-1, "[config_map::import(%s)]", filename);
	std::string full_key(std::string(m_keyname.c_str()) + "/DataBase");
	config.set_section(full_key.c_str());
	const char *value;

	direct_io = 1;
	if(config.get_value("DirectIO", value))
		_INFO("load DirectIO fail, will use default value(1)");
	else
		direct_io = atoi(value);

	if (config.get_value("TFlagConf", value))
		_ERROR_RETURN(-1,"load TFlagConf fail");
	tflag_conf = value;

	if (config.get_value("DicTermPair", value))
		_ERROR_RETURN(-1,"load DicTermPair fail");
	dic_termpair = value;

	if(config.get_value("TermWeightModelParameter",value))
		_ERROR_RETURN(-1,"load TermWeightModelParameter fail");
	termWeight_model_parameter=value;

	if(config.get_value("AddEntityPath",value))
		_ERROR_RETURN(-1,"load add entity fail");
	add_entity_path=value;

	if(config.get_value("TermImportPath",value))
		_ERROR_RETURN(-1,"load term import path fail");
	term_import_model=value;

	if(config.get_value("StopDic",value))
		 _ERROR_RETURN(-1,"load stopdic path fail");
	stop_dic=value;

	if(config.get_value("CaseDeleDic",value))
		_ERROR_RETURN(-1,"load CaseDeleDic path fail");
	case_dele_dic=value;

	if(config.get_value("KCWeightPath",value))
		_ERROR_RETURN(-1,"load kcWeight model path fail");
	kc_weight_model=value;

	if(config.get_value("SolidSynonymPath",value))
		_ERROR_RETURN(-1,"load solid synonym path fail");
	solid_synonym=value;

	if (config.get_value("RemoteIndexServerAddr", value)) {
		remote_index_server_addrstring = "";
	} else {
		remote_index_server_addrstring = value;
	}
    */
	return 0;
}
