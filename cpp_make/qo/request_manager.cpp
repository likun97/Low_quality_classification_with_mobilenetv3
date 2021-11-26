/*************************************************************************
    > File Name: request_manager.cpp
    > Author:  
    > Created Time: Thu 22 Mar 2018 04:22:56 PM CST
 ************************************************************************/
#include "common.h"
#include "exception"
#include "request_manager.hpp"
#include "qo_manager.hpp"
#include "reply_manager.hpp"
#include <Platform/log.h>
//#include <cv.h>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>

cv::Mat load_im_from_memory(unsigned char* data, int size);
int request_manager::svc()
{
    cv::Mat newMat;
    // process one server_request once iteration !
    while(active)
    {
        server_request *r = (server_request *)_list.get_from_head();
        // process the infos from server_request, include the type, image data, image w & h
        if(r != NULL)
        {
            if(r->data){
                // newMat = load_im_from_memory((unsigned char*)r->data, r->img_len);
                LowQualityFeatureRequest* data = (LowQualityFeatureRequest*)r->data;
                try
                {
                    newMat = load_im_from_memory((unsigned char*)data->req_cont, r->img_len);
                }
                catch (...)
                {
                    free(r->data);
                    free(r);
                    continue;
                }
            }
            r->h = newMat.rows;
            r->w = newMat.cols;
            if(r->h <= small_pic_height && r->w <= small_pic_width)
            {
                r->type = small_pic;
            }
            else
            {
                r->type = big_pic;
            }

            if(r->h <= 0 || r->w <= 0)
            {
                fprintf(stderr, "img wrong format,req_id %x w %d h %d\n", r->request_id, r->w, r->h);
                qo_request  *qo = new qo_request(r);
                qo->num = 1;
                qo->request_ids.push_back(r->request_id);
                qo->fds.push_back(r->fd);
                qo->predict_results.push_back(0);
                qo->predict_probs.push_back(0);
                reply_manager::Instance()->put(qo);
                free(r->data);
                free(r);
                continue;
            }
            int type = r->type;
            // new qo_request
            // will send to the qo_manager !
            qo_request *qo = NULL;
            pthread_mutex_lock(&qo_mutex);
            std::list<void *>::iterator i;
            for(i = _qo_request.begin(); i != _qo_request.end(); i++)
            {
                //type check
                if(((qo_request *)*i)->num < batch_num[type]) //batch num
                {
                    qo = (qo_request *)*i;
                    break;
                }
            }
            if(qo == NULL)
            {
                qo = new qo_request(r);
                qo->mark_time(QUEUE_BEGIN);
                _qo_request.push_back(qo);
            }
            // store the received image & server_request to the new created qo_request, then used by qo_manager!
            // because this qo is push_backed into the _qo_request (pointer as the element) ....
            qo->insert_request(r, newMat);
            fprintf(stderr, "insert query %x into %x,w %d h %d\n", r->request_id, qo->request_id, r->w, r->h);
            if(qo->num == (batch_num[type]))  //batch num do it now? i.e. not wait for the time manager to call the qo_manager::put_big()
            {
                // is the request is processed at now, then the qo is ereased from _qo_request, such that time_manager will not process this request
                if(i != _qo_request.end())
                    _qo_request.erase(i);
                else
                    _qo_request.pop_back();
                qo->mark_time(QUEUE_END);
                //
                if(qo->type == big_pic)
                    qo_manager::Instance()->put_big(qo);
                else
                    qo_manager::Instance()->put_small(qo);
            }
            pthread_mutex_unlock(&qo_mutex);
            free(r->data);
            free(r);
        }
    }
    return 0;
}

cv::Mat load_im_from_memory(unsigned char* data, int size)
{
    FreeImage_Initialise();
    cv::Mat im;
    if (!data)
        return im;

    FIMEMORY* stream = FreeImage_OpenMemory(data, size);
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromMemory(stream, size);
    if (fif == FIF_UNKNOWN)
        return im;

    FIBITMAP* fib_frame = nullptr;
    FIBITMAP* fibmp = nullptr;
    if (fif == FIF_GIF)
    {
        FIMULTIBITMAP* fimbmp = FreeImage_LoadMultiBitmapFromMemory(fif, stream, GIF_DEFAULT);
        fibmp = FreeImage_LockPage(fimbmp, 0);
        if (FreeImage_GetBPP(fibmp) == 4 || FreeImage_GetColorType(fibmp) == FIC_PALETTE)
        {
            if (FreeImage_IsTransparent(fibmp)) {
                fib_frame = FreeImage_ConvertTo32Bits(fibmp);
                if (FreeImage_GetColorType(fib_frame) == FIC_RGB)
                    fib_frame = FreeImage_ConvertTo24Bits(fibmp);
            }
            else {
                fib_frame = FreeImage_ConvertTo24Bits(fibmp);
            }
        }
        FreeImage_UnlockPage(fimbmp, fibmp, false);
        FreeImage_CloseMultiBitmap(fimbmp, 0);
    }
    else
    {
        fibmp = FreeImage_LoadFromMemory(fif, stream);
        if (fif == FIF_PNG && FreeImage_IsTransparent(fibmp))
        {
            RGBQUAD   rgbQuad = { 255, 255, 255, 0 };
            // Replace the transparency with white
            FIBITMAP* fibmp_bak = fibmp; //backup pointer to fibitmap, we need release memory later
            fibmp = FreeImage_Composite(fibmp, false, &rgbQuad);
            FreeImage_Unload(fibmp_bak);
        }
        fib_frame = FreeImage_ConvertTo24Bits(fibmp);
        FreeImage_Unload(fibmp);
    }

    FREE_IMAGE_COLOR_TYPE color_type = FreeImage_GetColorType(fib_frame);
    int width = FreeImage_GetWidth(fib_frame);
    int height = FreeImage_GetHeight(fib_frame);
    int mat_type = 0;
    int channels = 0;

    if (color_type == FIC_RGBALPHA) {
        mat_type = CV_8UC4;
        channels = 4;
    }
    else if (color_type == FIC_RGB) {
        mat_type = CV_8UC3;
        channels = 3;
    }

    auto step = FreeImage_GetPitch(fib_frame);
    auto bits = FreeImage_GetBits(fib_frame);
    im = cv::Mat(height, width, mat_type, bits, step).clone();
    cv::flip(im, im, 0);
    if (channels == 4)
    {
        cv::Mat split_img[4];
        cv::split(im, split_img);
        cv::bitwise_not(split_img[3], split_img[3]);
        cv::add(split_img[3], split_img[0], split_img[0]);
        cv::add(split_img[3], split_img[1], split_img[1]);
        cv::add(split_img[3], split_img[2], split_img[2]);
        cv::merge(split_img, 3, im);
    }

    FreeImage_Unload(fib_frame);
    FreeImage_CloseMemory(stream);
    FreeImage_DeInitialise();

    if (im.empty())
    {
        std::vector<uchar> vdata;
        for (int i = 0; i < size; ++i) {
            vdata.push_back(data[i]);
        }
        im = cv::imdecode(vdata, cv::IMREAD_COLOR);
    }

    return im;
}
