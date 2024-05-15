#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh = 0.5);

void draw_bbox_keypoints_line(cv::Mat& img_batch, std::vector<Detection>& res_batch);

void process_decode_ptr_host(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element, cv::Mat& img,
                             int count);

void cuda_decode(float* predict, int num_bboxes, float confidence_threshold, float* parray, int max_objects,
                 cudaStream_t stream);

void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);
