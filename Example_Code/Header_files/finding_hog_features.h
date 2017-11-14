#pragma once
#ifndef __FINDING_HOG_FEATURES_H
#define __FINDING_HOG_FEATURES_H

#include "opencv2\opencv.hpp"
using namespace cv;

void export_hog_value_to_file(vector<float> hog_value_motorcycle_roi, String input_filename_to_write);
void calculate_gradient_image(Mat motorcycle_roi);
vector<float> calculate_hog_image(Mat motorcycle_roi);
Mat get_hogdescriptor_visual(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);

#endif // !__FINDING_HOG_FEATURES_H
