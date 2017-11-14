#pragma once

#ifndef __EXTRACT_FEATURES_FROM_IMAGES_H
#define __EXTRACT_FEATURES_FROM_IMAGES_H

#include "opencv2/opencv.hpp"
using namespace cv;

void detect_save_display(Mat motorcycle_frameame);
string generate_filename(string filename, int filenum);
int image_processing(String input_filename);
int video_processing(String input_filename);
string save_roi_file(Mat motorcycle_roi, int filenum);
vector<Point> check_correct_tl_br(Point tl, Point br);
vector<int> check_correct_w_h(Point tl_rect_roi, int width, int height);

#endif //!__EXTRACT_FEATURES_FROM_IMAGES_H
