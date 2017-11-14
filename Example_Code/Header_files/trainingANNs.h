#pragma once
#ifndef __TRAINING_ANNS_H
#define __TRAINING_ANNS_H

#include "opencv2\opencv.hpp"
#include "ml.h"

using namespace ml;
using namespace cv;

int read_data_from_file_training(const char* filename, Mat data, int n_samples, int n_samples_attributes);
static bool read_num_class_data(const string& filename, int var_count, Mat* _data, Mat* _responses);
template<typename T> static Ptr<T> load_classifier(const string& filename_to_load);
inline TermCriteria TC(int iters, double eps);
static int build_mlp_classifier(const string& data_in_filename,
	const string& data_out_filename,
	const string& filename_to_save,
	const string& filename_to_load,
	int n_samples,
	int in_attributes,
	int out_attributes);
void Training_ANNs_display(void);
#endif // !__TRAINING_ANNS_H
