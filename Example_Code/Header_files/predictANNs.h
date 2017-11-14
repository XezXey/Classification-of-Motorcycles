#pragma once
#ifndef __PREDICT_ANNS_H
#define __PREDICT_ANNS_H

#include "opencv2\opencv.hpp"
#include <ml.hpp>

using namespace ml;
using namespace cv;

int read_data_from_file_predict(const char* filename, Mat data, int n_samples, int n_samples_attributes);
static bool read_num_class_data(const string& filename, int var_count, Mat* _data, Mat* _responses);
template<typename T> static Ptr<T> load_classifier(const string& filename_to_load);
inline TermCriteria TC(int iters, double eps);
static void classifier_predict(const Ptr<StatModel>& model, const Mat& data, const Mat& responses, int n_samples, int out_attributes);
static int load_mlp_classifier(const string& data_in_filename,
	const string& data_out_filename,
	const string& filename_to_save,
	const string& filename_to_load,
	int n_samples,
	int in_attributes,
	int out_attributes);
void Predict_ANNs_display(void);
#endif // !__PREDICT_ANNS_H
