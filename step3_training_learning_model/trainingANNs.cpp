//Training Neural Networks

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <io.h>

using namespace cv;
using namespace std;
using namespace cv::ml;


/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 797
#define ATTRIBUTES_PER_SAMPLE 256
#define NUMBER_OF_TESTING_SAMPLES 796
#define NUMBER_OF_CLASSES 10

// N.B. classes are integer handwritten digits in range 0-9

/******************************************************************************/
// loads the sample database from file (which is a CSV text file)
int read_data_from_csv(const char* filename, Mat data, int n_samples, int n_samples_attributes)
{
	float tmpf;
	// if we can't read the input file then return 0
	FILE* f = fopen(filename, "r");
	if (!f)
	{
		printf("ERROR: cannot read file %s\n", filename);
		return 0; // all not OK
	}

	// for each sample in the file
	for (int line = 0; line < n_samples; line++)
	{

		// for each attribute on the line in the file
		for (int attribute = 0; attribute < n_samples_attributes; attribute++)
		{

			// first 256 elements (0-255) in each line are the attributes
			fscanf_s(f, "%f,", &tmpf);
			data.at<float>(line, attribute) = tmpf;
		}
		fscanf_s(f, "\n");
	}

	fclose(f);

	return 1; // all OK
}

// This function reads data and responses from the file <filename>
static bool
read_num_class_data(const string& filename, int var_count,
	Mat* _data, Mat* _responses)
{
	const int M = 1024;
	char buf[M + 2];

	Mat el_ptr(1, var_count, CV_32F);
	int i;
	vector<int> responses;

	_data->release();
	_responses->release();

	FILE* f = fopen(filename.c_str(), "rt");
	if (!f)
	{
		cout << "Could not read the database " << filename << endl;
		return false;
	}

	for (;;)
	{
		char* ptr;
		if (!fgets(buf, M, f) || !strchr(buf, ','))
			break;
		responses.push_back((int)buf[0]);
		ptr = buf + 2;
		for (i = 0; i < var_count; i++)
		{
			int n = 0;
			sscanf(ptr, "%f%n", &el_ptr.at<float>(i), &n);
			ptr += n + 1;
		}
		if (i < var_count)
			break;
		_data->push_back(el_ptr);
	}
	fclose(f);
	Mat(responses).copyTo(*_responses);

	cout << "The database " << filename << " is loaded.\n";

	return true;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
	const Mat& data, const Mat& responses,
	int ntrain_samples,
	const string& filename_to_save)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	Ptr<ANN_MLP> test_model = load_classifier<ANN_MLP>(filename_to_save);
	if (test_model.empty())
		return;

	// compute prediction error on train and test data
	int correct = 0;
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);
		Mat predict(1, 10, CV_32F);
		float r = test_model->predict(sample, predict);

		float max = predict.at<float>(0, 0);
		int max_i = 0;
		for (int j = 1; j<10; j++)
		{
			if (predict.at<float>(0, j) > max)
			{
				max = predict.at<float>(0, j);
				max_i = j;
			}
		}

		if (responses.at<float>(i, max_i) == 1.0)
		{
			correct++;
		}
	}

	printf("CORRECT = %.2f\n", correct / (float)nsamples_all);
}


static int build_mlp_classifier(const string& data_in_filename,
	const string& data_out_filename,
	const string& filename_to_save,
	const string& filename_to_load,
	int samples,
	int in_attributes,
	int out_attributes)
{
	Mat data(samples, in_attributes, CV_32F);
	Mat responses(samples, out_attributes, CV_32F);

	bool ok1 = read_data_from_csv(data_in_filename.c_str(), data, samples, in_attributes);
	bool ok2 = read_data_from_csv(data_out_filename.c_str(), responses, samples, out_attributes);
	if (!ok1 || !ok2)
		return -1;

	Ptr<ANN_MLP> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*1.0);

	// Create or load MLP classifier
	if (!filename_to_load.empty())
	{
		model = load_classifier<ANN_MLP>(filename_to_load);
		if (model.empty())
			return false;
		ntrain_samples = 0;
	}
	else
	{
		// 1. unroll the responses

		Mat train_data = data.rowRange(0, ntrain_samples);
		Mat train_responses = responses.rowRange(0, ntrain_samples);

		// 2. train classifier
		int layer_sz[] = { in_attributes, 15, out_attributes };
		int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
		Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

		int method_RPROP = 1;		//Training method is RPROP = faster backpropagation
		double method_param = 0.001;	//
		int max_iter = 3000;

		Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

		cout << "Training the classifier (may take a few minutes)...\n";
		model = ANN_MLP::create();
		/*
		ANN_MLP::Params p(layer_sizes, ANN_MLP::SIGMOID_SYM, 0, 0, TC(max_iter, 0), method, method_param);
		model->setParams(p);
		*/
		model->setLayerSizes(layer_sizes);
		model->setTrainMethod(method_RPROP, method_param);
		model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
		model->setTermCriteria(TC(max_iter, 0));


		model->train(tdata);
		cout << endl;

		if (!filename_to_save.empty())
		{
			model->save(filename_to_save);
		}
	}

	test_and_save_classifier(model, data, responses, ntrain_samples, filename_to_save);

	return true;
}


/******************************************************************************/

int main(int argc, char** argv)
{
	string filename_to_save = "";
	string filename_to_load = "";
	string data_in_filename = "../data/letter-recognition.data";
	string data_out_filename = "../data/letter-recognition.data";
	int method = 0;
	int samples = 0;
	int in_attributes = 0;
	int out_attributes = 0;

	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-in") == 0) // flag "-data letter_recognition.xml"
		{
			i++;
			data_in_filename = argv[i];
		}
		if (strcmp(argv[i], "-out") == 0) // flag "-data letter_recognition.xml"
		{
			i++;
			data_out_filename = argv[i];
		}
		else if (strcmp(argv[i], "-save") == 0) // flag "-save filename.xml"
		{
			i++;
			filename_to_save = argv[i];
		}
		else if (strcmp(argv[i], "-load") == 0) // flag "-load filename.xml"
		{
			i++;
			filename_to_load = argv[i];
		}
		else if (strcmp(argv[i], "-samples") == 0) // flag "-load filename.xml"
		{
			i++;
			samples = atoi(argv[i]);
		}
		else if (strcmp(argv[i], "-in_attributes") == 0) // flag "-load filename.xml"
		{
			i++;
			in_attributes = atoi(argv[i]);
		}
		else if (strcmp(argv[i], "-out_attributes") == 0) // flag "-load filename.xml"
		{
			i++;
			out_attributes = atoi(argv[i]);
		}
	}

	printf("argv1: %s, argv2: %s, argv3: %s, argv4: %s, argv5: %d, argv6: %d, argv7: %d\n", data_in_filename.c_str(), data_out_filename.c_str(), filename_to_save.c_str(), filename_to_load.c_str(), samples, in_attributes, out_attributes);

	build_mlp_classifier(data_in_filename, data_out_filename, filename_to_save, filename_to_load, samples, in_attributes, out_attributes);

	return -1;
}