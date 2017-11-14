//Training Neural Networks

#include "stdafx.h"
#include "trainingANNs.h"

#include <opencv2/opencv.hpp>
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
#define ATTRIBUTES_PER_SAMPLE 3780
#define NUMBER_OF_TESTING_SAMPLES 796
#define NUMBER_OF_CLASSES 3


/*Functions Header*/
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


/******************************************************************************/
// loads the sample database from file (which is a CSV text file)	==> CSV is Comma Separate Value
int read_data_from_file_training(const char* filename, Mat data, int n_samples, int n_samples_attributes)
{
	float read_value_temp;	//For storing each value from CSV file
	// if we can't read the input file then return 0
	FILE* fstream = fopen(filename, "r");
	if (!fstream)
	{
		printf("ERROR: cannot read file : %s\n", filename);
		return 0; // Cannot read file.
	}

	// for each sample in the file
	for (int line = 0; line < n_samples; line++)
	{

		// for each attribute on the line in the file
		for (int attribute = 0; attribute < n_samples_attributes; attribute++)
		{

			// first 3780 elements (0-3780) in each line are the attributes
			fscanf_s(fstream, "%f,", &read_value_temp);
			data.at<float>(line, attribute) = read_value_temp;
		}
		fscanf_s(fstream, "\n");
	}

	fclose(fstream);

	return 1; // Reading value from file process is OK.
}

// This function reads data and responses from the file <filename>
static bool read_num_class_data(const string& filename, int var_count, Mat* _data, Mat* _responses)
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

template<typename T> static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded\n";

	return model;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static int build_mlp_classifier(const string& data_in_filename, 
	const string& data_out_filename, 
	const string& filename_to_save, 
	const string& filename_to_load, 
	int n_samples, 
	int in_attributes, 
	int out_attributes)
{

	Mat data(n_samples, in_attributes, CV_32F);
	Mat responses(n_samples, out_attributes, CV_32F);

	//Reading data from file both of input and output file
	bool input_file_load_status = read_data_from_file_training(data_in_filename.c_str(), data, n_samples, in_attributes);
	bool output_file_load_status = read_data_from_file_training(data_out_filename.c_str(), responses, n_samples, out_attributes);
	if (!input_file_load_status || !output_file_load_status) {	//Cannot Load input or output file
		return -1;
	}

	Ptr<ANN_MLP> model;
	
	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*1.0);

	// Load MLP classifier for re-training it.
	if (!filename_to_load.empty())
	{
		model = load_classifier<ANN_MLP>(filename_to_load);
		if (model.empty())
			cout << "Cannot Load ANNs Model for re-training." << endl;
			return false;
	}
	
	//Trainig Process
	// 1. unroll the responses and data
	Mat train_data = data.rowRange(0, ntrain_samples);
	Mat train_responses = responses.rowRange(0, ntrain_samples);


	// 2. Adjust the classifier settings
	int nlayers = 3;
	vector<int> layer_Size = { in_attributes, 2, out_attributes };	
	/*
	Each value represent to number of neuron in each layer
	1.First is Input Layer.
	2.Middles is Hidden Layer.
	3.Last is Output Layer
	*/
	int method_RPROP = 1;		//Training method is RPROP = faster backpropagation
	double method_param = 0.001;	//
	int max_iter = 3000;

	//Training Data 
	Ptr<TrainData> trainData = TrainData::create(train_data, ROW_SAMPLE, train_responses);

	model = ANN_MLP::create();
	model->setLayerSizes(layer_Size);
	model->setTrainMethod(method_RPROP, method_param);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
	model->setTermCriteria(TC(max_iter, 0));

	//3.Train the network.
	cout << "Training the classifier (may take a few minutes)..." << endl;;
	model->train(trainData);

	//Save trained model
	if (!filename_to_save.empty())
	{
		model->save(filename_to_save);
	}

	return true;
}

/******************************************************************************/
void Training_ANNs_display(void) {
	cout << "***********************************************************************************" << endl;
	cout << "    _____          _       _                      _    _   _ _   _        " << endl;
	cout << "   |_   _| __ __ _(_)_ __ (_)_ __   __ _   _     / \\  | \\ | | \\ | |___    " << endl;
	cout << "     | || '__/ _` | | '_ \\| | '_ \\ / _` | (_)   / _ \\ |  \\| |  \\| / __|   " << endl;
	cout << "     | || | | (_| | | | | | | | | | (_| |  _   / ___ \\| |\\  | |\\  \\__ \\   " << endl;
	cout << "     |_||_|  \\__,_|_|_| |_|_|_| |_|\\__, | (_) /_/   \\_\\_| \\_|_| \\_|___/   " << endl;
	cout << "                                   |___/                                  " << endl;
	cout << "***********************************************************************************" << endl;
}