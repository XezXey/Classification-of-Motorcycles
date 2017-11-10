//Predicting the Neural Networks

#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <ml.hpp>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <io.h>


using namespace cv;
using namespace std;
using namespace ml;

// loads the sample database from file (which is a CSV text file)
int read_data_from_file(const char* filename, Mat data, int n_samples, int n_samples_attributes)
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

			// first 256 elements (0-255) in each line are the attributes
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
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void classifier_predict(const Ptr<StatModel>& model, const Mat& data, const Mat& responses, int ntrain_samples, int rdelta)
{
	Mat _responses = responses;
	int j,i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;
	float r;
	int true_positive = 0;
	cout << "nsamples_all = " << nsamples_all << endl;
	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);


		for (int j = 0; j < 3; j++) {
			r = model->predict(sample, responses);
			r = abs(1 - responses.at<float>(i, j)) <= 0.1 ? 1.f : 0.f;
			cout << "R per Responses : " << r << endl;
			//cout << "OK1 : " << ((r - 1) <= FLT_EPSILON) << endl;
			//cout << "OK2 : " << (r - _responses.at<float>(i, j) <= 0.1) << endl;
			bool OK1 = (abs(r - 1) <= FLT_EPSILON);
			bool OK2 = abs(r - _responses.at<float>(i, j) <= 0.1) ? true:false;
			cout << "OK1 : " << OK1 << endl;
			cout << "OK2 : " << OK2 << endl;

			if (OK1 && OK2) {
				true_positive++;
			}
		}
		if (i < ntrain_samples) {
			train_hr += r;
			test_hr += r;
		}


		cout << "Test_hr : " << test_hr << endl;
		test_hr /= nsamples_all - ntrain_samples;
		cout << "Test_hr : " << test_hr << endl;
		train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;
	}
	cout << "Result : " << responses << endl;
	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n", train_hr*100., test_hr*100.);
	printf("Predict : %d%%", true_positive * 100);

}


static int load_mlp_classifier(const string& data_in_filename,
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
	bool input_file_load_status = read_data_from_file(data_in_filename.c_str(), data, n_samples, in_attributes);
	bool output_file_load_status = read_data_from_file(data_out_filename.c_str(), responses, n_samples, out_attributes);
	if (!input_file_load_status || !output_file_load_status) {	//Cannot Load input or output file
		return -1;
	}

	Ptr<ANN_MLP> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*1.0);

	// Load MLP classifier
	if (!filename_to_load.empty())
	{
		model = load_classifier<ANN_MLP>(filename_to_load);
		classifier_predict(model, data, responses, ntrain_samples, 'A');
	}
	
	return true;
}


/******************************************************************************/

void Predict_ANNs_display(void) {
	cout << "***********************************************************************************" << endl;
	cout << "  ____		       _ _      _            _    _   _ _   _" << endl;
	cout << " |  _ \\ _  __ ___  __ | (_) ___| |_   _     / \\  | \\ | | \\ | |___ " << endl;
	cout << " | |_) |  '__/ _ \\ / _` | |/ __| __| (_)   / _ \\ |  \\| |  \\| / __| " << endl;
	cout << " |  __/|  | |  __/  (_| | | (__| |_   _   / ___ \\| |\\  | |\\  \\__ \\ " << endl;
	cout << " |_|   |_ |  \\___ |\\__,_|_|\\___|\\__| (_) /_/   \\_\\_| \\_|_| \\_|___/ " << endl;
	cout << "***********************************************************************************" << endl;
}

int main(int argc, char** argv)
{
	string filename_to_save = "";
	string filename_to_load = "";
	string data_in_filename = "";
	string data_out_filename = "";
	int method = 0;
	int samples = 0;
	int in_attributes = 0;
	int out_attributes = 0;

	Predict_ANNs_display();

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

	load_mlp_classifier(data_in_filename, data_out_filename, filename_to_save, filename_to_load, samples, in_attributes, out_attributes);

	return -1;
}