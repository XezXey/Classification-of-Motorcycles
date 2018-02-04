//Training Support Vector Machines

#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <io.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


using namespace cv;
using namespace std;
using namespace cv::ml;


/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 797
#define ATTRIBUTES_PER_SAMPLE 3780
#define NUMBER_OF_TESTING_SAMPLES 796
#define NUMBER_OF_CLASSES 3


/******************************************************************************/
// loads the sample database from file (which is a CSV text file)	==> CSV is Comma Separate Value
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
	//model->setTermCriteria(TC(max_iter, 0));
	//COUNT == 1
	//EPS == 2
	//COUNT + EPS == 3
	//In this program We using Stop condition is COUNT or reach the iterations round.
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
	//cout << data.size() << endl;
	Mat responses(n_samples, out_attributes, CV_32S);
	//cout << responses.size() << endl;

	//Reading data from file both of input and output file
	bool input_file_load_status = read_data_from_file(data_in_filename.c_str(), data, n_samples, in_attributes);
	bool output_file_load_status = read_data_from_file(data_out_filename.c_str(), responses, n_samples, out_attributes);
	if (!input_file_load_status || !output_file_load_status) {	//Cannot Load input or output file
		return -1;
	}

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*1.0);

	// Load MLP classifier for re-training it.
	Ptr<SVM> model_svm;

	if (!filename_to_load.empty())
	{
		model_svm = load_classifier<SVM>(filename_to_load);
		if (model_svm.empty()) {
			cout << "Cannot Load SVM Model for re-training." << endl;
			return false;
		}
		else cout << "Load Model for re-training Success." << endl;
	}

	else {	// 2. Adjust the classifier settings if cannot load
	

		int max_iter = 3000;
		model_svm = SVM::create();
		model_svm->setType(SVM::C_SVC);
		model_svm->setKernel(SVM::SIGMOID);
		model_svm->setTermCriteria(TC(max_iter, 0));

	}

	//Trainig Process
	// 1. unroll the responses and data
	Mat train_data = data.rowRange(0, ntrain_samples);
	//cout << "train_data : " << train_data << endl;
	Mat train_responses = responses.rowRange(0, ntrain_samples);
	//cout << "train_responses : " << train_responses << endl;
	//cout << "train_data : " << train_data << endl;

	//Training Data 
	Ptr<TrainData> trainData = TrainData::create(train_data, ROW_SAMPLE, train_responses);


	//3.Train the network.
<<<<<<< HEAD
	cout << "Training the SVM_Kernel=" << model_svm->getKernelType() << " classifier (may take a few minutes)..." << endl;;
	model_svm->train(train_data, ROW_SAMPLE, train_responses);
	//model_svm->trainAuto(train_data, ROW_SAMPLE, train_responses);
=======
	cout << "Training the classifier (may take a few minutes)..." << endl;;
	model_svm->train(train_data, ROW_SAMPLE, train_responses);
>>>>>>> parent of 526f166... Use autoTrain method and add some example

	//Save trained model
	if (!filename_to_save.empty())
	{
		model_svm->save(filename_to_save);
	}

	return true;
}

/******************************************************************************/
void Training_SVM_display(void) {
	cout << "***********************************************************************************" << endl;
	cout << "    _____          _       _                      _    _   _ _   _        " << endl;
	cout << "   |_   _| __ __ _(_)_ __ (_)_ __   __ _   _     / \\  | \\ | | \\ | |___    " << endl;
	cout << "     | || '__/ _` | | '_ \\| | '_ \\ / _` | (_)   / _ \\ |  \\| |  \\| / __|   " << endl;
	cout << "     | || | | (_| | | | | | | | | | (_| |  _   / ___ \\| |\\  | |\\  \\__ \\   " << endl;
	cout << "     |_||_|  \\__,_|_|_| |_|_|_| |_|\\__, | (_) /_/   \\_\\_| \\_|_| \\_|___/   " << endl;
	cout << "                                   |___/                                  " << endl;
	cout << "***********************************************************************************" << endl;
}

int main(int argc, char** argv)
{
	string filename_to_save = "";
	string filename_to_load = "";
	string data_in_filename = "";
	string data_out_filename = "";

	int method = 0;
	int n_samples = 0;
	int in_attributes = 0;
	int out_attributes = 0;

	Training_SVM_display();
	cout << endl << endl << "Parameter for Training the SVM" << endl;;
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-in") == 0) // flag "-in <input_csv_file>.txt"
		{
			i++;
			data_in_filename = argv[i];
			cout << "	Input_data_filename : " << data_in_filename << endl;
		}
		else if (strcmp(argv[i], "-out") == 0) // flag "-out <output_csv_file>.txt"
		{
			i++;
			data_out_filename = argv[i];
			cout << "	Output_data_filename : " << data_out_filename << endl;

		}
		else if (strcmp(argv[i], "-save") == 0) // flag "-save <model_name>.xml"
		{
			i++;
			filename_to_save = argv[i];
			cout << "	Save to filename : " << filename_to_save << endl;
		}
		else if (strcmp(argv[i], "-load") == 0) // flag "-load <model_name>.xml"
		{
			i++;
			filename_to_load = argv[i];
			cout << "	Load from filename : " << filename_to_load << endl;
		}
		else if (strcmp(argv[i], "-samples") == 0) // flag "-samples <number_of_samples>"
		{
			i++;
			n_samples = atoi(argv[i]);
			cout << "	Number of Samepls : " << n_samples << endl;
		}
		else if (strcmp(argv[i], "-in_attributes") == 0) // flag "-in_attributes <number_of_input_attributes_of_training_data>"
		{
			i++;
			in_attributes = atoi(argv[i]);
			cout << "	Number of Input's Attributes : " << in_attributes << endl;
		}
		else if (strcmp(argv[i], "-out_attributes") == 0) // flag "-out_attributes <number_of_output_attributes_of_training_data>"
		{
			i++;
			out_attributes = atoi(argv[i]);
			cout << "	Number of Output's Attributes : " << out_attributes << endl;
		}
	}

	//printf("argv1: %s, argv2: %s, argv3: %s, argv4: %s, argv5: %d, argv6: %d, argv7: %d\n", data_in_filename.c_str(), data_out_filename.c_str(), filename_to_save.c_str(), filename_to_load.c_str(), samples, in_attributes, out_attributes);

	cout << "********************************************************************************************" << endl;
	if (build_mlp_classifier(data_in_filename, data_out_filename, filename_to_save, filename_to_load, n_samples, in_attributes, out_attributes)) {
		cout << "Result ===> POOf, Training the SVM Classifier Succesfully!!!" << endl;
		getchar();
		return 0;
	}
	else {
		cout << "Result ===> Crash, Training the SVM Classifier is not complete." << endl;
		getchar();
		return -1;
	}
}