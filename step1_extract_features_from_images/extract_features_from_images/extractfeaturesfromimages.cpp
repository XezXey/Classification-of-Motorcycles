// extractfeaturesfromimages.cpp : Defines the entry point for the console application.
// This Program start coding at 2/9/2017 15:59 GMT+7
/*
	Process:
	1. This Program will take set of images and use cascade detection for detect the plate of motorcycle
	2. Draw the Rectangle of the ROI(region of interest)
	3. export only the ROI of image
*/
#include "stdafx.h"

#include "opencv2/opencv.hpp"

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <string>

#define DEBUG

using namespace std;
using namespace cv;

/** Function Headers */
void detect_save_display(Mat motorcycle_frameame);
string generate_filename(string filename, int filenum);
int image_preprocessing(String input_filename);
int video_preprocessing(String input_filename);
string save_roi_file(Mat motorcycle_roi, int filenum);



/** Global variables */
String motorcycle_plate_cascade_name = "cascade_motor-1.xml";
CascadeClassifier motorcycle_plate_cascade;
String window_name = "Capture - Motorcycle Plate detection";
String window_roi = "ROI - Motorcycle";
String filename = "";

/** @function main */
int main(int argc, char** argv)
{
	Mat motorcycle_roi;
	int input_file_type = atoi(argv[1]);
	String input_filename = argv[2];
	//-- 1. Load the cascades
	if (!motorcycle_plate_cascade.load(motorcycle_plate_cascade_name)) {
		printf("--(!)Error loading face cascade\n");
		return -1;
	};
	
	//-- 2. Choose input method
	cout << "Choose your input file type" << endl;
	cout << "===========================" << endl;
	cout << "1. Image" << endl;
	cout << "2. Video" << endl;
	cout << "You choose : " << argv[1] << endl;
	
	filename += argv[2];
	cout << argv[1] << endl;

	//-- 3. Select Mode of operation from input file type
	
	switch (input_file_type) {
		case 1 :
			cout << "Input file is image." << endl;
			image_preprocessing(input_filename);
			break;
		case 2 :
			cout << "Input file is video." << endl;
			video_preprocessing(input_filename);
			break;
		default :
			cout << "Unknown input file type" << endl;
			break;
	}
	
	while (true) { 
		if(waitKey(30) >= 0)
			break;
	}
	return 0;
}

int video_preprocessing(String input_filename)
{
	VideoCapture capt_input_video(input_filename);
	if (!capt_input_video.isOpened()) {
		return -1;
	}
	
	for (;;)
	{
		Mat motorcycle_frame;
		capt_input_video >> motorcycle_frame; // get a new frame from camera
		imshow(window_name, motorcycle_frame);
		detect_save_display(motorcycle_frame);
		if (waitKey(100) >= 0) 
			break;
	}
	// the camera will be deinitialized automatically in
	VideoCapture destructor;
	return 0;
}

int image_preprocessing(String input_filename)
{
	Mat motorcycle_roi;
	Size img_size(640, 480);
	Mat motorcycle_image;
	motorcycle_image = imread(input_filename);
	if (!motorcycle_image.data) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	resize(motorcycle_image, motorcycle_image, img_size);

	detect_save_display(motorcycle_image);
	return 0;
}

//generate_filename function will delete the extension of filename and add a filenum to make it difference and can be save
string generate_filename(string filename, int filenum)
{
	size_t len_filename_extension = 4;
	filename.erase(filename.length() - len_filename_extension, len_filename_extension);
	filename += "_plate_" + to_string(filenum);

	#ifdef DEBUG
	cout << filenum << endl;
	cout << "Plate no." << to_string(filenum) << endl;
	cout << "Generated filename : " << filename << endl;
	#endif
	return filename;
}

string save_roi_file(Mat motorcycle_roi, int filenum)
{
	//Generate filename for saving multiple files in case of there are more than one plate can detected.
	char buffer_filename[50];
	char saved_filename[50];

	strcpy_s(buffer_filename, generate_filename(filename, (int)filenum).c_str());
	sprintf_s(saved_filename, "%s.JPG", buffer_filename);

	#ifdef DEBUG
		cout << "****************************************************************" << endl;
		cout << "Filename : " << generate_filename(filename, (int)filenum).c_str() << endl;
	#endif // DEBUG

	
	imwrite(saved_filename, motorcycle_roi);
	return string(saved_filename);
}


/** @function detect_save_display */
void detect_save_display(Mat motorcycle_frame)
{
	string saved_filename;
	vector<Rect> plates;
	Mat motorcycle_grey;
	//Convert to Gray Scale for processing
	cvtColor(motorcycle_frame, motorcycle_grey, COLOR_BGR2GRAY);	
	equalizeHist(motorcycle_grey, motorcycle_grey);

	//-- Detect motorcycle plates
	motorcycle_plate_cascade.detectMultiScale(motorcycle_grey, plates, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	Mat motorcycle_frame_original = motorcycle_frame.clone();
	//After use cascade dectection we get the ROI
	//plates.size() is number of ROI		
	cout << "Plate No.: " << "  x    " << "y   " << "width   " << "height   " << endl;
	for (size_t i = 0; i < plates.size(); i++)
	{
		//Delete old Rectangle and draw it on a new image
		motorcycle_frame = motorcycle_frame_original;

		//Declare 2 Point of top-left corner and bottom right corner for drawing a rectangle of ROI
		//Now use 1:1 scale up for covering the ROI
		//tl is top left corner
		//br is bottom right corner
		Point tl_rect_roi(plates[i].x - plates[i].width - 30, plates[i].y - plates[i].height - 40);
		Point br_rect_roi(plates[i].x + (2 * plates[i].width), plates[i].y + (2 * plates[i].height));
		Mat motorcycle_roi;

		#ifdef DEBUG
			cout << "Filename : " << generate_filename(filename, (int)i) << ".JPG" << endl;
			cout << "Plate[" << i + 1 << "] : " << plates[i].x << "   " << plates[i].y << "   " << plates[i].width << "      " 
				<< plates[i].height << endl;
			cout << "Each Plate Value [" << i + 1 << "] : " << plates[i] << endl;
			cout << "topcorner_rect_roi : " << tl_rect_roi << endl;
			cout << "bottomcorner_rect_roi : " << br_rect_roi << endl;
			cout << "=====================================================================================" << endl;
 		#endif

		//Draw the lines around the ROI by using red color with thickness = 2 , lineType is 8 and no shift
		//1. draw by use tl_rect_roi and br_rect_roi
		//2. draw by use roi that get an information from tl_rect_roi, br_rect_roi and some estimate how big it is
		//	 multiple 3 cause i use the 1:1 of plate so we will get the full image from tl and br point that need
		//   to triple extended
		//Use 2 lines for check that it's the same lines on same roi
		rectangle(motorcycle_frame, tl_rect_roi, br_rect_roi, Scalar(0, 0, 255), 2, 8, 0);
		Rect roi(tl_rect_roi.x, tl_rect_roi.y, (plates[i].width * 3) + 30, (plates[i].height * 3) + 40);
		rectangle(motorcycle_frame, roi, Scalar(128, 128, 255), 2, 8, 0);

	
		motorcycle_roi = motorcycle_frame_original(roi);

		saved_filename = save_roi_file(motorcycle_roi, (int)i+1);

		//Display all ROI
		#ifdef DEBUG
			imshow(saved_filename, motorcycle_roi);
		#endif
	}
	//-- Show the output (ROI that can detect on the image by using motorcylce_cascade.xml)
	imshow(window_name, motorcycle_frame_original);
}