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

//#define DEBUG
#define PROCESS

using namespace std;
using namespace cv;

/** Function Headers */
void detect_save_display(Mat motorcycle_frameame);
string generate_filename(string filename, int filenum);
int image_processing(String input_filename);
int video_processing(String input_filename);
string save_roi_file(Mat motorcycle_roi, int filenum);
vector<Point> check_correct_tl_br(Point tl, Point br);
vector<int> check_correct_w_h(Point tl_rect_roi, int width, int height);



/** Global variables */
String motorcycle_plate_cascade_name = "cascade_motor_plate.xml";
CascadeClassifier motorcycle_plate_cascade;
String window_name = "Capture - Motorcycle Plate detection";
String window_roi = "ROI - Motorcycle";
String filename = "";
int image_size_x = 640, image_size_y = 480;
int image_size_export_x = 64, image_size_export_y = 128;
int origin_point = 0;
int width_shift = 30;
int height_shift = 25;

/** @function main */
int main(int argc, char** argv)
{
	int input_file_type = atoi(argv[1]);
	String input_filename = argv[2];
	//-- 1. Load the cascades
	if (!motorcycle_plate_cascade.load(motorcycle_plate_cascade_name)) {
		printf("--(!)Error loading plates cascade\n");
		return -1;
	};
	
#ifdef DEBUG
	//-- 2. Choose input method
	cout << "Choose your input file type" << endl;
	cout << "===========================" << endl;
	cout << "1. Image" << endl;
	cout << "2. Video" << endl;
	cout << "You choose : " << argv[1] << endl;
#endif

	filename += argv[2];
	cout << argv[1] << endl;

	//-- 3. Select Mode of operation from input file type
	
	switch (input_file_type) {
		case 1 :
			cout << "Input file is image." << endl;
			image_processing(input_filename);
			break;
		case 2 :
			cout << "Input file is video." << endl;
			video_processing(input_filename);
			break;
		default :
			cout << "Unknown input file type" << endl;
			break;
	}
	
	#ifndef PROCESS
	while (true) { 
		if(waitKey(30) >= 0)
			break;
	}
	#endif
	return 0;
}

int video_processing(String input_filename)
{
	VideoCapture capt_input_video(input_filename);
	if (!capt_input_video.isOpened()) {
		return -1;
	}
	Size img_size(image_size_x, image_size_y);

	for (;;)
	{
		Mat motorcycle_frame;
		capt_input_video >> motorcycle_frame; // get a new frame from camera

		//Flip video in to right direction
		transpose(motorcycle_frame, motorcycle_frame);
		flip(motorcycle_frame, motorcycle_frame, 1);	//1 is flip the video around
		
		resize(motorcycle_frame, motorcycle_frame, img_size);	//Resize the image into 640x480

		imshow(window_name, motorcycle_frame);
		detect_save_display(motorcycle_frame);
		if (waitKey(1) > 0 || motorcycle_frame.empty()) {	//delay 25 ms before show a next frame.
			break;
		}
	}
	// the camera will be deinitialized automatically in
	VideoCapture destructor;
	return 0;
}

int image_processing(String input_filename)
{
	Mat motorcycle_roi;
	Size img_size(image_size_x, image_size_y);
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
	
	/*
	#ifdef DEBUG
		cout << filenum << endl;
		cout << "Plate no." << to_string(filenum) << endl;
		cout << "Generated filename : " << filename << endl;
	#endif
	*/
	return filename;
}

string save_roi_file(Mat motorcycle_roi, int filenum)
{
	//Generate filename for saving multiple files in case of there are more than one plate can detected.
	char buffer_filename[50];
	char saved_filename[50];

	strcpy_s(buffer_filename, generate_filename(filename, (int)filenum).c_str());
	sprintf_s(saved_filename, "%s.JPG", buffer_filename);

	/*
	#ifdef DEBUG
		cout << "****************************************************************" << endl;
		cout << "Filename : " << generate_filename(filename, (int)filenum).c_str() << endl;
	#endif // DEBUG
	*/
	
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

#ifdef PROCESS
	if (plates.size() != 0) {
		cout << "=====================================================================================" << endl;
		cout << "Filename : " << filename << endl;
		cout << "We have : " << plates.size() << " plates." << endl;
		cout << "=====================================================================================" << endl;
	}
#endif
	for (size_t i = 0; i < plates.size(); i++)
	{
		//Delete old Rectangle and draw it on a new image
		motorcycle_frame = motorcycle_frame_original;

		//Declare 2 Point of top-left corner and bottom right corner for drawing a rectangle of ROI
		//Now use 1:1 scale up for covering the ROI
		//tl is top left corner
		//br is bottom right corner

		Point tl_rect_roi(plates[i].x - plates[i].width - width_shift, plates[i].y - plates[i].height - height_shift);
		Point br_rect_roi(plates[i].x + (2 * plates[i].width), plates[i].y + (2 * plates[i].height));
		cout << plates[i].size() << " " << plates[i].x << " " << plates[i].y << " " << plates[i].width << " " << 
			plates[i].height << endl;
		//Point tl_rect_roi(plates[i].x, plates[i].y);
		//Point br_rect_roi(plates[i].x, plates[i].y);

		vector<Point> tl_br_corrected = check_correct_tl_br(tl_rect_roi, br_rect_roi);
		tl_rect_roi = tl_br_corrected[0];
		br_rect_roi = tl_br_corrected[1];

		//Draw the lines around the ROI by using red color with thickness = 2 , lineType is 8 and no shift
		//1. draw by use tl_rect_roi and br_rect_roi
		//2. draw by use roi that get an information from tl_rect_roi, br_rect_roi and some estimate how big it is
		//	 multiple 3 cause i use the 1:1 of plate so we will get the full image from tl and br point that need
		//   to triple extended
		//Use 2 lines for check that it's the same lines on same roi
		//rectangle(motorcycle_frame, tl_rect_roi, br_rect_roi, Scalar(0, 0, 255), 2, 8, 0);

		//Rect roi(tl_rect_roi.x, tl_rect_roi.y, (plates[i].width), (plates[i].height));

		//We Use tl(x,y) to draw and RoI and control size by using width and height
		vector<int> width_height_corrected = check_correct_w_h(tl_rect_roi,	plates[i].width, plates[i].height);

		//Rect roi(tl_rect_roi.x, tl_rect_roi.y, (plates[i].width * 3) + 30, (plates[i].height * 3) + 40);
		Rect roi(tl_rect_roi.x, tl_rect_roi.y, width_height_corrected[0], width_height_corrected[1]);
#ifdef DEBUG
		cout << width_height_corrected[0] << " " << width_height_corrected[1] << endl;
		rectangle(motorcycle_frame, roi, Scalar(128, 128, 255), 2, 8, 0);

#endif

		Mat motorcycle_roi;
		motorcycle_roi = motorcycle_frame_original(roi);


#ifdef DEBUG
		cout << "**********************Plate[" << i + 1 << "]********************** " << endl;
		cout << "Filename : " << generate_filename(filename, (int)i + 1) << ".JPG" << endl;
		cout << "Image size : ";
		cout << "Width = " << motorcycle_frame.size().width;
		cout << ", Height = " << motorcycle_frame.size().height << endl;
		cout << "Plate No.: " << "  x    " << "y   " << "width   " << "height   " << endl;
		cout << "Plate[" << i + 1 << "] : " << plates[i].x << "   " << plates[i].y << "   " << plates[i].width << "      "
			<< plates[i].height << endl;
		cout << "Each Plate Value [" << i + 1 << "] : " << plates[i] << endl;
		cout << "topcorner_rect_roi : " << tl_rect_roi << endl;
		cout << "bottomcorner_rect_roi : " << br_rect_roi << endl;
		cout << "Corrected TL : " << tl_rect_roi << endl;
		cout << "Corrected BR : " << br_rect_roi << endl;
		cout << "Region of Interest size : " << motorcycle_roi.size() << endl;
		cout << "=====================================================================================" << endl;
#endif

		Size img_size(image_size_export_x, image_size_export_y);

		resize(motorcycle_roi, motorcycle_roi, img_size);
		saved_filename = save_roi_file(motorcycle_roi, (int)i+1);
		
		imshow(window_name, motorcycle_frame);
		//Display all ROI
		#ifdef DEBUG
			imshow(saved_filename, motorcycle_roi);
		#endif

		#ifdef PROCESS
			if (plates.size() != 0) {
				cout << "Filename : " << saved_filename << ".JPG" << endl;
				cout << "==============================================================================" << endl;
			}
		#endif	
	}
	//-- Show the output (ROI that can detect on the image by using motorcylce_cascade.xml)
	//imshow(window_name, motorcycle_frame_original);

}

vector<Point> check_correct_tl_br(Point tl, Point br) 
{
	if (tl.x <= origin_point)	tl.x = origin_point;
	if (tl.y <= origin_point)	tl.y = origin_point;
	if (br.x >= image_size_x)	br.x = image_size_x - 1;
	if (br.y >= image_size_y)	br.y = image_size_y - 1;
	return { tl, br };
}

vector<int> check_correct_w_h(Point tl_rect_roi, int width, int height)
{
	int width_shift = 30;
	int height_shift = 25;
	//Width check and adjust into proper size
	if ((width * 3) + width_shift > image_size_x || tl_rect_roi.x + (width * 3) + width_shift > image_size_x)
		width = image_size_x - tl_rect_roi.x;
	else
		width = (width * 3) + width_shift;



	//Height check and adjust into proper size and auto get rid of license plate
	if ((height * 2) + height_shift > image_size_y || tl_rect_roi.y + (height) + height_shift  > image_size_y)
		height = image_size_y - tl_rect_roi.y;
	else
		height = (height) + height_shift;

	return { width, height };
}
