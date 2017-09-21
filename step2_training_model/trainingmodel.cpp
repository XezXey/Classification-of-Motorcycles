// trainingmodel.cpp : Defines the entry point for the console application.
//This program take image as an input and calculate the HoG for each block then visualize it.

#define DEBUG
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

using namespace std;
using namespace cv;

void calculate_gradient_image(Mat motorcycle_roi);
vector<float> calculate_hog_image(Mat motorcycle_roi);
Mat get_hogdescriptor_visual(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);

String window_name = "HoG_Visualization";


int main(int argc, char** argv)
{
	Mat motorcycle_roi = imread(argv[1]);
	Mat motorcycle_roi_hog_visual;
	vector<float> hog_value_motorcycle_roi;
	hog_value_motorcycle_roi = calculate_hog_image(motorcycle_roi);
#ifdef DEBUG
	for (int i = 0; i < hog_value_motorcycle_roi.size(); i++) {
		cout << i << " : " <<hog_value_motorcycle_roi[i] << endl;
	}
#endif
	motorcycle_roi_hog_visual = get_hogdescriptor_visual(motorcycle_roi, hog_value_motorcycle_roi, Size(64, 128));
	imshow(window_name, motorcycle_roi_hog_visual);
#ifdef DEBUG
	while (true) {
		if (waitKey(30) >= 0)
			break;
	}
#endif
    return 0;
}


void calculate_gradient_image(Mat motorcycle_roi) {
	//This function will take the image of roi and calculate the gradient 

	/*Convert datatype form CV_8U to CV_32F (CV_32F is float - the pixel can have any value between 0-1.0,
	this is useful for some sets of calculations on data - but it has to be converted into 8bits
	to save or display by multiplying each pixel by 255.)
	*/
	motorcycle_roi.convertTo(motorcycle_roi, CV_32F, 1 / 255.0);

	// Calculate gradients (gx, gy)
	Mat gx, gy;
	Sobel(motorcycle_roi, gx, CV_32F, 1, 0, 1);
	Sobel(motorcycle_roi, gy, CV_32F, 0, 1, 1);

	// Calculate gradient magnitude and direction (in degrees)
	Mat magnitude, angle;
	cartToPolar(gx, gy, magnitude, angle, 1);
}

vector<float> calculate_hog_image(Mat motorcycle_roi)
{
	Mat motorcycle_roi_grey;
	//motorcycle_roi.convertTo(motorcycle_roi, CV_32F, 1 / 255.0);
	cvtColor(motorcycle_roi, motorcycle_roi_grey, COLOR_BGR2GRAY);
	vector<float> hog_descriptors_value;
	HOGDescriptor hog_descriptor(
		Size(64, 128), //winSize
		Size(16, 16), //blocksize
		Size(8, 8), //blockStride,
		Size(8, 8), //cellSize,
		9, //nbins,
		1, //derivAper,
		-1, //winSigma,
		0, //histogramNormType,
		0.2, //L2HysThresh,
		1,//gammal correction,
		64,//nlevels=64
		0);//Use signed gradients 

	hog_descriptor.compute(motorcycle_roi_grey, hog_descriptors_value);
	cout << "HOG 's size : " << hog_descriptors_value.size() << endl;
	cout << "Visualize!!!" << endl;
	return hog_descriptors_value;
}

Mat get_hogdescriptor_visual(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu
