
#include "stdafx.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;



int main2(int argc, char** argv)
{

	double alpha; /**< Simple contrast control */
	int beta;  /**< Simple brightness control */

	VideoCapture	cap(0);	//Open default camera
	if (!cap.isOpened()) {	//Cannot open camera
		return -1;
	}
	namedWindow("frame", 1);

	/// Initialize values
	std::cout << " Basic Linear Transforms " << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "* Enter the alpha value [1.0-3.0]: "; std::cin >> alpha;
	std::cout << "* Enter the beta value [0-100]: "; std::cin >> beta;

	/*
	Mat camera_frame;
	cap >> camera_frame;
	Mat camera_new_frame = Mat::zeros(camera_frame.size(), camera_frame.type());
	*/
	Mat camera_frame, camera_new_frame;
	cap >> camera_frame;
	camera_new_frame = Mat::zeros(camera_frame.size(), camera_frame.type());
	//cout << "Initialize Mat : " << "Rows: " << camera_frame.rows << "Cols: " << camera_frame.cols << endl;
	for (;;) {
		cap >> camera_frame;
		//cout << "Rows: " << camera_frame.rows << "Cols: " << camera_frame.cols << endl;
		/// Do the operation new_image(i, j) = alpha * image(i, j) + beta
		for (int y = 0; y < camera_frame.rows; y++)	//Rows
		{
			for (int x = 0; x < camera_frame.cols; x++)	//Cols
			{
				for (int c = 0; c < 3; c++)			//Channel(Red, Green, Blue)
				{
					camera_new_frame.at<Vec3b>(y, x)[c] =
						saturate_cast<uchar>(alpha*(camera_frame.at<Vec3b>(y, x)[c]) + beta);
					/*
					Transform Value at any x, y
					I'(y, x) = alpha * I(y, x) + beta
					- beta is Brightness control
					- alpha is Contrast control
					- Convert back to unsigned char (0-255) by using saturate_cast
					*/

				}
			}
		}
		imshow("frame", camera_new_frame);
		if (waitKey(30) >= 0)	break;
	}

	return 0;
}
