// MC_GUIDlg.cpp : implementation file
//

#include "stdafx.h"
#include <stdexcept>
#include <vector>
#include "MC_GUI.h"
#include "MC_GUIDlg.h"
#include "afxdialogex.h"
#include <opencv2/opencv.hpp>
#include <ml.hpp>
#include <windows.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <io.h>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <chrono>
#include <thread>

#define SVMS

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
using namespace cv;
using namespace cv::ml;

/** Function Headers */
//1.Extract Features From Images
string generate_filename(string filename, int filenum);
int video_processing(String input_filename);
string save_roi_file(Mat motorcycle_roi, int filenum);
vector<Point> check_correct_tl_br(Point tl, Point br);
vector<int> check_correct_w_h(Point tl_rect_roi, int width, int height);
void localize_operation_area(void);
void onMouse(int event, int x, int y, int f, void*);


//2.Find HOG
vector<float> calculate_hog_image(Mat motorcycle_roi);

//3.Predict
template<typename T> static Ptr<T> load_classifier(const string& filename_to_load);
Mat push_hog_to_data_for_predict(vector<float>hog_value_motorcycle_roi, Mat data);

//4.Change ROI analysis area
Rect convert_roi_from_onlyarea_to_frame(Rect roi);

Mat src, src_temp, img, ROI;
Rect cropRect(0, 220, 320, 260);
Point P1(0, 0);
Point P2(0, 0);
bool clicked = false;
String winName = "Cropped Image";


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
int n_samples_attributes = 3780;
string classifier_filename = "Finale_Model_SVM_autos250poly.xml";

Ptr<SVM> model_svm;

int video_processing(String input_filename)
{
	Mat motorcycle_roi;
	vector<float> hog_value_motorcycle_roi;

	VideoCapture capt_input_video(input_filename);
	//VideoCapture capt_input_video(0);

	capt_input_video.set(CV_CAP_PROP_FPS, 60);
	capt_input_video.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capt_input_video.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	if (!capt_input_video.isOpened()) {
		return -1;
	}
	Size img_size(image_size_x, image_size_y);

	for (;;)
	{
		Mat motorcycle_frame, motorcycle_frame_original, motorcycle_show;	
		//motorcycle_frame use for drawing and labeling, motorcycle_frame_original use for analysis

		capt_input_video.read(motorcycle_frame_original); // get a new frame from camera

												 //Flip video in to right direction
												 //transpose(motorcycle_frame, motorcycle_frame);
												 //flip(motorcycle_frame, motorcycle_frame, 1);	//1 is flip the video around

		resize(motorcycle_frame_original, motorcycle_frame_original, img_size);	//Resize the image into 640x480
		motorcycle_frame_original.copyTo(src_temp);	//Use for crop a roi operate area

		motorcycle_frame_original.copyTo(motorcycle_show);
		rectangle(motorcycle_show, cropRect, Scalar(0, 0, 255), 2, 8, 0);


		vector<Rect> plates;

		//Change the rect_area_operate
		Rect rectCrop_area_operate = Rect(cropRect.x, cropRect.y, cropRect.width, cropRect.height);
		motorcycle_frame = Mat(motorcycle_frame_original, rectCrop_area_operate);
		//resize(motorcycle_frame, motorcycle_frame, img_size);	//Resize the image into 640x480

		//imshow("cropped area operate", motorcycle_frame);

		//motorcycle_plate_cascade.detectMultiScale(motorcycle_grey, plates, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		motorcycle_plate_cascade.detectMultiScale(motorcycle_frame, plates, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		//motorcycle_frame = motorcycle_frame_original;


		for (size_t i = 0; i < plates.size(); i++)
		{
			//Delete old Rectangle and draw it on a new image

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
			vector<int> width_height_corrected = check_correct_w_h(tl_rect_roi, plates[i].width, plates[i].height);

			//Rect roi(tl_rect_roi.x, tl_rect_roi.y, (plates[i].width * 3) + 30, (plates[i].height * 3) + 40);
			Rect roi(tl_rect_roi.x, tl_rect_roi.y, width_height_corrected[0], width_height_corrected[1]);
			Rect roi_show(tl_rect_roi.x, tl_rect_roi.y, width_height_corrected[0], width_height_corrected[1]);


			rectangle(motorcycle_frame, roi, Scalar(255, 0, 0), 2, 8, 0);

			//motorcycle_roi = motorcycle_frame_original(roi);
			motorcycle_roi = motorcycle_frame(roi);
			Size img_size_roi(image_size_export_x, image_size_export_y);
			resize(motorcycle_roi, motorcycle_roi, img_size_roi);
			imshow(window_roi, motorcycle_roi);

			vector<float>hog_value_motorcycle_roi = calculate_hog_image(motorcycle_roi);	//Find HOG
			Mat data(1, n_samples_attributes, CV_32F);
			data = push_hog_to_data_for_predict(hog_value_motorcycle_roi, data);


			String motorcyclename = "";
			//cout << "PREDICTING_SVM : ";
			int result_svm = 0;
			int labels;
			int nsamples_all = data.rows;
			//cout << nsamples_all << endl;
			int true_positive = 0;		//Classifier can classifier data correctly.

										// compute prediction error on train and test data

			for (int i = 0; i < nsamples_all; i++)
			{
				result_svm = model_svm->predict(data);
				if (result_svm == -1082130432) {
					result_svm += 1082130431;	//Normalized to [-1, 0 ,1]
					motorcyclename = "Fino";
				}
				else if (result_svm == 0) {
					motorcyclename = "Scoopy-i";
				}
				else if (result_svm == 1065353216) {
					result_svm -= 1065353215;	//Normalized to [-1, 0 ,1]
					motorcyclename = "Wave";
				}
				else motorcyclename = "Unknown";
			}



			//Put the label name on the picture
			putText(motorcycle_frame, motorcyclename, Point(roi.x, roi.y),
				FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255, 0, 0), 2, CV_AA);


			// For show on the GUI screen that convert from operated area to original area
			roi_show = convert_roi_from_onlyarea_to_frame(roi);
			rectangle(motorcycle_show, roi_show, Scalar(255, 0, 0), 2, 8, 0);
			putText(motorcycle_show, motorcyclename, Point(roi_show.x, roi_show.y),
				FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255, 0, 0), 2, CV_AA);
			
			

		}
		//imshow(window_name, motorcycle_frame);
		imshow(window_name, motorcycle_show);
		//imshow("Operate Area and Detection", motorcycle_show); for debug that show on other screen

		if (waitKey(30) > 0 || motorcycle_frame_original.empty()) {	//delay 25 ms before show a next frame.
			break;
		}
	}
	// the camera will be deinitialized automatically in
	VideoCapture destructor;
	return 0;
}
//Rect cropRect(0, 220, 640, 260);
Rect convert_roi_from_onlyarea_to_frame(Rect roi) {
	Rect roi_adjusted = roi;
	roi_adjusted.x = roi.x + cropRect.x;
	roi_adjusted.y = roi.y + cropRect.y;
	return roi_adjusted;
}

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

vector<Point> check_correct_tl_br(Point tl, Point br)
{
	if (tl.x <= origin_point)	tl.x = origin_point;
	if (tl.y <= origin_point)	tl.y = origin_point;
	if (br.x >= cropRect.width)	br.x = cropRect.width - 1;
	if (br.y >= cropRect.height)	br.y = cropRect.height - 1;
	return { tl, br };
}

vector<int> check_correct_w_h(Point tl_rect_roi, int width, int height)
{
	int width_shift = 30;
	int height_shift = 25;
	//Width check and adjust into proper size
	if ((width * 3) + width_shift > cropRect.width || tl_rect_roi.x + (width * 3) + width_shift > cropRect.width)
		width = cropRect.width - tl_rect_roi.x;
	else
		width = (width * 3) + width_shift;



	//Height check and adjust into proper size and auto get rid of license plate
	if ((height)+height_shift > cropRect.height || tl_rect_roi.y + (height) + height_shift  > cropRect.height)
		height = cropRect.height - tl_rect_roi.y;
	else
		height = (height) + height_shift;

	return { width, height };
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
	//cout << "HOG 's size : " << hog_descriptors_value.size() << endl;
	//cout << "Visualize!!!" << endl;
	return hog_descriptors_value;
}

Mat push_hog_to_data_for_predict(vector<float>hog_value_motorcycle_roi, Mat data)
{
	// for each sample in the file
	for (int line = 0; line < 1; line++)
	{

		// for each attribute on the line in the file
		for (int attribute = 0; attribute < n_samples_attributes; attribute++)
		{

			// first 256 elements (0-255) in each line are the attributes
			data.at<float>(line, attribute) = hog_value_motorcycle_roi.at(attribute);
		}
	}


	return data; // Reading value from file process is OK.
}

template<typename T> static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);

	return model;
}


void checkBoundary() {
	//check croping rectangle exceed image boundary
	if (cropRect.width>img.cols - cropRect.x)
		cropRect.width = img.cols - cropRect.x;

	if (cropRect.height>img.rows - cropRect.y)
		cropRect.height = img.rows - cropRect.y;

	if (cropRect.x<0)
		cropRect.x = 0;

	if (cropRect.y<0)
		cropRect.height = 0;
}


void showImage() {
	img = src.clone();
	String roi_coor;
	checkBoundary();
	if (cropRect.width>0 && cropRect.height>0) {
		ROI = src(cropRect);
		imshow("cropped", ROI);
	}

	rectangle(img, cropRect, Scalar(0, 255, 0), 1, 8, 0);
	roi_coor += "W : " + to_string(cropRect.width);
	roi_coor += " H : " + to_string(cropRect.height);
	putText(img, roi_coor, Point(cropRect.x, cropRect.y),
		FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255, 0, 0), 2, CV_AA);
	imshow(winName, img);
}

void onMouse(int event, int x, int y, int f, void*) {

	switch (event) {

	case  CV_EVENT_LBUTTONDOWN:
		clicked = true;

		P1.x = x;
		P1.y = y;
		P2.x = x;
		P2.y = y;
		break;

	case  CV_EVENT_LBUTTONUP:
		P2.x = x;
		P2.y = y;
		clicked = false;
		break;

	case  CV_EVENT_MOUSEMOVE:
		if (clicked) {
			P2.x = x;
			P2.y = y;
		}
		break;

	default:   break;


	}

	if (clicked) {
		if (P1.x>P2.x) {
			cropRect.x = P2.x;
			cropRect.width = P1.x - P2.x;
		}
		else {
			cropRect.x = P1.x;
			cropRect.width = P2.x - P1.x;
		}

		if (P1.y>P2.y) {
			cropRect.y = P2.y;
			cropRect.height = P1.y - P2.y;
		}
		else {
			cropRect.y = P1.y;
			cropRect.height = P2.y - P1.y;
		}

	}
	showImage();


}



void localize_operation_area(void) {

	if (src_temp.empty()) {
		return;
	}
	src = src_temp;	//For show only current picture from video -> not show following the video = just one!!!
	ofstream roi_config_file;
	roi_config_file.open("roi_config_file.txt");
	imshow(winName, src);
	setMouseCallback(winName, onMouse, NULL);

	while (1) {
		char c = waitKey();
		cout << "H : " << cropRect.height << endl << "W : " << cropRect.width << endl;
		if (c == 's'&&ROI.data) {
			cout << "H : " << cropRect.height << endl << "W : " << cropRect.width << endl;
			roi_config_file << "TL : " << cropRect.x << ", " << cropRect.y << endl << "BR : " << cropRect.x + cropRect.width
				<< ", " << cropRect.y + cropRect.height << endl;
			break;
		}
		if (c == '6') cropRect.x++;
		if (c == '4') cropRect.x--;
		if (c == '8') cropRect.y--;
		if (c == '2') cropRect.y++;

		if (c == 'w') { cropRect.y--; cropRect.height++; }
		if (c == 'd') cropRect.width++;
		if (c == 'x') cropRect.height++;
		if (c == 'a') { cropRect.x--; cropRect.width++; }

		if (c == 't') { cropRect.y++; cropRect.height--; }
		if (c == 'h') cropRect.width--;
		if (c == 'b') cropRect.height--;
		if (c == 'f') { cropRect.x++; cropRect.width--; }

		if (c == 'r') { cropRect.x = 0; cropRect.y = 0; cropRect.width = 0; cropRect.height = 0; }

		showImage();

	}

	roi_config_file.close();
	destroyWindow(winName);
	destroyWindow("cropped");
}

UINT MyThreadProc(LPVOID pParam)
{
	cout << "Begin Thread..." << endl;
	model_svm = load_classifier<SVM>(classifier_filename);
	if (!motorcycle_plate_cascade.load("cascade_motor_plate.xml")) {
		return -1;
	};

	video_processing("PJ2_Test_Finale.mp4");

	return 0;
}

// CAboutDlg dialog used for App About
class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMC_GUIDlg dialog



CMC_GUIDlg::CMC_GUIDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MC_GUI_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMC_GUIDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMC_GUIDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_STN_CLICKED(IDC_STATIC_VIDEO, &CMC_GUIDlg::OnStnClickedStaticVideo)
	ON_EN_CHANGE(IDC_MFCEDITBROWSE1, &CMC_GUIDlg::OnEnChangeMfceditbrowse1)
	ON_BN_CLICKED(IDC_BUTTON_REFRESH, &CMC_GUIDlg::OnBnClickedButtonRefresh)
	ON_BN_CLICKED(IDOK, &CMC_GUIDlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON_START, &CMC_GUIDlg::OnBnClickedButtonStart)
	ON_BN_CLICKED(IDC_CHECK_IPCAMERA, &CMC_GUIDlg::OnBnClickedCheckIpcamera)
	ON_BN_CLICKED(IDC_MFCBUTTON_CHANGE_ROI, &CMC_GUIDlg::OnBnClickedMfcbuttonChangeRoi)
END_MESSAGE_MAP()


// CMC_GUIDlg message handlers

BOOL CMC_GUIDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	
	cvNamedWindow("Capture - Motorcycle Plate detection", 0);
	cvResizeWindow("Capture - Motorcycle Plate detection", 822, 480); //420, 240);
	HWND hWnd_video = (HWND)cvGetWindowHandle("Capture - Motorcycle Plate detection");
	HWND hParent_video = ::GetParent(hWnd_video);
	HWND hDlg_video = GetDlgItem(IDC_STATIC_VIDEO)->m_hWnd;
	::SetParent(hWnd_video, hDlg_video);
	::ShowWindow(hParent_video, SW_HIDE);
	
	cvNamedWindow("ROI - Motorcycle", 0);
	cvResizeWindow("ROI - Motorcycle", 64, 128); //420, 240);
	HWND hWnd_roi = (HWND)cvGetWindowHandle("ROI - Motorcycle");
	HWND hParent_roi = ::GetParent(hWnd_roi);
	HWND hDlg_roi = GetDlgItem(IDC_STATIC_ROI)->m_hWnd;
	::SetParent(hWnd_roi, hDlg_roi);
	::ShowWindow(hParent_roi, SW_HIDE);

	//AfxBeginThread(MyThreadProc, 0);


	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CMC_GUIDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CMC_GUIDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CMC_GUIDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


//Video Player Window
void CMC_GUIDlg::OnStnClickedStaticVideo()
{
	// TODO: Add your control notification handler code here
}

//File Browser (ID : IDC_MFCEDITBROWSE1)
void CMC_GUIDlg::OnEnChangeMfceditbrowse1()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialogEx::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}


//Refresh Button (ID : IDC_BUTTON_REFRESH)
void CMC_GUIDlg::OnBnClickedButtonRefresh()	
{
	// TODO: Add your control notification handler code here
}

//Ok Button (ID : IDC_BUTTON_OK)
void CMC_GUIDlg::OnBnClickedOk()
{
	// TODO: Add your control notification handler code here
	CDialogEx::OnOK();
}

//Start Button (ID : IDC_BUTTON_START)
void CMC_GUIDlg::OnBnClickedButtonStart()
{
	// TODO: Add your control notification handler code here
	AfxBeginThread(MyThreadProc, 0);

}

//Switch to IP Camera (ID : IDC_CHECK_IPCAMERA
void CMC_GUIDlg::OnBnClickedCheckIpcamera()
{
	// TODO: Add your control notification handler code here
}


//Change ROI analysis Button (ID : IDC_MFCBUTTON_CHANGE_ROI)
void CMC_GUIDlg::OnBnClickedMfcbuttonChangeRoi()
{
	// TODO: Add your control notification handler code here
	//SuspendThread(MyThreadProc);
	localize_operation_area();
	//ResumeThread(MyThreadProc);
}

