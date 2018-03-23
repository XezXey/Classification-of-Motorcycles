
// MC_GUIDlg.h : header file
//

#pragma once


// CMC_GUIDlg dialog
class CMC_GUIDlg : public CDialogEx
{
// Construction
public:
	CMC_GUIDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MC_GUI_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnStnClickedStaticVideo();
	afx_msg void OnEnChangeMfceditbrowse1();
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButtonRefresh();
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButtonStart();
	afx_msg void OnBnClickedCheck2();
	afx_msg void OnBnClickedCheckIpcamera();
	afx_msg void OnBnClickedMfcbutton3();
	afx_msg void OnBnClickedMfcbuttonChangeRoi();
	CString editbrowse_filename;
	int open_ipcamera;
};
