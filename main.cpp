//Note: This code is created by Bingshu Wang , 2019.08.31
//We proposed an effective method to remove shadows from document images.
//This is only for academic exchange. 
// If one wants use it for commercial purpose, please contact us right now by yb77408@umac.mo or philipchen@um.edu.mo.  
// or  https://www.fst.um.edu.mo/en/staff/pchen.html. 

//If you try to use this code, please cite our paper 
// "An Effective Background Estimation Method For Shadows Removal Of Document Images"  
// accepted by ICIP2019.

#include"doc_shadow_removal.h"

int main()
{
	Mat img = imread("000_010.png");
	Mat gt = imread("000_010gt.bmp");
	Mat result(img.size(), CV_8UC3, 3);
	
	ShadowRemoval(img, result);

	imshow("img", img);
	imshow("result",result);

	double dMSE1 = CalulateOneImgMSE(result, gt);
	double dMSE2 = CalulateOneImgMSE(img, gt);
	double  error_ratio = sqrt(dMSE1) / sqrt(dMSE2);
	cout << "  MSE1: " << dMSE1 << "    MSE2:"<< dMSE2<<"     Error_ratio:"<< error_ratio <<endl;

	waitKey(0);
	return 0;
}







 

 