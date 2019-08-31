#include"doc_shadow_removal.h"


/*******************************************
*Function: Binarization using integral method
which is from https://github.com/phryniszak/AdaptiveIntegralThresholding
*Input: inputMat, gray-scale image
*       thre, the threshold to classify the fg and bg
*Output: outputMat, gray-scale, (0 bg, 255 fg)
*
*return：void
*date: 2019.01.15   wangbingshu
********************************************/
void ThresholdIntegral(Mat &inputMat, double thre, Mat &outputMat)
{
	// accept only char type matrices
	CV_Assert(!inputMat.empty());
	CV_Assert(inputMat.depth() == CV_8U);
	CV_Assert(inputMat.channels() == 1);
	CV_Assert(!outputMat.empty());
	CV_Assert(outputMat.depth() == CV_8U);
	CV_Assert(outputMat.channels() == 1);

	// rows -> height -> y
	int nRows = inputMat.rows;
	// cols -> width -> x
	int nCols = inputMat.cols;

	// create the integral image
	cv::Mat sumMat;
	cv::integral(inputMat, sumMat);

	CV_Assert(sumMat.depth() == CV_32S);
	CV_Assert(sizeof(int) == 4);

	int S = MAX(nRows, nCols) / 8;
	double T = 0.15;

	// perform thresholding
	int s2 = S / 2;
	int x1, y1, x2, y2, count, sum;

	// CV_Assert(sizeof(int) == 4);
	int *p_y1, *p_y2;
	uchar *p_inputMat, *p_outputMat;

	for (int i = 0; i < nRows; ++i)
	{
		y1 = i - s2;
		y2 = i + s2;

		if (y1 < 0) {
			y1 = 0;
		}
		if (y2 >= nRows) {
			y2 = nRows - 1;
		}

		p_y1 = sumMat.ptr<int>(y1);
		p_y2 = sumMat.ptr<int>(y2);
		p_inputMat = inputMat.ptr<uchar>(i);
		p_outputMat = outputMat.ptr<uchar>(i);

		for (int j = 0; j < nCols; ++j)
		{
			// set the SxS region
			x1 = j - s2;
			x2 = j + s2;

			if (x1 < 0) {
				x1 = 0;
			}
			if (x2 >= nCols) {
				x2 = nCols - 1;
			}

			count = (x2 - x1)*(y2 - y1);

			// I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
			sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

			if ((int)(p_inputMat[j] * count) < (int)(sum*(1.0 - T)*thre))
				p_outputMat[j] = 0;
			else
				p_outputMat[j] = 255;
		}
	}
}


/*******************************************
*Function: Evalaute the bright/dark extent in an document image
*Input: src， RGB channels
*       size, neighboring size (2*size+1)*(2*size+1)
*      
*Output:dst， RGB channels, 
*
*return：void
*date: 2019.01.15   wangbingshu
********************************************/
void EvaluationIllumination(Mat& src, int size, Mat& dst)
{
	int height = src.rows;
	int width = src.cols;
	int value = 0;
	double fusionFactor = 0;
	int minH, minW, maxH, maxW;
	uchar* puSrcTemp;

	for (int i = 0; i<height; i++)
	{
		uchar* puSrc = src.ptr(i);
		uchar* puDst = dst.ptr(i);
		for (int j = 0; j<width; j++)
		{
			minH = max(i - size, 0);
			maxH = min(i + size, height - 1);
			minW = max(j - size, 0);
			maxW = min(j + size, width - 1);

			int max_valueBGR[3] = { 0 };
			int min_valueBGR[3] = { 0 };

			for (int ii = minH; ii <= maxH; ii++)
			{
				puSrcTemp = src.ptr(ii);
				for (int jj = minW; jj <= maxW; jj++)
				{
					//B通道
					max_valueBGR[0] = max_valueBGR[0]>puSrcTemp[3 * jj] ? max_valueBGR[0] : puSrcTemp[3 * jj];
					min_valueBGR[0] = min_valueBGR[0]<puSrcTemp[3 * jj] ? min_valueBGR[0] : puSrcTemp[3 * jj];
					//G通道
					max_valueBGR[1] = max_valueBGR[1]>puSrcTemp[3 * jj + 1] ? max_valueBGR[1] : puSrcTemp[3 * jj + 1];
					min_valueBGR[1] = min_valueBGR[1]<puSrcTemp[3 * jj + 1] ? min_valueBGR[1] : puSrcTemp[3 * jj + 1];
					//R通道
					max_valueBGR[2] = max_valueBGR[2]>puSrcTemp[3 * jj + 2] ? max_valueBGR[2] : puSrcTemp[3 * jj + 2];
					min_valueBGR[2] = min_valueBGR[2]<puSrcTemp[3 * jj + 2] ? min_valueBGR[2] : puSrcTemp[3 * jj + 2];
				}
			}

			//计算融合因子以及反射率值
			for (int k = 0; k<3; k++)
			{
				if (max_valueBGR[k] > 0)
				{
					fusionFactor = (max_valueBGR[k] - min_valueBGR[k])*1.0 / max_valueBGR[k];
					value = max_valueBGR[k] * fusionFactor + min_valueBGR[k] * (1 - fusionFactor);
					puDst[3 * j + k] = value /*(int)(255 * puSrc[3 * j + k] / value)*/;
				}
				else
				{
					puDst[3 * j + k] = puSrc[3 * j + k];
				}
				
			}

		}//j

	}//i

} 


/*******************************************
*Function: Calculate the Mean Square Error between two images
*Input:src, the original image
*      predict, an predicted image obtained by methods
*Output:
*return：MSE value, double
*date: 2019.01.15   wangbingshu
********************************************/
double CalulateOneImgMSE(Mat& src,Mat& predict)
{
	double MSE = 0;
	int height = src.rows;
	int width = src.cols; 
	int iChannel = src.channels();
	int temp = 0;

	if (iChannel == 3)
	{
		for (int i = 0; i<height; i++)
		{
			uchar* puSrc = src.ptr(i);
			uchar* puPredict = predict.ptr(i);
			for (int j = 0; j<width; j++)
			{
				for (int k = 0; k<3; k++)
				{
					temp = puSrc[3 * j + k] - puPredict[3 * j + k];
					MSE += temp*temp;
				}
			}
		}
		MSE = MSE / (height*width*3);
	}
	if (iChannel == 1)
	{
		for (int i = 0; i<height; i++)
		{
			uchar* puSrc = src.ptr(i);
			uchar* puPredict = predict.ptr(i);
			for (int j = 0; j<width; j++)
			{
				temp = puSrc[j] - puPredict[j];
				MSE += temp*temp;
			}
		}
		MSE = MSE / (height*width);
	}
	

	return MSE;
}

double GetMean(Mat& grayImg)
{
	int height = grayImg.rows;
	int width = grayImg.cols;
	double sum = 0;
	double mean = 0;
	for (int i = 0; i < height; i++)
	{
		uchar* puGrayImg = grayImg.ptr(i);
		for (int j = 0; j < width; j++)
		{
			sum += puGrayImg[j];
		}
	}
	mean = sum / (height*width);

	return mean;
}




/*******************************************
*Function: Find a reference global background color
*Input: bgImg, RGB channels, illumination bg 
*       binary, gray-scale, 0(text),255(bg)
*       shadowMap, gray-scale, 0(shadow), 255(nonshadow)
*Output: iRefBg, the referenced background
*return：void
*date: 2019.01.15   wangbingshu
********************************************/
void FindReferenceBg(Mat& bgImg, Mat& binary,Mat& shadowMap, int iRefBg[3])
 {
	int height = bgImg.rows;
	int width = bgImg.cols;
	double BGR[3] = { 0 };
	double countNum = 0;
	for (int i = 0; i < height; i++)
	{
		uchar* puBgImg = bgImg.ptr(i);
		uchar* puBinary = binary.ptr(i);
		uchar* puShadowMap = shadowMap.ptr(i);
		for (int j = 0; j < width; j++)
		{
			if (puShadowMap[j]>0 && puBinary[j]>0)
			{
				BGR[0] += puBgImg[3 * j];
				BGR[1] += puBgImg[3 * j+1];
				BGR[2] += puBgImg[3 * j+2];
				countNum++;
			}
		}
	}

	int avg_bgr[3];
	for (int i=0;i<3;i++)
	{
		avg_bgr[i] = BGR[i] / countNum; 
	}
	
	double curMin = 255 * 255 * 3;
	double diff=0,curMag=0;
	for (int i = 0; i < height; i++)
	{
		uchar* puBgImg = bgImg.ptr(i);
		uchar* puShadowMap = shadowMap.ptr(i);
		uchar* puBinary = binary.ptr(i);
		for (int j = 0; j < width; j++)
		{
			if (puShadowMap[j]>0 && puBinary[j]>0)
			{
				curMag = 0;
				for (int k = 0; k < 3; k++) 
				{
					diff = puBgImg[k] - avg_bgr[k];
					curMag += diff * diff;
				}
				if (curMag < curMin)
				{
					curMin = curMag;
					iRefBg[0] = puBgImg[3*j];
					iRefBg[1] = puBgImg[3*j+1];
					iRefBg[2] = puBgImg[3*j+2];
				}

			}
		}
	}

}


/*******************************************
*Function: Remove the shadow by bg color ratio
*Input: img, RGB channels, original image
*       localBgColorImg, RGB channels, local background image
*       iRefBg, reference global bg
*Output: result, RGB channels, image without shadows
*return：void
*date: 2019.01.15   wangbingshu
********************************************/
void RemovalShadowByBgColorRatio(Mat& img, Mat& localBgColorImg, int iRefBg[3], Mat& result)
{
	int height = img.rows;
	int width = img.cols;
	double ratio;
	for (int i = 0; i < height; i++)
	{
		uchar* puImg = img.ptr(i);
		uchar* puLocalRef = localBgColorImg.ptr(i);
		uchar* puResult= result.ptr(i);
		for (int j = 0; j < width; j++)
		{
			for (int k=0;k<3;k++)
			{
				ratio = 1.0*puLocalRef[3 * j+k] / iRefBg[k];
				puResult[3 * j+k] = puImg[3 * j+k]/ratio;
			}
		}
	}
}


/*******************************************
*Function: Calculate the shadow strength factor, 
*          The ratio of nonshaow region bakcground and shadow region background
*Input: localBgColorImg, RGB channels, local background image
*       binary, gray-scale, 0(text),255(bg)
*       shadowMap, gray-scale, 0(shadow), 255(nonshadow)
*Output:dSSF, shadow strength factor (>=1)
*return：void
*date: 2019.01.15   wangbingshu
********************************************/
void CalculateShadowStrengthFactor(Mat& localBgColorImg, Mat& binaryImg, Mat& shadowMap, double dSSF[3])
{
	dSSF[0] = 1;
	dSSF[1] = 1;
	dSSF[2] = 1;

	Mat erodeBinary(shadowMap.size(), CV_8UC1, 1);
	Mat dilateBinary(shadowMap.size(), CV_8UC1, 1);
	int size = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), Point(size, size));
	erode(shadowMap, erodeBinary, element);
	dilate(shadowMap, dilateBinary, element);

	int height = localBgColorImg.rows;
	int width = localBgColorImg.cols;
	double bright[3] = { 0 };
	double brightNum = 0;

	double dark[3] = { 0 };
	double darkNum = 0;
	for (int i = 0; i < height; i++)
	{
		uchar* puBgColorImg = localBgColorImg.ptr(i);
		uchar* puErodeBinary = erodeBinary.ptr(i);
		uchar* puDilateBinary = dilateBinary.ptr(i);
		uchar* puBinary = binaryImg.ptr(i);

		for (int j = 0; j < width; j++)
		{
			if (puErodeBinary[j]>0 && puBinary[j]>0)     //统计非阴影区域的背景参考值
			{
				bright[0] += puBgColorImg[3 * j];
				bright[1] += puBgColorImg[3 * j + 1];
				bright[2] += puBgColorImg[3 * j + 2];
				brightNum++;
			}
			if (puDilateBinary[j] == 0 && puBinary[j]>0) //统计阴影区域的背景参考值
			{
				dark[0] += puBgColorImg[3 * j];
				dark[1] += puBgColorImg[3 * j + 1];
				dark[2] += puBgColorImg[3 * j + 2];
				darkNum++;
			}
		}
	}
	if (darkNum>0&&brightNum>0)
	{
		for (int i=0;i<3;i++)
		{
			bright[i] = 1.0* bright[i] / brightNum;
			dark[i] = 1.0*dark[i] / darkNum;
			dSSF[i] = bright[i] / dark[i];
		}
	}
}


/*******************************************
*Function: Fill in the hole
*Input: src,RGB channels, original image
*       shadowMap, gray-scale, 0(shadow), 255(nonshadow)
*       dSSF, shadow strength factor
*       iRefBg, reference background color
*Output:result, RGB channels, image with some holes filled
*return:void
*date: 2019.01.15   wangbingshu
********************************************/
void FillHole(Mat &src, Mat& shadowMap, double dSSF[3], int iRefBg[3], Mat &result)
{
	Mat gray(src.size(), CV_8UC1, 1);
	Mat binary(src.size(), CV_8UC1, 1);
	cvtColor(src, gray, CV_BGR2GRAY);
	ThresholdIntegral(gray, 1.0, binary);

	Mat erodeBinary(src.size(), CV_8UC1, 1);
	int size = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), Point(size, size));
	gray.copyTo(erodeBinary);
	erode(binary, erodeBinary, element);

	double ratio[3] = { 1 };
	double countNum[3] = { 0 };
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
	{
		uchar* puSrc = src.ptr(i);
		uchar* puShadowMap = shadowMap.ptr(i);
		uchar* puErodeBinary = erodeBinary.ptr(i);
		uchar* puResult = result.ptr(i);
		for (int j = 0; j < width; j++)
		{
			if (puShadowMap[j] == 255 && puErodeBinary[j] == 255)
			{
				for (int k=0;k<3;k++)
				{
					if (puSrc[3 * j + k] > 0)
					{
						ratio[k] += 1.0*puResult[3 * j + k] / puSrc[3 * j + k];
						countNum[k]++;
					}
				}
			}
		}
	}

	for (int i=0;i<3;i++)
	{
		if (countNum[i]>0)
		{
			ratio[i] = ratio[i] / countNum[i];
		}
		if (dSSF[i] > 1.3)
		{
			dSSF[i] = (dSSF[i] + 1) / 2;
		}
	}
	 
	if (ratio[0]>0 && ratio[1]>0 && ratio[2]>0&& countNum[0]>0 &&countNum[1]>0 && countNum[2]>0)
	{
		for (int i = 0; i < height; i++)
		{
			uchar* puResult = result.ptr(i);
			uchar* puSrc = src.ptr(i);
			uchar* puBinary = binary.ptr(i);
			uchar* puShadowMap = shadowMap.ptr(i);
		 
			for (int j = 0; j < width; j++)
			{
				if (puShadowMap[j]==0 && puBinary[j] == 0)  //阴影（puShadowMap[j]==0 ）中的text(puBinary[j] == 0)
				{
				   for (int k=0;k<3;k++)
					{
						puResult[3 * j + k] =  dSSF[k] * puSrc[3 * j + k] / ratio[k];
					}
				}
				if (puShadowMap[j] == 255 && puBinary[j] == 0)   //非阴影（puShadowMap[j]==255 ）中的text(puBinary[j] == 0)
				{
					puResult[3 * j] = puSrc[3 * j] ;
					puResult[3 * j + 1] = puSrc[3 * j + 1];
					puResult[3 * j + 2] = puSrc[3 * j + 2];
				}
			}
		}
	}

}

void ToneAdjust(Mat& img, Mat& binaryImg,Mat& shadowMap, double dSSF[3], Mat& result)
{
	//cout << dSSF[0] << " " << dSSF[1] << " " << dSSF[2] << endl;
	int height = img.rows;
	int width = img.cols;
	int size = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), Point(size, size));
	if ((dSSF[0]+ dSSF[0]+ dSSF[0])/3<2)
	{
		erode(shadowMap, shadowMap, element);
	}

	//对比度调整 
	double ratioBGR[3] = {0};
	int diffBGR[3] = {0};
	double ratioTemp;
	for (int i = 0; i < height; i++)
	{
		uchar* puImg = img.ptr(i);
		uchar* puBinaryImg = binaryImg.ptr(i);
		uchar* puShadowMap = shadowMap.ptr(i);
		uchar* puResult = result.ptr(i);
		for (int j = 0; j < width; j++)
		{
			// 针对阴影区域的文本进行处理
			if (puBinaryImg[j] == 0 && puShadowMap[j]==0)
			{
				for (int k=0;k<3;k++)
				{
					diffBGR[k] = puResult[3 * j + k] - puImg[3 * j + k];
					ratioBGR[k] = 1.0*(puResult[3 * j + k] + 1) / (puImg[3 * j + k] + 1);
					
					if (puImg[3 * j + k]<8)
					{
						ratioTemp = dSSF[k]+1;
						puResult[3 * j + k] = puImg[3 * j + k] * ratioTemp;
					}
					else if (puImg[3 * j + k]<15)
					{
						ratioTemp = dSSF[k];
						puResult[3 * j + k] = puImg[3 * j + k] * ratioTemp;
					}else
				    if(puImg[3 * j + k]<30 )
					 {
						 ratioTemp = (1 + ratioBGR[k]) / 2;
						 puResult[3 * j + k] = puImg[3 * j + k] * ratioTemp;
					 }
					else if(puImg[3 * j + k]<60 && ratioBGR[k]>dSSF[k])
					{
						puResult[3 * j + k] = puImg[3 * j + k] * (1 + dSSF[k]) / 2;
					}else
					{
						 ;
					}
					
				}
			}

			//针对非阴影区域文本处理
			if (puBinaryImg[j] == 0 && puShadowMap[j] == 255)
			{
				for (int k = 0; k < 3; k++)
				{
					puResult[3 * j + k] = puImg[3 * j + k];
				}
			}


		}
	}

}

/*******************************************
*Function:Remova shadows from document images
*Input:img, RGB channels, original image
*Output:result, RGB channels, image without shadows
*return：void
*date: 2019.01.15   wangbingshu 
********************************************/
void ShadowRemoval(Mat& img, Mat& result)
{
	Mat img2(img.size(), CV_8UC3, 3);
	img.copyTo(img2);

	//1、Get binarization image, text(0) background(255)
	Mat gray(img.size(), CV_8UC1, 1);
	Mat binaryImg(img.size(), CV_8UC1, 1);
	cvtColor(img, gray, CV_BGR2GRAY);
	ThresholdIntegral(gray,1.0, binaryImg);  //求取二值化图像，文本（0）和背景（255）
	//imshow("binaryImg", binaryImg);
	//waitKey(0);

    //2、Get the local bg image and the shadow map
	Mat bg(Size(img.cols, img.rows), CV_8UC3, 3);
	Mat shadowMap(img.size(), CV_8UC1, 1);
	for (int i = 0; i < 3; i++)
	{
		EvaluationIllumination(img,2,bg);  //bg 背景亮暗分布图							  //imshow("imgResult", imgResult);
		bg.copyTo(img);
	}
	cvtColor(bg, gray, CV_BGR2GRAY);
	medianBlur(gray, gray, 3); 
	threshold(gray, shadowMap, 0, 255, CV_THRESH_OTSU);   
	 
    //3、Find a global bg reference，store it in iRefBg
	int iRefBg[3] = { 0 };
	FindReferenceBg(bg, binaryImg,shadowMap, iRefBg);   

	//4、Evaluate the local bg color image based on neighboring information
	Mat localBgColorImg(Size(img.cols, img.rows), CV_8UC3, 3);
	EvaluationIllumination(img2, 1, localBgColorImg);  
	 
	//5、Remove the shadows from original image by bg color ratio
	RemovalShadowByBgColorRatio(img2, localBgColorImg, iRefBg, result);
	//imshow("result", result);

	//6、Fill in the possible holes.
	double dSSF[3] = { 0 };
	CalculateShadowStrengthFactor(localBgColorImg, binaryImg,shadowMap, dSSF);
	//cout << dSSF[0] << " " << dSSF[1] << " " << dSSF[2] << endl;
	double avg = (dSSF[0] + dSSF[1] + dSSF[2]) / 3;
	if (avg<1.39 )
	{
		FillHole(img2,  shadowMap, dSSF, iRefBg, result);
	}

	//用原图进行调整，三个通道变化太大的
	ToneAdjust(img2,binaryImg, shadowMap,dSSF,result);
	img2.copyTo(img);
}




