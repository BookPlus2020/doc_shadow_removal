#pragma once
#ifndef _DOC_SHADOW_REMOVAL_H
#define _DOC_SHADOW_REMOVAL_H
#include"common.h"
 
 

/*******************************************
*Function: Binarization using integral method
which is from https://github.com/phryniszak/AdaptiveIntegralThresholding
*Input: inputMat, gray-scale image
*       thre, the threshold to classify the fg and bg
*Output: outputMat, gray-scale, (0 bg, 255 fg)
*
*return£ºvoid
*date: 2019.01.15   wangbingshu
********************************************/
void ThresholdIntegral(Mat &inputMat, double thre, Mat &outputMat);

 
/*******************************************
*Function: Evalaute the bright/dark extent in an document image
*Input: src£¬ RGB channels
*       size, neighboring size (2*size+1)*(2*size+1)
*
*Output:dst£¬ RGB channels,
*
*return£ºvoid
*date: 2019.01.15   wangbingshu
********************************************/
void EvaluationIllumination(Mat& src, int size,Mat& dst);

/*******************************************
*Function: Calculate the Mean Square Error between two images
*Input:src, the original image
*      predict, an predicted image obtained by methods
*Output:
*return£ºMSE value, double
*date: 2019.01.15   wangbingshu
********************************************/
double CalulateOneImgMSE(Mat& src, Mat& predict);

double GetMean(Mat& grayImg);

 

/*******************************************
*Function: Find a reference global background color
*Input: bgImg, RGB channels, illumination bg
*       binary, gray-scale, 0(text),255(bg)
*       shadowMap, gray-scale, 0(shadow), 255(nonshadow)
*Output: iRefBg, the referenced background
*return£ºvoid
*date: 2019.01.15   wangbingshu
********************************************/
void FindReferenceBg(Mat& bg, Mat& binary, Mat& shadowMap, int iRefBg[3]);

/*******************************************
*Function: Remove the shadow by bg color ratio
*Input: img, RGB channels, original image
*       localBgColorImg, RGB channels, local background image
*       iRefBg, reference global bg
*Output: result, RGB channels, image without shadows
*return£ºvoid
*date: 2019.01.15   wangbingshu
********************************************/
void RemovalShadowByBgColorRatio(Mat& img,Mat& localBgColorImg,int iRefBg[3], Mat& result);

/*******************************************
*Function: Calculate the shadow strength factor,
*          The ratio of nonshaow region bakcground and shadow region background
*Input: localBgColorImg, RGB channels, local background image
*       binary, gray-scale, 0(text),255(bg)
*       shadowMap, gray-scale, 0(shadow), 255(nonshadow)
*Output:dSSF, shadow strength factor (>=1)
*return£ºvoid
*date: 2019.01.15   wangbingshu
********************************************/
void CalculateShadowStrengthFactor(Mat& imgResult, Mat& binaryImg, Mat& shadowMap,double dRef[3]);

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
void FillHole(Mat &src,  Mat& shadowMap, double dRef[3], int iRef[3], Mat &result);


void ToneAdjust(Mat& img, Mat& binaryImg, Mat& shadowMap, double dSSF[3], Mat& result);

/*******************************************
*Function:Remova shadows from document images
*Input:img, RGB channels, original image
*Output:result, RGB channels, image without shadows
*return£ºvoid
*date: 2019.01.15   wangbingshu
********************************************/
void ShadowRemoval(Mat& img, Mat& result);

#endif