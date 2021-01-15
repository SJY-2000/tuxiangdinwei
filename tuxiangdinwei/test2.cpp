#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int calcHOG(Mat src, float *Hist, int nAngle, int cellsize_x, int cellsize_y)
{
	Mat gx, gy;
	Mat mag, angle;

	Sobel(src, gx, CV_32F, 1, 0, 1);
	Sobel(src, gy, CV_32F, 0, 1, 1);
	cartToPolar(gx, gy, mag, angle, true);

	int nx = src.cols / cellsize_x;
	int ny = src.rows / cellsize_y;

	Rect roi(0, 0, cellsize_x, cellsize_y);
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			Mat roiMat(src.rows, src.cols, CV_8UC1);
			Mat roiMag(src.rows, src.cols, CV_8UC1);
			Mat roiAgl(src.rows, src.cols, CV_8UC1);

			roi.x = j*cellsize_x;
			roi.y = i*cellsize_y;
		
			roiMat = src(roi);
			roiMag = mag(roi);
			roiAgl = angle(roi);
		
			int head = (i*nx + j)*nAngle;

	
			for (int n = 0; n < roiMat.rows; n++)
			{
				uchar* Agl = roiAgl.ptr<uchar>(n);
				uchar* Mag = roiMag.ptr<uchar>(n);
				for (int m = 0; m < roiMat.cols; m++)
				{
					int c = (int)Agl[m] / 45;
					Hist[head + c] += (float)Mag[m];
				}
			}
		}
	}

	return 0;
}

float normL2(float *sHist, float *dHist, int bins)
{
	float distance = 0;
	for (int i = 0; i < bins; i++)
	{
		distance += pow((sHist[i] - dHist[i]), 2);
	}
	distance = sqrt(distance);

	return distance;
}

int HOGmatchTemplate(Mat frame, Mat refMat, Mat &resultMat)//,Point roi) 
{
	
	int cellsize_x = refMat.cols;
	int cellsize_y = refMat.rows;

	int nx = frame.cols - cellsize_x;
	int ny = frame.rows - cellsize_y;

	int nAngle = 8;

	int bins = nx*ny*nAngle;

	float * ref_hist = new float[bins];
	memset(ref_hist, 0, sizeof(float)*bins);

	int bins_1 = 1 * nAngle;
	float * hist = new float[bins_1];
	memset(hist, 0, sizeof(float)*bins_1);

	int reCode = 0;

	reCode = calcHOG(refMat, hist, nAngle, cellsize_x, cellsize_y);
	
	resultMat = Mat::zeros(frame.rows - cellsize_y, frame.cols - cellsize_x, CV_32F);
	for (int i = 0; i < frame.rows - cellsize_y; i++)
	{
		float *data = resultMat.ptr<float>(i);
		for (int j = 0; j < frame.cols - cellsize_x; j++)
		{
			Rect roi = Rect(j, i, cellsize_x, cellsize_y);
		
			reCode = calcHOG(frame(roi), ref_hist + (i*nx + j)*nAngle, nAngle, cellsize_x, cellsize_y);
			float dis = normL2(ref_hist + (i*nx + j)*nAngle, hist, bins_1);

			data[j] = dis;
	
		}
	}

	if (reCode != 0)
	{
		delete[] ref_hist;
		delete[] hist;
		return -1;
	}

	return 0;

}


int main()
{
	Mat refMat = imread("G:\\picture\\181.jpg", 0);
	Mat frame = imread("G:\\picture\\18.jpg", 0);

	Mat resultMat;
	Mat rframe;
	Mat rrefMat;

	float scale_small = 1;
	float scale_lager = 1 / scale_small;

	Size dsize = Size(round(scale_small * frame.cols), round(scale_small * frame.rows));
	resize(frame, rframe, dsize, 0, 0, INTER_AREA);
	Size dsize_1 = Size(round(scale_small * refMat.cols), round(scale_small * refMat.rows));
	resize(refMat, rrefMat, dsize_1, 0, 0, INTER_AREA);

	HOGmatchTemplate(rframe, rrefMat, resultMat);

	normalize(resultMat, resultMat, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal; Point minLoc, maxLoc; Point matchLoc;
	minMaxLoc(resultMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	matchLoc.x = minLoc.x * scale_lager;
	matchLoc.y = minLoc.y * scale_lager;


	rectangle(frame, matchLoc, Point(matchLoc.x + refMat.cols, matchLoc.y + refMat.rows), Scalar(255, 255, 0), 2, 8);

	imshow("frame", frame);
	imshow("refMat", refMat);
	waitKey(0);

}