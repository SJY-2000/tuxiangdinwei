#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap(0);

	int cnt = 0;
	Mat frame;
	Mat tempMat, refMat;
	Mat resultMat;

	if (!cap.isOpened())
	{
		std::cout << "不能打开视频文件" << std::endl;
		return -1;
	}
	
	cap >> frame;

	while (1)
	{

		cap >> frame;
		if (cnt == 0)
		{
			Rect2d r;
			r = selectROI(frame, true);
			tempMat = frame(r);
			tempMat.copyTo(refMat);
			destroyAllWindows();
		}
		int match_method = 0;

		matchTemplate(frame, refMat, resultMat, match_method);

		normalize(resultMat, resultMat, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal, maxVal; Point minLoc, maxLoc; Point matchLoc;
		minMaxLoc(resultMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		if ((match_method == TM_SQDIFF) || (match_method == TM_SQDIFF_NORMED))
		{
			matchLoc = minLoc;
		}
		else
		{
			matchLoc = maxLoc;
		}
	
		rectangle(frame, matchLoc, Point(matchLoc.x + refMat.cols, matchLoc.y + refMat.rows), Scalar(255, 255, 0), 2, 8);

		imshow("frame", frame);
		imshow("refMat", refMat);
		imshow("resultMat", resultMat);
		waitKey(30);
		cnt++;
	}
}