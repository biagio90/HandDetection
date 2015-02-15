//============================================================================
// Name        : HandTracking.cpp
// Author      : Biagio Brattoli
// Version     : 1.0
// Copyright   :
// Description : HandTracking in a video stream using OpenCV
//============================================================================

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define DEFAULT_CAMERA 0

using namespace std;
void calculateSkinColor(cv::VideoCapture capture, vector<cv::Vec3f> *avgs_r, vector<cv::Vec3b> *mins_r, vector<cv::Vec3b> *maxs_r);
void calculateRectAverage(cv::Mat image, cv::Rect rect, cv::Vec3b *min_r, cv::Vec3b *max_r, cv::Vec3f *avg_r);
cv::Mat bigComponent(cv::Mat binary);
cv::Mat getHandSample(cv::VideoCapture capture, vector<cv::Vec3i> areas);
vector<cv::Vec3i> getAreas();
cv::Mat segmentation(cv::Mat image, vector<cv::Vec3f> avg, vector<cv::Vec3b> mins, vector<cv::Vec3b> maxs);
cv::Mat thresholding(cv::Mat image, cv::Vec3f avg, cv::Vec3b min, cv::Vec3b max);
cv::Mat getImage(cv::VideoCapture capture);

int main( int argc, char **argv ) {
	cv::Mat image;
	cv::VideoCapture capture(DEFAULT_CAMERA);

	// GET SKIN COLOUR
	vector<cv::Vec3f> avgs;
	vector<cv::Vec3b> mins;
	vector<cv::Vec3b> maxs;
	calculateSkinColor(capture, &avgs, &mins, &maxs);
	//cout << "(" << avg[0] << ", " << avg[1] << ", " << avg[2] << ")" << endl;

	// HAND TRAKING
	while(cv::waitKey(1)==-1){
		capture.read(image);
		cv::Mat imageHSV;
		cv::cvtColor(image, imageHSV, CV_BGR2YCrCb);
		//cv::cvtColor(image, imageHSV, CV_BGR2YCrCb);
		//cv::GaussianBlur(imageHSV, imageHSV, cv::Size(9, 9), 0, 0);
		//cout << "(" << (int)imageHSV.at<cv::Vec3b>(240, 320)[0] << ", " << (int)imageHSV.at<cv::Vec3b>(240, 320)[1] << ", "<< (int)imageHSV.at<cv::Vec3b>(240, 320)[2] << endl;

		cv::Mat binary = segmentation(imageHSV, avgs, mins, maxs);
		cv::imshow("binary", binary);
		//cv::inRange(imageHSV, cv::Scalar(0, 131, 80), cv::Scalar(255, 185, 135), binary);
		cv::Mat contour = bigComponent(binary);

		cv::Mat output = cv::Mat::zeros(image.rows, image.cols, image.type());
		contour.copyTo(image, contour);
		cv::imshow("frame", image);
		//cv::imshow("mask", contour);

	}
	capture.release();

	return 0;
}

cv::Mat segmentation(cv::Mat image, vector<cv::Vec3f> avg, vector<cv::Vec3b> mins, vector<cv::Vec3b> maxs) {
	cv::Mat binaries = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	string name[9] = {"center", "up1", "down left", "down right", "down2", "left", "right"};
	for(uint i=0; i<avg.size(); i++){
		cv::Mat th = thresholding(image, avg[i], mins[i], maxs[i]);
		cv::imshow(name[i], th);
		cv::add(binaries, th, binaries);
	}
	cv::medianBlur(binaries, binaries, 15);
	cv::Mat kernel(9, 9, CV_8UC1);
	//cv::erode(binaries, binaries, kernel);
	cv::dilate(binaries, binaries, kernel);
	cv::erode(binaries, binaries, kernel);

	return binaries;
}

cv::Mat thresholding(cv::Mat image, cv::Vec3f avg, cv::Vec3b min, cv::Vec3b max){
	cv::Mat binary;
	cv::inRange(image, cv::Scalar(min[0], min[1], min[2]),
					   cv::Scalar(max[0], max[1], max[2]), binary);
	cv::GaussianBlur(binary, binary, cv::Size(15,15), 0, 0);
	return binary;
}


void calculateSkinColor(cv::VideoCapture capture, vector<cv::Vec3f> *avgs_r, vector<cv::Vec3b> *mins_r, vector<cv::Vec3b> *maxs_r){
	vector<cv::Vec3f> avgs;
	vector<cv::Vec3b> mins;
	vector<cv::Vec3b> maxs;
	vector<cv::Vec3i> areas = getAreas();

	cv::Mat image = getHandSample(capture, areas);
	cv::cvtColor(image, image, CV_BGR2YCrCb);

	for(vector<cv::Vec3i>::iterator iter = areas.begin();iter != areas.end();iter++){
		cv::Rect rect((*iter)[0]-(*iter)[2], (*iter)[1]-(*iter)[2], (*iter)[2], (*iter)[2]);
		cv::Vec3f avg;
		cv::Vec3b min;
		cv::Vec3b max;
		calculateRectAverage(image, rect, &min, &max, &avg);
		//cout << "(" << avg[0] << ", " << avg[1] << ", "<< avg[2] << endl;
		cout << "(" << (int)min[0] << ", " << (int)min[1] << ", "<< (int)min[2] << endl;
		cout << "(" << (int)max[0] << ", " << (int)max[1] << ", "<< (int)max[2] << endl;

		avgs.push_back(avg);
		mins.push_back(min);
		maxs.push_back(max);
	}

	*avgs_r = avgs;
	*mins_r = mins;
	*maxs_r = maxs;

	return;
}

cv::Mat getHandSample(cv::VideoCapture capture, vector<cv::Vec3i> areas){
	cv::Mat image;
	while(cv::waitKey(1)==-1){
		capture.read(image);
		cv::Mat imageRect(image);
		for(vector<cv::Vec3i>::iterator iter = areas.begin();iter != areas.end();iter++){
			cv::Rect rect((*iter)[0]-(*iter)[2], (*iter)[1]-(*iter)[2], (*iter)[2], (*iter)[2]);
			cv::rectangle(imageRect, rect, cv::Scalar(0, 0, 255), 2, 1);
		}
		cv::imshow("frame", imageRect);
	}

	return image;
}

void calculateRectAverage(cv::Mat image, cv::Rect rect, cv::Vec3b *min_r, cv::Vec3b *max_r, cv::Vec3f *avg_r){
	cv::Vec3f avg(0, 0, 0);
	cv::Vec3f min(256, 256, 256);
	cv::Vec3f max(0, 0, 0);

	int count=0;
	for(int i=rect.x; i<rect.x+rect.width;i++) {
		for(int j=rect.y; j<rect.y + rect.height; j++){
			cv::Vec3b pixel = image.at<cv::Vec3b>(i,j);
			avg[0] += pixel[0];
			avg[1] += pixel[1];
			avg[2] += pixel[2];
			count++;

			for(int k=0; k<3; k++){
				if(pixel[k] < min[k]){
					min[k] = pixel[k];
				}
				if(pixel[k] > max[k]){
					max[k] = pixel[k];
				}

			}

		}
	}

	avg[0] = avg[0]/count;
	avg[1] = avg[1]/count;
	avg[2] = avg[2]/count;

	*avg_r = avg;
	*min_r = min;
	*max_r = max;
	return;
}


cv::Mat bigComponent(cv::Mat binary) {
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	findContours( binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0) );

	/// Draw contours
	cv::Mat drawing = cv::Mat::zeros( binary.size(), CV_8UC3 );
	int max_area = 0, big_contour=-1;
	for( uint i = 0; i< contours.size(); i++ ){
		double area =contourArea( contours[i],false);  //  Find the area of contour
		if(area>max_area){
			max_area=area;
			big_contour=i;                //Store the index of largest contour
		}
	}

	drawContours(drawing, contours,big_contour, cv::Scalar( 0,0,255), 5, 8, hierarchy ); // Draw the largest contour using previously stored index.

	return drawing;
}

cv::Mat getImage(cv::VideoCapture capture){
	cv::Mat image;
	capture.read(image);
	//cv::resize(image, image, cv::Size(240, ));
	return image;
}

vector<cv::Vec3i> getAreas(){

	vector<cv::Vec3i> areas;
	areas.push_back(cv::Vec3i(320, 240, 20)); //center
	//areas.push_back(cv::Vec3i(320, 130, 20)); //up2
	areas.push_back(cv::Vec3i(320, 350, 20)); //up1
	//areas.push_back(cv::Vec3i(320, 180, 20)); //down1
	areas.push_back(cv::Vec3i(280, 300, 20)); //down left
	areas.push_back(cv::Vec3i(360, 300, 20)); //down right
	areas.push_back(cv::Vec3i(320, 300, 20)); //down2
	areas.push_back(cv::Vec3i(280, 240, 20)); //left
	areas.push_back(cv::Vec3i(360, 240, 20)); //right

	return areas;
}
