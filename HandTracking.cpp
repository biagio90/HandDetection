//============================================================================
// Name        : HandTracking.cpp
// Author      : Biagio Brattoli
// Version     : 1.0
// Description : HandTracking in a video stream using OpenCV
//============================================================================

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define DEFAULT_CAMERA 0
#define ALPHA 3.0
#define BETA  50

using namespace std;
void calculateSkinColor(cv::VideoCapture capture, vector<cv::Vec3f> *avgs_r, vector<cv::Vec3b> *mins_r, vector<cv::Vec3b> *maxs_r);
void calculateRectAverage(cv::Mat image, cv::Rect rect, cv::Vec3b *min_r, cv::Vec3b *max_r, cv::Vec3f *avg_r);
cv::Mat bigComponent(cv::Mat binary);
cv::Mat getHandSample(cv::VideoCapture capture, vector<cv::Vec3i> areas);
vector<cv::Vec3i> getAreas();
cv::Mat segmentation(cv::Mat image, vector<cv::Vec3f> avg, vector<cv::Vec3b> mins, vector<cv::Vec3b> maxs);
cv::Mat thresholding(cv::Mat image, cv::Vec3f avg, cv::Vec3b min, cv::Vec3b max);
cv::Mat getImage(cv::VideoCapture capture);
void showImage(string name, cv::Mat image);
void showHand(cv::Mat image);
void increaseContrast(cv::Mat image, float alpha, int beta);

int main( int argc, char **argv ) {
	cv::Mat image;
	cv::VideoCapture capture(DEFAULT_CAMERA);

	// GET SKIN COLOUR using min and max value per each small sample squares
	vector<cv::Vec3f> avgs;
	vector<cv::Vec3b> mins;
	vector<cv::Vec3b> maxs;
	calculateSkinColor(capture, &avgs, &mins, &maxs);

	cv::Mat contour;
	// HAND TRAKING
	while(cv::waitKey(1)==-1){
		image = getImage(capture);

		cv::Mat binary = segmentation(image, avgs, mins, maxs); //find pixels of skin colour
		contour = bigComponent(binary); //find the biggest object

		// HERE some feature extraction and machine learning method can be used to check if it is a hand shape
		// for example, Fourier descriptors + SVM

		cv::Mat output = cv::Mat::zeros(image.rows, image.cols, image.type());
		contour.copyTo(image, contour); //draw the red contour around the hand
		showImage("frame", image);
	}
	capture.release();

	return 0;
}

cv::Mat segmentation(cv::Mat image, vector<cv::Vec3f> avg, vector<cv::Vec3b> mins, vector<cv::Vec3b> maxs) {
	cv::Mat imageHSV(image.clone());
	increaseContrast(imageHSV, ALPHA, BETA);
	cv::cvtColor(imageHSV, imageHSV, CV_BGR2HSV); //change image to HSV
	cv::Mat binaries = cv::Mat::zeros(imageHSV.rows, imageHSV.cols, CV_8UC1);

	for(uint i=0; i<mins.size(); i++){	//create a different binary map per each sample square
		cv::Mat th = thresholding(imageHSV, avg[i], mins[i], maxs[i]);
		cv::add(binaries, th, binaries); //add all result in one map
	}
	//image processing to clean the binary map
	cv::medianBlur(binaries, binaries, 15);
	cv::Mat kernel(9, 9, CV_8UC1);
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

/*
 *Calculate min and max of pixels in different sample squares. This will be used for segmentation
*/
void calculateSkinColor(cv::VideoCapture capture, vector<cv::Vec3f> *avgs_r, vector<cv::Vec3b> *mins_r, vector<cv::Vec3b> *maxs_r){
	vector<cv::Vec3f> avgs;
	vector<cv::Vec3b> mins;
	vector<cv::Vec3b> maxs;
	vector<cv::Vec3i> areas = getAreas(); //create sample squares position

	cv::Mat image = getHandSample(capture, areas); //wait until the user give the image with the hand on the sample squares
	increaseContrast(image, ALPHA, BETA);
	cv::cvtColor(image, image, CV_BGR2HSV);

	//Calculate min and max for each sample squares
	for(vector<cv::Vec3i>::iterator iter = areas.begin();iter != areas.end();iter++){
		cv::Rect rect((*iter)[0]-(*iter)[2], (*iter)[1]-(*iter)[2], (*iter)[2], (*iter)[2]);
		cv::Vec3f avg;
		cv::Vec3b min;
		cv::Vec3b max;
		calculateRectAverage(image, rect, &min, &max, &avg);

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
		image = getImage(capture);
		cv::Mat imageRect(image.clone());
		//draw sample squares
		for(vector<cv::Vec3i>::iterator iter = areas.begin();iter != areas.end();iter++){
			cv::Rect rect((*iter)[0]-(*iter)[2], (*iter)[1]-(*iter)[2], (*iter)[2], (*iter)[2]);
			cv::rectangle(imageRect, rect, cv::Scalar(0, 0, 255), 2, 1);
		}
		showHand(imageRect); //draw hand position
		showImage("frame", imageRect);
	}

	return image;
}

/*
 * draw hand on the image which helps the user for the initial position
 */
void showHand(cv::Mat image){
	cv::Mat hand = cv::imread("hand.jpg");
	if(hand.empty()) return;

	cv::Mat th;
	cv::cvtColor(hand, th, CV_BGR2GRAY);
	cv::threshold(th, th, 50, 255, CV_THRESH_BINARY);
	hand.copyTo(image, th);
}

/*
 * For each rect, finds max and min. In this version of the software the average is not used.
 */
void calculateRectAverage(cv::Mat image, cv::Rect rect, cv::Vec3b *min_r, cv::Vec3b *max_r, cv::Vec3f *avg_r){
	cv::Vec3f avg(0, 0, 0);
	cv::Vec3f min(256, 256, 256);
	cv::Vec3f max(0, 0, 0);

	int count=0;
	for(int i=rect.x; i<rect.x+rect.width;i++) {
		for(int j=rect.y; j<rect.y + rect.height; j++){
			cv::Vec3b pixel = image.at<cv::Vec3b>(j,i);
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

/*
 * Finds the biggest connected component of a binary image
 */
cv::Mat bigComponent(cv::Mat binary) {
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	findContours( binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0) );

	/// Draw contours
	cv::Mat drawing = cv::Mat::zeros( binary.size(), CV_8UC3 );
	int max_area = 0, big_contour=-1;
	for( uint i = 0; i< contours.size(); i++ ){
		double area =contourArea( contours[i],false); //  Find the area of contour
		if(area>max_area){
			max_area=area;
			big_contour=i; //Store the index of largest contour
		}
	}

	drawContours(drawing, contours,big_contour, cv::Scalar( 0,0,255), 5, 8, hierarchy ); // Draw the largest contour using previously stored index.

	return drawing;
}

cv::Mat getImage(cv::VideoCapture capture){
	cv::Mat image;
	capture.read(image);
	cv::resize(image, image, cv::Size(640, 480));
	return image;
}

void increaseContrast(cv::Mat image, float alpha, int beta) {

	image.convertTo(image, -1, alpha, beta); // increase contrast by multiple each pixel for a value bigger than 1

	// increase contrast by using histogram equalization on the Y channel of YCrCb color space
/*	cv::cvtColor(image, image, CV_BGR2YCrCb);
	vector<cv::Mat> channels(3);
	cv::split(image, channels);
	//for( int c = 0; c < 3; c++ ) {
		cv::equalizeHist(channels[0], channels[0]);
	//}
	cv::merge(channels, image);
	cv::cvtColor(image, image, CV_YCrCb2BGR);
*/
}

void showImage(string name, cv::Mat image){
	cv::flip(image, image, 1); //mirroring
	cv::imshow(name, image);
}

/*
 * Create the sample squares for skin color extraction.
 * In this version, there are 7 squares hard coded based on a 640x480 image.
 */
vector<cv::Vec3i> getAreas(){
	cv::Vec2i center(320, 240);
	int size = 20;
	vector<cv::Vec3i> areas;
	areas.push_back(cv::Vec3i(center[0], center[1], size)); //center
	areas.push_back(cv::Vec3i(center[0], 	center[1]-40, size)); //down1
	areas.push_back(cv::Vec3i(center[0]-40, center[1]+60, size)); //down left
	areas.push_back(cv::Vec3i(center[0]+40, center[1]+60, size)); //down right
	areas.push_back(cv::Vec3i(center[0], 	center[1]+60, size)); //down2
	areas.push_back(cv::Vec3i(center[0]-40, center[1]	, size)); //left
	areas.push_back(cv::Vec3i(center[0]+40, center[1]	, size)); //right

	return areas;
}
