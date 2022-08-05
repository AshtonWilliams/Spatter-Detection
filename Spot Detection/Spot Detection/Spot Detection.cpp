#include <opencv2\opencv.hpp>
#include <iostream>
#include <stack>
#include <fstream>
#include<numeric>

using namespace cv;
using namespace std;


//Count the number of spatter per category, and total
//Count the amount of spatter area per category, and total
//Calculate the spatter to build ratio per category, and total.
//saves the detected spot stats to a text file
void SaveStats(string filepath, Mat croppedImg, vector<int> areasSm, vector<int>areasMd, vector<int>areasLg) {

	filepath.resize(filepath.size() - 4);
	string fileName = filepath + "_SpatterQuantization.txt";
	ofstream statsFile;
	statsFile.open(fileName);

	//----------------------the number of spatter per category, and total----------------------
	statsFile << "Number of Spatter (small/red): " << areasSm.size() << endl;
	statsFile << "Number of Spatter (medium/green): " << areasMd.size() << endl;
	statsFile << "Number of Spatter (large/blue): " << areasLg.size() << endl;
	statsFile << "Total Number of Spatter: " << areasSm.size() + areasMd.size() + areasLg.size() << endl << endl;


	//----------------------the amount of spatter area per category, and total----------------------
	int smArea = accumulate(areasSm.begin(), areasSm.end(), 0);
	int mdArea = accumulate(areasMd.begin(), areasMd.end(), 0);
	int lgArea = accumulate(areasLg.begin(), areasLg.end(), 0);
	int areaTotal = smArea + mdArea + lgArea;

	statsFile << "Amount of Spatter Area (small/red): " << smArea << endl;
	statsFile << "Amount of Spatter Area (medium/green): " << mdArea << endl;
	statsFile << "Amount of Spatter Area (large/blue): " << lgArea << endl;
	statsFile << "Total Amount of Spatter Area: " << areaTotal << endl << endl;


	//----------------------the spatter to build ratio per category, and total----------------------
	
	Mat imgCrop = croppedImg;
	Mat mask;
	cvtColor(imgCrop, imgCrop, COLOR_BGR2GRAY);
	threshold(imgCrop, mask, 50, 255, THRESH_BINARY); //separate the part from the background
	//imwrite("C:\\Users\\ashto\\source\\repos\\OpenCV Projects\\Part Images\\test.png", mask);

	double buildGeometry = countNonZero(mask);
	double spatToBrSm = smArea / buildGeometry;
	double spatToBrMd = mdArea / buildGeometry;
	double spatToBrLg = lgArea / buildGeometry;
	double spatToBrTotal = areaTotal / buildGeometry;

	statsFile << "Spatter to Build Ratio (small/red): " << spatToBrSm << endl;
	statsFile << "Spatter to Build Ratio (medium/green): " << spatToBrMd << endl;
	statsFile << "Spatter to Build Ratio (large/blue): " << spatToBrLg << endl;
	statsFile << "Total Spatter to Build Ratio: " << spatToBrTotal << endl << endl;

	statsFile.close();
}

//finds all the spots, assigns them a size based on their area, then colors in the areas based on their assigned sizes
Mat ColorSpots(Mat preparedImg, Mat croppedImg, string filepath) {

	//so I dont overwrite the original images
	Mat mask = preparedImg;
	Mat img = croppedImg;

	const int connectivity_8 = 8; //use 8 so it includes diagonals
	Mat labels, stats, centroids, grayMask;
	vector<int> areasSm, areasMd, areasLg; // contains the value used for that area
	vector<int> areasValueSm, areasValueMd, areasValueLg; //contains the value of that area for stats

	cvtColor(mask, grayMask, COLOR_BGR2GRAY); //converts image to grayscale so connectedComponents can be used on it

	int nLabels = connectedComponentsWithStats(grayMask, labels, stats, centroids, connectivity_8, CV_32S);
	int area;

	//loads the spots into a vector based on its size/area
	for (int i = 1; i < nLabels; i++) { //0 = the entire background

		area = stats.at<int>(i, CC_STAT_AREA);

		//if the area is small, load it into the areasSm vector
		if (area > 1 && area <= 10) {
			areasSm.push_back(i);
			areasValueSm.push_back(area);
		}

		//else if the area is medium, load it into the areasMd vector
		else if (area > 10 && area <= 30) {
			areasMd.push_back(i);
			areasValueMd.push_back(area);
		}

		//else if the area is large, load it into the areasLg vector
		else if (area > 30){
			areasLg.push_back(i);
			areasValueLg.push_back(area);
		}

	}

	SaveStats(filepath, croppedImg, areasValueSm, areasValueMd, areasValueLg); //calculates the stats on the detected areas and writes them out to a file

	int point;
	//iterate over every pixel in labels and color that location its color based on what vector it's in (the vector it's in is based on its size)
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {

			point = labels.at<int>(i, j);

			//if the area is small, color it red
			if (find(areasSm.begin(), areasSm.end(), labels.at<int>(i, j)) != areasSm.end()) {

				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 255;
			}

			//else if the area is medium, color it green
			else if (find(areasMd.begin(), areasMd.end(), labels.at<int>(i, j)) != areasMd.end()) {

				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 255;
				img.at<Vec3b>(i, j)[2] = 0;
			}

			//else if the area is large, color it blue
			else if(find(areasLg.begin(), areasLg.end(), labels.at<int>(i, j)) != areasLg.end()){

				img.at<Vec3b>(i, j)[0] = 255;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 0;
			}

		}

	}

	return img;
}

//function that prepares the image for spot detection by cropping and thresholding the image
Mat ThresholdImg(Mat croppedImg) {
	
	Mat imgCrop = croppedImg; //so I dont overwrite the inputted image
	Mat mask;

	//Image Thresholding (image manipulation)
	Scalar mean, stddev;
	meanStdDev(imgCrop, mean, stddev);
	int meanStd = (int)(mean[0] + 2 * stddev[0]); //use this to find the brighest objects
	//int meanStd = (int)(mean[0] + stddev[0]);
	threshold(imgCrop, mask, meanStd, 255, THRESH_BINARY);

	return mask; //return cropped and thresholded image
}

//crops the image so it only includes the part
Mat CropImg(Mat originalImg) {

	//cropping & resizing of originalImg so it only includes the part in the image
	Rect roi(96, 230, 1811, 1503); //(x,y,height,width) of full image
	Mat imgCrop = originalImg(roi);
	//resize(imgCrop, imgCrop, Size(), 0.5, 0.5); // image = 625x500 so it fits on my screen when displayed
	return imgCrop;
}

//takes in a filepath to an image and saves the image with the white spots detected with the same name in another directory
void DetectWhiteSpots(string image) {

	//function calls needed to preform the detection and coloring of the white spots
	Mat originalImg = imread(image); //reads in the entered image
	Mat croppedImg = CropImg(originalImg); //crops (and resizes if needed) the img to prepare it for thresholding and detecting
	Mat preparedImg = ThresholdImg(croppedImg); //prepares image by thresholding to receive and return the mask 
	Mat labeledImg = ColorSpots(preparedImg, croppedImg, image); //detects the spots and colors them in based on their area/size on a copy of the original img

	image.resize(image.size() - 4);
	imwrite( image + "_labeled.png", labeledImg);

	//if you want to display the image
	/*imshow("DetectWhiteSpots", labeledImg);
	waitKey(0);*/

}

int main() {

	string image = "C:\\Users\\ashto\\source\\repos\\OpenCV Projects\\Part Images\\Layer62.png";
	//string image = "C:\\Users\\ashto\\source\\repos\\OpenCV Projects\\Part Images\\Layer126.png";
	//string image = "C:\\Users\\ashto\\source\\repos\\OpenCV Projects\\Part Images\\Layer134.png";
	DetectWhiteSpots(image);
}