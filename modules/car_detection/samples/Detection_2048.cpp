#include "car_detection.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::car_Detection;

int main() {
	string model_path = "../models";
	string data_path = "../data";
	int model_num = 0;

	CarDetector cd = CarDetector(data_path, model_path, model_num);
	cd.preprocessor();
	cd.infer();
}

