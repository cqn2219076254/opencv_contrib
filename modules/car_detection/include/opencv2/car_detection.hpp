#ifndef __OPENCV_CAR_DETECTION_HPP__
#define __OPENCV_CAR_DETECTION_HPP__

#include <opencv2/core.hpp>
#include <ostream>

namespace cv {
	namespace car_Detection {

		class CV_EXPORTS_W CarDetector {
		public:
			CV_WRAP CarDetector(const std::string& data_path = "", const std::string& model_path = "", const int& model_num = 0);

			~CarDetector();

			void preprocessor();

			void infer();

		private:
			std::string model_path;
			std::string data_path;
		};
	}
}

#endif