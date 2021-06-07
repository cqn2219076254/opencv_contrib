#include "car_detection.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn/layer.details.hpp> 
#include <ostream>
#include <fstream>
#include <sstream>
#include "src/opencvCustomOps/GreaterLayer.hpp"
#include "src/opencvCustomOps/Sampling.hpp"
#include "src/opencvCustomOps/Grouping.hpp"
#include "src/opencvCustomOps/Calculate.hpp"
#include <opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::car_Detection;

void importLayer() {
	CV_DNN_REGISTER_LAYER_CLASS(Max, MaxLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Sin, SinLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Cos, CosLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Square, SquareLayer);
	CV_DNN_REGISTER_LAYER_CLASS(QueryBallPoint, QueryBallPointLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MySqueeze, MySqueezeLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GreaterThenCast, GreaterThenCastLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyExpandDims, MyExpandDimsLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyExpandDimsF, MyExpandDimsFLayer);
	CV_DNN_REGISTER_LAYER_CLASS(FarthestPointSample, FarthestPointSampleLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GatherPoint, GatherPointLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GroupPoint, GroupPointLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyTranspose, MyTransposeLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyTransposeBack, MyTransposeBackLayer);
	CV_DNN_REGISTER_LAYER_CLASS(CiSum, CiSumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(CiiSum, CiiSumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(NonMaxSuppressionV2, NonMaxSuppressionV2Layer);
	CV_DNN_REGISTER_LAYER_CLASS(Cast, CastLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Unpack, UnpackLayer);
	CV_DNN_REGISTER_LAYER_CLASS(StackWithZero, StackWithZeroLayer);
	CV_DNN_REGISTER_LAYER_CLASS(OneHot, OneHotLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Maximum, MaximumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(Minimum, MinimumLayer);
	CV_DNN_REGISTER_LAYER_CLASS(MyConcat, MyConcatLayer);
	CV_DNN_REGISTER_LAYER_CLASS(ArgMax, ArgMaxLayer);
	CV_DNN_REGISTER_LAYER_CLASS(FarthestPointSampleWithDistance, FarthestPointSampleWithDistanceLayer);
	CV_DNN_REGISTER_LAYER_CLASS(BatchMatMul, BatchMatMulLayer);
	CV_DNN_REGISTER_LAYER_CLASS(QueryBallPointDilated, QueryBallPointDilatedLayer);
	CV_DNN_REGISTER_LAYER_CLASS(QueryBallPointDilatedV2, QueryBallPointDilatedV2Layer);
	CV_DNN_REGISTER_LAYER_CLASS(Gather, GatherLayer);
	CV_DNN_REGISTER_LAYER_CLASS(GatherV2, GatherV2Layer);
}

CarDetector::CarDetector(const std::string& data_path, const std::string& model_path, const int& model_num) {
	importLayer();
	if (model_num == 1)
		CarDetector::model_path = model_path + "/3DSSD-4096.pb";
	else if (model_num == 0)
		CarDetector::model_path = model_path + "/3DSSD-2048.pb";
	CarDetector::data_path = data_path;
}

CarDetector::~CarDetector() {}

template <class Type>
Type stringToNum(const string& str) {
    istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

void pre(string idx, string data_path) {
    int MAX_POINT_NUMBER = 16384;
    string POINT = data_path + "/point/";
    string CALIB = data_path + "/calib/";
    string IMAGE = data_path + "/image/";
    string PRE_DATA = data_path + "/pre_data/";
    int m, l, number;
    int extents[3][2] = { {-40, 40}, {-5, 3}, {0, 70} };

    Mat img = imread(IMAGE + idx + ".png");
    if (!img.data)
    {
        cout << "图像加载失败!" << endl;
        exit(0);
    }
    cvtColor(img, img, COLOR_BGR2RGB);
    int* shape = new int[2];
    shape[0] = img.rows;
    shape[1] = img.cols;

    //read
    ifstream Bin(POINT + idx + ".bin", ios_base::binary);
    l = Bin.tellg();
    Bin.seekg(0, ios::end);
    m = Bin.tellg();
    number = (m - l) / 16;
    Bin.close();
    ifstream file(POINT + idx + ".bin", ios_base::binary);
    float* data = new float[number * 4]();
    file.read(reinterpret_cast<char*>(data), sizeof(float) * (number * 4));
    file.close();
    float* points = new float[number * 4]();
    float* points_intensity = new float[number]();
    for (int i = 0; i < number; i++) {
        for (int j = 0; j < 3; j++) points[4 * i + j] = data[4 * i + j];
        points[4 * i + 3] = 1;
        points_intensity[i] = data[4 * i + 3];
    }
    Mat Ps = Mat(2, new int[] { number, 4 }, CV_32F, points);
    Mat PIs = Mat(2, new int[] { number, 1 }, CV_32F, points_intensity);

    ifstream Txt(CALIB + idx + ".txt");
    String temp;
    float* P = new float[12];
    float* V2C = new float[12];
    float* R0 = new float[9];
    for (int i = 0; i < 27; i++) Txt >> temp;
    for (int i = 0; i < 12; i++) {
        Txt >> temp;
        P[i] = stringToNum<float>(temp);
    }
    for (int i = 0; i < 14; i++) Txt >> temp;
    for (int i = 0; i < 9; i++) {
        Txt >> temp;
        R0[i] = stringToNum<float>(temp);
    }
    Txt >> temp;
    for (int i = 0; i < 12; i++) {
        Txt >> temp;
        V2C[i] = stringToNum<float>(temp);
    }
    Mat p = Mat(2, new int[] { 3, 4 }, CV_32F, P);
    Mat v2c = Mat(2, new int[] { 3, 4 }, CV_32F, V2C);
    Mat r0 = Mat(2, new int[] { 3, 3 }, CV_32F, R0);
    Ps = Ps * v2c.t();
    Ps = r0 * Ps.t();
    Mat img_coord = Mat(4, number, CV_32F);
    float* img_data = (float*)img_coord.data;
    memcpy(img_data, (float*)Ps.data, sizeof(float) * Ps.total());
    for (int i = 3 * number; i < 4 * number; i++) img_data[i] = 1;
    Ps = Ps.t();
    img_coord = img_coord.t() * p.t();
    img_data = (float*)img_coord.data;
    vector<int> filter;
    float* Ps_data = (float*)Ps.data;
    float* PIs_data = (float*)PIs.data;
    for (int i = 0; i < number; i++) {
        img_data[i * 3] /= img_data[i * 3 + 2];
        img_data[i * 3 + 1] /= img_data[i * 3 + 2];
        if (img_data[i * 3] >= 0 &&
            img_data[i * 3] < shape[1] &&
            img_data[i * 3 + 1] >= 0 &&
            img_data[i * 3 + 1] < shape[0] &&
            Ps_data[i * 3] > extents[0][0] &&
            Ps_data[i * 3] < extents[0][1] &&
            Ps_data[i * 3 + 1] > extents[1][0] &&
            Ps_data[i * 3 + 1] < extents[1][1] &&
            Ps_data[i * 3 + 2] > extents[2][0] &&
            Ps_data[i * 3 + 2] < extents[2][1]) {
            filter.push_back(i);
        }
    }
    int num = filter.size(), index, r;
    bool* isChoosen = new bool[num]();
    ofstream outfile(PRE_DATA + idx + ".txt");
    if (filter.size() < MAX_POINT_NUMBER) {
        for (int i = 0; i < num; i++) {
            index = filter[i];
            outfile << Ps_data[index * 3] << " "
                << Ps_data[index * 3 + 1] << " "
                << Ps_data[index * 3 + 2] << " "
                << PIs_data[index] << endl;
        }
        for (int i = 0; i < MAX_POINT_NUMBER - num; i++) {
            do {
                r = rand() % num;
            } while (isChoosen[r]);
            isChoosen[r] = true;
            index = filter[r];
            outfile << Ps_data[index * 3] << " "
                << Ps_data[index * 3 + 1] << " "
                << Ps_data[index * 3 + 2] << " "
                << PIs_data[index] << endl;
        }
    }
    else {
        for (int i = 0; i < MAX_POINT_NUMBER; i++) {
            do {
                r = rand() % num;
            } while (isChoosen[r]);
            isChoosen[r] = true;
            index = filter[r];
            outfile << Ps_data[index * 3] << " "
                << Ps_data[index * 3 + 1] << " "
                << Ps_data[index * 3 + 2] << " "
                << PIs_data[index] << endl;
        }
    }
}

void CarDetector::preprocessor()
{
	ifstream list(data_path + "/list.txt");
	vector<string> Idx;
	string idx;
	while (getline(list, idx)) {
		Idx.push_back(idx);
	}
	for (string i : Idx) {
		pre(i, data_path);
	}
}

cv::Mat getInput(string idx, string data_path)
{
	string inputDir = data_path + "/pre_data/" + idx + ".txt";
	ifstream ifs(inputDir);
	string str;
	float* inp = new  float[65536];
	for (int i = 0; i < 65536; i++) {
		ifs >> str;
		inp[i] = stringToNum<float>(str);
	}
	int size[] = { 1,16384,4 };
	cv::Mat C = cv::Mat(3, size, CV_32F, inp);
	return C;
}

void CarDetector::infer()
{
	ifstream in(data_path + "/list.txt");
	vector<string> Idx;
	string idx;
	while (getline(in, idx)) {
		Idx.push_back(idx);
	}
	for (auto i : Idx) {
		std::cout << "Running: " << i << std::endl;
		Mat blob = getInput(i, data_path);
		Net net = readNetFromTensorflow(model_path);
		net.setInput(blob);
		std::vector<String> outNames(2);
		outNames[0] = "import/Gather";
		outNames[1] = "import/Gather_1";
		std::vector<Mat> outs(2);
		net.forward(outs, outNames);
		std::ofstream result_3d(data_path + "/aft_data/"+i+".txt", std::ios::out);
		for (int k = 0; k < 100; k++) {
			for (int j = 0; j < 7; j++) {
				result_3d << outs[0].ptr<float>(k)[j] << " ";
			}
			result_3d << outs[1].ptr<float>(0)[k] << " " << std::endl;
		}
	}
}
