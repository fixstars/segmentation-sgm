#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <segmentation_sgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

static bool exists(const std::string& path)
{
	std::ifstream ifs(path);
	return !ifs.fail();
}

static void draw_segmentation(cv::Mat& img, const std::vector<sgm::SegmentationSGM::Segment>& segments)
{
	CV_Assert(img.type() == CV_8UC3);

	cv::Mat segimg = cv::Mat::zeros(img.size(), CV_8UC3);

	for (const auto& segment : segments)
	{
		const int d = static_cast<int>(segment.disp);
		const int radius = std::max(segment.width / 2, 1);

		const cv::Rect region(segment.u - radius, segment.vT, segment.width, segment.vB - segment.vT);

		const int H = std::max(0, 64 - 7 * d / 4);
		const int S = 255;
		const int V = d < 5 ? 64 : 128;

		segimg(region) = cv::Vec3b(H, S, V);
		cv::rectangle(segimg, region, cv::Scalar(0, 0, 255), 1);
	}

	cv::cvtColor(segimg, segimg, cv::COLOR_HSV2BGR);
	cv::addWeighted(img, 1, segimg, 0.5, 0, img);
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml [dizp_size]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const int disp_size = argc > 4 ? std::stoi(argv[4]) : 64;
	const int first_frame = 1;

	cv::Mat I1 = cv::imread(cv::format(argv[1], first_frame), cv::IMREAD_UNCHANGED);
	cv::Mat I2 = cv::imread(cv::format(argv[2], first_frame), cv::IMREAD_UNCHANGED);

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");

	// stereo sgbm
	const int wsize = 11;
	auto ssgbm = cv::StereoSGBM::create(0, disp_size, wsize);
	ssgbm->setP1(8 * wsize * wsize);
	ssgbm->setP2(32 * wsize * wsize);
	ssgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

	cv::Mat disparity;

	// read camera parameters
	const std::string cameraFile(argv[3]);
	if (!exists(cameraFile))
	{
		std::cerr << "File " << cameraFile << " not found." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	const cv::FileStorage fs(cameraFile, cv::FileStorage::READ);

	// input parameters
	sgm::SegmentationSGM::Parameters param;
	param.camera.fu = fs["FocalLengthX"];
	param.camera.fv = fs["FocalLengthY"];
	param.camera.u0 = fs["CenterX"];
	param.camera.v0 = fs["CenterY"];
	param.camera.baseline = fs["BaseLine"];
	param.maxDisparity = disp_size;

	sgm::SegmentationSGM segsgm(param);
	std::vector<sgm::SegmentationSGM::Segment> segments;

	for (int frame_no = first_frame;; frame_no++)
	{
		I1 = cv::imread(cv::format(argv[1], frame_no), cv::IMREAD_UNCHANGED);
		I2 = cv::imread(cv::format(argv[2], frame_no), cv::IMREAD_UNCHANGED);
		if (I1.empty() || I2.empty())
		{
			std::cout << "imread failed." << std::endl;
			break;
		}

		if (I1.type() == CV_16U)
		{
			// conver to CV_8U
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
			I2.convertTo(I2, CV_8U);
		}

		const auto t1 = std::chrono::steady_clock::now();

		ssgbm->compute(I1, I2, disparity);

		disparity.setTo(0, disparity < 0);
		disparity.convertTo(disparity, CV_8U, 1. / cv::StereoSGBM::DISP_SCALE);

		const auto t2 = std::chrono::steady_clock::now();

		segsgm.compute(disparity, segments);

		const auto t3 = std::chrono::steady_clock::now();

		const auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

		// colorize disparity
		cv::Mat disparity_color;
		disparity.convertTo(disparity_color, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_color, disparity_color, cv::COLORMAP_JET);
		disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == 0);

		// put processing time
		cv::putText(disparity_color, cv::format("sgm execution time : %4.1f [msec]", 1e-3 * duration12), cv::Point(100, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::putText(disparity_color, cv::format("segmentation  time : %4.1f [msec]", 1e-3 * duration23), cv::Point(100, 80), 2, 0.75, cv::Scalar(255, 255, 255));

		// draw segmentation
		cv::Mat segmentation;
		cv::cvtColor(I1, segmentation, cv::COLOR_GRAY2BGR);
		draw_segmentation(segmentation, segments);

		cv::imshow("disparity", disparity_color);
		cv::imshow("segmentation sgm", segmentation);

		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;

		if (c == 'p')
			cv::waitKey(0);
	}

	return 0;
}
