#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "segmentation_sgm.h"

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

static void draw_segments(cv::Mat& img, const std::vector<cpu::SegmentationSGM::Segment>& segments)
{
	CV_Assert(img.type() == CV_8UC3);
	img = 0;

	for (const auto& segment : segments)
	{
		const int d = (int)segment.disp;
		const int radius = std::max(segment.width / 2, 1);
		const cv::Rect region(segment.u - radius, segment.vT, segment.width, segment.vB - segment.vT);
		cv::Vec3b color;
		color[0] = std::max(0, 64 - 7 * d / 4);
		color[1] = 255;
		color[2] = d < 5 ? 64 : 128;
		img(region) = color;
	}

	cv::cvtColor(img, img, cv::COLOR_HSV2BGR);

	const cv::Scalar borderColor(255, 255, 255);
	for (const auto& segment : segments)
	{
		const int radius = std::max(segment.width / 2, 1);
		const cv::Point tl(segment.u - radius, segment.vT);
		const cv::Point br(segment.u + radius, segment.vB);
		cv::rectangle(img, cv::Rect(tl, br), cv::Scalar(255, 255, 255), 1);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml [dizp_size]" << std::endl;
		return -1;
	}

	const int disp_size = argc >= 5 ? std::stoi(argv[3]) : 64;
	const int first_frame = 1;

	cv::Mat I1 = cv::imread(format_string(argv[1], first_frame), -1);
	cv::Mat I2 = cv::imread(format_string(argv[2], first_frame), -1);

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");

	// stereo sgbm
	const int wsize = 11;
	const int P1 = 8 * wsize * wsize;
	const int P2 = 32 * wsize * wsize;
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(0, disp_size, wsize, P1, P2, 0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);
	cv::Mat disparity;

	// read camera parameters
	const cv::FileStorage cvfs(argv[3], cv::FileStorage::READ);
	CV_Assert(cvfs.isOpened());
	const cv::FileNode node(cvfs.fs, NULL);
	
	// input parameters
	cpu::SegmentationSGM::Parameters param;
	param.camera.fu = node["FocalLengthX"];
	param.camera.fv = node["FocalLengthY"];
	param.camera.u0 = node["CenterX"];
	param.camera.v0 = node["CenterY"];
	param.camera.baseline = node["BaseLine"];
	param.maxDisparity = disp_size;
	cpu::SegmentationSGM segsgm(param);
	std::vector<cpu::SegmentationSGM::Segment> segments;

	for (int frame_no = first_frame;; frame_no++)
	{
		I1 = cv::imread(format_string(argv[1], frame_no), -1);
		I2 = cv::imread(format_string(argv[2], frame_no), -1);
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

		const auto t1 = std::chrono::system_clock::now();

		ssgbm->compute(I1, I2, disparity);

		disparity.setTo(0, disparity < 0);
		disparity.convertTo(disparity, CV_8U, 1. / cv::StereoSGBM::DISP_SCALE);

		const auto t2 = std::chrono::system_clock::now();

		segsgm.compute(disparity, segments);

		const auto t3 = std::chrono::system_clock::now();

		const auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
		// std::cout << "segments: " << segments.size() << std::endl;

		cv::Mat dispImg;
		disparity.convertTo(dispImg, CV_8U, 255. / disp_size);
		cv::applyColorMap(dispImg, dispImg, cv::COLORMAP_JET);
		cv::putText(dispImg, format_string("sgm execution time : %4.1f [msec]", 1e-3 * duration12), cv::Point(100, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::putText(dispImg, format_string("segmentation  time : %4.1f [msec]", 1e-3 * duration23), cv::Point(100, 80), 2, 0.75, cv::Scalar(255, 255, 255));
		dispImg.setTo(cv::Scalar(0, 0, 0), disparity == 0);

		// draw segments
		cv::Mat draw;
		cv::cvtColor(I1, draw, cv::COLOR_GRAY2BGRA);
		cv::Mat segmentImg(I1.size(), CV_8UC3);
		draw_segments(segmentImg, segments);
		cv::cvtColor(segmentImg, segmentImg, cv::COLOR_BGR2BGRA);
		draw = draw + 0.5 * segmentImg;

		//cv::imshow("left image", I1);
		cv::imshow("disparity", dispImg);
		cv::imshow("segments", draw);

		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;

		if (c == 'p')
			cv::waitKey(0);
	}

	return 0;
}
