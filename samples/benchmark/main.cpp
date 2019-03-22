/*
Copyright 2019 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime.h>

#include <libsgm.h>
#include <segmentation_sgm.h>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

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
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml [dizparity-size] [segment-width] [iterations]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	cv::Mat I1 = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
	cv::Mat I2 = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
	const cv::FileStorage fs(argv[3], cv::FileStorage::READ);
	const int disp_size = argc > 4 ? std::stoi(argv[4]) : 128;
	const int segment_width = argc > 5 ? std::stoi(argv[5]) : 7;
	const int iterations = argc > 6 ? std::stoi(argv[6]) : 100;

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(fs.isOpened(), "camera.xml read failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");

	// stereo sgbm
	const int width = I1.cols;
	const int height = I1.rows;

	const int input_depth = I1.type() == CV_8U ? 8 : 16;
	const int input_bytes = input_depth * width * height / 8;
	const int output_depth = 8;
	const int output_bytes = output_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	// input parameters
	sgm::SegmentationSGM::Parameters param;
	param.camera.fu = fs["FocalLengthX"];
	param.camera.fv = fs["FocalLengthY"];
	param.camera.u0 = fs["CenterX"];
	param.camera.v0 = fs["CenterY"];
	param.camera.baseline = fs["BaseLine"];
	param.maxDisparity = disp_size;
	param.segmentWidth = segment_width;

	sgm::SegmentationSGM segsgm(param);
	std::vector<sgm::SegmentationSGM::Segment> segments;

	device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);
	cudaMemcpy(d_I1.data, I1.data, input_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_I2.data, I2.data, input_bytes, cudaMemcpyHostToDevice);

	cudaDeviceProp prop;
	int version;
	cudaGetDeviceProperties(&prop, 0);
	cudaRuntimeGetVersion(&version);

	// show settings
	std::cout << "# Settings" << std::endl;
	std::cout << "device name         : " << prop.name << std::endl;
	std::cout << "CUDA runtime version: " << version << std::endl;
	std::cout << "image size          : " << I1.size() << std::endl;
	std::cout << "disparity size      : " << disp_size << std::endl;
	std::cout << "segment width       : " << segment_width << std::endl;
	std::cout << "iterations          : " << iterations << std::endl;
	std::cout << std::endl;

	// run benchmark
	std::cout << "Running benchmark..." << std::endl;
	uint64_t sum[3] = { 0, 0, 0 };
	for (int i = 0; i <= iterations; i++)
	{
		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();

		segsgm.compute(reinterpret_cast<const uchar*>(d_disparity.data), width, height, width, segments,
			sgm::SegmentationSGM::INPUT_PTR_DEVICE);
		cudaDeviceSynchronize();

		const auto t3 = std::chrono::system_clock::now();

		if (i > 0)
		{
			sum[0] += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			sum[1] += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
			sum[2] += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count();
		}
	}
	std::cout << "Done." << std::endl << std::endl;

	// show results
	const std::string names[3] = { "Stereo SGM", "Segmentation SGM", "Total" };
	for (int i = 0; i < 3; i++)
	{
		const double time_millisec = 1e-3 * sum[i] / iterations;
		const double fps = 1e3 / time_millisec;
		std::cout << "# " << names[i] << " Results" << std::endl;
		std::cout.setf(std::ios::fixed);
		std::cout << std::setprecision(1) << "Processing Time[Milliseconds]: " << time_millisec << std::endl;
		std::cout << std::setprecision(1) << "FPS                          : " << fps << std::endl;
		std::cout << std::endl;
	}

	cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);
	cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

	// colorize disparity
	cv::Mat disparity_color;
	disparity.convertTo(disparity_color, CV_8U, 255. / disp_size);
	cv::applyColorMap(disparity_color, disparity_color, cv::COLORMAP_JET);
	disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == 0);

	// draw segmentation
	cv::Mat segmentation;
	if (I1.type() != CV_8U)
	{
		cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
		I1.convertTo(I1, CV_8U);
	}
	cv::cvtColor(I1, segmentation, cv::COLOR_GRAY2BGR);
	draw_segmentation(segmentation, segments);

	// save disparity and segmentation
	cv::imwrite("disparity.png", disparity_color);
	cv::imwrite("segmentation.png", segmentation);

	return 0;
}
