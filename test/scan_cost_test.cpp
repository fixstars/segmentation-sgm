#include <gtest/gtest.h>

#include <fstream>
#include <opencv2/highgui.hpp>

#include "test_utility.h"
#include "reference.h"
#include "internal.h"

TEST(ScanCostTest, RandomU8)
{
	using disp_type = uchar;

	const int segmentWidth = 10;
	const int umax = 1024 / segmentWidth;
	const int vmax = 333;
	const int dmax = 128;

	const auto D = randomMat<disp_type>(umax, vmax, 0, dmax);
	const auto DR = randomVector<float>(vmax, 0, dmax - 1);
	const int vhor = vmax / 2;

	cv::Mat1w labels1, labels2;
	ref::scanCost(D, dmax, DR, vhor, labels1);
	sgm::scanCost(D, dmax, DR, vhor, labels2);

	EXPECT_TRUE(equals(labels1, labels2));
}

TEST(ScanCostTest, RandomU16)
{
	using disp_type = ushort;

	const int segmentWidth = 10;
	const int umax = 1024 / segmentWidth;
	const int vmax = 333;
	const int dmax = 128;

	const auto D = randomMat<disp_type>(umax, vmax, 0, dmax);
	const auto DR = randomVector<float>(vmax, 0, dmax - 1);
	const int vhor = vmax / 2;

	cv::Mat1w labels1, labels2;
	ref::scanCost(D, dmax, DR, vhor, labels1);
	sgm::scanCost(D, dmax, DR, vhor, labels2);

	EXPECT_TRUE(equals(labels1, labels2));
}

TEST(ScanCostTest, FromFile)
{
	const std::string testDataDir(TEST_DATA_DIR);

	// Load data
	const cv::Mat1b D = cv::imread(testDataDir + "/disparity_reduced.png");
	std::ifstream ifs(testDataDir + "/line.txt");

	CV_Assert(!D.empty());
	CV_Assert(!ifs.fail());

	const int vmax = D.cols;
	const int dmax = 128;

	float a, b;
	ifs >> a >> b;

	// compute expected road disparity
	std::vector<float> DR(vmax);
	for (int v = 0; v < vmax; v++)
		DR[v] = a * v + b;

	// horizontal row from which road dispaliry becomes negative
	const int vhor = cvRound(-b / a);

	cv::Mat1w labels1, labels2;
	ref::scanCost(D, dmax, DR, vhor, labels1);
	sgm::scanCost(D, dmax, DR, vhor, labels2);

	EXPECT_TRUE(equals(labels1, labels2));
}
