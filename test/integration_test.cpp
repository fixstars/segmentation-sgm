#include <gtest/gtest.h>

#include <fstream>
#include <opencv2/highgui.hpp>

#include "test_utility.h"
#include "reference.h"
#include "internal.h"

TEST(IntegrationTest, FromFile)
{
	// Load data
	const std::string testDataDir(TEST_DATA_DIR);
	cv::Mat1b disparity = cv::imread(testDataDir + "/disparity.png");
	std::ifstream ifs(testDataDir + "/line.txt");

	CV_Assert(!disparity.empty());
	CV_Assert(!ifs.fail());

	const int vmax = disparity.cols;
	const int dmax = 128;

	float a, b;
	ifs >> a >> b;

	// compute horizontal median
	const int segmentWidth = 10;
	cv::Mat1b colmuns1, colmuns2;
	ref::horizontalMedian(disparity, colmuns1, segmentWidth);
	sgm::horizontalMedian(disparity, colmuns2, segmentWidth);

	// compute expected road disparity
	std::vector<float> DR(vmax);
	for (int v = 0; v < vmax; v++)
		DR[v] = a * v + b;

	// horizontal row from which road dispaliry becomes negative
	const int vhor = cvRound(-b / a);

	// scan cost
	cv::Mat1w labels1, labels2;
	ref::scanCost(colmuns1, dmax, DR, vhor, labels1);
	sgm::scanCost(colmuns2, dmax, DR, vhor, labels2);

	EXPECT_TRUE(equals(labels1, labels2));
}
