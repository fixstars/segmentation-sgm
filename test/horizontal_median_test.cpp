#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "test_utility.h"
#include "reference.h"
#include "internal.h"

TEST(HorizontalMedianTest, RandomU8)
{
	using disp_type = uchar;
	using Mat = cv::Mat_<disp_type>;

	const int segmentWidth = 10;
	const int width = 1024;
	const int height = 333;

	Mat src = randomMat<disp_type>(height, width);
	Mat dst1, dst2;

	ref::horizontalMedian(src, dst1, segmentWidth);
	sgm::horizontalMedian(src, dst2, segmentWidth);

	EXPECT_TRUE(equals(dst1, dst2));
}

TEST(HorizontalMedianTest, RandomU16)
{
	using disp_type = ushort;
	using Mat = cv::Mat_<disp_type>;

	const int segmentWidth = 10;
	const int width = 1024;
	const int height = 333;

	Mat src = randomMat<disp_type>(height, width);
	Mat dst1, dst2;

	ref::horizontalMedian(src, dst1, segmentWidth);
	sgm::horizontalMedian(src, dst2, segmentWidth);

	EXPECT_TRUE(equals(dst1, dst2));
}

TEST(HorizontalMedianTest, FromFile)
{
	const int segmentWidth = 10;

	// Load data
	const std::string testDataDir(TEST_DATA_DIR);
	cv::Mat1b src = cv::imread(testDataDir + "/disparity.png");
	cv::Mat1b dst1, dst2;

	ref::horizontalMedian(src, dst1, segmentWidth);
	sgm::horizontalMedian(src, dst2, segmentWidth);

	EXPECT_TRUE(equals(dst1, dst2));
}

class HorizontalMedianSegmentWidthTest : public ::testing::TestWithParam<int> {};
INSTANTIATE_TEST_CASE_P(TestDataIntRange, HorizontalMedianSegmentWidthTest, ::testing::Range(3, 10, 1));

TEST_P(HorizontalMedianSegmentWidthTest, RangeTest)
{
	using disp_type = ushort;
	using Mat = cv::Mat_<disp_type>;

	const int segmentWidth = GetParam();
	const int width = 1024;
	const int height = 333;

	Mat src = randomMat<disp_type>(height, width);
	Mat dst1, dst2;

	ref::horizontalMedian(src, dst1, segmentWidth);
	sgm::horizontalMedian(src, dst2, segmentWidth);

	EXPECT_TRUE(equals(dst1, dst2));
}
