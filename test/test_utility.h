#ifndef __TEST_UTILITY_H__
#define __TEST_UTILITY_H__

#include <opencv2/core.hpp>

template <typename T>
static bool equals(const cv::Mat_<T>& a, const cv::Mat_<T>& b)
{
	return cv::countNonZero(a != b) == 0;
}

template <typename T>
static cv::Mat_<T> randomMat(int rows, int cols, T lo, T hi)
{
	cv::Mat_<T> m(rows, cols);
	cv::randu(m, lo, hi);
	return m;
}

template <typename T>
static cv::Mat_<T> randomMat(int rows, int cols)
{
	return randomMat(rows, cols, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

template <typename T>
static std::vector<T> randomVector(int size, T lo, T hi)
{
	std::vector<T> vec(size);
	cv::randu(vec, lo, hi);
	return vec;
}

template <typename T>
static std::vector<T> randomVector(int size)
{
	return randomVector(size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

#endif // !__TEST_UTILITY_H__
