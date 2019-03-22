#ifndef __REFERENCE_H__
#define __REFERENCE_H__

#include <opencv2/core.hpp>

namespace ref
{

template <typename T>
void horizontalMedian(const cv::Mat_<T>& src, cv::Mat_<T>& dst, int segmentWidth);

template <typename T>
void scanCost(const cv::Mat_<T>& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels);

} // namespace ref

#endif // !__REFERENCE_H__
