#ifndef __INTERNAL_H__
#define __INTERNAL_H__

#include <cstdint>

#include <opencv2/core.hpp>
#include <driver_types.h>

namespace sgm
{

template <typename T>
void horizontalMedian(const cv::Mat_<T>& src, cv::Mat_<T>& dst, int segmentWidth);

template <typename T>
void horizontalMedianGPU(const T* D, int pitch, T* columns, int umax, int vmax, int segmentWidth);

template <typename T>
void scanCost(const cv::Mat_<T>& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels);

template <typename T>
void scanCostForeGPU(const T* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L,
	cudaStream_t stream);

template <typename T>
void scanCostBackGPU(const T* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L,
	cudaStream_t stream);

void winnerTakesAllGPU(uint16_t* LFore, uint16_t* LBack, uint16_t* labels, int umax, int vmax, int dmax, int vhor);

}

#endif // !__INTERNAL_H__
