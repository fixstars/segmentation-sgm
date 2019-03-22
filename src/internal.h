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
