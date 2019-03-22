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

#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include <cuda_runtime.h>

#include "macro.h"

namespace sgm
{

template <typename T>
struct DeviceBuffer
{
	DeviceBuffer() : data(nullptr), size(0) {}
	DeviceBuffer(size_t size) : data(nullptr), size(0) { allocate(size); }
	~DeviceBuffer() { destroy(); }

	void allocate(size_t _size)
	{
		if (data && size >= _size)
			return;

		destroy();
		CUDA_CHECK(cudaMalloc(&data, sizeof(T) * _size));
		size = _size;
	}

	void destroy()
	{
		if (data)
			CUDA_CHECK(cudaFree(data));
		data = nullptr;
		size = 0;
	}

	T* data;
	size_t size;
};

} // namespace sgm

#endif // !__DEVICE_BUFFER_H__
