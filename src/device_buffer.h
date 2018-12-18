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
