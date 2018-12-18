#include "internal.h"

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "parameters.h"
#include "macro.h"

#if CUDA_VERSION >= 9000
#define SHFL_UP(var, delta, width) __shfl_up_sync(0xffffffff, (var), (delta))
#define SHFL_DOWN(var, delta, width) __shfl_down_sync(0xffffffff, (var), (delta))
#define SHFL(var, srcLane, width) __shfl_sync(0xffffffff, (var), (srcLane))
#else
#define SHFL_UP(var, delta, width) __shfl_up((var), (delta))
#define SHFL_DOWN(var, delta, width) __shfl_down((var), (delta))
#define SHFL(var, srcLane, width) __shfl((var), (srcLane))
#endif

namespace sgm
{

enum { FORE = 0, BACK = 1 };

constexpr int WARP_SIZE = 32;
constexpr int MC_MAX_DIFF = 64;
constexpr float SIGMA_LINE_A = 0.01f;
constexpr float SIGMA_LINE_B = 1.f;
constexpr int MAX_WARPS_PER_DMAX = MAX_DISPARITY / WARP_SIZE;
constexpr int WTA_WARPS_PER_BLOCK = 2;

static int divUp(int total, int grain)
{
	return (total + grain - 1) / grain;
}

template <typename T>
__inline__ __device__ T partialselect(T* buf, int n, int m)
{
	for (int i = 0; i <= m; i++)
	{
		int mi = i;
		T mv = buf[i];
		for (int j = i + 1; j < n; j++)
		{
			if (buf[j] < mv)
			{
				mi = j;
				mv = buf[j];
				T t = buf[i]; buf[i] = buf[mi]; buf[mi] = t;
			}
		}
	}
	return buf[m];
}

template <typename T>
__inline__ __device__ T median(T* buf, int n)
{
	const int m = n / 2;
	return partialselect(buf, n, m);
}

template <typename T>
__global__ void horizontalMedianKernel(const T* __restrict__ disparity, int pitch,
	T* columns, int umax, int vmax, int segmentWidth)
{

	T buf[MAX_SEGMENT_WIDTH];

	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int u = blockIdx.x * blockDim.x + threadIdx.x;

	if (v >= vmax || u >= umax)
		return;

	for (int du = 0; du < segmentWidth; du++)
		buf[du] = disparity[v * pitch + u * segmentWidth + du];

	// get median
	const T m = median(buf, segmentWidth);

	// store with transposed
	columns[u * vmax + v] = m;
}

__inline__ __device__ void warpShuffleReduction(const uint16_t* __restrict__ L, uint16_t minCostsOO[2][MAX_WARPS_PER_DMAX],
	uint16_t minCostsGO[2], int& minCostOO, int dmax, int droad, int vp, int tid)
{
	const int tmax = dmax / WARP_SIZE;
	
	int prevMinWarp[2], currMinWarp[2];
	prevMinWarp[FORE] = P_INF;
	prevMinWarp[BACK] = P_INF;

	for (int t = 0, k = tid; t < tmax; t++, k += WARP_SIZE)
	{
		uint16_t localL[2];
		localL[FORE] = L[k];
		localL[BACK] = L[dmax - 1 - k];

		// warp shfl reduction
		for (int d = 1; d < 32; d <<= 1)
		{
			localL[FORE] = min(localL[FORE], SHFL_UP(localL[FORE], d));
			localL[BACK] = min(localL[BACK], SHFL_UP(localL[BACK], d));
		}

		// calc min of all(before current warp) 
		localL[FORE] = min(localL[FORE], prevMinWarp[FORE]);
		localL[BACK] = min(localL[BACK], prevMinWarp[BACK]);

		currMinWarp[FORE] = SHFL(localL[FORE], 31);
		currMinWarp[BACK] = SHFL(localL[BACK], 31);

		// shift up by 1 elem
		localL[FORE] = SHFL_UP(localL[FORE], 1);
		localL[BACK] = SHFL_UP(localL[BACK], 1);

		// first elem of warp is last elem value of one before warp
		if (tid == 0)
		{
			localL[FORE] = t == 0 ? P_INF : prevMinWarp[FORE];
			localL[BACK] = t == 0 ? P_INF : prevMinWarp[BACK];
		}

		// update last elem value
		prevMinWarp[FORE] = currMinWarp[FORE];
		prevMinWarp[BACK] = currMinWarp[BACK];

		minCostsOO[FORE][t] = localL[FORE];
		minCostsOO[BACK][tmax - 1 - t] = SHFL(localL[BACK], 31 - tid); // reversing [BACK] elems

		if (droad <= 0 && t == tmax - 1)
			minCostsGO[BACK] = currMinWarp[BACK];
		else if (droad > 0 && t == tmax - 1 - droad / WARP_SIZE)
			minCostsGO[BACK] = SHFL(localL[BACK], WARP_SIZE - 1 - (droad % WARP_SIZE));
	}

	minCostOO = SHFL(prevMinWarp[BACK], 0);
}

template <typename T>
__global__ void scanCostForeKernel(const T* __restrict__ disparity, int umax, int vmax, int dmax,
	const float* __restrict__ road, int vhor, uint16_t* __restrict__ L)
{
	const int uc = blockIdx.y * blockDim.y + threadIdx.y;
	if (uc > umax)
		return;

	const int tid = threadIdx.x;
	const int n = dmax + 1;

	// cost buffers
	L += uc * vmax * n;
	extern __shared__ uint16_t Lp[];
	uint16_t* tmpLp = Lp + n;

	uint16_t minCostsOO[2][MAX_WARPS_PER_DMAX];
	uint16_t minCostsGO[2];
	int minCostOO = P_INF;

	for (int vc = 0; vc < vmax; vc++)
	{
		uint16_t* Lc = L + vc * n;

		const int vp = vc - 1;
		const bool inside = vp >= 0 && vp < vmax;
		const int d = disparity[uc * vmax + vc];
		const int droad = lroundf(road[vc]);
		const float sigmad = sqrtf(SIGMA_LINE_A * SIGMA_LINE_A * vc * vc + SIGMA_LINE_B * SIGMA_LINE_B);

		if (inside)
			warpShuffleReduction(Lp, minCostsOO, minCostsGO, minCostOO, dmax, droad, vp, tid);

		const int minCost = inside ? min(minCostOO, static_cast<int>(Lp[dmax])) : 0;

		for (int k = tid, t = 0; k < n; k += WARP_SIZE, t++)
		{
			const bool object = k < dmax;

			// matching cost
			const int dhat = object ? k : droad;
			const int diff = object ? abs(d - dhat) : lroundf(max(d - dhat, 0) / sigmad);
			const int MC = min(diff, MC_MAX_DIFF);

			// transition cost
			int PT = 0;
			if (inside)
			{
				const int cost0 = Lp[k]; // no penalty
				int cost1 = P_INF, cost2 = P_INF, cost3 = P_INF;
				if (object)
				{
					// Object from Object
					cost1 = minCostsOO[FORE][t] + P_OO1;
					cost2 = minCostsOO[BACK][t] + P_OO2;

					// Object from Ground
					cost3 = k > droad ? P_OG2 : P_INF;
				}
				else
				{
					// Ground from Object
					cost3 = droad >= 0 && droad < dmax ? Lp[droad] + P_OG1 : P_INF;
				}

				PT = min(min(cost0, cost1), min(cost2, cost3));
			}

			const int cost = MC + PT - minCost;
			Lc[k] = cost;
			tmpLp[k] = cost;
		}

		for (int k = tid; k < n; k += WARP_SIZE)
			Lp[k] = tmpLp[k];
	}
}

template <typename T>
__global__ void scanCostBackKernel(const T* __restrict__ disparity, int umax, int vmax, int dmax,
	const float* __restrict__ DR, int vhor, uint16_t* __restrict__ L)
{
	const int uc = blockIdx.y * blockDim.y + threadIdx.y;
	if (uc > umax)
		return;

	const int tid = threadIdx.x;
	const int n = dmax + 1;

	// cost buffers
	L += uc * vmax * n;
	extern __shared__ uint16_t Lp[];

	uint16_t minCostsOO[2][MAX_WARPS_PER_DMAX];
	uint16_t minCostsGO[2];
	int minCostOO = P_INF;

	for (int vc = vmax - 1; vc >= 0; vc--)
	{
		uint16_t* Lc = L + vc * n;

		const int vp = vc + 1;
		const bool inside = vp >= 0 && vp < vmax;
		const int d = disparity[uc * vmax + vc];
		const int droad = lroundf(DR[vc]);
		const float sigmad = sqrtf(SIGMA_LINE_A * SIGMA_LINE_A * vc * vc + SIGMA_LINE_B * SIGMA_LINE_B);

		if (inside)
			warpShuffleReduction(Lp, minCostsOO, minCostsGO, minCostOO, dmax, droad, vp, tid);

		const int minCost = inside ? min(minCostOO, static_cast<int>(Lp[dmax])) : 0;

		for (int k = tid, t = 0; k < n; k += WARP_SIZE, t++)
		{
			const bool object = k < dmax;

			// matching cost
			const int dhat = object ? k : droad;
			const int diff = object ? abs(d - dhat) : lroundf(max(d - dhat, 0) / sigmad);

			const int MC = min(diff, MC_MAX_DIFF);

			// transition cost
			int PT = 0;
			if (inside)
			{
				const int cost0 = Lp[k]; // no penalty
				int cost1 = P_INF, cost2 = P_INF, cost3 = P_INF;
				if (object)
				{
					// Object from Object
					cost1 = minCostsOO[FORE][t] + P_OO2;
					cost2 = minCostsOO[BACK][t] + P_OO1;

					// Object from Ground
					cost3 = k == droad ? Lp[dmax] + P_OG1 : P_INF;
				}
				else
				{
					// Ground from Object
					cost3 = minCostsGO[BACK] + P_OG2;
				}

				PT = min(min(cost0, cost1), min(cost2, cost3));
			}

			const int cost = MC + PT - minCost;
			Lc[k] = cost;
			Lp[k] = cost;
		}
	}
}

__global__ void winnerTakesAllKernel(const uint16_t* __restrict__ LFore, const uint16_t* __restrict__ LBack,
	uint16_t* __restrict__ labels, int umax, int vmax, int dmax, int vhor)
{
	const int warpIdx = threadIdx.x / WARP_SIZE;
	const int laneIdx = threadIdx.x % WARP_SIZE;

	const int uc = blockIdx.y * blockDim.y + threadIdx.y;
	const int vc = blockIdx.x * WTA_WARPS_PER_BLOCK + warpIdx;
	if (uc >= umax || vc >= vmax)
		return;

	const int n = dmax + 1;

	int minSCost = INT_MAX;
	int label = -1;

	LFore += (uc * vmax + vc) * n;
	LBack += (uc * vmax + vc) * n;

	for (int t = 0, k = laneIdx; t * WARP_SIZE < n; t++, k += WARP_SIZE)
	{
		const bool object = k < dmax;
		const bool invalid = (!object && vc <= vhor) || k >= n;

		int scost = invalid ? P_INF : LFore[k] + LBack[k];
		int mink = k;

		for (int d = 16; d > 0; d >>= 1)
		{
			int _scost = SHFL_DOWN(scost, d);
			int _mink = SHFL_DOWN(mink, d);

			if (_scost < scost)
			{
				scost = _scost;
				mink = _mink;
			}
			else if (_scost == scost)
				mink = min(mink, _mink);
		}

		if (laneIdx == 0 && scost < minSCost)
		{
			minSCost = scost;
			label = mink;
		}
	}

	if (laneIdx == 0)
		labels[uc * vmax + vc] = label;
}

template <typename T>
void horizontalMedianGPU(const T* D, int pitch, T* columns, int umax, int vmax, int segmentWidth)
{
	const dim3 block(32, 32);
	const dim3 grid(divUp(umax, block.x), divUp(vmax, block.y));
	horizontalMedianKernel<<<grid, block>>>(D, pitch, columns, umax, vmax, segmentWidth);
	CUDA_CHECK(cudaGetLastError());
}
template void horizontalMedianGPU<uint8_t>(const uint8_t* D, int pitch, uint8_t* columns, int umax, int vmax, int segmentWidth);
template void horizontalMedianGPU<uint16_t>(const uint16_t* D, int pitch, uint16_t* columns, int umax, int vmax, int segmentWidth);

template <typename T>
void scanCostForeGPU(const T* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L,
	cudaStream_t stream)
{
	const dim3 grid(1, umax);
	const dim3 block(32, 1);

	const int n = dmax + 1;
	const size_t sharedSize = sizeof(uint16_t) * 2 * n;

	scanCostForeKernel<<<grid, block, sharedSize, stream>>>(D, umax, vmax, dmax, DR, vhor, L);
	CUDA_CHECK(cudaGetLastError());
}
template void scanCostForeGPU<uint8_t>(const uint8_t* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L, cudaStream_t stream);
template void scanCostForeGPU<uint16_t>(const uint16_t* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L, cudaStream_t stream);

template <typename T>
void scanCostBackGPU(const T* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L,
	cudaStream_t stream)
{
	const dim3 grid(1, umax);
	const dim3 block(32, 1);

	const int n = dmax + 1;
	const size_t sharedSize = sizeof(uint16_t) * n;

	scanCostBackKernel<<<grid, block, sharedSize, stream>>>(D, umax, vmax, dmax, DR, vhor, L);
	CUDA_CHECK(cudaGetLastError());
}
template void scanCostBackGPU<uint8_t>(const uint8_t* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L, cudaStream_t stream);
template void scanCostBackGPU<uint16_t>(const uint16_t* D, int umax, int vmax, int dmax, const float* DR, int vhor, uint16_t* L, cudaStream_t stream);

void winnerTakesAllGPU(uint16_t* LFore, uint16_t* LBack, uint16_t* labels,
	int umax, int vmax, int dmax, int vhor)
{
	const dim3 grid(divUp(vmax, WTA_WARPS_PER_BLOCK), umax);
	const dim3 block(WTA_WARPS_PER_BLOCK * WARP_SIZE, 1);
	winnerTakesAllKernel<<<grid, block>>>(LFore, LBack, labels, umax, vmax, dmax, vhor);
	CUDA_CHECK(cudaGetLastError());
}

} // namespace sgm
