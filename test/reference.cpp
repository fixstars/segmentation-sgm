#include "reference.h"

#include "parameters.h"

namespace ref
{

template <typename T>
static inline T median(T* x, size_t n)
{
	CV_Assert(n > 0);
	if (n == 1) return x[0];
	const auto m = n / 2;
	std::nth_element(x, x + m, x + n);
	return x[m];
}

template <typename T>
static inline T median(std::vector<T>& x) { return median(x.data(), x.size()); }

template <typename T>
void horizontalMedian(const cv::Mat_<T>& src, cv::Mat_<T>& dst, int segmentWidth)
{
	const int umax = src.cols / segmentWidth;
	const int vmax = src.rows;

	CV_Assert(umax * segmentWidth <= src.cols);

	// compute horizontal median of each column
	dst.create(umax, vmax);
	std::vector<T> buf(segmentWidth);
	for (int v = 0; v < vmax; v++)
	{
		for (int u = 0; u < umax; u++)
		{
			// compute horizontal median
			for (int du = 0; du < segmentWidth; du++)
			{
				const T d = src(v, u * segmentWidth + du);
				buf[du] = d < sgm::MAX_DISPARITY ? d : 0;
			}
			const T m = median(buf);

			// store with transposed
			dst(u, v) = m;
		}
	}
}
template void horizontalMedian<uchar>(const cv::Mat1b& src, cv::Mat1b& dst, int segmentWidth);
template void horizontalMedian<ushort>(const cv::Mat1w& src, cv::Mat1w& dst, int segmentWidth);

constexpr int P_OO1 = sgm::P_OO1;
constexpr int P_OO2 = sgm::P_OO2;
constexpr int P_OG1 = sgm::P_OG1;
constexpr int P_OG2 = sgm::P_OG2;
constexpr int P_INF = sgm::P_INF;

template <typename T>
void scanOneColumn(int uc, const cv::Mat_<T>& D, int dmax, const std::vector<float>& DR, int vhor,
	cv::Mat1w& labels)
{
	const int vmax = D.cols;
	//const int umax = D.rows;
	const int n = dmax + 1;
	enum { FORE = 0, BACK = 1 };

	const int MAX_DIFF = 64;
	const float sigmaa = 0.01f;
	const float sigmab = 1.f;

	//////////////////////////////////////////////////////////////////////////////
	// compute cost tables
	//////////////////////////////////////////////////////////////////////////////
	cv::Mat1w L = cv::Mat1w::zeros(vmax, n);
	cv::Mat1w minCostsOO(2, dmax);
	cv::Mat1w minCostsGO(2, 1);

	for (int dir = 0; dir < 2; dir++)
	{
		const bool forward = dir == 0;
		const int v0 = forward ? 0 : vmax - 1;
		const int v1 = forward ? vmax : -1;
		const int rv = forward ? 1 : -1;

		for (int vc = v0; vc != v1; vc += rv)
		{
			const int vp = vc - rv;
			const bool inside = vp >= 0 && vp < vmax;

			const int d = D(uc, vc);
			const int droad = cvRound(DR[vc]);
			const float sigmad = sqrtf(sigmaa * sigmaa * vc * vc + sigmab * sigmab);

			int minCostOO = P_INF, minCostGO = P_INF;
			for (int k = 0; k < dmax && inside; k++)
			{
				// Object from Object
				minCostsOO(FORE, k) = minCostOO;
				minCostOO = std::min(minCostOO, (int)L(vp, k));

				// Ground from Object
				minCostsGO(FORE) = minCostGO;
				if (k < droad)
					minCostGO = std::min(minCostGO, (int)L(vp, k));
			}

			minCostOO = P_INF, minCostGO = P_INF;
			for (int k = dmax - 1; k >= 0 && inside; k--)
			{
				// Object from Object
				minCostsOO(BACK, k) = minCostOO;
				minCostOO = std::min(minCostOO, (int)L(vp, k));

				// Ground from Object
				minCostsGO(BACK) = minCostGO;
				if (k > droad)
					minCostGO = std::min(minCostGO, (int)L(vp, k));
			}

			const int minCost = inside ? std::min(minCostOO, (int)L(vp, dmax)) : 0;

			int minSCost = std::numeric_limits<int>::max();
			int mink = -1;

			for (int k = 0; k < n; k++)
			{
				const bool object = k < dmax;

				// matching cost
				const int dhat = object ? k : droad;
				const int diff = object ? std::abs(d - dhat) : cvRound(std::max(d - dhat, 0) / sigmad);
				const int MC = std::min(diff, MAX_DIFF);

				// transition cost
				int PT = 0;
				if (inside)
				{
					const int cost0 = L(vp, k); // no penalty
					int cost1 = P_INF, cost2 = P_INF, cost3 = P_INF;
					if (object)
					{
						// Object from Object
						cost1 = minCostsOO(FORE, k) + (forward ? P_OO1 : P_OO2);
						cost2 = minCostsOO(BACK, k) + (forward ? P_OO2 : P_OO1);

						// Object from Ground
						if (forward)
							cost3 = k > droad ? L(vp, dmax) + P_OG2 : P_INF;
						else
							cost3 = k == droad ? L(vp, dmax) + P_OG1 : P_INF;
					}
					else
					{
						// Ground from Object
						if (forward)
							cost3 = droad >= 0 && droad < dmax ? L(vp, droad) + P_OG1 : P_INF;
						else
							cost3 = minCostsGO(BACK) + P_OG2;
					}

					PT = std::min(std::min(cost0, cost1), std::min(cost2, cost3));
				}

				const int Lc = MC + PT - minCost;

				///////////////////////////////////////////////////////////////////////////
				// Winner Takes All
				///////////////////////////////////////////////////////////////////////////
				if (dir > 0)
				{
					const int cost = !object && vc <= vhor ? P_INF : L(vc, k) + Lc;
					if (cost < minSCost)
					{
						minSCost = cost;
						mink = k;
					}
				}

				L(vc, k) = Lc;
			}

			if (dir > 0)
				labels(uc, vc) = mink;
		}
	}
}

template <typename T>
void scanCost(const cv::Mat_<T>& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels)
{
	const int umax = D.rows;
	const int vmax = D.cols;

	labels.create(umax, vmax);

	for (int uc = 0; uc < umax; uc++)
		scanOneColumn(uc, D, dmax, DR, vhor, labels);
}

template void scanCost<uchar>(const cv::Mat1b& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels);
template void scanCost<ushort>(const cv::Mat1w& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels);

} // namespace ref
