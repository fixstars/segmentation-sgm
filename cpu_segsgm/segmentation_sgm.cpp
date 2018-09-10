#include "segmentation_sgm.h"

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#ifdef _WIN32
#define OMP_PARALLEL_FOR __pragma(omp parallel for)
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#endif
#else
#define OMP_PARALLEL_FOR
#endif

static const int P_OO1 = 10;        // object <=> object transition cost
static const int P_OO2 = 2 * P_OO1; // object <=> object transition cost when violate object ordering
static const int P_OG1 = 5;         // object <=> ground transition cost
static const int P_OG2 = 2 * P_OG1; // object <=> ground transition cost
static const int INF = std::numeric_limits<short>::max();

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
static inline T median(std::vector<T>& x)
{
	return median(x.data(), x.size());
}

using CameraParameters = SegmentationSGM::CameraParameters;

struct Line
{
	Line(float a = 0, float b = 0) : a(a), b(b) {}
	Line(const cv::Point2f& pt1, const cv::Point2f& pt2)
	{
		a = (pt2.y - pt1.y) / (pt2.x - pt1.x);
		b = -a * pt1.x + pt1.y;
	}
	float a, b;
};

// Transformation between pixel coordinate and world coordinate
struct CoordinateTransform
{
	CoordinateTransform(const CameraParameters& camera) : camera(camera)
	{
		sinTilt = (sinf(camera.tilt));
		cosTilt = (cosf(camera.tilt));
		B = camera.baseline * camera.fu / camera.fv;
	}

	inline float toY(float d, int v) const
	{
		return (B / d) * ((v - camera.v0) * cosTilt + camera.fv * sinTilt);
	}

	inline float toZ(float d, int v) const
	{
		return (B / d) * (camera.fv * cosTilt - (v - camera.v0) * sinTilt);
	}

	inline float toV(float Y, float Z) const
	{
		return camera.fv * (Y * cosTilt - Z * sinTilt) / (Y * sinTilt + Z * cosTilt) + camera.v0;
	}

	inline float toD(float Y, float Z) const
	{
		return camera.baseline * camera.fu / (Y * sinTilt + Z * cosTilt);
	}

	CameraParameters camera;
	float sinTilt, cosTilt, B;
};

// estimate road model from camera tilt and height
static Line calcRoadModelCamera(const CameraParameters& camera)
{
	const float sinTilt = sinf(camera.tilt);
	const float cosTilt = cosf(camera.tilt);
	const float a = (camera.baseline / camera.height) * cosTilt;
	const float b = (camera.baseline / camera.height) * (camera.fu * sinTilt - camera.v0 * cosTilt);
	return Line(a, b);
}

// estimate road model from v-disparity
template <typename disp_type>
static Line calcRoadModelVD(const cv::Mat_<disp_type>& disparity, const CameraParameters& camera,
	int samplingStep = 2, int minDisparity = 10, int maxIterations = 32, float inlierRadius = 3, float maxCameraHeight = 5)
{
	const int umax = disparity.rows;
	const int vmax = disparity.cols;

	// sample v-disparity points
	std::vector<cv::Point2f> points;
	points.reserve(vmax * umax);
	for (int u = 0; u < umax; u += samplingStep)
		for (int v = 0; v < vmax; v += samplingStep)
			if (disparity(u, v) >= minDisparity)
				points.push_back(cv::Point2f(static_cast<float>(v), disparity(u, v)));

	if (points.empty())
		return Line(0, 0);

	// estimate line by RANSAC
	cv::RNG random;
	Line bestLine;
	int maxInliers = 0;
	for (int iter = 0; iter < maxIterations; iter++)
	{
		// sample 2 points and get line parameters
		const cv::Point2f& pt1 = points[random.next() % points.size()];
		const cv::Point2f& pt2 = points[random.next() % points.size()];
		if (pt1.x == pt2.x)
			continue;

		const Line line(pt1, pt2);

		// estimate camera tilt and height
		const float tilt = atanf((line.a * camera.v0 + line.b) / (camera.fu * line.a));
		const float height = camera.baseline * cosf(tilt) / line.a;

		// skip if not within valid range
		if (height <= 0.f || height > maxCameraHeight)
			continue;

		// count inliers within a radius and update the best line
		int inliers = 0;
		for (const auto& pt : points)
			if (fabs(line.a * pt.x + line.b - pt.y) <= inlierRadius)
				inliers++;

		if (inliers > maxInliers)
		{
			maxInliers = inliers;
			bestLine = line;
		}
	}

	// apply least squares fitting using inliers around the best line
	double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
	int n = 0;
	for (const auto& pt : points)
	{
		const float x = pt.x;
		const float y = pt.y;
		const float yhat = bestLine.a * x + bestLine.b;
		if (fabs(yhat - y) <= inlierRadius)
		{
			sx += x;
			sy += y;
			sxx += x * x;
			syy += y * y;
			sxy += x * y;
			n++;
		}
	}

	const float a = static_cast<float>((n * sxy - sx * sy) / (n * sxx - sx * sx));
	const float b = static_cast<float>((sxx * sy - sxy * sx) / (n * sxx - sx * sx));
	return Line(a, b);
}

template <typename disp_type>
void scanOneColumn(int uc, const cv::Mat_<disp_type>& D, const std::vector<float>& DR, int vhor,
	cv::Mat1w& labels, int dmax, const CameraParameters& camera)
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
			const int dRoad = cvRound(DR[vc]);
			const float sigmad = sqrtf(sigmaa * sigmaa * vc * vc + sigmab * sigmab);

			int minCostOO = INF, minCostGO = INF;
			for (int k = 0; k < dmax && inside; k++)
			{
				// Object from Object
				minCostsOO(FORE, k) = minCostOO;
				minCostOO = std::min(minCostOO, (int)L(vp, k));

				// Ground from Object
				minCostsGO(FORE) = minCostGO;
				if (k < dRoad)
					minCostGO = std::min(minCostGO, (int)L(vp, k));
			}

			minCostOO = INF, minCostGO = INF;
			for (int k = dmax - 1; k >= 0 && inside; k--)
			{
				// Object from Object
				minCostsOO(BACK, k) = minCostOO;
				minCostOO = std::min(minCostOO, (int)L(vp, k));

				// Ground from Object
				minCostsGO(BACK) = minCostGO;
				if (k > dRoad)
					minCostGO = std::min(minCostGO, (int)L(vp, k));
			}

			const int minCost = inside ? std::min(minCostOO, (int)L(vp, dmax)) : 0;

			int minSCost = std::numeric_limits<int>::max();
			int mink = -1;

			for (int k = 0; k < n; k++)
			{
				const bool object = k < dmax;

				// matching cost
				const int dhat = object ? k : dRoad;
				int diff = 0;
				if (d > 0)
					diff = object ? std::abs(d - dhat) : cvRound(std::max(d - dhat, 0) / sigmad);
				diff = std::min(MAX_DIFF * diff / dmax, MAX_DIFF);

				const int PL = object && vc > vhor ? 1 : 0;
				const int MC = diff + PL;

				// transition cost
				int PT = 0;
				if (inside)
				{
					const int cost0 = L(vp, k); // no penalty
					int cost1 = INF, cost2 = INF, cost3 = INF;
					if (object)
					{
						// Object from Object
						cost1 = minCostsOO(FORE, k) + forward ? P_OO1 : P_OO2;
						cost2 = minCostsOO(BACK, k) + forward ? P_OO2 : P_OO1;

						// Object from Ground
						if (forward)
							cost3 = k > dRoad ? P_OG2 : INF;
						else
							cost3 = k == dRoad ? L(vp, dmax) + P_OG1 : INF;
					}
					else
					{
						// Ground from Object
						if (forward)
							cost3 = dRoad >= 0 && dRoad < dmax ? L(vp, dRoad) + P_OG1 : INF;
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
					const int cost = !object && vc <= vhor ? INF : L(vc, k) + Lc;
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

void extractSegments(const cv::Mat1w& labels, int dmax, const CameraParameters& camera,
	std::vector<SegmentationSGM::Segment>& segments, int segmentWidth)
{
	// post-processing (region merging) for segmentation

	struct Segment
	{
		Segment(int vT = 0, int vB = 0, int d = 0) : vT(vT), vB(vB), d(d) {}
		int H() const { return vB - vT; }
		int vT, vB, d;
	};

	const int vmax = labels.cols;
	const int umax = labels.rows;
	const int TOLERANCE_D = 1;
	const float TOLERANCE_Z = 5;
	const int MIN_HEIGHT = 10;
	auto isObject = [=](int d) { return d > 0 && d < dmax; };
	CoordinateTransform tf(camera);

	std::vector<Segment> tmpSegments1, tmpSegments2;
	tmpSegments1.reserve(vmax);
	tmpSegments2.reserve(vmax);

	std::vector<int> buf(vmax);
	int sumd = 0, n = 0;

	for (int u = 0; u < umax; u++)
	{
		tmpSegments1.clear();
		tmpSegments2.clear();

		// initial region merging
		int vT = 0;
		for (int vc = 0; vc <= vmax; vc++)
		{
			if (vc == 0)
			{
				const int dc = labels(u, vc);
				n = 0;
				if (isObject(dc)) buf[n++] = dc;
				continue;
			}
			if (vc == vmax)
			{
				if (n && vc - vT >= MIN_HEIGHT)
					tmpSegments1.push_back(Segment(vT, vc, median(buf.data(), n)));
				continue;
			}

			const int vp = vc - 1;
			const int dp = labels(u, vp);
			const int dc = labels(u, vc);
			bool merge = false;
			if (isObject(dp) && isObject(dc))
			{
				const float dz = tf.toZ(static_cast<float>(dp), vp) - tf.toZ(static_cast<float>(dc), vc);
				merge = fabsf(dz) <= TOLERANCE_Z || std::abs(dp - dc) <= TOLERANCE_D;
			}
			if (merge)
			{
				buf[n++] = dc;
			}
			else
			{
				if (isObject(dp) && vc - vT >= MIN_HEIGHT)
					tmpSegments1.push_back(Segment(vT, vc, median(buf.data(), n)));

				vT = vc;
				n = 0;
				if (isObject(dc)) buf[n++] = dc;
			}
		}

		// second region merging
		for (size_t i = 0; i <= tmpSegments1.size(); i++)
		{
			if (i == 0)
			{
				vT = tmpSegments1[0].vT;
				sumd = tmpSegments1[0].d;
				n = 1;
				continue;
			}
			if (i == tmpSegments1.size())
			{
				const Segment& sp = tmpSegments1[i - 1];
				tmpSegments2.push_back(n == 1 ? sp : Segment(vT, sp.vB, cvRound(1. * sumd / n)));
				continue;
			}

			const Segment& sp = tmpSegments1[i - 1];
			const Segment& sc = tmpSegments1[i];
			bool merge = false;
			if (std::abs(sp.vB - sc.vT) <= 10)
			{
				const float dz = tf.toZ(static_cast<float>(sp.d), sp.vB) - tf.toZ(static_cast<float>(sc.d), sc.vT);
				merge = fabsf(dz) <= TOLERANCE_Z || std::abs(sp.d - sc.d) <= TOLERANCE_D;
			}

			if (merge)
			{
				sumd += sc.d;
				n++;
			}
			else
			{
				tmpSegments2.push_back(n == 1 ? sp : Segment(vT, sp.vB, cvRound(1. * sumd / n)));
				vT = sc.vT;
				sumd = sc.d;
				n = 1;
			}
		}

		for (const auto& tmpSegment : tmpSegments2)
		{
			SegmentationSGM::Segment segment;
			segment.u = segmentWidth * u + segmentWidth / 2;
			segment.vT = tmpSegment.vT;
			segment.vB = tmpSegment.vB;
			segment.width = segmentWidth;
			segment.disp = static_cast<float>(tmpSegment.d);
			segments.push_back(segment);
		}
	}
}

template <typename disp_type>
static void compute(const cv::Mat_<disp_type>& disparity, std::vector<SegmentationSGM::Segment>& segments,
	const SegmentationSGM::Parameters& param)
{
	static_assert(std::is_same<disp_type, uchar>::value || std::is_same<disp_type, ushort>::value, "");

	segments.clear();

	const int segmentWidth = param.segmentWidth;
	const int umax = disparity.cols / segmentWidth;
	const int vmax = disparity.rows;
	const int dmax = param.maxDisparity;
	CameraParameters camera = param.camera;

	// compute horizontal median of each column
	cv::Mat_<disp_type> columns(umax, vmax);
	std::vector<disp_type> buf(segmentWidth);
	for (int v = 0; v < vmax; v++)
	{
		for (int u = 0; u < umax; u++)
		{
			// compute horizontal median
			for (int du = 0; du < segmentWidth; du++)
				buf[du] = disparity(v, u * segmentWidth + du);
			const disp_type m = median(buf);

			// store with transposed
			columns(u, v) = m;
		}
	}

	// compute road model (assumes planar surface)
	Line line;
	if (param.roadEstimation == SegmentationSGM::ROAD_ESTIMATION_AUTO)
	{
		line = calcRoadModelVD(columns, camera);

		// when AUTO mode, update camera tilt and height
		camera.tilt = atanf((line.a * camera.v0 + line.b) / (camera.fu * line.a));
		camera.height = camera.baseline * cosf(camera.tilt) / line.a;
	}
	else if (param.roadEstimation == SegmentationSGM::ROAD_ESTIMATION_CAMERA)
	{
		line = calcRoadModelCamera(camera);
	}
	else
	{
		CV_Error(cv::Error::StsInternal, "No such mode");
	}

	// compute expected road disparity
	std::vector<float> DR(vmax);
	for (int v = 0; v < vmax; v++)
		DR[v] = line.a * v + line.b;

	// horizontal row from which road dispaliry becomes negative
	const int vhor = cvRound(-line.b / line.a);

	// scan cost
	cv::Mat1w labels(umax, vmax);

	int uc;
OMP_PARALLEL_FOR
	for (uc = 0; uc < umax; uc++)
		scanOneColumn(uc, columns, DR, vhor, labels, dmax, camera);

	// segment extraction
	extractSegments(labels, dmax, param.camera, segments, segmentWidth);
}

SegmentationSGM::SegmentationSGM(const Parameters& param) : param_(param)
{
}

void SegmentationSGM::compute(const cv::Mat& disparity, std::vector<SegmentationSGM::Segment>& segments)
{
	CV_Assert(disparity.type() == CV_8U || disparity.type() == CV_16U);
	if (disparity.type() == CV_8U)
		::compute(cv::Mat_<uchar>(disparity), segments, param_);
	else if (disparity.type() == CV_16U)
		::compute(cv::Mat_<ushort>(disparity), segments, param_);
}
