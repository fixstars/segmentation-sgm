#include "segmentation_sgm.h"

#include <memory>

#include <cuda_runtime.h>

#include "internal.h"
#include "parameters.h"
#include "device_buffer.h"
#include "macro.h"

namespace sgm
{

using CameraParameters = SegmentationSGM::CameraParameters;

class CudaStream
{
public:

	CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
	~CudaStream() { CUDA_CHECK(cudaStreamDestroy(stream_)); }

	operator cudaStream_t() const { return stream_; }
	void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
	cudaStream_t stream_;
};

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

template <typename T>
void horizontalMedian(const cv::Mat_<T>& src, cv::Mat_<T>& dst, int segmentWidth)
{
	// compute horizontal median of each column

	const int umax = src.cols / segmentWidth;
	const int vmax = src.rows;
	const int pitch = static_cast<int>(src.step / sizeof(T));

	CV_Assert(umax * segmentWidth <= src.cols);

	dst.create(umax, vmax);

	DeviceBuffer<T> d_src(vmax * pitch);
	DeviceBuffer<T> d_dst(vmax * umax);

	CUDA_CHECK(cudaMemcpy(d_src.data, src.data, sizeof(T) * vmax * pitch, cudaMemcpyHostToDevice));

	horizontalMedianGPU(d_src.data, pitch, d_dst.data, umax, vmax, segmentWidth);

	CUDA_CHECK(cudaMemcpy(dst.data, d_dst.data, sizeof(T) * vmax * umax, cudaMemcpyDeviceToHost));
}

template void horizontalMedian<uchar>(const cv::Mat1b& src, cv::Mat1b& dst, int segmentWidth);
template void horizontalMedian<ushort>(const cv::Mat1w& src, cv::Mat1w& dst, int segmentWidth);

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
template <typename T>
static Line calcRoadModelVD(const cv::Mat_<T>& disparity, const CameraParameters& camera,
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

template <typename T>
class ScanCost
{

public:

	struct CostSize
	{
		int umax, vmax, nlabels;
		CostSize(int vmax = 0, int umax = 0, int nlabels = 0) : vmax(vmax), umax(umax), nlabels(nlabels) {}
		bool operator==(const CostSize& rhs) const
		{
			const CostSize& lhs(*this);
			return lhs.vmax == rhs.vmax && lhs.umax == rhs.umax && lhs.nlabels == rhs.nlabels;
		}
		bool operator!=(const CostSize& rhs) const
		{
			const CostSize& lhs(*this);
			return !(lhs == rhs);
		}
	};

	void init(int umax, int vmax, int nlabels)
	{
		d_labels_.allocate(umax * vmax);
		d_LFore_.allocate(umax * vmax * nlabels);
		d_LBack_.allocate(umax * vmax * nlabels);

		cudaMemset(d_LFore_.data, 0, sizeof(ushort) * d_LFore_.size);
		cudaMemset(d_LBack_.data, 0, sizeof(ushort) * d_LBack_.size);

		costSize_ = CostSize(umax, vmax, nlabels);
	}

	void compute(const T* D, int umax, int vmax, int dmax, const float* DR, int vhor, cv::Mat1w& labels)
	{
		const int n = dmax + 1;

		labels.create(umax, vmax);

		if (CostSize(umax, vmax, n) != costSize_)
			init(umax, vmax, n);

		CudaStream stream1, stream2;

		scanCostForeGPU(D, umax, vmax, dmax, DR, vhor, d_LFore_.data, stream1);
		scanCostBackGPU(D, umax, vmax, dmax, DR, vhor, d_LBack_.data, stream2);

		stream1.synchronize();
		stream2.synchronize();

		winnerTakesAllGPU(d_LFore_.data, d_LBack_.data, d_labels_.data, umax, vmax, dmax, vhor);

		CUDA_CHECK(cudaMemcpy(labels.data, d_labels_.data, sizeof(ushort) * umax * vmax, cudaMemcpyDeviceToHost));
	}

private:

	DeviceBuffer<ushort> d_labels_, d_LFore_, d_LBack_;
	CostSize costSize_;
};

template <typename T>
void scanCost(const cv::Mat_<T>& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels)
{
	const int umax = D.rows;
	const int vmax = D.cols;

	DeviceBuffer<T> d_disparity(umax * vmax);
	DeviceBuffer<float> d_road(vmax);
	CUDA_CHECK(cudaMemcpy(d_disparity.data, D.data, sizeof(T) * umax * vmax, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_road.data, DR.data(), sizeof(float) * vmax, cudaMemcpyHostToDevice));

	ScanCost<T> scan;
	scan.compute(d_disparity.data, umax, vmax, dmax, d_road.data, vhor, labels);
}

template void scanCost<uchar>(const cv::Mat1b& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels);
template void scanCost<ushort>(const cv::Mat1w& D, int dmax, const std::vector<float>& DR, int vhor, cv::Mat1w& labels);

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

	struct ToDepth
	{
		ToDepth(const CameraParameters& camera)
		{
			bf = camera.baseline * camera.fu;
			v0 = camera.v0;
			sinTilt = sinf(camera.tilt);
			cosTilt = cosf(camera.tilt);
		}
		float operator()(int d, int v) const
		{
			return (bf / d) * cosTilt - (v - v0) * sinTilt;
		}
		float bf, v0, sinTilt, cosTilt;
	};

	const int vmax = labels.cols;
	const int umax = labels.rows;
	const int TOLERANCE_D = 1;
	const float TOLERANCE_Z = 5;
	const int MIN_HEIGHT = 10;
	auto isObject = [=](int d) { return d > 0 && d < dmax; };
	ToDepth toDepth(camera);

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
				const float dz = toDepth(dp, vp) - toDepth(dc, vc);
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
				const float dz = toDepth(sp.d, sp.vB) - toDepth(sc.d, sc.vT);
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

template <typename T>
class SegmentationSGMEngine
{

public:

	using Segment = SegmentationSGM::Segment;
	using Parameters = SegmentationSGM::Parameters;

	void compute(const T* disparity, int width, int height, int pitch,
		std::vector<Segment>& segments, const Parameters& param)
	{
		static_assert(std::is_same<T, uchar>::value || std::is_same<T, ushort>::value, "");

		segments.clear();

		const int segmentWidth = param.segmentWidth;
		const int umax = width / segmentWidth;
		const int vmax = height;
		const int dmax = param.maxDisparity;

		CameraParameters camera = param.camera;

		// compute horizontal median of each column
		d_columns_.allocate(umax * vmax);
		horizontalMedianGPU(disparity, pitch, d_columns_.data, umax, vmax, segmentWidth);

		// copy to host
		h_columns_.create(umax, vmax);
		CUDA_CHECK(cudaMemcpy(h_columns_.data, d_columns_.data, sizeof(T) * umax * vmax, cudaMemcpyDeviceToHost));

		// compute road model (assumes planar surface)
		Line line;
		if (param.roadEstimation == SegmentationSGM::ROAD_ESTIMATION_AUTO)
		{
			line = calcRoadModelVD(h_columns_, camera);

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
		h_road_.resize(vmax);
		for (int v = 0; v < vmax; v++)
			h_road_[v] = line.a * v + line.b;

		// copy to device
		d_road_.allocate(vmax);
		CUDA_CHECK(cudaMemcpy(d_road_.data, h_road_.data(), sizeof(float) * vmax, cudaMemcpyHostToDevice));

		// horizontal row from which road dispaliry becomes negative
		const int vhor = cvRound(-line.b / line.a);

		// scan cost
		scanCost_.compute(d_columns_.data, umax, vmax, dmax, d_road_.data, vhor, labels_);

		// segment extraction
		extractSegments(labels_, dmax, camera, segments, segmentWidth);
	}

private:

	DeviceBuffer<T> d_disparity_, d_columns_;
	cv::Mat_<T> h_columns_;

	DeviceBuffer<float> d_road_;
	std::vector<float> h_road_;

	cv::Mat1w labels_;
	ScanCost<T> scanCost_;
};

class SegmentationSGMBase
{
public:

	using Segment = SegmentationSGM::Segment;
	using Parameters = SegmentationSGM::Parameters;

	virtual void compute(const void* disparity, int width, int height, int pitch,
		std::vector<Segment>& segments, const Parameters& param) = 0;

	virtual ~SegmentationSGMBase() {}
};

template <typename T>
class SegmentationSGMImpl : public SegmentationSGMBase
{
public:

	void compute(const void* disparity, int width, int height, int pitch,
		std::vector<Segment>& segments, const Parameters& param) override
	{
		engine_.compute(reinterpret_cast<const T*>(disparity), width, height, pitch, segments, param);
	}

private:
	SegmentationSGMEngine<T> engine_;
};

class SegmentationSGM::Impl
{

public:

	Impl(const Parameters& param) : type_(-1), param_(param)
	{
		CV_Assert(param.maxDisparity <= MAX_DISPARITY);
		CV_Assert(param.maxDisparity % 32 == 0);
		CV_Assert(param.segmentWidth > 0 && param.segmentWidth <= MAX_SEGMENT_WIDTH);
	}

	template <typename T>
	void compute(const T* disparity, int width, int height, int pitch,
		std::vector<Segment>& segments, InputPtrType ptr)
	{
		static_assert(std::is_same<T, uchar>::value || std::is_same<T, ushort>::value, "");

		if (cv::DataType<T>::type != type_)
		{
			segmentation_.reset(new SegmentationSGMImpl<T>());
			type_ = cv::DataType<T>::type;
		}

		const T* d_disparity = disparity;

		if (ptr == INPUT_PTR_HOST)
		{
			// copy to device
			d_disparity_.allocate(height * pitch);
			CUDA_CHECK(cudaMemcpy(d_disparity_.data, disparity, sizeof(T) * height * pitch, cudaMemcpyHostToDevice));
			d_disparity = reinterpret_cast<const T*>(d_disparity_.data);
		}

		segmentation_->compute(d_disparity, width, height, pitch, segments, param_);
	}

	void compute(const cv::Mat& disparity, std::vector<Segment>& segments)
	{
		const int type = disparity.type();

		CV_Assert(type == CV_8U || type == CV_16U);

		const void* data = disparity.data;
		const int w = disparity.cols;
		const int h = disparity.rows;
		const int pitch = static_cast<int>(disparity.step / disparity.elemSize());

		if (type == CV_8U)
			compute(reinterpret_cast<const uchar*>(data), w, h, pitch, segments, INPUT_PTR_HOST);
		else
			compute(reinterpret_cast<const ushort*>(data), w, h, pitch, segments, INPUT_PTR_HOST);
	}

private:

	DeviceBuffer<ushort> d_disparity_;
	std::unique_ptr<SegmentationSGMBase> segmentation_;
	int type_;

	Parameters param_;
};

SegmentationSGM::SegmentationSGM(const Parameters& param)
{
	impl_ = new Impl(param);
}

SegmentationSGM::~SegmentationSGM()
{
	delete impl_;
}

void SegmentationSGM::compute(const cv::Mat& disparity, std::vector<Segment>& segments)
{
	impl_->compute(disparity, segments);
}

void SegmentationSGM::compute(const uint8_t* disparity, int width, int height, int pitch,
	std::vector<Segment>& segments, InputPtrType ptr)
{
	impl_->compute(disparity, width, height, pitch, segments, ptr);
}

void SegmentationSGM::compute(const uint16_t* disparity, int width, int height, int pitch,
	std::vector<Segment>& segments, InputPtrType ptr)
{
	impl_->compute(disparity, width, height, pitch, segments, ptr);
}

SegmentationSGM::CameraParameters::CameraParameters()
{
	// default settings
	fu = 1.f;
	fv = 1.f;
	u0 = 0.f;
	v0 = 0.f;
	baseline = 0.2f;
	height = 1.f;
	tilt = 0.f;
}

SegmentationSGM::Parameters::Parameters(int segmentWidth, int maxDisparity, int roadEstimation, const CameraParameters& camera)
	: segmentWidth(segmentWidth), maxDisparity(maxDisparity), roadEstimation(roadEstimation), camera(camera) {}

} // namespace sgm
