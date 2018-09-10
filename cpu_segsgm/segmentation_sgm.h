#ifndef __SEGMENTATION_SGM_H__
#define __SEGMENTATION_SGM_H__

#include <opencv2/opencv.hpp>

class SegmentationSGM
{
public:

	enum
	{
		ROAD_ESTIMATION_AUTO = 0, //!< road disparity are estimated by input disparity
		ROAD_ESTIMATION_CAMERA    //!< road disparity are estimated by camera tilt and height
	};

	/** @brief segment struct
	*/
	struct Segment
	{
		int u;                        //!< segment center x position
		int vT;                       //!< segment top y position
		int vB;                       //!< segment bottom y position
		int width;                    //!< segment width
		float disp;                   //!< segment average disparity
	};

	/** @brief CameraParameters struct
	*/
	struct CameraParameters
	{
		float fu;                 //!< focal length x (pixel)
		float fv;                 //!< focal length y (pixel)
		float u0;                 //!< principal point x (pixel)
		float v0;                 //!< principal point y (pixel)
		float baseline;           //!< baseline (meter)
		float height;             //!< height position (meter), ignored when ROAD_ESTIMATION_AUTO
		float tilt;               //!< tilt angle (radian), ignored when ROAD_ESTIMATION_AUTO

		// default settings
		CameraParameters()
		{
			fu = 1.f;
			fv = 1.f;
			u0 = 0.f;
			v0 = 0.f;
			baseline = 0.2f;
			height = 1.f;
			tilt = 0.f;
		}
	};

	struct Parameters
	{
		int segmentWidth;         //!< segment width
		int maxDisparity;         //!< maximum value of input disparity (half-open interval [0, max))
		int roadEstimation;       //!< road disparity estimation mode
		CameraParameters camera;  //!< camera parameters

		// default settings
		Parameters()
		{
			segmentWidth = 10;
			maxDisparity = 64;
			roadEstimation = ROAD_ESTIMATION_AUTO;
			camera = CameraParameters();
		}
	};

	/** @brief The constructor
	@param param input parameters
	*/
	SegmentationSGM(const Parameters& param);

	/** @brief Computes segments in a disparity map
	@param disparity 8-bit or 16-bit single-channel disparity map
	@param output array of stixels
	*/
	void compute(const cv::Mat& disparity, std::vector<Segment>& segments);

private:

	Parameters param_;
};

#endif // !__SEGMENTATION_SGM_H__