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

#ifndef __SEGMENTATION_SGM_H__
#define __SEGMENTATION_SGM_H__

#include <opencv2/opencv.hpp>

namespace sgm
{

class SegmentationSGM
{
public:

	enum
	{
		ROAD_ESTIMATION_AUTO = 0, //!< road disparity are estimated by input disparity
		ROAD_ESTIMATION_CAMERA    //!< road disparity are estimated by camera tilt and height
	};

	enum InputPtrType
	{
		INPUT_PTR_HOST = 0,       //!< input pointer is host pointer
		INPUT_PTR_DEVICE          //!< input pointer is device pointer
	};

	/** @brief segment struct
	*/
	struct Segment
	{
		int u;                    //!< segment center x position
		int vT;                   //!< segment top y position
		int vB;                   //!< segment bottom y position
		int width;                //!< segment width
		float disp;               //!< segment average disparity
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

		CameraParameters();
	};

	struct Parameters
	{
		int segmentWidth;         //!< segment width
		int maxDisparity;         //!< maximum value of input disparity (half-open interval [0, max))
		int roadEstimation;       //!< road disparity estimation mode
		CameraParameters camera;  //!< camera parameters

		Parameters(int segmentWidth = 7, int maxDisparity = 64, int roadEstimation = ROAD_ESTIMATION_AUTO,
			const CameraParameters& camera = CameraParameters());
	};

	/** @brief The constructor
	@param param input parameters
	*/
	SegmentationSGM(const Parameters& param = Parameters());

	/** @brief The destructor
	*/
	~SegmentationSGM();

	/** @brief Computes segments in a disparity map
	@param disparity Input 8-bit or 16-bit single-channel disparity map
	@param segments Output array of segments
	*/
	void compute(const cv::Mat& disparity, std::vector<Segment>& segments);

	/** @brief Computes segments in a disparity map (Low-level interface)
	@param disparity Pointer to 8-bit single-channel disparity map
	@param width Width of the disparity map
	@param height Height of the disparity map
	@param pitch Pitch of the disparity map (in pixels)
	@param segments Output array of segments
	@param ptr Input pointer type. see InputPtrType
	*/
	void compute(const uint8_t* disparity, int width, int height, int pitch, std::vector<Segment>& segments,
		InputPtrType ptr = INPUT_PTR_DEVICE);

	/** @brief Computes segments in a disparity map (Low-level interface)
	@param disparity Pointer to 8-bit single-channel disparity map
	@param width Width of the disparity map
	@param height Height of the disparity map
	@param pitch Pitch of the disparity map (in pixels)
	@param segments Output array of segments
	@param ptr Input pointer type. see InputPtrType
	*/
	void compute(const uint16_t* disparity, int width, int height, int pitch, std::vector<Segment>& segments,
		InputPtrType ptr = INPUT_PTR_DEVICE);

private:

	class Impl;
	Impl* impl_;
};

} // namespace cpu

#endif // !__SEGMENTATION_SGM_H__