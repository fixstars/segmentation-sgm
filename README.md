# Segmentation SGM
A CUDA implementation of Segmentation SGM

<div style="text-align: center;">
<img src="https://github.com/fixstars/segmentation-sgm/wiki/images/segmentation_001.png" width=1024><br/>
</div>

## Description
**Segmentation SGM** is an implementation of segmentation algorithm that utilize Semi-Global Matching.  
From a given disparity map, it extracts segments that cover obstacles. It is implemented using C++ and CUDA.

## Algorithm
- The detailed algorithm is [here](https://github.com/fixstars/segmentation-sgm/wiki/Segmentation-SGM) (currently Japanese only).

## Requirement
- CUDA
- OpenCV 3.0 or later
- CMake 3.10 or later

## How to build
```
$ git clone https://github.com/fixstars/segmentation-sgm.git
$ cd segmentation-sgm
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Enable sample with libSGM
If you have installed [libSGM](https://github.com/fixstars/libSGM), you can run `sample/movie_with_libsgm` and `sample/benchmark` by following command.

```
cmake -DWITH_LIBSGM=ON -DDCMAKE_MODULE_PATH=path/to/libSGM -DCMAKE_INSTALL_PREFIX=path/to/libSGM ..
```

## How to run
```
./segmentation_sgm_sample_movie left-image-format right-image-format camera.xml
```
- left-image-format
    - the left image sequence
- right-image-format
    - the right image sequence
- camera.xml
    - the camera intrinsic and extrinsic parameters

### Example
 ```
./segmentation_sgm_sample_movie img_c0_%09d.pgm img_c1_%09d.pgm daimler_gt_stixel.xml
```

### Data
The sample images available at [Daimler Urban Scene Segmentation Benchmark Dataset 2014](http://www.6d-vision.com/scene-labeling) and [Daimler Ground Truth Stixel Dataset](http://www.6d-vision.com/ground-truth-stixel-dataset) are used to test the software.

## Performance
The Segmentation SGM performance obtained from benchmark sample

|Device \ Width x Height x Max Disparity|1024 x 440 x 64|1024 x 440 x 128|
|--|--|--|
|Tegra X2    |16 [ms] (64FPS) |23[ms] (44FPS) |
  
## Author
The "SGM Team"  
[Fixstars Corporation](http://www.fixstars.com/)

## License
Apache License 2.0
