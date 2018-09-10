# segmentation-sgm
An implementation of segmentation SGM

## Requirement
- OpenCV
- OpenMP (optional)

## How to build
```
$ git clone ssh://git@gitlab.fixstars.com:8022/tech/adaskit/segmentation-sgm.git
$ cd segmentation-sgm
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to use
```
./segmentation-sgm left-image-format right-image-format camera.xml
```
- left-image-format
    - the left image sequence
- right-image-format
    - the right image sequence
- camera.xml
    - the camera intrinsic and extrinsic parameters

### Example
 ```
./segmentation-sgm images/img_c0_%09d.pgm images/img_c1_%09d.pgm camera.xml
```

### Data
- The sample movie images available at http://www.6d-vision.com/scene-labeling under "Daimler Urban Scene Segmentation Benchmark Dataset" are used to test the software.