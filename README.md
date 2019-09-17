# SFND 3D Object Tracking

Understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, know how to detect objects in an image using the YOLO deep-learning framework. And finally, know how to associate regions in a camera image with Lidar points in 3D space. 

**Implementation:
 
First, match 3D objects in the bounding box by using keypoints.
 
Second, compute the TTC based on Lidar measurements. 

Third, compute the TTC based on the camera, which requires to match keypoints to regions of interest and then to compute the TTC. 

Fourth, conduct various tests with the framework. The goal is to identify the most suitable detector/descriptor combination and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

