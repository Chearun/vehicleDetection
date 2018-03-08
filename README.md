## Report

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chessboardImage]: ./reportPict/chessBoard.PNG "Undistorted chessboard"
[undistortImage]: ./reportPict/undistort.PNG "Undistorted project image"
[binaryImage]: ./reportPict/binary.PNG "Binary project image"
[wrappedImage]: ./reportPict/wrapped.PNG "Wrapped project image"
[markedImage]: ./reportPict/markedLines.PNG "marked project image"
[markedUndistImage]: ./reportPict/markedUndist.PNG "marked undistorted project image"
[projectVideo]: ./output_videos/project_video.mp4 "Project Video"
[challengeVideo]: ./output_videos/challenge_video.mp4 "challenge Video"


### Report

#### 1. This report describes the solution for [goals](https://review.udacity.com/#!/rubrics/571/view)  

Whole python code is located in [IPython](./code.ipynb). Executed version of project can be viewed [here](./code.html).

### Camera Calibration

#### 1. Camera matrix and distortion coefficients

First cell of IPython project uses cv2 method findChessboardCorners to detect all corners on distorted chess boards. 
Next cell uses cv2 method calibrateCamera to calculate camera distortion based on points detected in previous cell. Finally distortion is stored to 'camera_cal/wide_dist_pickle.p' so this calculations can be skipped when tuning image processing pipeline. 
Example chessboard before and after calibration 

![Example chessboard before and after calibration][chessboardImage]

### Pipeline (single images)

#### 1. Distortion correction

In third IPython cell distortion of input images are corrected using parameters stored in pickle file. 
Example of project image before and after distortion correction:

![Undistorted project image][undistortImage]

#### 2. Binary image creation

In forth IPython cell I tried several approaches to detect lines. first I tried to use combination of colour and gradient methods in convertToBinMix function, this method worked well for project video but it failed in challenge video because of very distinct surface colour change in the middle of the line.
Next I tried to detect yellow and white lines just by colours in convertToBinColors but this method was also impossible to apply on both challenge and project videos.
In the end I decided to use colour detection for find all greyish colours convertToBinNotGray which after some experiments started to work quite reliably on both project and challenge video.
Example of binary image:
 
![Binary project image][binaryImage]

#### 3. Perspective transform

In the fifth IPython cell Images are transformed to "bird eye view" using cv2 warpPerspective method I chose to hardcode the source and destination points in the following manner:

```python
offset = 300
src = np.float32([[580,460], [702,460], [233,700], [1075,700]]) # Project video
dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                 [offset, img_size[1]], 
                 [img_size[0]-offset, img_size[1]]])
```
For challenge video camera position seems to change so different source points were selected
```python
src = np.float32([[588,490], [740,490], [309,710], [1117,710]]) # Challenge video
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Wrapped project image][wrappedImage]

#### 4. Identifying line pixels and fitting polynomial

In sixth IPython cell binary images are processed to mark lines and detect their features with function markLines. 
First, lines pixels are identified with sliding windows (green rectangles) in for loop
```python
for window in range(nwindows):
```
Next, Marked points are go through sanity check pixelsSanityCheck and if are passing, fitted with quadratic polynomial using numpy polyfit method. This fits are verified in lineFitSanitycheck for correct position and distance and approximate perpendicularity. 

![marked project image][markedImage]

#### 5. Line radius and car position calculations

Apart from identifying lines function markLines also calculates radius using calculateRadius function. To calculate radius first polynomial coefficients are converted from pixels to meters based on dashed line length and traffic lane width. Next radius is calculated close to car hood (the bottom of the picture) using formula:
Rcurve=((1+(2Ay+B)^2)^3/2)/|2A|
The position of the car on the line is calculated as a deviation of the middle of the line from middle of the picture and converted to meters, This parameters together with detected lines are overlaid on input picture and returned.

Lines marked on distortion corrected image:

![marked undistorted project image][markedUndistImage]

---

### Pipeline (video)

#### 1. Video processing modifications

For video processing I decide to reuse functions from image processing pipeline. markLines function was modified to accept previous line position as an input. This significantly speeds up marking left and right line pixels and if passes sanity checks is overlaid on the output image. 
When no line passing sanity check is found, previously detected line is used.

Here is my project video: 

![project video][projectVideo]

Here is challenge video:

![challenge video][challengeVideo]


---

### Discussion

#### 1. problems / issues

The biggest problem is image binarization under different light conditions. It can be work around on challenge video by taking lane detected in previous frames but it is not possible for harder challenge video. Also radius calculation seems to be a bit noisy.
