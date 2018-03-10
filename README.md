## Report

---

**Vehicle Finding Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[hogCarImage]: ./output_images/hogCar.PNG "Hog Car"
[hogNonCarImage]: ./output_images/hogNotCar.PNG "Hog non car"
[fixedWindowsImage]: ./output_images/fixedWindows.PNG "Fixed Windows"
[variableWindowsImage]: ./output_images/variableWindows.PNG "Variable window vehicle"
[variableWindowsHeatMapImage]: ./output_images/variableWindowsHeatMap.PNG "Variable window heat map"
[finalVideo]: ./output_videos/final_projct_video.mp4 "Final video"
### Report

#### 1. This report describes the solution for [goals](https://review.udacity.com/#!/rubrics/513/view)

Whole python code is located in [IPython](./code.ipynb). Executed version of project can be viewed [here](./code.html).

#### 2. Camera calibration
First and second cell of IPython was copied from Advance Line detection and is used to correct the camera distortions.

#### 3. Line detection pipeline
Third cell of IPython was also copied from Advance Line detection and consist of full line detection pipeline.

#### 4. Getting optimal parameters of HOG
In 4th cell of IPython optimal HOG parameters for dataset are determined. I decided to use only grey scale and checked all the combinations of following parameters:
HOG orientations [6,8,10]
HOG pixels per cell [8,10,16]
HOG cells per block [1,2,3]
C for linear SVM = [1,3,6,10,20]
The best parameters are:
Best orient: 10
Best pix_per_cell: 10
Best cell_per_block: 3
Best C param: 1

This parameters are showed on one Vehicle picture and one non-Vehicle picture in 5th IPython cell

![Hog Car][hogCarImage]

![Hog non car][hogNonCarImage]
#### 5. Train classifier
in 6th IPython cell histogram and spacial features are added to best HOG parameters and classifier is trained the resulting accuracy is 0.9598.

#### 6. Classification with fixed size windows

In 8th cell classier is tested with fixed 100x100(400,680) pixel window and it does quite good job although there are quire a few false positives. 
Example of fixed window vehicle detection:

![Fixed window vehicle][fixedWindowsImage]

#### 7. Classification with variable size windows

In 9th cell classier is tested with variable windows of scale 1.2(390,490), 1.4(450,590), 1.7(560,680). It looks much better than fixed windows. 
Example of variable window vehicle detection:

![Variable window vehicle][variableWindowsImage]

#### 8. Heat map

In 10th IPython cell I convert detections from variable windows to heat map. to avoid too many false positives threshold is set to 2.
Example of variable window heat map:

![Variable window Heat Map vehicle][variableWindowsHeatMapImage]

#### 9. Video pipeline

Finally the same pipeline is applied to the video together with line detection. To reduce the number of false positives heat map is summed over 40 frames and threshold is applied this value.
Next only labels which were detected over 40 frames ale displayed. Here is video with lines and vehicles detected: 

![Final video][finalVideo]


---

### Discussion

#### 1. problems / issues

The biggest problem are false positives. It was greatly improved by using information from previous frames. Solution deep learning algorithm could be implemented.
