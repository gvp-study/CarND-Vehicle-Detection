
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/8-car-noncar.jpg
[image2]: ./examples/car-noncar-features.jpg
[image3]: ./examples/slide-search-windows.jpg
[image4]: ./examples/car-detect-slide-windows.jpg
[image5]: ./examples/car-bboxes-heatmap.jpg
[image6]: ./examples/car-labeled-heatmap.jpg
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

All the code for this project has been obtained from the example code in the course and also from watching Ryan Keene's video referred to in it. The resulting code is in this python notebook: [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/Vehicle-Detection.ipynb)
---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of a random set of eight  `vehicle` and eight `non-vehicle` classes is shown below. I noted the variety of both the cars and noncars in the images. Obviously, the examples differ in color, lighting, shape and where they appear in the image. I also noted that all these images were 64x64 color images.

![alt text][image1]

I then collected all the functions related to this project in the fourth cell for later use. Some of these functions are:
* get_hog_features: Returns the histogram of gradients for a given image and given parameters
* bin_spatial: Returns an unravelled array of the given image after it is resized.
* color_hist: Returns the stacked color histograms of all the channels of a given image.
* extract_features: Extracts all the three feature vectors (spatial, color and HOG) for every image in the given image list using the previous functions.
* slide_window: Finds a list of sub windows that result from sliding on the image in both rows and columns.
* draw_boxes: Draws the given bounding boxes into the given image.
* single_img_features: This function takes a given image and extracts all the three feature vectors (spatial, color and HOG) using the previous functions.
* search_windows: Takes an image and a set of windows to classify them as having a car on noncar. It uses the single_img_features and a trained sklearn.classifier to do this.
* visualize: Plotting function for multiple images and their titles.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

The first row of the figure below shows the result of using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The second row shows the first channel of the (32x32) resized image which bins the red color. The third row shows the Y channel of a YCrCb representation of the image. The fourth row shows the corresponding images related to Cr channel of the images.


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of histogram bins from 8, 16, 32 and 64. The HOG images that it produced  seemed to be differentiate cars better at 32.
I chose the number of orientation bins to be 9 as suggested by the course.
I kept the pixels_per_cell at 8 and the cells_per_block to be 2 as suggested by the course.
I experimented with RGB, HSV and YCrCb spaces and found that the YCrCb color space is more robust to downsampling and manipulation by definition.
It also seemed that using 'ALL' the channels for HOG transform instead of choosing just one of the many would help with the classification by not throwing out valuable gradient data from the remaining two channels.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the sixth code cell in the notebook to do this.

I trained a linear SVM using the following parameters.
* colorspace = 'YCrCb'
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL'
* spatial_size = (32, 32)
* hist_bins = 32
* spatial_feat = True
* hist_feat = True
* hog_feat = True

For 4000 samples
134.80552291870117 Seconds to compute features...
Using: 9 orientations 8 pixels per cell 2 cell per block 32 histogram bins, and (32, 32) spatial sampling
Feature vector length 8460
14.71 Seconds to train SVC...
Test Accuracy of the SVC =  0.9875

I used the default values for all the parameters for classifier using the LinearSVC() function.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

From the FAQ video, I confirmed that most cars seem to lie in a 256 pixel strip between the 400 and 656 rows. I restricted the windows to remain within these bounds while sliding. I also tested scales of 1, 1.5 and 2.0 and settled on 1.5.
I tried a search window size of (64x64) and (32x32) which increased the number of search windows and the time taken to process the image. I finally settled on a search window size of (96x96) which seemed to detect the cars better.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
he code to do this lies in the eighth code cell in the notebook. The function that generates a list of sliding windows over the 256 high strip in every image is called slide_window. This function takes in the image and the bounds of the search and the search window size (96x96) and an overlap of 50%. The actual function that predicts the cars in the image is using the classifier is called search_windows. This function takes as input the image and the list of windows to check for cars. It also takes all the parameters that were found to be best for the classifying from the earlier tests.
I tried a search window size of (64x64) and (32x32) which increased the number of search windows and the time taken to process the image. I finally settled on a search window size of (96x96) which seemed to detect the cars better.
The results are shown in the figure below. The green windows are all (100 in number) the windows being searched and the thick blue windows show the windows with cars detected in them. Note that the search correctly finds all but one of the cars in the six test images. It misses the white car in the first image.


![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
