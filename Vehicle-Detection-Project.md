
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
[image6]: ./examples/six-frames-heatmap.jpg
[image7]: ./examples/sum-of-six-frames-heatmap.jpg
[video1]: ./test.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

All the code for this project has been obtained from the example code in the course and also from watching Ryan Keene's video referred to in it. The resulting code is in this python notebook:
[Here](https://github.com/gvp-study/CarND-Vehicle-Detection/blob/master/Vehicle-Detection.ipynb)
---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step starts from the third code cell of the IPython notebook.  

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

The first row of the figure below shows the result of using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The second row shows the first channel of the `(32x32)` resized image which bins the red color. The third row shows the `Y` channel of a `YCrCb` representation of the image. The fourth row shows the corresponding images related to `Cr` channel of the images.


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of histogram bins from 8, 16, 32 and 64. The HOG images that it produced  seemed to be differentiate cars better at 32.
I chose the number of orientation bins to be 9 as suggested by the course.
I kept the pixels_per_cell at 8 and the cells_per_block to be 2 as suggested by the course.
I experimented with RGB, HSV and YCrCb spaces and found that the YCrCb color space is more robust to downsampling and manipulation by definition.
It also seemed that using 'ALL' the channels for HOG transform instead of choosing just one of the many would help with the classification by not throwing out valuable gradient data from the remaining two channels.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the sixth code cell in the notebook to do this. I used all three feature lists (HOG features, color histogram and spatial features) for classifying the image windows.

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

I tried to run the whole set of training data on my IMac and it failed due to lack of memory or CPU. So I was forced to use only a set of 4000 samples from the 8000+ training data. This might affect the final results.
Despite only using 4000 samples, it took around 98 seconds to train with 99% accuracy as shown below.

97.94166016578674 Seconds to compute features...
Using: 9 orientations 8 pixels per cell 2 cell per block 32 histogram bins, and (32, 32) spatial sampling
Feature vector length 8460
17.61 Seconds to train SVC...
Test Accuracy of the SVC =  0.9938

I used the default values for all the parameters for classifier when using the LinearSVC() function.

### Sliding Window Search

#### 4. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


From the FAQ video, I confirmed that most cars seem to lie in a 256 pixel strip between the 400 and 656 rows. I restricted the windows to remain within these bounds while sliding. I also tested scales of 1, 1.5 and 2.0 and settled on 1.5.
I tried a search window size of ``(64x64)`` and ``(32x32)`` which increased the number of search windows and the time taken to process the image. I finally settled on a search window size of ``(96x96)`` which seemed to detect the cars better. The larger size windows also saved on the processing time considerably.

![alt text][image3]

#### 5. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two values of scale (1.5 and 2.0) using `YCrCb` 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. I settled on a scale of 1.5 which provided a nice result shown in the test images shown below.

The code to do this lies in the eighth code cell in the notebook. The function that generates a list of sliding windows over the 256 high strip in every image is called slide_window. This function takes in the image and the bounds of the search and the search window size (96x96) and an overlap of 50%.

The key function that predicts the cars in the image is using the classifier is called search_windows. This function takes as input the image and the list of windows to check for cars. It also takes all the parameters that were found to be best for the classifying from the earlier tests.

I tried a search window size of (64x64) and (32x32) which increased the number of search windows and the time taken to process the image. I finally settled on a search window size of (96x96) which seemed to detect the cars better.

The results are shown in the figure below. The green windows are all (100 in number) the windows being searched and the thick blue windows show the windows with cars detected in them. Note that the search correctly finds all but one of the cars in the six test images. It misses the white car in the first image.


![alt text][image4]
---

#### 6. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  The threshold is the key factor in filtering out the false positives in the frames. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the thresholded heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to bound the area of each blob detected. The result for the test images is shown below.
![alt text][image5]
When dealing with a continuous stream of images from a video, I wanted to use a global variable for the heatmap 'sum_heat' to hold on to the car detections from the previous frames. I could then threshold the summed heatmap and label the resulting 'sum_heat'. This could eliminate false positives. I was unable to implement this for the videos for now.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames. The resulting bounding boxes drawn on the last frame is also shown.
Note that the difference in the number and position of the windows which detected cars in the frames. In these cases it would be best to integrate the heatmap over several frames before labeling and then bounding the resulting summed labeled image. The result is as shown below. Note that the resulting heatmap and bounding box envelopes the white car well.

![alt text][image7]


### Video Implementation

#### 7. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The processing of the video is accomplished using the pipeline function called process_image. This function finds the cars in a given image and then labels the resulting heatmap and then draws distinct bounding boxes around the segmented regions in the labeled heatmap image. The process_image is then fed to the video class member function fl_image and the output is then saved to test.mp4.
The main 3 functions I use in the process_image are find_cars_video, apply_threshold, label and draw_labeled_bboxes as shown below. The apply_threshold with a threshold of 5 eliminates some false positives.

One other thing I did to reduce false positives was to keep an array of six heatmaps. These heatmaps are filled cyclically from the frames and hold a history of the past five frames. The sum_heat heatmap sums these six heatmaps and then threshold them at a high threshold of 5. This eliminates most of the false positives. I also noted that the false positive that show up at around second 40+ in the project_video.mp4 is due to the fact that the classifier sees a truck in the opposite lane seen past the left guard rail.

```python
# Hold the heat map from 6 frames.
sum_heat = np.zeros_like(last_img[:,:,0]).astype(np.float)
heat_array = [np.copy(sum_heat), np.copy(sum_heat), np.copy(sum_heat), np.copy(sum_heat), np.copy(sum_heat),np.copy(sum_heat) ]
img_count = 0
def process_image(img):
    global sum_heat
    global img_count
    global heat_array
    out_img, heat_map, img_boxes = find_cars_video(img, scale)
    heat_array[img_count % 6] = heat_map
    sum_heat = np.zeros_like(last_img[:,:,0]).astype(np.float)
    for i in range(6):
        sum_heat = sum_heat + heat_array[i]
    # Threshold by at least 3
    sum_heat = apply_threshold(sum_heat, 5)
    labels = label(sum_heat)
    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    img_count += 1
    return draw_img

```

Here's a [link to my video result](./test.mp4)


---

### Discussion

#### 8. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I was unable to implement the enhancements suggested by Ryan Keene. I hope to complete this project in the future if time permits.
The main disadvantage of the simplistic approach I have used is that there is only a simple averaging of the heatmap of the vehicle detected from frame to frame. This will result in a noisy detection and will prevent the tracking of detected cars when they get occluded by another vehicle that passes in between the camera and the original tracked vehicle. This can be improved by tracking the actual bounding boxes between frames and tracking its velocity in pixel space.
