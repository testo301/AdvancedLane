# **Finding Lane Lines on the Road** 

## Writeup Template

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

The code is placed in the AdvanceLane_Submission.ipynb file.

[//]: # (Image References)
[image1]: ./test_images_output/1CameraCalibrationIn.jpg "Camera Calibration Input"
[image2]: ./test_images_output/1CameraCalibrationOut.jpg "Camera Calibration Output"

[image3]: ./test_images_output/2CalibrationExIn.jpg "Undistortion Example Input"
[image4]: ./test_images_output/2CalibrationExOut.jpg "Undistortion Example Output"

[image5]: ./test_images_output/3AbsSobelIn.jpg "Absolute Sobel Example Input"
[image6]: ./test_images_output/3AbsSobelOut.jpg "Absolute Sobel Example Output"

[image7]: ./test_images_output/4DirSobelIn.jpg "Directional Sobel Example Input"
[image8]: ./test_images_output/4DirSobelOut.jpg "Directional Sobel Example Output"

[image9]: ./test_images_output/5MagSobelIn.jpg "Magnitude Sobel Example Input"
[image10]: ./test_images_output/5MagSobelOut.jpg "Magnitude Sobel Example Output"

[image11]: ./test_images_output/6HLSOut.jpg "HLS Thresholding Example Output"
[image12]: ./test_images_output/7combined.jpg "Combined Thresholding Example Output"

[image13]: ./test_images_output/8perspectiveIn.jpg "Perspective transform Example Input"
[image14]: ./test_images_output/8perspectiveOut.jpg "Perspective transform Example Output"

[image15]: ./test_images_output/9linePixelsOut.jpg "Identification of line pixels Example Output"
[image16]: ./test_images_output/10searchAroundPolyOut.jpg "Searching around the line Example Output"

[image17]: ./test_images_output/11InversePerspectiveOut.jpg "Inverse Perspective Overlay Example Output"

[image18]: ./test_images_output/12InitialPipelineIN.jpg "Initial Pipeline Example Input"
[image19]: ./test_images_output/12InitialPipelineOut.jpg "Initial Pipeline Example Output"

[image20]: ./test_images_output/13VideoPipelineIN.jpg "Video Pipeline Example Input"
[image21]: ./test_images_output/13VideoPipelineOut.jpg "Video Pipeline Example Output"




---

### Project Pipeline

### 1. Camera Calibration

Image processing starts with the camera calibration which is placed outside the pipeline as the one-off event.

Calibration is performed by processing all available chessboard images from the 'camera_cal' folder. The approach assumes fixed object points grid. The image points are detected using the findChessboardCorners() function. The function captures the corners on the distorted images and calibration is performed using the CV2 function calibrateCamera().

Chessboard image prior to correction:
![alt text][image1]

Chessboard image post correction:
![alt text][image2]

Based on the calibration outputs, the function undist(img, mtx, dist) is defined.

Sample image prior to undistortion:
![alt text][image3]

Sample image prior after undistortion:
![alt text][image4]

### 2. Thresholding

After undistortion, a sub-pipeline is defined comprising of Sobel and color filter. The pipelineA() gathers the steps together. The components of this pipeline are:

- Absolute Sobel thresholding
Sample image after absolute Sobel thresholding:
![alt text][image6]

- Directional Sobel thresholding
Sample image after absolute Sobel thresholding:
![alt text][image8]

- Magnitude Sobel thresholding
Sample image after absolute Sobel thresholding:
![alt text][image10]

- HLS color thresholding 
Sample image after the HLS color thresholding:
![alt text][image11]

The pipeline based on the Sobel and color thresholding is wrapped in the pipelineA() function. The function assumes the following logical operators to overlay the thresholding layers.

    # Gradient thresholding
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=sthresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=sthresh)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=sthresh)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=dirthresh)
    
    # COLOR thresholding
    hls_binary = hls_select(image, thresh=cthresh)
    
    # Combined thresholding
    combined[ ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))  | hls_binary == 1 ] = 1

The pipeline fucntion provides the following set of inputs for the respective kernels, absolute threshold, directional threshold and color threshold

    # Thresholding Pipeline inputs  
    pipelineA(image, ksize = 3, sthresh=(20, 100), dirthresh=(0.7, 1.0), cthresh=(90, 255) )

Sample image after the combined color and Sobel thresholding pipeline application:
![alt text][image12]

### 3. Perspective Transformation

The next step consists of the transformation of the perspective. The trasformation is simplistic and utilizes only two parameters.

The function transforms the image, taking the offset serving as cropping and perspective constant, along with scalar for correcting horizon line.

    def perspective(image, offset = 50, horizon_scalar = 1.25 )

Sample image after the combined color and Sobel thresholding pipeline application PRIOR to perspective transform:
![alt text][image13]

Sample image after the combined color and Sobel thresholding pipeline application AFTER the perspective transform:
![alt text][image14]

### 4. Line Detection

The first step of line detection is identification of line pixels using the histogram. The expectation is that the image (after proper perspective transform) would be characterized by bimodality that would indicate approximate left and right lane locations.

The function find_lane_pixels() identifies the initial location of lines and outlines the windows (9 windows defined by default). The margin for the window was defined by default as +/- 100.

The function fit_polynomial() leverages the identified pixels and fits the second order polynomial.

Sample image after the initial line detection with the fixed size windows:
![alt text][image15]

The next function search_around_poly() 

Sample image after the polynomial fitting within the margin around the previous line:
![alt text][image16]

The function polynomial_def() is an auxiliary function used later in the video pipeline for the extraction of the definition of the left and right polynomial. The function leverages the find_lane_pixels() function for initial search of line pixels.

### 4. Curvature Measurement

Two functions are provided for the curvature calculation. 
- measure_curvature_pixels()
    Outputs the left and right line curvature radius in pixels
- measure_curvature_real()
    Outputs the left and right line curvature radius in meters
    
The functions don't take any arguments and take directly the perspective transformed image as an input.

The outputs of the functions will be overlayed on the individual frames in the videos.

### 5. Position Measurement

The function center_position() is used to calculate the distance of the left and right lanes from the image center (in meters). The third output value is the distaince of the identified lane center from the image center. 

For readability the function takes no arguments other than perspective transformed image, but could directly take left and right lane fits for quicker processing if necessary.

### 6. Reverse Perspective Transformation

The function perspective_reverse() takes as an argument the original image and the perspective transformed image from the previous step.

The function leverages the perspective() function, the second output of which provides the inverse transform parameters, i.e. projecting from destination back to source defined in the perspective().

The output of the function is the color filled area overlayed on the original image, filling the space between the lanes.

Sample image after the inverse transform layer overlay:
![alt text][image19]

### 7. Initial Test Pipeline (before the class Line())

The function pipeline() combines the previously defined steps:
- Undistorting the image with the camera calibration parameters
- Applying Sobel and color thresholding
- Perspective transformation
- Reverse perspective transformation
- Curvature and position information layer on the output image/frame

Sample image before the initial pipeline application:
![alt text][image18]

Sample image after the initial pipeline application:
![alt text][image19]

### 7. Class Line()

The class Line() was re-used from the lesson 2. Tips and Trick for the Project.

    class Line():
        def __init__(self):
            # was the line detected in the last iteration?
            self.detected = False  
            # x values of the last n fits of the line
            self.recent_xfitted = [] 
            #average x values of the fitted line over the last n iterations
            self.bestx = None     
            #polynomial coefficients averaged over the last n iterations
            self.best_fit = None  
            #polynomial coefficients for the most recent fit
            self.current_fit = [np.array([False])]  
            #radius of curvature of the line in some units
            self.radius_of_curvature = None 
            #radius values over the last n iterations
            self.radius_hist = [np.array([False])] 
            #radius average over the last n iterations
            self.radius_best = None
            #distance in meters of vehicle center from the line
            self.line_base_pos = None 
            #difference in fit coefficients between last and new fits
            self.diffs = np.array([0,0,0], dtype='float') 
            #x values for detected line pixels
            self.allx = None  
            #y values for detected line pixels
            self.ally = None         
            
The key attributes defined for lane detection were
    - current_fit - the historical array of the 10 most recent polynomial coefficients
    - best_fit - the average of 10 last frames for the fitted coefficients of the second order polynomial
    - detected - the flag indicating whether the lines were properly detected in the last iteration.

The Line() class has one method defined. The method updates the key attributes by:
- updating the 'detected' state of the current fit
- managing the length of the historical lists current_fit, radius_hist
- computing the average for the best_fit and radius_best

       def update(self, line_fit, curverad_real,position):
        # Removal of the first empty current_fit and radius_hist values to avoid diluting the average
        if self.current_fit[0].any() == [False]:
            self.current_fit = self.current_fit[1:]
            
        if self.radius_hist[0].any() == [False]:
            #self.radius_hist = self.radius_hist[:len(self.radius_hist)-1]
            self.radius_hist = self.radius_hist[1:]
                
        # When the line is not detected in the current frame the detected flag is modified to False and historical average is taken
        if line_fit is None:
            self.detected = False
            self.best_fit = np.average(self.current_fit,axis=0)
            self.radius_best = np.average(self.radius_hist,axis=0)
        else:
            self.detected = True
            # Updating and averaging the fitted model parameters
            self.current_fit.append(line_fit)
            # Subsetting the current_fit list ONLY to the last 10 frames
            if len(self.current_fit) > 10:
                self.current_fit = self.current_fit[len(self.current_fit)-10:]
            self.best_fit =   np.average(self.current_fit,axis=0)  

            # Updating and averaging the curvature of the line in meters
            self.radius_hist.append(curverad_real)
            # Subsetting the current_fit list ONLY to the last 10 
            if len(self.radius_hist) > 10:
                self.radius_hist = self.radius_hist[len(self.radius_hist)-10:]
            #self.radius_best =   np.average(self.radius_hist,axis=0)  
            # Median for more stable results
            self.radius_best =   np.median(self.radius_hist,axis=0) 
            
            self.radius_of_curvature = curverad_real
            self.line_base_pos = position
            
            #TODO - update parametrow x z left_fitx
            #self.recent_xfitted.append(fitx)
                        
            if self.best_fit is not None:
                self.diffs  = abs(self.best_fit - line_fit)

The Line() instances will be initialized separately for right and left lanes:

        rlane = Line()
        llane = Line()

### 8. Video Pipeline

The introduction of class Line() provides additional features related to the sanity checks and parameter smoothing.

The function perspective_reverse_video(image, warped,left_fit, right_fit) takes the previous provided polynomial model specification and only then perspective transforms the filled between lane area back into the original perspective.

The function last_fit_search(binary_warped, left_fit, right_fit) provides a way to search around the previously identified lines within a given margin. The function is modified comparing to the initial pipeline to directly take the polynomial definitions.

The function video_pipeline() combines the following step to produce the final processed image/frame output.
- Undistorting the image with the camera calibration parameters
- Applying Sobel and color thresholding
- Perspective transformation
- Performing sanity checks on the previous best fits:
    - are polynomial parameters responsible for direction not diverging (e.g. defaulted by max 50%)
    - is the curvature of the lines in meters represented as the radius not diferging (e.g. defaulted by max 50%)
- if tests are satisfied and the previous line set was detected:
    - performing search around previous polynomial with the last_fit_search()
- if previous lines were not detected or sanity checks failed:
    - define new polynomials based on newly identified lane points
- updating the Line() class attributes
    - passing the new attributes via update() method
    - smoothing out the fit and curvature parameters within the update() method
- Reverse perspective transformation
- Curvature and position information layer on the output image/frame

Sample image before the initial pipeline application:
![alt text][image20]

Sample image after the initial pipeline application:
![alt text][image21]

The video output is located in the \test_videos_output folder

### 9. Sanity Checks

The video_pipeline() contains a set of sanity check to make sure that: 
    - the polynomial parameters responsible for direction not diverging (e.g. defaulted by max 50%)
    - the curvature of the lines in meters represented as the radius not diferging (e.g. defaulted by max 50%)
    
If any of the checks are breached, the detected flag is set to False and information is displayed on the text layer of the frame.

### 10. Identify potential shortcomings with your current pipeline

The main shortcomings are:
- Arbitrary nature of some of the parameters:
    1. Margin of search around the polynomial
    2. Horizon scalar 
    3. Boundaries of the source and destination points for the perspective transform
    4. Parameters for the Sobel and color thresholding   
    5. Sanity check thresholds
    6. Length of the historical window for fit / curvature smoothing via average
- No control for the width of the lane. The challenge video proved that the more pronounced concrete barrier became a 'better' left lane and the solution stabilized on the wider lane than expected.
- Slowness of the solution application on the frames (tested on older PC). There is one point of redundancy for code readability when the polonomial fit is re-applied within another function.
- Image/frame highlights and shaded areas in the challenge videos exposed limitations of the current approach in providing the stable solution

### 11. Suggest possible improvements to your pipeline

Potential improvements involve:
- Introducing smart selection of the arbitrary parameters (e.g. cross-validation, tuning dependent on the neighbors, etc.)
- Adding more pronounced color filters to detect specific lines and separating them from passing vehicles (e.g. on the harder_challenge_video and a better yellow lane detection)
- Capturing the expected width of the lane and forcing it as a constraint for model fitting
- Optimizing the code in terms of minimizing number of necessary transformations
- Lengthening the historical window for smoothing to stabilize the solution, but at the same time capture the most current incoming information
- Identification of instantaneous road bumps where the currently captured lines diverge from the historical averages (e.g. parallel shift).