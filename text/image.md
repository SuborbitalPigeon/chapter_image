---
author: Bruce Cowan
bibliography: text/refs.bib
csl: ieee.csl
documentclass: scrbook
geometry:
- top: 2cm
- left: 4cm
- bottom: 4cm
- right: 2.5cm
lang: en-GB
linestretch: 1.5
papersize: a4
title: Defect recognition using image processing of C-scan images
---

# Introduction

*Image processing* algorithms are a group of methods which operate on images to modify them, or to extract information from them.
When a photograph is taken with a standard phone or camera, many of these processes are applied to the raw image from the sensor.
Operations range from simple ones such as rescaling or rotation, up to complex ones such as those which operate in the frequency domain.

In the domain of defect recognition, the main task to be carried out is to separate out regions of the image which indicate defects, and areas that are not defects.
This is then a problem of image *segmentation*.

![An example ultrasound image](images/stepped.png){#fig:stepped}

For example, see the image of a reference sample part in [@fig:stepped].
This sample has thirty deliberately-introduced defects, which have been inserted at known locations and depths.
The part is made up of five sections of differing thicknesses.
By visual inspection, it is quite easy to see most of the defect locations based on the contrast difference between the background and the small mostly circular regions.

The question to be answered by the end of this chapter becomes 'is it possible for an algorithm to segment these defect regions from the background?'

## Digital images

<!-- Forget about time-varying stuff here? -->

A *digital image* can be represented as a matrix of values.
This matrix can be 2D in the case of a greyscale image, 3D in the case of a multi-channel image or a set of greyscale images; or 4D for a set of multi-channel images.
Sets of images are often used to represent a variable time.

A *greyscale* image represents the values of a single variable (often lightness) in a spatial scene.
Multi-channel images are a set of greyscale images stacked together, a common example would be making use of separate colour channels for colour images.
For computer display purposes, these are commonly stored as red, green and blue values.
However, the commonly-used JPEG image storage format actually stores the image with channels which represent brightness (Y), blueness (Cb) and redness (Cr), this colour space being known as *YCbCr*.
This is to take advantage of the fact that the human vision system is more sensitive to changes in brightness than in colour.
The colour channels are often stored at a lower resolution than the luma, and this can bring a 50 % savings in storage space of the image, before the main part of the JPEG compression algorithm is applied.

Ultrasound data represents a single variable, and therefore images derived from it are single-channel (greyscale).
, many of these images in this chapter have been assigned colourmaps.
These mappings between pixel value and colour are *perceptually uniform*, which means that there is a linear relation between underlying value and pixel lightness.
This removes the biasses inherent in colourmaps which are often used for ultrasound visualisation^[For example, the infamous 'rainbow'], which can affect the effective contrast.
It would be possible to just use greyscale, however the human vision can determine only about 30 shades of grey, but millions of colours [@nunez_optimizing_2018].

## Pixel connectivity

The *connectivity* of a pixel describes its relationship with its neighbouring pixels.
For a 2D image, it is possible to define *4-connectivity* and *8-connectivity*, with the numbers indicating how many of the neighbouring pixels are considered to be connected to the central one.
More complex structures can be implemented for higher dimension data such as stacks of images (3D).

These different types of connectivity can be demonstrated as:
$$
\begin{align}
C_4 = \begin{matrix}
  0 & 1 & 0 \\
  1 & X & 1 \\
  0 & 1 & 0 \\
\end{matrix}
&
C_8 = \begin{matrix}
  1 & 1 & 1 \\
  1 & X & 1 \\
  1 & 1 & 1 \\
\end{matrix}
\end{align}
$$
where $X$ indicates the central pixel of the region.

This is an important concept to understand for several image processing algorithms, particularly *region filling*.

## Histograms

<!-- Put an image here showing original image, underexposed, overexposed; then their histograms underneath. -->

A *histogram* is a representation of the distribution of pixel values in an image.
It is common to show the lightness channel of an image in order to determine how well exposed the image is.

See [@fig:histogram] for a demonstration of an image's histogram^[This figure has yet to be created].
Also shown are underexposed and overexposed images and their respective histograms.
It can be seen that the histogram is moved to the left-hand side for the underexposed image, and to the right for the overexposed.

An ideally-exposed image would have an equal number of pixels for each value, and this would result in a flat histogram.
Flattening the histogram of an image can be achieved with histogram equalisation, and this is described in [@sec:histographequalisation].
However, for photographic purposes, this often causes displeasing results.

##Ultrasonic testing

By making use of probes containing *piezoelectric* materials, it is possible to inject ultrasound into a medium.
Piezoelectric materials deform when a voltage is applied across them, and a voltage appears across them when they are deformed.

![A diagram of an ultrasonic transducer](An ultrasonic transducer, source: [@echocardiographer.org_transducer_nodate](images/transducer.png){#fig:transducer}

An example transducer is shown in [@fig:transducer].
The piezoelectric crystal is excited by an AC voltage, which causes it to vibrate at the source's frequency.
In order to get the largest oscillations, it is necessary to use a frequency which is as close as possible to the natural frequency of the crystal.
This is dependent on the dimensions of the crystal.
The crystal is mounted on a backing material, which is used to reduce the oscillations of the crystal once the pulse has been sent, to reduce *ringing effects*.

### Representations of ultrasonic data

There are three main representations of ultrasonic scans:

A-scan
  ~ Can be represented as a graph of amplitude of echo against time.
  
B-scan
  ~ Displays ultrasonic response along a line, probe moves in one axis.
    Can be visualised as A-scans put on their side.
    
C-scan
  ~ Displays an ultrasonic metric as a function of 2D position, where the probe or beam moves in two axes.
    Various metrics can be used, such as thickness or maximum amplitude.

Using a single element probe, mechanical movement of the probe is required to acquire B and C-scans, namely 1D movement for B and 2D movement for C-scans.
With phased array probes, which contain multiple elements, it is possible to acquire a B-scan without moving the probe, and a C-scan by moving in one axis.

![An example A-scan, taken from an oscilloscope](images/ascan.png){#fig:ascan}

An ultrasound signal can be represented in the time domain in an *A-scan*, as shown in [@fig:ascan].
A-scans show the amplitude of the ultrasound response with respect to the time since the pulse was sent.
By using the speed of propagation in the material, it is possible to convert time to depth.
This means that the A-scan can also be seen as echo amplitude vs. depth.

To simplify processing, the ultrasonic propagation speed is commonly assumed to be a constant value.
For isotropic materials, this is a reasonable assumption.
However, due to the significant anisotropy of composite materials, this can become an issue.
As long as the ultrasonic wave is propagating normally through the lamina layers, the speed should be relatively constant and independent of the ply layup sequence.

A *C-scan* is a top down view of ultrasound data, where the axes of the image represent a 2D surface of a part.
It is therefore necessary for a probe to be moved in two dimensions on the part, or for an array to be used.
The image shown in [@fig:stepped] is a C-scan of a sample composite part which contains deliberate defects.

Each pixel in the image represents a single A-scan location.
This requires a decision in which value to assign to the pixel, given that each A-scan is made of several time domain samples.
One approach is to find the Time of Flight (ToF), which is the amount of time taken for the wave to propagate from the front surface of the part to the most prominent echo inside it, or the back surface, depending on the relative amplitudes.
This can then be visualised as a depth map making use of the speed of propagation.
For a part with no internal defects, this would correspond to a thickness map.
However, when defects are introduced, the echoes from them mask the echo from the back surface, and this has the effect of appearing to change the apparent depth of the part.

## Image processing

*Image processing* methods are algorithms which operate on matrix-based image data.
Some of these operations are:

* Simple operations such as scaling, rotation and general warping
* Histogram manipulations (equalisation, tone curves)
* Colour processing (changing hue, saturation)
* Edge detection
* Feature detection
* Denoising

Many of these will be run when taking a photograph with a standard camera or a phone, without any intervention from the user.

The specifics of several algorithms are discussed in [@sec:imageprocessing].

# Image processing techniques {#sec:imageprocessing}

This section contains details of different image processing techniques.

## Image transforms

In this section, *general image transforms* refer to operations which affect the shape, position and dimensions of the image.
This includes the following *affine transformations*:

* Translation
* Scaling
* Rotation
* Shear

Move generalised image warping can be performed using a *projective transform*, whereby the corners of the image can be moved to any position in the $xy$ plane.
These operations can be performed by the application of $3 \times 3$ transformation matrices.
By making use of such matrices, it is possible to chain together several operations which can be applied by using a single matrix multiplication.

The coordinates of the output image can be found by multiplying the input coordinates by the transformation matrix.
The input coordinates must be converted to *homogeneous coordinates* for the multiplication to be possible (i.e. $[x, y, 1]^T$).

### Translation

A *translation* moves the origin point of the image.
It can be represented by the following matrix [@vanderwalt_scikitimage_2014]:

$$
T =
\begin{bmatrix}
  1 & 0 & t_x \\
  0 & 1 & t_y \\
  0 & 0 & 1 \\
\end{bmatrix}
$$
where $t_x$ and $t_y$ are the movements in the $x$ and $y$ axes respectively.

### Scaling

*Scaling* refers to resizing the image matrix in order to change the image's resolution.
This can be represented by the following matrix:

$$
T =
\begin{bmatrix}
  s_x & 0   & 0 \\
  0   & s_y & 0 \\
  0   & 0   & 1 \\
\end{bmatrix}
$$
where $s_x$ and $s_y$ are the scaling factors (values $\gt 1$ denote scaling up) in the $x$ and $y$ directions respectively.

It is common in image processing to reduce the size of images in order to speed up more complex operations later in the processing pipeline.
In the early days of digital photography, it was quite common for digital cameras to *upscale* their output images in order to achieve more marketable *megapixel* counts.
However, trying to add information which is not in the original image actually cause the effective resolution to reduce.^[This brings to mind the quote 'Prediction is very difficult, especially if it's about the future'[@anker_forecasting_2017], which is attributed to Neils Bohr]

### Rotation

*Rotation* of an image can be represented by the following transformation matrix [@vanderwalt_scikitimage_2014]:

$$
T =
\begin{bmatrix}
  \cos \theta & -\sin \theta & 1 \\
  \sin \theta &  \cos \theta & 0 \\
  0           & 0            & 1 \\
\end{bmatrix}
$$
where $\theta$ is the anti-clockwise rotation angle referred to the positive $x$ axis (in radians).

<!-- It is common in photography to convert between portrait and landscape orientation of photographs^[Also for the far too often required conversion of vertically phone-captured videos to landscape for display on non-phone devices], this requires a positive or negative rotation of 90°. -->

It should be noted that this matrix refers to a rotation where the pivot point is the origin point of the image (top left).
In order to use another pivot point, a translation procedure must also be performed.

### Shear

*Shear* refers to changing the shape of the image along the $x$ or $y$ axes.
It can be represented by the following matrix:

<!-- Use phi for this -->

$$
T =
\begin{bmatrix}
  1 & \sin \phi & 1 \\
  0 & \cos \phi & 0 \\
  0 & 0         & 1 \\
\end{bmatrix}
$$
where $\phi$ is the shear angle.

### Combined transforms

As has been mentioned previously, it is possible to combine all the above operations into a single matrix transform, known as an *affine transform*.
By chaining these simple operations together, the combined transform matrix can be [@vanderwalt_scikitimage_2014]:

$$
T = 
\begin{bmatrix}
  s_x \cos \theta & -s_y \sin (\theta + \phi) & t_x \\
  s_x \sin \theta &  s_y \cos (\theta + \phi) & t_y \\
  0               & 0                          & 1   \\
\end{bmatrix}
$$
where the symbols have the same meanings as defined previously.

It is also possible to perform more general operations, such as a *projective transform*, which allows the possibility of freely moving the corners of the image in $xy$ space.
A projective transform matrix can be constructed from that of an affine transform.

![An example affine transform](images/affine.png){#fig:affine}

The image shown in [@fig:affine] shows an example of an affine transform, which includes:

* Scale factor of 0.7 in $x$, and 0.6 in $y$.
* Rotation of 15°.
* Shear of -30°.
* Translation of 50 pixels right and 300 pixels down.

This results in the following transformation matrix:

$$
T = 
\begin{bmatrix}
  \frac{1.3 \sqrt{3}}{2} & 0   & -50  \\
  1.5 \sin \theta        & 1.5 & -300 \\
  0                      & 0   & 1    \\
\end{bmatrix}
$$

### Estimating transforms

Provided a set of input coordinates, and their desired output positions, it is possible to estimate the transformation required.
This can be done by using methods such as least-squares or RANSAC (RANdom SAmple Consensus).
RANSAC is less affected by outliers when compared with least-squares.

By making use of frames from a video stream (where each frame is an image), it is possible to make use of generalised image transforms to perform video stabilisation or object tracking [@cowan_performance_2016].
Finding points in each frame using a point detection algorithm, and by matching the corresponding ones, a transform can be estimated.
Point correspondence can be estimated using a description of local area around each point.
See [#sec:pointdetection] for more details on point detection.

#### RANSAC

*Random Sample Consensus (RANSAC)* is a method similar to a least-squares fit, however it is less susceptible to outliers [@fischler_random_1981].
It operates by choosing a random subset of the total samples, and producing a fit model using only these samples.
The number of inliers using this model is established, using a preset distance threshold.
Other random samples are then selected, and this process is repeated until the number of inliers to the model reaches a threshold.
The parameters of the model with the highest number of inliers is chosen as the final model.

### Filtering and boundary modes

The pixels which make up an image are located at integer coordinates.
By applying geometric transforms, it is highly likely the new locations of pixels would ideally be at floating-point coordinates, which is not possible due to the matrix representations of images, so a *filtering* algorithm must be applied.

The simplest technique is to assign each output pixel to the *nearest* floating-point position.
This can result in significant artefacts (for example, severe aliasing for small angle rotations), and so more complex methods such as *linear* or *cubic* filtering can be used, which interpolate between the floating-point coordinate values to generate the output pixels.

<!-- Does this bit belong here? -->

Image processing techniques which make use of matrices have a problem when being run on pixels at the edges of an image, due to the fact that not all the neighbourhood pixels exist.
To deal with this problem, the image is temporarily resized by a few pixels.
The values of the newly-created pixels can be set in different ways, for example choosing a constant value, or reflecting the pixels adjacent to them in the original image.

## Histogram manipulation

An image's *histogram* is a representation of the frequency of pixel values in the image.

In an ideal image, the full range of possible pixel values would appear, and would have equal frequency.

### Tone curves

A function can be applied which maps between input pixel value and output pixel value.
These functions could be simple linear operations in order to change the exposure value of an image, or any other general function.
*S-shaped* curves are often used to increase the contrast in the mid-tones of an image, which can result in more pleasing photographs for example.
Normally, curves are only applied to the lightness channel of colour images, as manipulating the hues is usually not required.

### Histogram equalisation {#sec:histographequalisation}

![Original image, and after the application of global and adaptive equalisation](images/equalise.png){#fig:equalise}

*Histogram equalisation* is a process whereby the values are modified to create as close to a flat histogram as possible.
This is achieved by increasing the distance between values assigned to the most common input values.
This is a way of improving the contrast in an image, and is useful for image processing for scientific purposes.
It however leads to poor quality results for photographs.

See [@fig:equalise] for an example of an ultrasound C-scan before and after equalisation, both global and adaptive.
In this case, the block size for the adaptive equalisation is such that the image is split into 8 × 8 blocks.
It can be seen that the contrast between the background, and the defects is higher, especially in the case of the image which has undergone local equalisation.

![Histograms of image, and after global and adaptive equalisation](images/equalise_hist.png){#fig:equalise_hist}

See [@fig:equalise_hist] for histograms of the original image, and after the application of global and adaptive equalisation.
It can be seen that the pixel values in the original images are predominantly that of the five background steps of the sample, as can be expected.
After a global equalisation process has been carried out, it can be seen that the peaks have been 'spread out' across the full range of possible values.
Adaptive equalisation preserves the values of the peaks, but they have been spread out in value.
This has increased the contrast in the non-defect regions.

## Denoising

*Noise* in an image manifests itself as random fluctuations in the values of pixels when compared to their neighbours.
It is a by-product of several physical phenomena, such as electrically generated noise, pixel leakage^[Find a source for this].

In digital photography, noise is a particular problem with images taken with high ISO values (higher amplifier gains), which can be required in low-light situations.
The pixel values are amplified according to the camera's ISO setting, which has the by-product of reducing the effective dynamic range of the camera.

![A noisy image](images/noisy.png){#fig:noisy}

An example of an image with noise is shown in [@fig:noisy].
Random colour noise can be seen in this image, as a result of the high gain which was used when taking this photograph, due to the dark conditions.

In ultrasound C-scans, noise appears as a result of the gain applied to the underlying A-scans.
Electrical noise present in the acquisition hardware will also have an effect.

Noise can be reduced using several methods.
Some of these are described in the following sections.

### Median filtering

For this technique, a simple statistical test is applied to the neighbourhood of each pixel in the input image.
The value that each pixel is set to in the output image is the median value of this pixel's neighbourhood.
This therefore rejects random high and low values in each neighbourhood.
By making the region larger, it is possible to increase the strength of the smoothing effect.

The image must first be padded to remove the problem of not having enough pixels to form neighbourhoods at the edges of the image.

This is a very simple operation, and performs relatively poorly as it has a strong blurring effect on the image.
However, it is computationally cheap to perform.

### Blurring

A *blur* operation is the equivalent to a low-pass filter in signal processing.
Each pixel in the output image becomes a weighted average of the surrounding pixels.

A common blurring algorithm is the *Gaussian blur*, which uses an approximation of a 2D Gaussian distribution to create a weighted average.
Such matrices are digital approximations to the function $G(x, y) = \frac{1}{2 \pi \sigma^2} \exp{-\frac{x^2 + y^2}{2 \sigma^2}}$, where $\sigma$ sets the strength of the blur.
The size of the matrix is dependent upon $\sigma$, in order to achieve an acceptable approximation.
It is possible to convert this 2D convolution into the application of two 1D convolutions, which speeds up the implementation.

### Non-local means denoising

The *non-local means* method also makes use of pixel regions.
However, instead of considering the whole region for the averaging operation, it also takes into account the neighbourhood of these pixels.
If the region is similar to the original pixel's region, then this pixel will be considered in the averaging operation.
By doing this, the texture content of the image is retained [@buades_nonlocal_2005].

### Wavelet denoising

*Wavelet transforms* are a type of frequency domain transformation, similar in purpose to the Fourier Transform or Discrete Cosine Transform (DCT).
They have particular advantages in the field of image processing, due to the fact they encode not only frequency data, but also time dependence [@gonzalez_digital_2002].

By making use of wavelet decomposition (see [@sec:waveletdecompose] for further detail), it is possible to convert an image into a *pyramid* of separate images which contain different levels of detail.
Noise is generally a high-frequency variation between individual pixels, and this means that it will be most prominent in the most detailed level of an image pyramid.
By running a smoothing algorithm (for example, one of the above or just reducing the amplitudes) on only the most detailed level(s), it is possible to reduce the noise of the image without having an effect on the general structure.

### Comparison

![Comparison of denoising methods](images/denoise_all.png){#fig:denoise_all}

Contained within [@fig:denoise_all] is a comparison of the different denoising methods when applied to the image of [@fig:noisy].
Displayed alongside these are an indication of processor time taken to compute.
The tuning parameters of each method have been set to achieve a similar level of denoising to each other, for the sake of visual comparison.

It can be seen that the simple methods

## Thresholding {#sec:thresholding}

*Thresholding* is a process which can be used to convert a full-scale image to a binary image (0 or 1).
The general working principle of this is to split the image into *dark* and *light* regions.
In order to divide the pixels into dark and light, a *threshold* value must be found.

There are several algorithms which can be used for this, but they can be grouped into *global* and *adaptive* methods.
Global methods find a single value for the threshold based on the whole image's histogram.
Adaptive methods use separate threshold values for each pixel in the image, making use of a pixel's neighbourhood.

![Global thresholding example](images/global_threshold.png){#fig:global_threshold}

An example of a global threshold being applied to a C-scan is shown in [@fig:global_threshold], specifically the *Otso* algorithm.
This finds the optimal threshold based by minimising the interclass variance^[Get a reference for this].

The red line shown in the histogram indicates the threshold that has been chosen.
As can be seen in the binary image, this algorithm has not performed well.
This is due to the fact that the threshold has been chosen such that an equal number of peaks have been separated.
It happens that the value of the pixels which make up defects on the first two steps of the part have sufficently low values to be below the threshold.
However, the background of the third, fourth and fifth steps are also below this threshold.

It is expected that an algorithm which makes use of local information would yield better results for C-scan images of parts of differing thickness such as this.

![Adaptive thresholding example](images/adaptive_threshold.png){#fig:adaptive_threshold}

Shown in [@fig:adaptive_threshold] is a demonstration of the application of a adaptive thresholding algorithm, in this case *Sauvola* [@sauvola_adaptive_2000].
What is immediately obvious is that the defect segmentation performance using this method is much better, as it deals with the varying background values.
The red bars in the histogram is a histogram which represents the distribution of pixel threshold values, as this now varies by pixel.
Interesting to note is the fact that the peaks in the threshold histogram are generally in-between the peaks in the image histogram.

Also shown is an image which shows the threshold value for each pixel.
Note that the background of each step in the image is different, this explains the increased performance of this method when compared to that demonstrated in [@fig:adaptive_threshold].

Segmentation performance of thresholding is fair, but the main advantage is that of processing time.
For the images above, the global method took roughly 5 ms, and the adaptive method 22 ms.
Adaptive methods are more complex, so this has an impact on processing time, but comes with the advantages mentioned previously.

## Point detection {#sec:pointdetection}

The purpose of *point detection* is to find individual pixels which have a significantly different value than their neighbours.
A simple way of detecting such points is to find approximate derivatives around each pixel's neighbourhood.
More complex algorithms involving point *detectors* and neighbourhood *descriptors* can be used to find key points in an image along with a summary of their neighbourhood for applications such as video stabilisation or object detection and tracking [@cowan_performance_2016].

In the highly-likely case of a noisy input image, it is required that denoising be carried out before finding points, as the algorithm will likely detect noise pixels as salient points.

## Edge detection

The purpose of *edge detection* is to find the pixels which represent significant discontinuities in an image, or *edges*.
The simplest way to achieve this is to find the derivative of the pixels in a region.
Due to the discrete grid-like nature of images, the derivative must be approximated by a $3 \times 3$ matrix which is convolved with the image.

![Demonstration of the Sobel operator](images/sobel.png){#fig:sobel}

Horizontal and vertical edges can be detected separately using two different convolution matrices.
The resultant edge responses can then be used to find a total edge magnitude and edge direction.
Examples of these types of edge operators are *Sobel*, *Prewitt* and *Roberts*.
An example of applying the Sobel operator on a greyscale spiral is shown in [@fig:sobel].

### Laplacian

The *Laplacian* makes use of the second derivative of the image.
Due to the fact that second derivative is used, the raw Laplacian is very sensitive to noise.
Therefore, for image processing purposes, it is paired with a blurring operation, such as the Gaussian.
This combination becomes known as the Laplacian of Gaussian (LoG).

<!--
What's the point of using the LoG over the boring ones?
-->

### Canny edge detection

The aforementioned edge detection algorithms only find the edge responses (magnitude and direction) in an image.
There is no guarantee that these represent actual connected edges, or if they are just local gradients.
An example of a method which can find connected edges in an image is the *Canny* algorithm [@canny_computational_1986].

Its operation can be summarised as:

1. Apply a Gaussian smoothing operator to the image.
2. Apply Sobel edge detection to find edge amplitude.
3. Reduce edges found in the previous step to be 1 pixel wide.
4. Hysteresis thresholding, where an initial (high) threshold is used to select *strong* edges, and 8-connectivity is used to connect these edges with weaker ones up to a lower threshold value.

### Hough transform

The Hough transform can be used to detect lines, circles or ellipses in an image (depending on which parameters are used).
In order to find these features, a boolean image is used (0 for no potential feature, 1 for a potential feature).

A straight line can be parameterised as $y = mx + c$, however this has the problem of not being able to represent vertical lines (as $m = \infty$).
Therefore, a polar form of a line ($x \cos \theta + y \sin \theta = \rho$) is used for the detection of lines.
To find the potential values of $\rho$ and $\theta$, the following process is carried out [@duda_use_1972]:

1. A positive pixel is chosen from the image.
2. The parameters for potential lines which pass through this pixel are found, using other random positive pixels.

These steps are followed for other positive pixels in the image.
Looking at the values of $\rho$ for corresponding values of $\theta$ across these pixels, the best-fitting line will be chosen based on the similarity in $\rho$ value.

Finding circles and ellipses follows a similar process, however the parameters to be found are different (e.g. radius, position, eccentricity).

## Wavelet decomposition {#sec:waveletdecompose}

![Example of a wavelet decomposition](images/decompose.png){#fig:decompose}

A *wavelet decomposition* is a method which can be used to separate the differing frequency components in an image.
This is very useful for noise reduction, as noise is mainly a high-frequency effect.
An example decomposition is shown in [@fig:decompose].
In this case, the 'Daubechies' with length 1 wavelet function was used.
The top row presents the low frequency 'approximation', which looks similar to a blurred version of the original image.
Going down the other rows are increasing 'levels', or detail components, where the higher the level are made up of higher frequencies.

Recombining these individual levels, it is possible to recreate the original image.
This is a useful technique for retouching images, as it enables the possibility to isolate edits to only certain frequency ranges without having an effect on the others.
For example, by reducing the noise on only the highest levels of the decomposition, the denoising process does not affect the overall structure of the image.

## Region filling

<!-- The idea of region filling algorithms using random (or specified) seed points. -->

# Acquisition of C-scan images

<!---
Details of how C-scans are generated from the underlying raw_data.
Things about VIEWS and how the gating parameters work and so on.
-->

# Previous work on image processing for defect recognition

## Chang et al., 2003

This work concerns the detection of breast tumours from ultrasound images.
A comparison was carried out between the use of SVM and a neural network setup.
A database of 250 images was compiled, which contained 140 of benign tumours and 110 of cancerous, where each image only contained the tumour area.

The features which were extracted from the images were texture-based features, namely autocorrelation and autocovariance.
An SVM model was trained and tested using the $k$-fold method with $k$ in this case being set to 5.
A neural network model was also trained using the sample process for testing.
It was noted that the classification performance of these two methods was comparable, but SVM took significantly less time to go through the training process.

It should be noted that it is not possible to assign a class probability to a sample using the SVM method.
This is due to how SVM works, where an optimal dividing hyperplane is found to separate two classes, and a sample's class is decided by which side of the plane it is on.

## Belaid et al., 2011

The main innovation of this work is the use of the *monogenic signal* rather than the commonly-used Hilbert transform for separating amplitude and phase information in order to generate ultrasonic C-scans.
This was shown to improve edge detection performance at the cost of reducing the detail components of the images.

## Kechida et al., 2012

Weld inspection using TOFD was the target for this research.
The approach used was to make use of texture information.
Multi-resolution methods (wavelet decomposition), Gabor functions, PCA and a form of clustering (called c-means) were employed.
The end result was to carry out a binary classification (defect or no defect).

The images were first separated into small windows, and these were decomposed into four detail levels using wavelet decomposition.
2D Gabor functions were also used to perform this decomposition process.
Several statistics were extracted from each of the level, specifically:

1. Angular moment, second order
2. Contrast
3. Inverse difference moment
4. Sum mean
5. Entropy

PCA was then carried out to reduce the amount of data generated from the extracted features.
Once this dimensionality reduction has been carried out, a fuzzy c-means clustering method was applied.
C-means is similar to *k*-means, however a given point can be a member of multiple classes (a probability is assigned to them).

By varying the size of the image windows used, the performance of the method will vary according to the size of defects to be found.
It was noted that larger windows will increase the processing time required.
Different wavelet methods were used, and the performance of the classification was found to be invariant to the chosen technique.
The number of PCA components was also optimised to provide good classification performance with an acceptable processing time.
It was concluded that wavelets would be more useful due to the possibility of adding defect sizing using this method.

# Application of image processing to C-scan images

This section concerns the application of several of the image processing techniques detailed in [@sec:imageprocessing] to C-scan images.

A common framework was created for image loading and processing, mainly based on the methods in the `scikit-image` Python package.
The framework (`cueimgproc`) provides a consistent interface for image processing, and exposes a significant amount of image processing algorithms.
`cueimgproc` is not specific to C-scan images, the only requirement is that greyscale images are used as input.

The general principle of operation is that of *graphs of nodes*, i.e. an image is created, and several *filters* can be applied in turn to the image.
Some of the nodes are so-called *filters*, which take an input image, and create an output which has been processed.
Others are *transforms*, which take an input image and create several non-image outputs, these are often used for feature detection.

For defect recognition purposes, several *pipelines* were created, and these are discussed in the following sections.

## Simple thresholding

A simple approach to region segmentation can be performed by using *thresholding* methods, such as those described in [@sec:thresholding].
Thresholding has the advantage of being very simple to implement and fast to compute.

The task to be completed by using image thresholding on a ultrasonic C-scan image is to separate the image into two classes, namely *defect* and *non defect*.
This is a binary segmentation process.

As mentioned previously, there are two broad classes of thresholding algorithm, *global* and *adaptive*.
It is expected that the segmentation performance of global thresholds will be worse than adaptively thresholded samples, due to thickness differences.

The main process to be followed in the thresholding segmentation pipeline is:

1. Load the image
2. Optional preprocessing (denoising, histogram equalisation, resizing)
3. Apply a thresholding algorithm, which results in a binary image (0 for below threshold pixels, 1 for above).
4. Separate regions of connected high-valued pixels into labelled regions
5. Calculate properties of these regions (specifically centroid, area, eccentricity, perimeter)

![Comparison of thresholding methods](images/all_threshold.png){#fig:threshold_compare}

Shown in [@fig:threshold_compare] is a comparison between all the thresholding algorithms which are available in the `scikit-image` image processing package.
This includes both global and adaptive methods, where the adaptive methods can be identified by the indication of an *average threshold*, which is the simple mean of the individual pixel thresholds.
Given that the expected number of defects in the sample is 30, it can be seen that the *Sauvola* algorithm has performed the best in this case.
This so happens to be a adaptive thresholding algorithm.

The Sauvola technique was intended to be used for the binarisation of images of documents, to segment text from a background [@sauvola_adaptive_2000].
Perhaps its good performance on defect segmentation can be explained by defects being relatively small spots on a large background, somewhat like text on a page.
A window surrounding each pixel is analysed in order to determine the pixel's threshold value.

The pixel thresholds are set by the following:

$$
T(x, y) = m(x, y) \left[ 1 + k \left( \frac{s(x, y)}{R} - 1 \right) \right]
$$
where $m(x, y)$ is the window mean, $s(x, y)$ is the window standard deviation; and $k$, $R$ are parameters.

<!-- What do these parameters do? -->

The particular parameters used in the application of Sauvola in [@fig:threshold_compare] were:

* A window size of 15
* $k$ of 0.2
* $R$ of 0.5 (i.e. the half point of the 0-1 floating-point pixel value range)

### Pipeline

![Results of thresholding pipeline](images/pipelines/threshold.png){#fig:pipeline_threshold}

The thresholding pipeline is a simple one, containing a single threshold filter.
From the results of [@fig:threshold_compare], the Sauvola algorithm was chosen.

Shown in [@fig:pipeline_threshold] are the results of this pipeline.
The binary thresholded image is shown in the centre of this figure.
By making use of connectivity information, it is possible to separate the different regions out.
The image at the bottom of this figure show the results of the segmentation, where each region is shown in a different colour.

Details of these regions are determined, including:

* Centroid location in the image.
* Area of region.
* Shape of region (in terms of the eccentricity).
* Perimeter of the region.

The eccentricity is a measure of the circularity of the region.
It is expected that defects have a low eccentricity, and are relatively small.
By looking at [@fig:pipeline_threshold], it is possible to see that the *steps* between different depths are detected as separate regions.
These have large areas, and are also rectangular, and therefore have high eccentricity.

It can also be seen that numerous regions have been detected, many of which are very small, and due to noise in the C-scan.
A simple method of removing these noise regions would be to add an area threshold, and reject regions which are smaller than a certain size.
This can be viewed from the perspective of defining a necessary minimum defect size to reject in a part.
Of course, the areas in C-scan images are measured in pixels, so it would require conversion from physical units to pixels through an appropriate factor.
Also, applying a threshold on eccentricity values would be useful in order to remove large aspect ratio regions which are not likely to be defects.

![Results of thresholding pipeline](images/pipelines/threshold_triple.png){#fig:pipeline_threshold_triple}

All together, this method could be regarded as three layers of thresholding, based on pixel value, region size and region shape.
Adding these extra thresholds results in [@fig:pipeline_threshold_triple].
Regions with areas less than 32 pixels are discarded, as are regions with eccentricities of over 0.9.

## Edge detection

![Results of edge detection using the Canny algorithm](images/pipelines/canny.png){#fig:pipeline_canny}

![Results of edge detection using the Canny algorithm with additional thresholding](images/pipelines/canny_extra.png){#fig:pipeline_canny_extra}

## Region growing

# Defect recognition performance evaluation

<!---
Quantitative performance evaluations of the previous section's work.
-->

With reference to the dimensions of the sample part and the expected locations of the defects, it is possible to determine the precision and positional accuracy of the described segmentation methods.

<!-- Put a picture of the reference sample here -->

The engineering drawing of the reference part is shown in [@fig:reference_drawing].
Centroid coordinates (in mm) of the defect positions were created, resulting in a $30 \times 2$ matrix.

The C-scan however does not cover the entirely of the part, and there is a scaling factor involved to convert pixel space to physical space.
The image could also be rotated with respect the scanned part, due to the fact a physical process was involved in the scan.
In order to find a transformation from pixel space to physical space, a method such as least-squares can be used to match coordinates in the image to physical positions.

![Manually segmented defect regions](images/stepped-defects.png){#fig:stepped_defects}

Using the engineering drawing, the positions of defects were determined.
It is then required to find the corresponding points in the C-scan, and use these two matrices to calculate the image transformation required to reproject the image into physical coordinates.
To produce a *ground truth*, a binary image was created manually, with the defect pixels being marked by hand.
The C-scan showed no sign of the defect in the third row and third column, hence why it is not displayed.
Using `scikit-image`, the properties of these regions were determined, with the intention of finding the centroids.
Making use of these centroids and the expected defect locations in mm, an affine transformation matrix was found using a least-squares method:

$$
\begin{bmatrix}
  0.6618 & -0.0062 & 8.7165 \\
  0.0020 & 0.5248  & 76.5579 \\
  0.0000 & 0.0000  & 1.000 \\
\end{bmatrix}
$$
which corresponds to:

* Translation $(t_x, t_y) = (8.72, 76.6)$ mm.
* Scale $(s_x, s_y) = (0.662, 0.625)$ mm/px.
* Rotation $\theta = 0.173°$.
* Shear $\phi = 0.501°$.

![Estimated transform with errors](images/transforms/groundtruth.png){#fig:groundtruth_transform}

When finding this transformation, the points will have certain errors with respect to the ideal locations.
An image which shows these points with these errors is shown in [#fig:groundtruth_transform].

![Estimated transform errors](images/transforms/groundtruth_error.png){#fig:groundtruth_error}

The values of these errors are shown in a histogram in [#fig:groundtruth_error].

## Thresholding

![Results of transform using thresholding](images/transforms/threshold.png){#fig:transform_threshold}

![Error histogram for transform using thresholding](images/transforms/threshold_error.png){#fig:transform_threshold_error}

## Edge detection

![Results of transform using Canny edge detection](images/transforms/canny.png){#fig:transform_canny}

![Error histogram for transform using Canny edge detection](images/transforms/canny_error.png){#fig:transform_canny_error}

## Region filling

![Results of transform using region filling](images/transforms/fill.png){#fig:transform_fill}

![Error histogram for transform using Canny edge detection](images/transforms/fill_error.png){#fig:transform_fill_error}

# Conclusions and future work

<!---
The drawbacks of image-based methods (manual gating, contrast issues and so on).
-->

## Gating fusion

As has been discussed in this chapter, a major drawback with making use of C-scan images to carry out defect recognition is the requirement to select appropriate gating parameters.
Perhaps one set of values makes is possible to segment some defects in the sample, but makes it more difficult for others.
This would be the case for defects at different depths of varying depth.

One possible method of removing this problem could be to use a series of C-scan images use different gating parameters.
These individual images can be *fused* together to increase the defect/non-defect contrast for each defect in the sample.
This process could be similar to that of *exposure fusion*, which can be used to improve the quality of photographs of high dynamic range scenes [@mertens_exposure_2007].
Individual photographs are taken by *exposure bracketing*, where some images in the group are underexposed, and others overexposed.
Details from dark regions of the scene can be extracted from the overexposed photographs, and those from the bright parts of the scene from the underexposed.
Optimal weights for each image's contribution to the output on a pixel-by-pixel basis are calculated, and the output image is the sum of the individual bracketed exposure after these weights are applied.
A measure of quality is used to drive the elicitation of weights; for photographs this can be composed of saturation and contrast.

For ultrasonic C-scans, there is no such concept of 'exposure', but by varying the gating parameters used to generate the C-scan from the raw ultrasound data, it is possible to manipulate the effect tone curve of the C-scan.
Therefore, the gating parameters can be seen as an analogue of exposure.
A quality measure can then be contrast, as there is no chroma data.

# References
