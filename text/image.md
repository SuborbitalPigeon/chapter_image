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

*Image processing* algorithms are a group of methods which operate on images to produce other images.
They are commonly used in photography.
Operations range from simple ones such as rescaling or rotation, up to the complex (for example those which operate in the frequency domain).

![An example ultrasound image](images/stepped.png){#fig:stepped}

In the domain of defect recognition, the main task to be carried out is one of *segmentation*.
It is necessary to find the regions which represent defects in the image.

For example, see the image of a reference sample part in [@fig:stepped].
This sample has thirty deliberate defects, which have been inserted at known locations and depths.
The part is made of five sections of varying thickness.
By visual inspection, it is quite easy to see the defect locations based on the contrast difference between the background and the small mostly circular regions.
The question to be answered becomes 'is it possible for an algorithm to segment these defect regions from the background?'

## Digital images

A *digital image* can be represented as a matrix of values.
This matrix can be 2D in the case of a grey-scale image, 3D in the case of a multi-channel image or a set of greyscale images; or 4D for a set of multi-channel images.
Sets of images are often used to represent the effect of time.

A *greyscale* image represents the values of a single variable in a spatial scene.
Multi-channel images are a set of greyscale images stacked together, a common example would be making use of separate colour channels for colour images.
For general photography purposes, these are commonly displayed with red, green and blue values.
However, the commonly-used JPEG compression algorithm actually stores the image with channels which represent brightness (Y), blueness (Cb) and redness (Cr), this colour space being known as *YCbCr*.
This is to take advantage of the fact that the human vision system is more sensitive to changes in brightness than in colour.
The colour channels are often stored at a lower resolution than the luma, and this can bring a 50 % savings in storage space of the image, before the main JPEG compression algorithms are applied.

## Histogram

## C-scans

<!---
Discuss what C-scans are and how they are generated.
-->

<!---
![An example A-scan](images/ascan.png){#fig:ascan}
-->

An ultrasound signal can be represented in the time domain in an *A-scan*.
An A-scan shows the amplitude of the ultrasound response with respect to the time since the pulse was sent.
By using the speed of propagation in the material, it is possible to convert time to depth.
This means that the A-scan can also be seen as echo amplitude vs. depth.

To simplify processing, the ultrasonic propagation speed is commonly assumed to be a constant value.
For isotropic materials, this is a reasonable assumption, however due to the significant anisotropy of composite materials, this can become an issue.
However, as long as the ultrasonic wave is propagating normally through the lamina layers, the speed should be relatively constant and independent of the ply layup sequence.

A *C-scan* is a top down view of ultrasound data, where the axes of the image represent a 2D surface of a part.
It is therefore necessary for a probe to be moved in two dimensions on the part, or for an array to be used.
The image shown in [@fig:stepped] is a C-scan of a sample composite part, with deliberate defects.

Each pixel in the image represents a single A-scan location.
This requires a decision in which value to assign to the pixel, given that each A-scan is made of several time domain samples.
One approach is to find the Time of Flight (ToF), which is the amount of time taken for the wave to propagate from the front surface of the part to the most prominent echo inside it or the back surface, depending on the relative amplitudes.
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

## General image transforms

In this section, *general image transforms* refers to operations which affect the dimensions and shape of the image matrix.
This includes scaling, translation, rotation, shear and generalised warps.

All the operations discussed in this section can be described by $3 \times 3$ transformation matrices.
By making use of such matrices, it is possible to chain together several operations to be carried out at the same time.

### Scaling

*Scaling* refers to resizing the image matrix in order to change the image's resolution.
This can be represented by the following matrix:

$$
T =
\begin{bmatrix}
  S_x & 0 & 1 \\
  0 & S_y & 0 \\
  0 & 0 & 1 \\
\end{bmatrix}
$$
where $S_x$ and $S_y$ are the scaling factors (values $\lt 1$ are scaling down) in the $x$ and $y$ directions respectively.

It is common in image processing to reduce the size of images in order to speed up more complex operations later in the processing pipeline.
It was quite common in the early days of digital cameras for images to be *upscaled* in order to achieve more marketable *megapixel* counts.
However, adding information which is not in the original image can actually cause the effective resolution to reduce.^[This brings to mind the quote "Prediction is very difficult, especially if it's about the future"[@anker_forecasting_2017], which is attributed to Neils Bohr]

### Rotation

*Rotation* of an image can be represented by the following transformation matrix:

$$
T =
\begin{bmatrix}
  \cos \theta & \sin \theta & 1 \\
  -\sin \theta & \cos \theta & 0 \\
  0 & 0 & 1 \\
\end{bmatrix}
$$
where $\theta$ is the rotation angle referred to the positive $x$ axis (in radians).

<!-- It is common in photography to convert between portrait and landscape orientation of photographs^[Also for the far too often required conversion of vertically phone-captured videos to landscape for display on non-phone devices], this requires a positive or negative rotation of 90°. -->

It should be noted that this matrix refers to a rotation with a pivot point of the origin point of the image (top-left).
In order to use another pivot point, a translation procedure must also be performed.

### Filtering and boundary modes

<!-- I don't know where to put this section -->

The pixels which make up an image are located at integer coordinate values.
By applying transforms, it is possible that the ideal new locations of pixels would involve floating-point.
However, this is not possible in the matrix representation of an image, so a *filtering* algorithm must be applied.

The simplest technique is to assign a pixel's value to the *nearest* value of the floating-point coordinates.
This can result in significant artefacts (for example, aliasing for small angle rotations), and so more complex methods such as *linear* or *cubic* filtering can be used.^[Should I talk more about these?]

## Histogram manipulation

An image's *histogram* is a representation of the frequency of pixel values in the image.

For an ideally exposed photograph, the full range of possible pixel values would appear, and would have equal frequency.

### Tone curves

A function can be applied which maps between input tone in the image and output.
It is common to use so called *s-shaped* curves, these are used to increase the contrast in the mid-tones of an image.

### Histogram equalisation

![Original image, and global and local equalisation](images/equalise.png){#fig:equalise}

*Histogram equalisation* is a process whereby the values are modified to create as close to a flat histogram as possible.
This is achieved by increasing the distance between values assigned to more common input values.
This is a way of improving the contrast in an image, and is useful for image processing for scientific purposes.
It however leads to poor quality results for photographs, as it is a simplistic operation.

See [@fig:equalise] for an example of an ultrasound C-scan before and after equalisation, both global and local.
In this case, the block size for the adaptive equalisation is such that the image is split into 8 × 8 blocks.
It can be seen that the contrast between the background and the defects is higher, especially in the case of the image which has undergone local equalisation.

![Histograms of image, and after global and local equalisation](images/equalise_hist.png){#fig:equalise_hist}

See [@fig:equalise_hist] for histograms of the original image, and after the application of global and local equalisation.
It can be seen that the pixel values in the original images are predominantly that of the five background steps of the sample, as can be expected.
After a global equalisation process has been carried out, it can be seen that the peaks have been 'spread out' across the full range of possible values.
Local equalisation preserves the values of the peaks, but they have been spread out in value.
This has increased the contrast in the non-defect regions.

## Denoising

*Noise* in an image manifests itself as random fluctuations in the values of pixels when compared to their neighbours.
It is a by-product of several physical phenomena, such as electrically generated noise, pixel leakage^[Find a source for this].

In digital photography, noise is a particular problem with images taken with high ISO values (higher amplifier gains), which can be required in low-light situations.
The pixel values are amplified according to the camera's ISO setting, which has the by-product of reducing the effective dynamic range of the camera.

An example of an image with noise is shown in [@fig:noisy].

<!---
![A noisy image](images/noise.jpg)
-->

In ultrasound C-scans, noise appears as a result of the gain applied to the underlying A-scans for each pixel.
Also, electrical noise present in the A-scans will have an effect.

Noise can be reduced using several methods.
Some of these are described in the following sections.

### Median filtering

For this technique, a simple statistical test is applied to the neighbourhood of each pixel in the input image.
The value that each pixel is set to in the output image is the median value of this pixel's neighbourhood.
This therefore rejects random high and low values in the neighbourhood.
By making the region larger, it is possible to increase the effect.

The image must first be padded to remove the problem at the edges.

This is a very simple operation, and performs relatively poorly as it has a strong blurring effect on the image.
However, it is computationally cheap to perform.

### Blurring

<!---
Gaussian blurring
-->

### Non-local means denoising

### Wavelet denoising

*Wavelet transforms* are a type of frequency domain transformation, similar in purpose to the Fourier Transform or Discrete Cosine Transform (DCT).
They have particular advantages in the field of image processing, due to the fact they encode not only frequency data, but also time dependence [@gonzalez_digital_2002].

By making use of wavelet decomposition (see [@sec:waveletdecompose] for further detail), it is possible to convert an image into a *pyramid* of separate images which contain different levels of detail.
Noise is generally a variation between individual pixels, and this means that it will be most prominent in the most detailed level of an image pyramid.
By running a smoothing algorithm on only the most detailed level, it is possible to reduce the noise of the image without having an effect on the general structure.

## Thresholding

Binary threshold methods and their operation, and specific advantages and disadvantages.

## Point detection

The purpose of *point detection* is to find individual pixels which have a significantly different value than their neighbours.
A simple way of detecting such points is to find approximate derivatives around each pixel's neighbourhood.
More complex algorithms involving point *detectors* and neighbourhood *descriptors* can be used to find key points in an image along with a summary of their neighbourhood for applications such as video stabilisation or object detection and tracking [@cowan_performance_2016].

It is required to denoise images before finding points, as the algorithm will likely detect noise pixels as salient points.

## Edge detection

The purpose of *edge detection* is to find the pixels which represent significant discontinuities in an image, or *edges*.
The simplest way to achieve this is to find the derivative of the pixels in a region.
Due to the discrete grid-like nature of images, the derivative must be approximated by a $3 \times 3$ matrix which is convolved with the image.

Horizontal and vertical edges can be detected separately using two different convolution matrices.
The resultant edge responses can then be used to find a total edge magnitude and edge direction.
Examples of these types of edge operators are *Sobel*, *Prewitt* and *Roberts*.

### Laplacian

The *Laplacian* makes use of the second derivative of the image.
Due to the fact that second derivative is used, the raw Laplacian is very sensitive to noise.
Therefore, for image processing purposes, it is paired with a blurring operation, such as the Gaussian.
This combination becomes known as the Laplacian of Gaussian (LoG).

<!--
What's the point of using the LoG over the boring ones?
-->

### Canny edge detection

The aforementioned edge detection algorithms^[Check plural] only find the edge responses in an image.
There is no guarantee that these represent actual connected edges, or if they are just local gradients.
An example of a method which can find connected edges in an image is the *Canny* algorithm [@canny_computational_1986].

Its operation can be summarised as:

1. Apply a Gaussian smoothing operator to the image.
2. Apply Sobel edge detection to find edge amplitude.
3. Reduce edges found in the previous step to be 1 pixel wide.
4. Hysteresis thresholding, where an initial (high) threshold is used to select *strong* edges, and 8-connectivity is used to connect these edges with weaker ones up to a lower threshold value.

### Hough transform

The Hough transform can be used to detect lines, circles or ellipses in an image (depending on the parameters used).
The input is a boolean image (0 for no potential feature, 1 for a potential feature).

A straight line can be parameterised as $y = mx + c$, however this has the problem of not being able to represent vertical lines (as $m = \infty$).
Therefore, a polar form of a line ($x \cos \theta + y \sin \theta = \rho$) is used.
To find the potential values of $\rho$ and $\theta$, the following process is carried out [@duda_use_1972]:

1. A positive pixel is chosen from the image.
2. The parameters of potential lines which pass through this pixel are found.

<!---
This doesn't make sense
-->

These steps are followed for other positive pixels in the image.
Looking at the values of $\rho$ for corresponding values of $\theta$ across these pixels, the best line will be chosen based on the similarity in $\rho$ value.

Finding circles and ellipses follows a similar process, however the parameters to be found are different.

## Wavelet decomposition {#sec:waveletdecompose}

## Region filling

<!---
The idea of region filling algorithms using random (or specified) seed points.
-->

# Acquisition of C-scan images

<!---
Details of how C-scans are generated from the underlying data.
Things about VIEWS and how the gating parameters work and so on.
-->

# Previous work on image processing for finding defects

<!---
This section will contain details of previous work on this problem.
-->

# Application of image processing to C-scan images

<!---
What has been done, reasoning behind why these methods and so on.
-->

# Defect recognition performance evaluation

<!---
Quantitative performance evaluations of the previous section's work.
-->

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
This process could be similar to that of *exposure fusion*, which can be used to improve the quality of photographs of high dynamic range scenes[@mertens_exposure_2007].
Individual photographs are taken by *exposure bracketing*, where some images in the group are underexposed, and others overexposed.
Details from dark regions of the scene can be extracted from the overexposed photographs, and those from the bright parts of the scene from the underexposed.
Optimal weights for each image's contribution to the output on a pixel-by-pixel basis are calculated, and the output image is the sum of the individual bracketed exposure after these weights are applied.
A measure of quality is used to drive the elicitation of weights; for photographs this can be composed of saturation and contrast.

For ultrasonic C-scans, there is no such concept of 'exposure', but by varying the gating parameters used to generate the C-scan from the raw ultrasound data, it is possible to manipulate the effect tone curve of the C-scan.
Therefore, the gating parameters can be seen as an analogue of exposure.
A quality measure can then be contrast, as there is no chroma data.

# References
