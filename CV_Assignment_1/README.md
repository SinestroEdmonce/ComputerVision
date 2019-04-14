## Question Information

1. **For assignment No.2, students should do coding to solve the question below:**  
    - ***Write a function that convolves an image with a given convolution filter.***
    - *As input, the function takes a greyscale image $(img0)$ and a convolution filter stored in matrix $h$. The output of the function should be an image $img1$ of the same size as $img0$ which results from convolving $img0$ with $h$. You can assume that the filter $h$ is odd sized along both dimensions.*
    - *You will need to handle boundary cases on the edges of the image. For example, when you place a convolution mask on the top left corner of the image, most of the filter mask will lie outside the image. One solution is to output a zero value at all these locations, the better thing to do is to pad the image such that pixels lying outside the image boundary have the same intensity value as the nearest pixel that lies inside the image.*
    - *You are not allowed to use any system convolution function or something related to filters.*

2. **For assignment No.2, students should do coding to solve two question below:**  
    - ***Compute the gray level (luminance) histogram for an image and equalize it so that the tones look better (and the image is less sensitive to exposure settings). You may want to use the following steps:***
        - *Convert the color image to luminance.*
        - *Compute the histogram, the cumulative distribution, and the compensation transfer function.*
        - *(Optional) Try to increase the “punch” in the image by ensuring that a certain fraction of pixels (say, 5%) are mapped to pure black and white.*
        - *(Optional) Limit the local gain $f(I)$ in the transfer function. One way to do this is to limit $f(I) < \lambda I$ or $f(I) < \lambda$ while performing the accumulation, keeping any unaccumulated values "in reserve".*
        - *Compensate the luminance channel through the lookup table and re-generate the color image using color ratios.*
        - *(Optional) Color values that are clipped in the original image, i.e., have one or more saturated color channels, may appear unnatural when remapped to a non-clipped value. Extend your algorithm to handle this case in some useful way.*

    - ***Compute the gray level (luminance) histograms for each patch, but add to vertices based on distance (a spline).***
        - *Distribute values (counts) to adjacent vertices (bilinear).*
        - *Convert to CDF (look-up functions).*
        - *(Optional) Use low-pass filtering of CDFs.*
        - *Interpolate adjacent CDFs for final lookup.*  
        ***(Interploate adjacent CDFs as the picture shows below)***
         <div align=center><img width="400" height="210" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/img/adaptively_local_hist_eq.jpg"/></div>
        

## Expriments

1. The results of the convolution on the given image *``Lenna.png``* with different filters are shown below:  

    - Listed from left to right are the source, the average and the gaussian:  
    
    <div align=center><img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/resource/Lenna.png"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_average.jpg"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_gaussian.jpg"/></div>

    - Listed from left to right are the source, the sobel and the laplacian:  

    <div align=center><img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/resource/Lenna.png"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_sobel.jpg"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_laplacian.jpg"/></div>

2. The results of the global histogram equalization and local histogram equalization on the given image *``Lenna.png``* are shown below:

    - Global histogram equalization. Listed from left to right are the *common*, the *punch*, the *lambda*.  
    The results extracted luminance information from ``YCbCr`` and treated ``Y`` channel as the luminance level:

    <div align=center><img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_ycbcr_common.jpg"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_ycbcr_lambda0.7.jpg"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_ycbcr_punch0.4.jpg"/></div>

    - Local histogram equalization. Listed from left to right are the *gaussian*, the *bonus*, the *average*.  
    The results extracted luminance information from ``YCbCr`` and treated ``Y`` channel as the luminance level:

    <div align=center><img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_ycbcr_gaussian.jpg"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_ycbcr_bonus.jpg"/> <img width="150" height="150" src="https://github.com/SinestroEdmonce/ComputerVision/raw/master/CV_Assignment_1/results/Lenna_ycbcr_average.jpg"/></div>

## Usage

- For assignment No.2, you should firstly enter into the directory ``assignment_1_q2`` and open your terminal. Then you can easily execute the command ``python main4unit.py -m`` for the detailed usage.

- For assignment No.3,  you should firstly enter into the directory ``assignment_1_q3`` and open your terminal. Then you can easily execute the command ``python main4unit.py -m`` for the detailed usage.