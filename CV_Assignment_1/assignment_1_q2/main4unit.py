import numpy as np
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def image_filter_conv2d(img0, h):
    """Function that is used to impact a given filter on an image.

    :param img0: Mostly a np.ndarray object that represents an image
    :param h: A specific filter that given by users, also a np.ndarray object

    :return: A np.array object that represents a new image,
        transformed from the original image, and usually looks like an example below:

        [3 1 0 2]
        [3 3 4 1]
        [4 5 7 9]

        the value seems like a matrix and indeed it is matrix that describes an image.

    """

    # Flip the filter for convolution
    h_flipped_filter = h[::-1, ::-1]

    # Zero-padding
    padding_size4row, padding_size4col = h_flipped_filter.shape[0]-1, h_flipped_filter.shape[1]-1
    padding_image_height, padding_image_width = img0.shape[0] + padding_size4row, img0.shape[1] + padding_size4col
    padding_img = np.pad(img0, ((padding_size4row, padding_size4row), (padding_size4col, padding_size4col)),
                         'constant', constant_values=(0,0))

    # Calculate the convolution on the given image batch by batch
    # In every batch, we calculate the result by multiplying pixels by the filter and adding up the multiplications
    h_flipped_filter_flatten = h_flipped_filter.flatten()

    def conv2d_in_batch(loc):
        i, j = int(loc/padding_image_width), loc%padding_image_width
        img0_batch = padding_img[i:i+h_flipped_filter.shape[0], j:j+h_flipped_filter.shape[1]]

        return np.dot(img0_batch.flatten(), h_flipped_filter_flatten)

    # Use np.vectorize() to impact the function on every elment
    vectorzie_conv2d = np.vectorize(conv2d_in_batch)

    # Obtain the final result: padding_image_height * padding_image_width
    return np.array(vectorzie_conv2d(np.arange(start=0,
                                               stop=padding_image_height*padding_image_width,
                                               step=1)).reshape(padding_image_height,
                                                                padding_image_width),
                    dtype=np.uint8)



def generate_filter_kernel(kernel_type, kernel_size=3, sigma=0):
    """Function that generates a kernel

    :param kernel_type: The type of kernel or self defined kernel
    :param kernel_size: Size of the kernel, only designed for gaussian and average one
    :param sigma: Variance only designed for gaussian kernel

    :return: a kernel for convolution, perhaps looks like:

        [1, 1, 1]
        [1, 1, 1]  * 1/9
        [1, 1, 1]

    """

    if kernel_type=="gaussian":
        kernelx = cv2.getGaussianKernel(kernel_size, sigma)
        kernely = cv2.getGaussianKernel(kernel_size, sigma)
        return np.multiply(kernelx, np.transpose(kernely))

    if kernel_type=="laplacian":
        laplacian_kernel = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]))
        return laplacian_kernel

    if kernel_type=="average":
        average_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        return average_kernel

    if kernel_type=="sobel":
        sobel_kernel = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))
        return sobel_kernel

    if kernel_type=="prewitt":
        prewitt_kernel = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))
        return prewitt_kernel

    self_defined_kernel = list(map(float, kernel_type.strip().split()))
    return np.array(self_defined_kernel).reshape(kernel_size, kernel_size)


def main():
    """Function that is used to run this single unit program for testing or so.

    Users can just run this module for some specific intentions.

    :return:
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', help='Path to the image', default="./../resource/Lenna.png", type=str)
    parser.add_argument('--output', '-o', help='Path to the output',
                        default="./../results",
                        type=str)
    parser.add_argument('--filter', '-f', help='Filter to choose', default="gaussian", type=str)
    parser.add_argument('--kernel_size', '-ks', help='Length of a side of the filter', default=3, type=int)
    parser.add_argument('--gaussian_sigma', '-gs', help='Length of a side of the filter', default=0, type=int)
    parser.add_argument('--manual', '-m', help='Guidance for usage')

    args = parser.parse_args()

    if args.manual:
        print("Use -i to determine the path to your image.\n"
              "Use -o to determine the path to your output.\n"
              "Use -f to determine which filter you hope to use, including Gaussian.\n"
              "Uee -ks to determine the length of a side of the kernel.\n"
              "Uee -gs to determine the value of sigma in a gaussian function.\n"
              "Uee -m to refer to the guidance for this program\n")
        return

    # Show the arguments
    print("resource path: "+args.image+" , "
          + "output path: "+args.output+" , "
          + "filter: "+args.filter)
    # Obtain the kernel for the convolution
    kernel = generate_filter_kernel(kernel_type=args.filter,
                                    kernel_size=args.kernel_size,
                                    sigma=args.gaussian_sigma)

    # Obtain the image to be processed
    original_img = cv2.imread(args.image)
    img_np = np.array(original_img)

    # Process the image through RGB channels
    img_r = image_filter_conv2d(img0=img_np[:, :, 0],
                                h=kernel)
    img_g = image_filter_conv2d(img0=img_np[:, :, 1],
                                h=kernel)
    img_b = image_filter_conv2d(img0=img_np[:, :, 2],
                                h=kernel)

    # Stack three channels to get the filtered image and store it in the output directory
    filtered_image = cv2.merge([img_r, img_g, img_b])
    cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, args.filter), filtered_image)

    # Display the result and comparison between two images.
    # The code below should be executed with GUI.

    # filtered_image = cv2.imread("%s/Lenna-%s.jpg" % (args.output, args.filter))
    # cv2.imshow('input_image', original_img)
    # cv2.imshow('filtered_image', filtered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__=='__main__':
    main()
