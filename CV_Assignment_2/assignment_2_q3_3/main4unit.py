import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse as parse
import math


def canny_edge_detection(img0, ksize=5, sigma=1.44, percent=0.8, ratio=0.4):
    """Function that is used to apply Canny operator to the image

    :param img0: np.ndarray, an object that stores the original image
    :param ksize: int, an object that represents the size of the kernel in Gaussian
    :param sigma: float32, an object that represents the sigma in Gaussian, a variance
    :param percent: float32, an object that determine the high threshold
    :param ratio: float32, an object that determine the low threshold
    :return:
    """

    height, width = np.shape(img0)

    # Gaussian blur
    image = cv2.GaussianBlur(img0, (ksize, ksize), sigma)

    # Generate the image gradient and the gradient direction
    image_gradient = np.zeros((height, width))
    gradient_direction = np.zeros((height, width))

    # Generate the image gradient along the two axes, by using Sobel operator
    x_gradient = cv2.Sobel(image, -1, 1, 0)
    x_gradient = x_gradient.astype(np.uint8)

    y_gradient = cv2.Sobel(image, -1, 0, 1)
    y_gradient = y_gradient.astype(np.uint8)

    # Obtain the gradient direction
    for idx1 in range(height):
        for idx2 in range(width):
            image_gradient[idx1, idx2] = np.sqrt(x_gradient[idx1, idx2]**2+y_gradient[idx1, idx2]**2)
            theta = math.atan(y_gradient[idx1, idx2]/(x_gradient[idx1, idx2]))*180/math.pi + 90

            if 0 <= theta < 45:
                gradient_direction[idx1, idx2] = 2
            elif 45 <= theta < 90:
                gradient_direction[idx1, idx2] = 3
            elif 90 <= theta < 135:
                gradient_direction[idx1, idx2] = 0
            else:
                gradient_direction[idx1, idx2] = 1

    # Normalize
    matrix_max = np.max(image_gradient)
    image_gradient = image_gradient/matrix_max

    # Determine the threshold
    high_threshold = percent * np.max(image_gradient)
    low_threshold = ratio * high_threshold

    # Adjust the result, according to the high and low threshold
    gradient_nms_adjusted = np.zeros((height, width))
    result_image = np.zeros((height, width))

    # Interpolate to do non-maximum suppression
    for idx1 in range(1, height-1):
        for idx2 in range(1, width-1):
            east = image_gradient[idx1, idx2 + 1]
            south = image_gradient[idx1 + 1, idx2]
            west = image_gradient[idx1, idx2 - 1]
            north = image_gradient[idx1 - 1, idx2]
            north_east = image_gradient[idx1 - 1, idx2 + 1]
            north_west = image_gradient[idx1 - 1, idx2 - 1]
            south_west = image_gradient[idx1 + 1, idx2 - 1]
            south_east = image_gradient[idx1 + 1, idx2 + 1]

            # The real value of image gradient
            gradient_value, g1, g2 = image_gradient[idx1, idx2], 0, 0

            if gradient_direction[idx1, idx2] == 0:
                proportion = np.fabs(y_gradient[idx1, idx2] / x_gradient[idx1, idx2])
                g1 = east * (1 - proportion) + north_east * proportion
                g2 = west * (1 - proportion) + south_west * proportion
            elif gradient_direction[idx1, idx2] == 1:
                proportion = np.fabs(x_gradient[idx1, idx2] / y_gradient[idx1, idx2])
                g1 = north * (1 - proportion) + north_east * proportion
                g2 = south * (1 - proportion) + south_west * proportion
            elif gradient_direction[idx1, idx2] == 2:
                proportion = np.fabs(x_gradient[idx1, idx2] / y_gradient[idx1, idx2])
                g1 = north * (1 - proportion) + north_west * proportion
                g2 = south * (1 - proportion) + south_east * proportion
            elif gradient_direction[idx1, idx2] == 3:
                proportion = np.fabs(y_gradient[idx1, idx2] / x_gradient[idx1, idx2])
                g1 = west * (1 - proportion) + north_west * proportion
                g2 = east * (1 - proportion) + south_east * proportion

            # Judge whether it is possible to be an edge point
            if gradient_value >= g1 and gradient_value >= g2:
                gradient_nms_adjusted[idx1, idx2] = gradient_value
            else:
                gradient_nms_adjusted[idx1, idx2] = low_threshold

    # Double threshold detection
    for idx1 in range(1, height - 1):
        for idx2 in range(1, width - 1):
            # Selection by threshold
            if gradient_nms_adjusted[idx1, idx2] >= high_threshold:
                result_image[idx1, idx2] = 1
            elif gradient_nms_adjusted[idx1, idx2] <= low_threshold:
                result_image[idx1, idx2] = 0

    for idx1 in range(1, height - 1):
        for idx2 in range(1, width - 1):
            # Connection
            if low_threshold < gradient_nms_adjusted[idx1, idx2] < high_threshold:
                if (gradient_nms_adjusted[idx1 - 1, idx2 - 1: idx2 + 1] >= high_threshold).any() \
                        or (gradient_nms_adjusted[idx1 + 1, idx2 - 1: idx2 + 1] >= high_threshold).any() \
                        or (gradient_nms_adjusted[idx1, idx2 - 1: idx2 + 1] >= high_threshold).any():
                    result_image[idx1, idx2] = 1
                else:
                    result_image[idx1, idx2] = 0

    return result_image


def main():
    """"Function that is used to run this single unit program for testing or so.
    Users can just run this module for some specific intentions.


    :return:
    """

    parser = parse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default='./../resource/Lenna.png',
                        help='The path to the original image [Default: ./../resource/Lenna.png]')
    parser.add_argument('--sigma', '-s', type=float, default=1.2,
                        help='The sigma in Gaussian, a variance [Default: 1.2]')
    parser.add_argument('--kernel_size', '-ksize', type=int, default=3,
                        help='The size of the kernel in Gaussian [Default: 3]')
    parser.add_argument('--percent', '-p', type=float, default=0.1,
                        help='Determine the high threshold [Default: 0.8]')
    parser.add_argument('--ratio', '-l', type=float, default=0.05,
                        help='Determine the low threshold [Default: 0.4]')

    args = parser.parse_args()

    original_image = cv2.imread(args.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Generate the Canny edge detection
    result = canny_edge_detection(img0=original_image, ksize=args.kernel_size,
                                  percent=args.percent, ratio=args.ratio)

    # Actual Canny operator
    # result = cv2.Canny(original_image, 60, 180)

    # Save the figure
    plt.title('Edge Detection by using Canny operator',
              fontsize=8, fontweight='bold')
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.savefig('./../results/canny-edge-detection.png', dpi=400)
    plt.close('all')


if __name__ == '__main__':
    main()
