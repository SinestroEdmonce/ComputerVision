import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse as parse

from assignment_2_q3_1.main4unit import GaussPyramid


class SEdgel:

    def __init__(self, x, y, block):
        self.x = x
        self.y = y
        self.block = block


def apply_edge_detector(diff_image):
    """Function that is used to detect the edge

    :param diff_image:
    :return:
    """

    height, width = np.shape(diff_image)
    diff_image = diff_image.astype(np.float)

    # Store the edgel points
    edgel = []

    # Carry out the detection
    for idx1 in range(height-1):
        for idx2 in range(width-1):
            # Number of zero crossing
            zero_crossing_num = 0

            # Area for detection
            block = np.zeros((2, 2))

            if (diff_image[idx1, idx2]>0) != (diff_image[idx1, idx2+1]>0):
                zero_crossing_num += 1

                x_1 = np.array([idx1, idx2])
                x_2 = np.array([idx1, idx2+1])
                zero_crossing = (x_1 * diff_image[idx1, idx2 + 1] - x_2 * diff_image[idx1, idx2]) / \
                                (diff_image[idx1, idx2 + 1] - diff_image[idx1, idx2])

                block[0, :] = zero_crossing

            if (diff_image[idx1, idx2]>0) != (diff_image[idx1+1, idx2]>0):
                zero_crossing_num += 1

                x_1 = np.array([idx1, idx2])
                x_2 = np.array([idx1+1, idx2])
                zero_crossing = (x_1 * diff_image[idx1 + 1, idx2] - x_2 * diff_image[idx1, idx2]) / \
                                (diff_image[idx1 + 1, idx2] - diff_image[idx1, idx2])

                if zero_crossing_num == 1:
                    block[0, :] = zero_crossing
                else:
                    block[1, :] = zero_crossing

            if (diff_image[idx1, idx2+1]>0) != (diff_image[idx1+1, idx2+1]>0):
                zero_crossing_num += 1

                x_1 = np.array([idx1, idx2+1])
                x_2 = np.array([idx1+1, idx2+1])
                zero_crossing = (x_1 * diff_image[idx1 + 1, idx2 + 1] - x_2 * diff_image[idx1, idx2 + 1]) / \
                                (diff_image[idx1 + 1, idx2 + 1] - diff_image[idx1, idx2 + 1])

                if zero_crossing_num == 1:
                    block[0, :] = zero_crossing
                else:
                    block[1, :] = zero_crossing

            if (diff_image[idx1+1, idx2]>0) != (diff_image[idx1+1, idx2+1]>0):
                zero_crossing_num += 1

                x_1 = np.array([idx1 + 1, idx2])
                x_2 = np.array([idx1 + 1, idx2 + 1])
                zero_crossing = (x_1 * diff_image[idx1 + 1, idx2 + 1] - x_2 * diff_image[idx1 + 1, idx2]) / \
                                (diff_image[idx1 + 1, idx2 + 1] - diff_image[idx1 + 1, idx2])

                if zero_crossing_num == 1:
                    block[0, :] = zero_crossing
                else:
                    block[1, :] = zero_crossing

            # Judge the edge, according to the zero crossing number
            if zero_crossing_num == 2:
                edgel.append(SEdgel(x=idx1, y=idx2, block=block))

    # Generate the result
    result_image = np.zeros((height, width))
    for idx in range(len(edgel)):
        edge_loc1 = np.rint(edgel[idx].block[0, :]).astype(np.int)
        edge_loc2 = np.rint(edgel[idx].block[1, :]).astype(np.int)

        result_image[edge_loc1[0], edge_loc1[1]] = 1
        result_image[edge_loc2[0], edge_loc2[1]] = 1

    return result_image


def main():
    """"Function that is used to run this single unit program for testing or so.
    Users can just run this module for some specific intentions.


    :return:
    """
    parser = parse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default='./../resource/Lenna.png',
                        help='The path to the original image [Default: ./../resource/Lenna.png]')
    parser.add_argument('--k_coefficient', '-k', type=float, default=1.5,
                        help='The coefficient in Gaussian Pyramid [Default: 1.5]')
    parser.add_argument('--sigma', '-s', type=float, default=1.0,
                        help='The sigma in Gaussian, a variance [Default: 1.0]')
    parser.add_argument('--kernel_size', '-ksize', type=int, default=3,
                        help='The size of the kernel in Gaussian [Default: 3]')
    parser.add_argument('--octave', '-o', type=int, default=3,
                        help='The number of the octaves in Gaussian Pyramid [Default: 3]')
    parser.add_argument('--layer', '-l', type=int, default=5,
                        help='The number of the layers in one octave in Gaussian Pyramid [Default: 5]')

    args = parser.parse_args()

    original_image = cv2.imread(args.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    gauss_pyramid = GaussPyramid(img0=original_image,
                                 k=args.k_coefficient, kernel_size=args.kernel_size,
                                 sigma=args.sigma, octave=args.octave, layer=args.layer,
                                 color='gray')

    # Generate the Gaussian Pyramid
    gauss_pyramid.generate_gaussian_pyramid()
    diff_image = gauss_pyramid.generate_difference()

    # Apply edge detection
    result = apply_edge_detector(diff_image=diff_image)

    # Save the figure
    plt.title('Edge Detection by using zero-crossing',
              fontsize=8, fontweight='bold')
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.savefig('./../results/edge-detection.png', dpi=400)
    plt.close('all')


if __name__ == '__main__':
    main()

