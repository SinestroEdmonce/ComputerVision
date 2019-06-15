import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse as parse


class GaussPyramid:

    def __init__(self, img0, k, kernel_size=3, sigma=1.0, octave=3, layer=5, color='rgb'):
        self.image = img0
        self.k = k
        self.ksize = kernel_size
        self.sigma = sigma
        self.octave = octave
        self.layer = layer
        self.results = []
        self.color = color

        # Size of the original image
        if color == 'rgb':
            self.height, self.width, self.channels = np.shape(img0)
        # For gray only
        else:
            self.height, self.width = np.shape(img0)

        # Generate all matrices
        for idx1 in range(octave):
            self.results.append([])

    def generate_gaussian_pyramid(self):
        """Function that is used to generate Gaussian pyramid.

        :return:
        """

        # The size of the first layer in the first octave should be four times as the original
        # temp_image = cv2.resize(self.image, (self.height*2, self.width*2), cv2.INTER_LINEAR)

        # Apply the Gaussian blur to the first layer in the first octave
        temp_image = cv2.GaussianBlur(self.image, (self.ksize, self.ksize), self.sigma)

        # Generate the figure and grid
        block = 2**(self.octave-1)
        grid = plt.GridSpec(2**self.octave-1, block*self.layer, hspace=0.1)

        # The beginning location of y axis
        y_begin = 0

        for idx1 in range(self.octave):
            if idx1 == 0:
                self.results[idx1].append(temp_image)
            else:
                self.results[idx1].append(cv2.pyrDown(temp_image))

                # For RGB
                if self.color == 'rgb':
                    height, width, channels = np.shape(self.results[idx1][0])
                # For gray only
                else:
                    height, width = np.shape(self.results[idx1][0])
                self.results[idx1][0] = cv2.resize(self.results[idx1][0],
                                                   (int(height/2), int(width/2)),
                                                   cv2.INTER_LINEAR)

            # Obtain the location of the picture
            image_size = int(block/(2**idx1))

            # Add the picture to the subplot
            plt.subplot(grid[y_begin: y_begin+image_size, 0: image_size])
            plt.title('Octave: {}, Layer: {}'.format(idx1+1, 1), fontsize=6, fontweight='bold')
            plt.imshow(self.results[idx1][0])
            plt.axis('off')

            for idx2 in range(1, self.layer):
                blurred_image = cv2.GaussianBlur(self.results[idx1][0],
                                                 (self.ksize, self.ksize),
                                                 (self.k**(idx2-1))*self.sigma)

                self.results[idx1].append(blurred_image)

                # Add the picture to the subplot
                plt.subplot(grid[y_begin: y_begin+image_size, block*idx2: image_size+block*idx2])
                plt.title('Octave: {}, Layer: {}'.format(idx1+1, idx2+1), fontsize=6, fontweight='bold')
                plt.imshow(self.results[idx1][idx2])
                plt.axis('off')

            temp_image = self.results[idx1][self.layer-3]
            y_begin += image_size

        # Save the figure of Gaussian pyramid
        plt.savefig('./../results/gaussian-pyramid.png', dpi=400)
        plt.close('all')

        # Show all the picture
        # plt.show()

    def generate_down_sampling(self):
        """Function that shows down-sampling results

        :return:
        """

        block = 2**(self.octave-1)
        grid = plt.GridSpec(block, 2**self.octave-1)

        # The beginning location of x axis
        x_begin = 0

        for idx in range(self.octave):
            image_size = int(block/(2**idx))
            # Paint the figure
            plt.subplot(grid[block-image_size: block, x_begin: x_begin+image_size])
            plt.title('Octave: {}'.format(idx+1), fontsize=8, fontweight='bold')
            plt.imshow(self.results[idx][0])
            plt.axis('off')

            x_begin += image_size

        # Save the figure of down-sampling
        plt.savefig('./../results/down-sampling.png', dpi=400)
        plt.close('all')

        # Show all the picture
        # plt.show()

    def generate_difference(self):
        reconstruct = cv2.resize(self.results[self.octave-1][self.layer-1],
                                 (self.height, self.width),
                                 cv2.INTER_LINEAR)

        # Gaussian blurred Image
        if self.color == 'gray':
            temp_image = cv2.GaussianBlur(self.image, (self.ksize, self.ksize), self.sigma)
        else:
            # Not apply the Gaussian blurred image, better
            temp_image = self.image

        difference = np.abs(temp_image.astype(np.int) - reconstruct.astype(np.int)).astype(np.uint8)
        gray_image = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY) if self.color == 'rgb' else difference

        # Save the figure of down-sampling
        plt.title('Difference between the original and the last layer in the last octave',
                  fontsize=8, fontweight='bold')
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')
        plt.savefig('./../results/difference-to-origin.png', dpi=400)
        plt.close('all')

        # Show all the picture
        # plt.show()

        return temp_image.astype(np.int) - reconstruct.astype(np.int)


def main():
    """Function that is used to run this single unit program for testing or so.
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
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    gauss_pyramid = GaussPyramid(img0=original_image,
                                 k=args.k_coefficient, kernel_size=args.kernel_size,
                                 sigma=args.sigma, octave=args.octave, layer=args.layer)

    # Generate Gaussian Pyramid
    gauss_pyramid.generate_gaussian_pyramid()

    # Generate down sampling picture
    gauss_pyramid.generate_down_sampling()

    # Generate difference and its picture
    _ = gauss_pyramid.generate_difference()


if __name__ == '__main__':
    main()
