import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse as parse


def generate_gaussian_2nd_derivative(sigma=1.0):
    """Function that is used to generate 2nd derivative of 1-D Gaussian

    :param sigma: float32, an object that represents sigma in Gaussian, a variance.
    :return:
    """
    x_axis = np.linspace(-5, 5, 200)
    y_axis = -np.exp(-x_axis**2/(2*(sigma**2)))*(1-(x_axis**2)/(sigma**2))/(np.sqrt(2.*np.pi)*sigma**3)

    # Paint all curves on one figure
    plt.figure(num=1, figsize=(10, 10), dpi=60)

    plt.plot(x_axis, y_axis, label='2nd derivative')

    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('2nd Derivative of 1-D Gaussian')
    plt.legend()

    # Save the figure
    plt.savefig('./2nd-derivative.png', dpi=120)

    # Show the figure
    plt.show()


def main():
    """Function that is used to run this single unit program for testing or so.
    Users can just run this module for some specific intentions.

    :return:
    """

    parser = parse.ArgumentParser()
    parser.add_argument('--sigma', '-s', type=float, default=1.0,
                        help='The sigma in Gaussian, a variance [Default: 1.0]')

    args = parser.parse_args()

    # Show the arguments
    print('sigma: {}'.format(str(args.sigma)))

    # Show the result
    generate_gaussian_2nd_derivative(sigma=args.sigma)


if __name__=='__main__':
    main()
