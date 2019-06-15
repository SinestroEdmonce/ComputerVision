import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse as parse


def generate_gaussian_2nd_derivative(x_axis, sigma=1.0):
    """Function that is used to generate 2nd derivative of 1-D Gaussian.

    :param x_axis: [float32], an object that stores the x coordinates.
    :param sigma: float32, an object that represents sigma in Gaussian, a variance.
    :return:
    """
    y_axis = -np.exp(-x_axis**2/(2*(sigma**2)))*(1-(x_axis**2)/(sigma**2))/(np.sqrt(2.*np.pi)*sigma**3)

    return y_axis


def generate_DoG(x_axis, k, sigma=1.0):
    """Function that is used to generate image of DoG.

    :param x_axis: [float32], an object that stores the x coordinates.
    :param k: float32, an object that represents the coefficient of sigma.
    :param sigma: float32, an object that represents sigma in Gaussian, a variance.
    :return:
    """

    # Generate the Normal distribution
    gaussian = np.exp(-x_axis**2/(2*(sigma**2)))/(np.sqrt(2.*np.pi)*sigma)

    # DoG storage
    y_axis = np.zeros((len(k), len(x_axis)))

    # Different values of k
    for idx in range(0, len(k)):
        gaussian_k = np.exp(-x_axis**2/(2.*(k[idx]**2)*(sigma**2)))/(np.sqrt(2.*np.pi)*k[idx]*sigma)
        y_axis[idx, :] = (gaussian_k-gaussian)/(k[idx]*sigma-sigma)

    return y_axis


def compare_DoG_with_2nd_derivative(k, sigma=1.0):
    """Function that is used to compare the DoG with 2nd derivative of 1-D Gaussian.

    :param k: float32, an object that represents the coefficient of sigma.
    :param sigma: float32, an object that represents sigma in Gaussian, a variance.
    :return:
    """

    # Generate the X coordinates
    x_axis = np.linspace(-5, 5, 200)

    # Generate DoG
    y_axis_DoG = generate_DoG(x_axis=x_axis, k=k, sigma=sigma)

    # Generate 2nd derivative
    y_axis_2nd_derivative = generate_gaussian_2nd_derivative(x_axis=x_axis, sigma=sigma)

    # Paint all curves on one figure
    plt.figure(num=1, figsize=(10, 10), dpi=60)

    for idx in range(0, len(k)):
        plt.plot(x_axis, y_axis_DoG[idx, :], label='k={}'.format(str(k[idx])))
    plt.plot(x_axis, y_axis_2nd_derivative, label='2nd derivative', linestyle='--')

    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Comparison between DoG and 2nd Derivative')
    plt.legend()

    # Save the figure
    plt.savefig('./../results/comparison-DoG-2nd_derivative.png', dpi=120)

    # Show the figure
    plt.show()


def main():
    """Function that is used to run this single unit program for testing or so.
    Users can just run this module for some specific intentions.

    :return:
    """

    parser = parse.ArgumentParser()
    parser.add_argument('--k_coefficient', '-k', type=str, default='1.2 1.4 1.6 1.8 2.0',
                        help='The coefficient in DoG, input as a string [Default: 1.2 1.4 1.6 1.8 2.0]')
    parser.add_argument('--sigma', '-s', type=float, default=1.0,
                        help='The sigma in Gaussian, a variance [Default: 1.0]')

    args = parser.parse_args()

    # Show the arguments
    print('k: {}'.format(args.k_coefficient))
    print('sigma: {}'.format(str(args.sigma)))

    # Pre-process the coefficient
    k = [float(coef) for coef in args.k_coefficient.split(' ')]

    # Show the result
    compare_DoG_with_2nd_derivative(k=k, sigma=args.sigma)


if __name__ == '__main__':
    main()



