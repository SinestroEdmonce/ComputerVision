import numpy as np
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def calculate_histogram(img0):
    """Function that calculates the amount of every luminancce level in an image

    :param img: A np.ndarray object that store the luminance information of an image
    :return: A np.ndarray object that represents the total times of every color bit's appearance in an image
    """

    img_histogram = np.zeros(256)
    for element in img0.flatten():
        img_histogram[element] += 1

    return img_histogram


def generate_cumulative_func(img_hist):
    """Function that generates cumulative function

    :param img_hist: A np.ndarray object that represents the total times of every luminance level's appearance in an image
    :return: The cumulative function that is treated as a transforming function
    """

    total = sum(img_hist)

    # Normalize the histogram
    cumulative_func = np.zeros(len(img_hist)).astype(np.float64)

    # Calculate cumulative histogram
    cumulative_func[0] = img_hist[0]/total
    for idx in range(1, 256):
        cumulative_func[idx] = img_hist[idx]/total + cumulative_func[idx-1]

    return (cumulative_func*255).astype(int)


def execute_histogram_equalization_YCbCr(img0, flag):
    """Function that carries out histogram equalization

    In this function, we transform the original image into YCbCr pattern to obtain the luminance

    :param img0: A np.ndarray object that represents the original image's luminance
    :param flag: Determine which method of calculating cumulative function should be used
    :return:
    """

    # Transform the colored image to YCbCr pattern
    img_ycbcr = cv2.cvtColor(img0, cv2.COLOR_BGR2YCR_CB)
    img_y, _, _ = cv2.split(img_ycbcr)

    # Carry out histogram equalization
    img_hist = calculate_histogram(img0=img_y)
    if flag[0] == 0:
        cumulative_func = generate_cumulative_func(img_hist=img_hist)
    elif flag[0] == 1:
        cumulative_func = generate_punch_cumulative_func(img_hist=img_hist,
                                                         scale=flag[1])
    elif flag[0] == 2:
        cumulative_func = generate_lambda_cumulative_func(img_hist=img_hist,
                                                          xlambda=flag[1])
    img_ycbcr[:, :, 0] = cumulative_func[img_y]

    return cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)


def execute_histogram_equalization_HSV(img0, flag):
    """Function that carries out histogram equalization

    In this function, we transform the original image into HSV pattern to obtain the luminance

    :param img0: A np.ndarray object that represents the original image's luminance
    :param flag: Determine which method of calculating cumulative function should be used
    :return:
    """

    # Transform the colored image to HSV pattern
    img_hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    _, _, img_v = cv2.split(img_hsv)

    # Carry out histogram equalization
    img_hist = calculate_histogram(img0=img_v)
    if flag[0] == 0:
        cumulative_func = generate_cumulative_func(img_hist=img_hist)
    elif flag[0] == 1:
        cumulative_func = generate_punch_cumulative_func(img_hist=img_hist,
                                                         scale=flag[1])
    elif flag[0] == 2:
        cumulative_func = generate_lambda_cumulative_func(img_hist=img_hist,
                                                          xlambda=flag[1])
    img_hsv[:, :, 2] = cumulative_func[img_v]

    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def execute_histogram_equalization_RGB(img0, flag):
    """Function that carries out histogram equalization

    In this function, we treat the original image as RGB pattern to obtain the luminance.
    Then we separately carry out the histogram equalization on each channel

    :param img0: A np.ndarray object that represents the original image's luminance
    :param flag: Determine which method of calculating cumulative function should be used
    :return:
    """

    # Carry out histogram equalization separately
    for idx in range(0, 3):
        img_hist = calculate_histogram(img0=img0[:, :, idx])
        if flag[0] == 0:
            cumulative_func = generate_cumulative_func(img_hist=img_hist)
        elif flag[0] == 1:
            cumulative_func = generate_punch_cumulative_func(img_hist=img_hist,
                                                             scale=flag[1])
        elif flag[0] == 2:
            cumulative_func = generate_lambda_cumulative_func(img_hist=img_hist,
                                                              xlambda=flag[1])
        img0[:, :, idx] = cumulative_func[img0[:, :, idx]]

    return img0


def generate_punch_cumulative_func(img_hist, scale):
    """Function that calculates the cumulative function for 'punch'

    :param img_hist: A np.ndarray object that represents the total times of every luminance level's appearance in an image
    :param scale: A certain fraction of pixels
    :return:
    """

    total = sum(img_hist)
    scale = scale if scale < 0.5 else 1-scale
    white_threshold, black_threshold = 0, 0

    cumulative_func = np.zeros(len(img_hist)).astype(np.float64)

    # Calculate cumulative histogram
    cumulative_func[0] = img_hist[0] / total
    for idx in range(1, 256):
        cumulative_func[idx] = img_hist[idx] / total + cumulative_func[idx - 1]
        # Ensure that a certain fraction of pixels (say, 5%) are mapped to pure black and white
        if cumulative_func[idx] >= scale/2 and black_threshold == 0:
            black_threshold = idx
        if cumulative_func[idx] >= 1-scale/2 and white_threshold == 0:
            white_threshold = idx

    cumulative_func[range(0, black_threshold)] = 0
    cumulative_func[range(white_threshold, 256)] = 1

    return (cumulative_func * 255).astype(int)


def generate_lambda_cumulative_func(img_hist, xlambda):
    """Function that calculates the cumulative function for requirements that F(i) < lambda*i

    :param img_hist: A np.ndarray object that represents the total times of every luminance level's appearance in an image
    :param xlambda: A certain lambda value
    :return:
    """

    total = sum(img_hist)

    cumulative_func = np.zeros(len(img_hist)).astype(np.float64)

    # Calculate cumulative histogram
    cumulative_func[0] = img_hist[0] / total
    for idx in range(1, 256):
        cumulative_func[idx] = 255*img_hist[idx]/total + cumulative_func[idx - 1]
        # Force that cumulative_func[idx] < lambda*idx is always obeyed
        if cumulative_func[idx] >= xlambda*idx:
            cumulative_func[idx] = xlambda*idx

    return cumulative_func.astype(int)


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
    parser.add_argument('--method4_hist_eq', '-m4he', help='Methods to carry out histogram equalization',
                        default="ycbcr", type=str)
    parser.add_argument('--method4cdf', '-m4cdf', help='Methods to generate cumulative function',
                        default="punch", type=str)
    parser.add_argument('--alpha', '-a', help='Parameter for punch cumulative function',
                        default=0.4, type=float)
    parser.add_argument('--xlambda', '-l', help='Parameter for lambda cumulative function',
                        default=0.5, type=float)
    parser.add_argument('--manual', '-m', help='Guidance for usage')
    args = parser.parse_args()

    if args.manual:
        print("Use -i to determine the path to your image.\n"
              "Use -o to determine the path to your output.\n"
              "Uee -m4he to determine the method to carry out histogram equalization.\n"
              "Uee -m4cdf to determine the method to generate cumulative function.\n"
              "Uee -a to determine the value of alpha in punch cumulative function.\n"
              "Uee -l to determine the value of lambda in lambda cumulative function.\n"
              "Uee -m to refer to the guidance for this program\n")
        return

    # Show the arguments
    print("resource path: " + args.image + " , "
          + "output path: " + args.output + " , "
          + "method: " + args.method4_hist_eq + "_" + args.method4cdf)
    # Obtain the kernel for the convolution

    # Obtain the image to be processed
    original_img = cv2.imread(args.image)
    img_np = np.array(original_img)

    # Deal with the method for CDF
    # Process the image
    if args.method4_hist_eq == "ycbcr":
        if args.method4cdf == "lambda":
            hist_eq_image = execute_histogram_equalization_YCbCr(img_np, [2, args.xlambda])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "ycbcr"+"_"+"lambda"+str(args.xlambda)), hist_eq_image)
        elif args.method4cdf == "punch":
            hist_eq_image = execute_histogram_equalization_YCbCr(img_np, [1, args.alpha])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "ycbcr"+"_"+"punch"+str(args.alpha)), hist_eq_image)
        else:
            hist_eq_image = execute_histogram_equalization_YCbCr(img_np, [0, 0])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "ycbcr"+"_"+"common"), hist_eq_image)
    elif args.method4_hist_eq == "hsv":
        if args.method4cdf == "lambda":
            hist_eq_image = execute_histogram_equalization_HSV(img_np, [2, args.xlambda])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "hsv"+"_"+"lambda"+str(args.xlambda)), hist_eq_image)
        elif args.method4cdf == "punch":
            hist_eq_image = execute_histogram_equalization_HSV(img_np, [1, args.alpha])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "hsv"+"_"+"punch"+str(args.alpha)), hist_eq_image)
        else:
            hist_eq_image = execute_histogram_equalization_HSV(img_np, [0, 0])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "hsv"+"_"+"common"), hist_eq_image)
    else:
        if args.method4cdf == "lambda":
            hist_eq_image = execute_histogram_equalization_RGB(img_np, [2, args.xlambda])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "rgb"+"_"+"lambda"+str(args.xlambda)), hist_eq_image)
        elif args.method4cdf == "punch":
            hist_eq_image = execute_histogram_equalization_RGB(img_np, [1, args.alpha])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "rgb"+"_"+"punch"+str(args.alpha)), hist_eq_image)
        else:
            hist_eq_image = execute_histogram_equalization_RGB(img_np, [0, 0])
            cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, "rgb"+"_"+"common"), hist_eq_image)

    # Display the result and comparison between two images.
    # The code below should be executed with GUI.

    # hist_eq_image = cv2.imread("%s/Lenna-%s.jpg" % (args.output, "rgb"+"_"+"common" ))
    # cv2.imshow('input_image', original_img)
    # cv2.imshow('hist_eq_image', hist_eq_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__=='__main__':
    main()
