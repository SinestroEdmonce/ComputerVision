import numpy as np
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def apply_low_pass_filter2img(histogram, scale, threshold, method):
    """Fucntion that applies a given filter to the batch histogram

    :param histogram: np.ndarray, an object that stores the batch histogram
    :param scale: int, an object that represents the size of the given filter, plus being odd necessarily,
                        or represents bonus level
    :param threshold: float32, an object that denotes threshold of frequency
    :param method: str, an object the denotes the filter to use
    :return:
    """

    # Obtain the high frequency threshold and average of frequency
    average = sum(histogram)/256
    high_freq_threshold = average*threshold

    # Method "average" that applies an average filter to histogram
    if method == "average":
        # Initialize a container to store the filtered result
        result_hist = np.zeros(256 + 2 * int(scale/2))
        result_hist[int(scale/2):256 + int(scale/2)] = histogram

        for idx in range(int(scale/2), 256+int(scale/2)):
            if result_hist[idx] > high_freq_threshold:
                result_hist[idx] = sum(result_hist[idx-int(scale/2):idx+int(scale/2)])/scale
        return result_hist[int(scale/2):256 + int(scale/2)]
    # Method "bonus" that forces the high frequency to be smaller than the threshold,
    # and deliveries the difference to all
    elif method == "bonus":
        bonus = 0
        for idx in range(256):
            if histogram[idx] > high_freq_threshold:
                bonus += histogram[idx] - scale*average
                histogram[idx] = high_freq_threshold

        bonus = bonus/256
        for idx in range(256):
            histogram[idx] += bonus
        return histogram
    # Method "gaussian" that forces the high frequency to satisfy an gaussian multiplication
    elif method == "gaussian":
        for idx in range(256):
            if histogram[idx] > high_freq_threshold:
                histogram[idx] = histogram[idx]*np.exp((-0.5)*histogram[idx]**2/high_freq_threshold**2)
        return histogram


def calculate_local_histogram(img0: np.ndarray, batch, restrict_hgih_freq=False,
                              scale=3, threshold=3.0, method="average"):
    """Function that calculates the histogram batch by batch

    :param img0: np.ndarray, an object that is used to store the original image
    :param batch: int, an object that represents total batches for the local histogram equalization
    :param restrict_high_freq: bool, an object that determines whether to do the low-pass filtering
    :param scale: int, an object that represents the size of the given filter, plus being odd necessarily,
                        or represents bonus level
    :param method: str, an object that represents filter method
    :param threshold: float32, an object that denotes threshold of frequency
    :return:
    """

    # Initialize some basic variables
    img_height, img_width = img0.shape[0], img0.shape[1]
    batch_height, batch_width = int(img_height/batch), int(img_width/batch)

    batch_histogram = np.zeros((batch, batch, 256))

    for idx1 in range(batch):
        for idx2 in range(batch):
            # Calculate local histogram batch by batch
            x_begin, y_begin = idx1*batch_height, idx2*batch_width
            x_end, y_end = x_begin+batch_height, y_begin+batch_width

            img_batch = img0[x_begin:x_end, y_begin:y_end]
            for pixel in img_batch.flatten():
                batch_histogram[idx1][idx2][pixel] += 1

            # Restrict the high frequency part of the image, acting like a low-pass filter
            if restrict_hgih_freq:
                batch_histogram[idx1][idx2] = apply_low_pass_filter2img(histogram=batch_histogram[idx1][idx2],
                                                                        scale=scale, threshold=threshold,
                                                                        method=method)
    return batch_histogram


def generate_cumulative_func(batch_hist: np.ndarray, batch):
    """Function that generates the cumulative function for each batch

    :param batch_hist: np.ndarray, an object that stores all histograms in every batch
    :param batch: int, an object that denotes size of every batch
    :return:
    """

    # Normalize the histogram
    cumulative_func = np.zeros((batch, batch, 256)).astype(np.float64)

    # Calculate cumulative histogram
    for idx1 in range(batch):
        for idx2 in range(batch):
            total = sum(batch_hist[idx1][idx2])
            cumulative_func[idx1][idx2][0] = batch_hist[idx1][idx2][0]/total
            for idx3 in range(1, 256):
                cumulative_func[idx1][idx2][idx3] = batch_hist[idx1][idx2][idx3]/total + cumulative_func[idx1][idx2][idx3-1]

    return (cumulative_func*255).astype(np.uint8)


def execute_local_histogram_equalization(img0, batch,
                                         restrict=False, scale=3, threshold=3.0, method="average",
                                         luminance_type="ycbcr"):
    """Function that executes local histogram equalization on the given image

    :param img0: np.ndarray, an object that stores the original image
    :param batch: int, the size of batch
    :param restrict: bool, an object that determines whether to do the low-pass filtering
    :param scale: int, an object that represents the size of the given filter, plus being odd necessarily,
                        or represents bonus level
    :param threshold: float32, an object that denotes threshold of frequency
    :param method: str, an object that represents filter method
    :param luminance_type: str, an object that denotes image type to use for extract luminance level
    :return:
    """

    # YCbCr pattern
    if luminance_type == "ycbcr":
        # Transform the colored image to YCbCr pattern
        img_ycbcr = cv2.cvtColor(img0, cv2.COLOR_BGR2YCR_CB)
        img_y, _, _ = cv2.split(img_ycbcr)

        # Obtain the histogram of each batch
        batch_histogram = calculate_local_histogram(img0=img_y, batch=batch,
                                                    restrict_hgih_freq=restrict,
                                                    scale=scale, threshold=threshold, method=method)
        cumulative_func = generate_cumulative_func(batch_hist=batch_histogram, batch=batch)
        img_ycbcr[:, :, 0] = adaptively_batch_histogram_equalization(img0=img_y,
                                                                     cumulative_func=cumulative_func,
                                                                     batch=batch)
        # Execute the histogram equalization just independently in every batch
        # img_ycbcr[:, :, 0] = batch_histogram_equalization(img0=img_y,
        #                                                   cumulative_func=cumulative_func,
        #                                                   batch=batch)
        return cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)
    # RGB pattern
    elif luminance_type == "rgb":
        # Obtain the histogram of each batch
        for idx in range(0, 3):
            batch_histogram = calculate_local_histogram(img0=img0[:, :, idx], batch=batch,
                                                        restrict_hgih_freq=restrict,
                                                        scale=scale, threshold=threshold, method=method)
            cumulative_func = generate_cumulative_func(batch_hist=batch_histogram, batch=batch)
            img0[:, :, idx] = adaptively_batch_histogram_equalization(img0=img0[:, :, idx],
                                                                      cumulative_func=cumulative_func,
                                                                      batch=batch)
            return img0
    # HSV pattern
    elif luminance_type == "hsv":
        # Transform the colored image to HSV pattern
        img_hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
        _, _, img_v = cv2.split(img_hsv)

        # Obtain the histogram of each batch
        batch_histogram = calculate_local_histogram(img0=img_v, batch=batch,
                                                    restrict_hgih_freq=restrict,
                                                    scale=scale, threshold=threshold, method=method)
        cumulative_func = generate_cumulative_func(batch_hist=batch_histogram, batch=batch)
        img_hsv[:, :, 2] = adaptively_batch_histogram_equalization(img0=img_v,
                                                                   cumulative_func=cumulative_func,
                                                                   batch=batch)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def batch_histogram_equalization(img0, cumulative_func, batch):
    """Function that executes histogram equalization independently in every batch
    (A function for comparison)

    :param img0: np.ndarray, an object that stores the original image
    :param cumulative_func: np.ndarray, an object for look-up function:CDF
    :param batch: int, the size of batch
    :return:
    """
    img_height, img_width = img0.shape[0], img0.shape[1]
    batch_height, batch_width = int(img_height / batch), int(img_width / batch)

    for idx1 in range(img_height):
        for idx2 in range(img_width):
            index = img0[idx1][idx2]

            img0[idx1][idx2] = cumulative_func[int(idx1/batch_height)][int(idx2/batch_width)][index]
    return img0


def adaptively_batch_histogram_equalization(img0, cumulative_func, batch):
    """Function that executes adaptively histogram equalization batch by batch

    :param img0: np.ndarray, an object that stores the original image
    :param cumulative_func: np.ndarray, an object for look-up function:CDF
    :param batch: int, the size of batch
    :return:
    """
    img_height, img_width = img0.shape[0], img0.shape[1]
    batch_height, batch_width = int(img_height/batch), int(img_width/batch)

    for idx1 in range(img_height):
        for idx2 in range(img_width):
            index = img0[idx1][idx2]

            # Four corners
            if idx1 <= batch_height/2 and idx2 <= batch_width/2:
                img0[idx1][idx2] = cumulative_func[0][0][index]
            elif idx1 >= (batch-1)*batch_height+batch_height/2 and idx2 <= batch_width/2:
                img0[idx1][idx2] = cumulative_func[-1][0][index]
            elif idx1 <= batch_height and idx2 >= (batch-1)*batch_width+batch_width/2:
                img0[idx1][idx2] = cumulative_func[0][-1][index]
            elif idx1 >= (batch-1)*batch_height+batch_height/2 and idx2 >= (batch-1)*batch_width+batch_width/2:
                img0[idx1][idx2] = cumulative_func[-1][-1][index]
            # Four edges
            elif idx2 <= batch_width/2:
                batch_idx1, batch_idx2 = int((idx1 - batch_height / 2) / batch_height), 0
                # Coefficient for interpolated calculation
                p = (idx1 - (batch_idx1 * batch_height + batch_height / 2)) / batch_height
                img0[idx1][idx2] = p * cumulative_func[batch_idx1][batch_idx2][index] \
                                   + (1 - p) * cumulative_func[batch_idx1 + 1][batch_idx2][index]
            elif idx1 <= batch_height/2:
                batch_idx1, batch_idx2 = 0, int((idx2 - batch_width / 2) / batch_width)
                # Coefficient for interpolated calculation
                p = (idx2 - (batch_idx2 * batch_width + batch_width / 2)) / batch_width
                img0[idx1][idx2] = p * cumulative_func[batch_idx1][batch_idx2][index] \
                                   + (1 - p) * cumulative_func[batch_idx1][batch_idx2 + 1][index]
            elif idx2 >= (batch-1)*batch_width+batch_width/2:
                batch_idx1, batch_idx2 = int((idx1 - batch_height / 2) / batch_height), batch - 1
                # Coefficient for interpolated calculation
                p = (idx1 - (batch_idx1 * batch_height + batch_height / 2)) / batch_height
                img0[idx1][idx2] = p * cumulative_func[batch_idx1][batch_idx2][index] \
                                   + (1 - p) * cumulative_func[batch_idx1 + 1][batch_idx2][index]
            elif idx1 >= (batch-1)*batch_height+batch_height/2:
                batch_idx1, batch_idx2 = batch-1, int((idx2 - batch_width / 2) / batch_width)
                # Coefficient for interpolated calculation
                p = (idx2 - (batch_idx2 * batch_width + batch_width / 2)) / batch_width
                img0[idx1][idx2] = p * cumulative_func[batch_idx1][batch_idx2][index] \
                                   + (1 - p) * cumulative_func[batch_idx1][batch_idx2 + 1][index]
            # Inner area
            else:
                batch_idx1, batch_idx2 = int((idx1 - batch_height / 2) / batch_height), \
                                         int((idx2 - batch_width / 2) / batch_width)
                # Coefficient for interpolated calculation
                s = (idx2 - (batch_idx2 * batch_width + batch_width / 2)) / batch_width
                t = (idx1 - (batch_idx1 * batch_height + batch_height / 2)) / batch_height

                img0[idx1][idx2] = (1 - s) * (1 - t) * cumulative_func[batch_idx1][batch_idx2][index] \
                                   + s * (1 - t) * cumulative_func[batch_idx1 + 1][batch_idx2][index] \
                                   + (1 - s) * t * cumulative_func[batch_idx1][batch_idx2 + 1][index] \
                                   + s * t * cumulative_func[batch_idx1 + 1][batch_idx2 + 1][index]

    return img0


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
    parser.add_argument('--method4filter', '-m4f', help='Methods to generate low pass filter',
                        default="gaussian", type=str)
    parser.add_argument('--scale', '-s', help='Parameter for the size of the given filter or the bonus level',
                        default=3, type=int)
    parser.add_argument('--threshold', '-t', help='Parameter to determine the threshold for frequency',
                        default=3.0, type=float)
    parser.add_argument('--restrict', '-r', action="store_true",
                        help='Parameter to determine whether to do the restriction')
    parser.add_argument('--batch', '-b', help='Parameter for the number of batches', type=int)
    parser.add_argument('--manual', '-m', help='Guidance for usage')
    args = parser.parse_args()

    if args.manual:
        print("Use -i to determine the path to your image.\n"
              "Use -o to determine the path to your output.\n"
              "Uee -m4he to determine the method to extract the luminance level and carry out histogram equalization.\n"
              "Uee -m4f to determine the method to generate low pass filter.\n"
              "Uee -s to determine the size of the given filter or the bonus level.\n"
              "Uee -t to determine the threshold for frequency.\n"
              "Uee -r to determine whether to do the restriction.\n"
              "Uee -b to determine the number of batches.\n"
              "Uee -m to refer to the guidance for this program\n")
        return

    # Obtain the image to be processed
    original_img = cv2.imread(args.image)
    img_np = np.array(original_img)

    # Process the image
    # Store the image
    if args.restrict:
        # Show the arguments
        print("resource path: " + args.image + " , "
              + "output path: " + args.output + " , "
              + "method: " + args.method4_hist_eq + "_" + args.method4filter)

        hist_eq_image = execute_local_histogram_equalization(img0=img_np, batch=args.batch,
                                                             restrict=True,
                                                             scale=args.scale, threshold=args.threshold,
                                                             method=args.method4filter,
                                                             luminance_type=args.method4_hist_eq)
        cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, args.method4_hist_eq + "_" + args.method4filter), hist_eq_image)
    # No restriction for high frequency
    else:
        # Show the arguments
        print("resource path: " + args.image + " , "
              + "output path: " + args.output + " , "
              + "method: " + args.method4_hist_eq)

        hist_eq_image = execute_local_histogram_equalization(img0=img_np, batch=args.batch,
                                                             restrict=False,
                                                             scale=args.scale, threshold=args.threshold,
                                                             method=args.method4filter,
                                                             luminance_type=args.method4_hist_eq)
        cv2.imwrite("%s/Lenna_%s.jpg" % (args.output, args.method4_hist_eq), hist_eq_image)

    print("Histogram equalization succeeds! The result is stored in {}".format(args.output))

    # Display the result and comparison between two images.
    # The code below should be executed with GUI.

    # hist_eq_image = cv2.imread("%s/Lenna-%s.jpg" % (args.output, args.method4_hist_eq))
    # cv2.imshow('input_image', original_img)
    # cv2.imshow('hist_eq_image', hist_eq_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__=='__main__':
    main()
