from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import datetime
from concurrent.futures import ProcessPoolExecutor
from os import path


smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
sharpenElements = np.array((
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]), dtype="int")
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")
emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")
something = np.array((
    [0.1, 0, 0.1],
    [0, 0.6, 0],
    [0.1, 0, 0.1]), dtype="float")

kernels = (
    ("sharpenElements", sharpenElements),
    ("emboss", emboss),
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("something", something)
)


def filter_all(image, kernel, parts):
    img_size = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]
    pad = (kernel_width - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    with ProcessPoolExecutor() as executor:
        results = [executor.submit(filter_part, image, img_size, pad, kernel, i, parts) for i in range(parts)]
    whole_image = results[0].result()
    for i in range(1, parts):
        whole_image = np.concatenate((whole_image, results[i].result()))

    return whole_image


def filter_part(image, img_size, pad, kernel, part, all_parts):
    (img_height, img_width) = img_size
    div_height = img_height // all_parts
    if part == all_parts-1:
        output = np.zeros((int(img_height // all_parts) + img_height % all_parts, img_width), dtype="float32")
        (part_height, part_width) = (int(part * div_height + pad), int((part + 1) * div_height) +
                                     img_height % all_parts + pad)
    else:
        output = np.zeros((int(img_height // all_parts), img_width), dtype="float32")
        (part_height, part_width) = (int(part * div_height + pad), int((part + 1) * div_height) + pad)
    y_part = 0
    for y in np.arange(part_height, part_width):
        x_part = 0
        for x in np.arange(pad, img_width + pad):
            img_part = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            k = (img_part * kernel).sum()

            output[y_part, x_part] = k
            x_part += 1
        y_part += 1

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


def test_emboss():
    img_name = input("Write image name :")
    upper_range = int(input("How many : "))
    n = 1
    while path.exists(img_name) is False:
        img_name = input("Write correct image name :")
    image = cv2.imread(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    while n <= upper_range:
        print("Emboss for ", n, " processes")
        start = datetime.datetime.now()
        filter_all(gray, emboss, n)
        duration = datetime.datetime.now() - start
        print(duration)
        if n == 1:
            duration_1 = duration
        else:
            print("Faster : ", float(duration_1/duration))
        n *= 2


def test_kernals():
    img_name = input("Write image name :")
    n = int(input("How many processes :"))
    while path.exists(img_name) is False:
        img_name = input("Write correct image name :")
    image = cv2.imread(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for (kernel_name, kernel) in kernels:
        print("Now running ", kernel_name)
        start = datetime.datetime.now()
        convole_output = filter_all(gray, kernel, n)
        duration = datetime.datetime.now() - start
        print(duration)
        cv2.imshow("original", cv2.resize(gray, (800, 800)))
        cv2.imshow("filtered", cv2.resize(convole_output, (800, 800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # test_emboss()
    test_kernals()


if __name__ == "__main__":
    main()
