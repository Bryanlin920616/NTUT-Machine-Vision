import cv2
import numpy as np
import os


def mean_filter(image, kernel_size):
    '''
    @param: kernel_size must be odd number
    '''
    # zero padding
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    
    # init filtered_image
    filtered_image = np.zeros_like(image)
    
    # 濾波
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # extract the region
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # cal the mean of the region
            region_sum = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    region_sum += region[m, n]
            # assign value
            filtered_image[i, j] = region_sum / (kernel_size * kernel_size)
    
    return filtered_image

def median_filter(image, kernel_size):
    '''
    @param: kernel_size must be odd number
    '''
    # zero padding
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    
    # init filtered_image
    filtered_image = np.zeros_like(image)
    
    # 濾波
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # extract the region
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # cal the mean of the region
            # arr = [region[m, n] for m in range(kernel_size) for n in range(kernel_size)]
            arr = []
            for m in range(kernel_size):
                for n in range(kernel_size):
                    arr.append(region[m, n])
            # assign value
            med = int((kernel_size*kernel_size-1)/2)
            filtered_image[i, j] = sorted(arr)[med]
    
    return filtered_image

def gaussian_kernel(size, sigma):
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    
    s = 0
    # construct kernel
    for x in range(size):
        for y in range(size):
            x_distance = (x - k) ** 2
            y_distance = (y - k) ** 2 # shift the idx: from [0-(size-1) to -k ] -> (-k to k)
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x_distance + y_distance) / (2 * sigma**2))
            s += kernel[x, y]
    # normalization
    kernel /= s

    return kernel

def gaussian_filter(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    # print("Doing gaussian filter with kernel:")
    # print(kernel)
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_pixel = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    filtered_pixel += region[m, n] * kernel[m, n]
            filtered_image[i, j] = filtered_pixel
    
    return filtered_image


print(gaussian_kernel(5, 1))
#'''
os.makedirs('results', exist_ok=True)
os.makedirs('combination', exist_ok=True)
# List of image paths
image_paths = ['./images/img1.jpg', './images/img2.jpg', './images/img3.jpg']
for idx, p in enumerate(image_paths):
    # read image
    image = cv2.imread(p)[:,:,0]
    ## mean filter
    mean_filtered_image_3 = mean_filter(image, 3)
    cv2.imwrite(f'./results/img{idx+1}_q1_3.jpg', mean_filtered_image_3)
    mean_filtered_image_7 = mean_filter(image, 7)
    cv2.imwrite(f'./results/img{idx+1}_q1_7.jpg', mean_filtered_image_7)
    ## median filter
    median_filter_image_3 = median_filter(image, 3)
    cv2.imwrite(f'./results/img{idx+1}_q2_3.jpg', median_filter_image_3)
    median_filter_image_7 = median_filter(image, 7)
    cv2.imwrite(f'./results/img{idx+1}_q2_7.jpg', median_filter_image_7)
    ## gaussian filter
    gaussian_filter_image_5 = gaussian_filter(image, kernel_size=5, sigma=1)
    cv2.imwrite(f'./results/img{idx+1}_q3.jpg', gaussian_filter_image_5)

    mean_after_median_3 = mean_filter(median_filter_image_3, 3)
    cv2.imwrite(f'./combination/img{idx+1}_MedNMean_3.jpg', mean_after_median_3)
#'''
