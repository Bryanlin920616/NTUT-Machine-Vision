import cv2
import numpy as np
import os

# ImageConverter
class ImageConverter:
    @staticmethod
    def convert_color_to_greyscale(src_img):
        height, width, _ = src_img.shape
        grey_img = np.zeros((height, width), dtype=np.uint8)

        # convert every pixel to grey
        for i in range(height):
            for j in range(width):
                b, g, r = src_img[i, j]
                greyLevel = 0.3 * r + 0.59 * g + 0.11 * b
                grey_img[i, j] = greyLevel

        return grey_img

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
    pad_size = kernel_size // 2
    # padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    filtered_image = np.copy(image)
    
    for i in range(pad_size, image.shape[0]-pad_size):
        for j in range(pad_size, image.shape[1]-pad_size):
            region = image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            filtered_pixel = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    filtered_pixel += region[m, n] * kernel[m, n]
            filtered_image[i, j] = filtered_pixel
    
    return filtered_image

def sobel_operator(image):
    """計算 Sobel 梯度"""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h, w = image.shape
    Gx = np.zeros((h, w))
    Gy = np.zeros((h, w))

    for i in range(1, h-1):
        for j in range(1, w-1):
            region = image[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(Kx * region)
            Gy[i, j] = np.sum(Ky * region)

    G = np.sqrt(Gx**2 + Gy**2) # magnitude
    theta = np.arctan2(Gy, Gx) # return radian [-pi, pi]
    
    return G, theta

def non_maximum_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N))
    angle = np.rad2deg(theta) # radian to degree
    angle[angle < 0] += 180 # [-pi, pi] -> [0, 180]

    for i in range(1, M-1):
        for j in range(1, N-1):
             # skip the boundary pixels
            a = angle[i, j]
            
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                # dir: 0
                b = G[i, j-1]
                n = G[i, j+1]
            elif (22.5 <= a < 67.5):
                # dir: 45
                b = G[i-1, j-1]
                n = G[i+1, j+1]
            elif (67.5 <= a < 112.5):
                # dir: 90
                b = G[i-1, j]
                n = G[i+1, j]
            elif (112.5 <= a < 157.5):
                # dir: 135
                b = G[i-1, j+1]
                n = G[i+1, j-1]
            else:
                print(f"angle out of range?= {a}")

            # leave the gradient bigger than the two neighbor on the direction
            if (G[i, j] >= b) and (G[i, j] >= n):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    return Z

def double_threshold(grad_img, low_threshold, high_threshold):
    h, w = grad_img.shape
    result = np.zeros_like(grad_img)

    for i in range(h):
        for j in range(w):
            if grad_img[i, j] >= high_threshold:
                result[i, j] = 255
            elif grad_img[i, j] >= low_threshold:
                result[i, j] = 128

    return result

def edge_tracking_by_hysteresis(img):
    height, width = img.shape
    edge_visited = np.zeros((height, width), dtype=bool)
    
    def depth_first_search(stack):
        while stack:
            y, x = stack.pop()
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    # only do the points in the boundary
                    if 0 <= ny < height and 0 <= nx < width and not edge_visited[ny, nx]:
                        if img[ny, nx] == 128: # link the weaks(visit as well)
                            edge_visited[ny, nx] = True
                            img[ny, nx] = 255
                            stack.append((ny, nx)) # spread
    
    # Link weak edges to strong edges
    for y in range(1, height-1):
        for x in range(1, width-1):
            # If it is strong and not visited, recursive link to other neighbors(weak edges)
            if img[y, x] == 255 and not edge_visited[y, x]:
                edge_visited[y, x] = True
                depth_first_search([(y, x)])
    
    # Suppress remaining weak edges
    for y in range(1, height-1):
        for x in range(1, width-1):
            if img[y, x] == 128:
                img[y, x] = 0
    
    return img

    
def canny_edge_detection(image_path, output_path, i):
    image = cv2.imread(image_path)
    image = ImageConverter.convert_color_to_greyscale(image)

    # Step 1: Noise Reduction
    blurred_image = gaussian_filter(image, 5, 1.1) # Gaussian _filtering
    # cv2.imwrite(f"./temp/img{i+1}_gaussian.jpg", blurred_image)

    # Step 2: Gradient Calculation
    grad_mag, grad_angle = sobel_operator(blurred_image)
    # cv2.imwrite(f"./temp/img{i+1}_grad.jpg", grad_mag.astype(np.uint8))

    # Step 3: Non-maximum Suppression(leave the max gradients)
    non_max_img = non_maximum_suppression(grad_mag, grad_angle)
    # cv2.imwrite(f"./temp/img{i+1}_non-max.jpg", non_max_img)

    # Step 4: Double Threshold
    threshold_img = double_threshold(non_max_img, 60, 170) #60, 170
    # cv2.imwrite(f"./temp/img{i+1}_d-Thres.jpg", threshold_img)

    # Step 5: Edge Tracking by Hysteresis
    final_img = edge_tracking_by_hysteresis(threshold_img)
    
    cv2.imwrite(output_path, final_img)
    return final_img

def main():
    input_folder = './images'
    output_folder = './results'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    i = 0
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.jpg', '_sobel.jpg'))
            canny_edge_detection(input_path, output_path, i)
            i += 1

if __name__ == '__main__':
    main()
