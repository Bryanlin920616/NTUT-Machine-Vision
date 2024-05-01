import cv2
import numpy as np
from typing import Tuple
from copy import deepcopy

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

    @staticmethod
    def convert_greyscale_to_binary(grey_img, threshold=128):
        height, width = grey_img.shape
        binary_img = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                if grey_img[i, j] <= threshold:
                    binary_img[i, j] = 255 # white
                else:
                    binary_img[i, j] = 0 # black

        return binary_img

    @staticmethod
    def convert_color_to_index_color(src_img, color_map):
        height, width, _ = src_img.shape
        index_color_img = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                pixel = src_img[i, j] #取出像素
                min_dist = float('inf')
                min_index = -1
                for idx, color in enumerate(color_map): #找出color map上最接近的color
                    dist = ((pixel[0]-color[0])**2 + (pixel[1]-color[1])**2 + (pixel[2]-color[2])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        min_index = idx
                index_color_img[i, j] = color_map[min_index]

        return index_color_img




def distance_transform(image, connect):
    distance = np.full(image.shape, np.inf)  # Initialize distance array with infinity
    
    # First pass: Check from top-left to bottom-right
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0:
                distance[i, j] = 0
            else:
                min_dist = distance[i, j]  # Start with the current distance (infinity)
                # Check top and left neighbors (including diagonal)
                if i > 0:
                    min_dist = min(min_dist, distance[i-1, j] + 1)  # Top
                    if connect == 8:
                        if j > 0:
                            min_dist = min(min_dist, distance[i-1, j-1]+1)# + np.sqrt(2))  # Top-left
                        if j < image.shape[1] - 1:
                            min_dist = min(min_dist, distance[i-1, j+1]+1)# + np.sqrt(2))  # Top-right
                if j > 0:
                    min_dist = min(min_dist, distance[i, j-1] + 1)  # Left
                distance[i, j] = min_dist

    # Second pass: Check from bottom-right to top-left
    for i in range(image.shape[0]-1, -1, -1):
        for j in range(image.shape[1]-1, -1, -1):
            min_dist = distance[i, j]
            # Check bottom and right neighbors (including diagonal)
            if i < image.shape[0]-1:
                min_dist = min(min_dist, distance[i+1, j] + 1)  # Bottom
                if connect == 8:
                    if j > 0:
                        min_dist = min(min_dist, distance[i+1, j-1]+1)# + np.sqrt(2))  # Bottom-left
                    if j < image.shape[1] - 1:
                        min_dist = min(min_dist, distance[i+1, j+1]+1)# + np.sqrt(2))  # Bottom-right
            if j < image.shape[1]-1:
                min_dist = min(min_dist, distance[i, j+1] + 1)  # Right

            distance[i, j] = min_dist
    return distance



def medial_axis_skeletonization(distance, binary_img):
    ''' Compute the medial axis of the binary object based on the distance transform '''
    skeleton = deepcopy(binary_img)
    max_dist = int(np.max(distance))

    for d in range(1, max_dist+1):
        for i in range(1, distance.shape[0]-1):
            for j in range(1, distance.shape[1]-1):
                if distance[i, j] == d:
                    val = get_eight_neighbor_val(i, j, distance)
                    if i != skeleton.shape[0]-1:
                        grid = skeleton[i-1:i+2, j-1:j+2]
                    else:
                        # Create a 3x3 grid where the last row is filled with zeros if i is at the last row
                        grid = np.vstack((skeleton[i-1:i+1, j-1:j+2], np.zeros((1, 3))))
                            
                    if d < max(val) and not will_break_connectivity(grid):
                        skeleton[i, j] = 0  # Remove the point

    for i in range(1, skeleton.shape[0]): # bottom edge case
        for j in range(1, skeleton.shape[1]-1):
            if skeleton[i][j] != 0:
                if i != skeleton.shape[0]-1:
                    grid = skeleton[i-1:i+2, j-1:j+2]
                else:
                    # Create a 3x3 grid where the last row is filled with zeros if i is at the last row
                    grid = np.vstack((skeleton[i-1:i+1, j-1:j+2], np.zeros((1, 3))))
                if not will_break_connectivity(grid):
                    if (skeleton[i][j+1]) or (skeleton[i][j-1]):
                        skeleton[i][j] = 0

    
    return skeleton

def will_break_connectivity(ske):
    grid = deepcopy(ske)
    grid[1, 1] = 0
    grid[grid > 1] = 1

    # remove redundant branch
    if np.sum(grid) < 2:
        return False
    # two side been separated
    if np.sum(grid[0, :]) != 0 and np.sum(grid[2, :]) != 0  and np.sum(grid[1, :]) == 0:
        return True
    if np.sum(grid[:,0]) != 0 and np.sum(grid[:,2]) != 0  and np.sum(grid[:,1]) == 0:
        return True
    # corner point been separated
    if grid[0, 0] and not grid[1, 0] and not grid[0, 1]:
        return True
    if grid[2, 0] and not grid[1, 0] and not grid[2, 1]:
        return True
    if grid[0, 2] and not grid[0, 1] and not grid[1, 2]:
        return True
    if grid[2, 2] and not grid[2, 1] and not grid[1, 2]:
        return True

    return False


def get_four_neighbor_val(i, j, arr):
    ''' Return four neighbor. '''
    seq = [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]
    val = [arr[s[0], s[1]] for s in seq if 0 <= s[0] < arr.shape[0] and 0 <= s[1] < arr.shape[1]]
    return val 

def get_eight_neighbor_val(i, j, arr):
    ''' Return eight neighbors from top-left clockwisely the center. '''
    seq = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]
    val = [arr[s[0], s[1]] for s in seq if 0 <= s[0] < arr.shape[0] and 0 <= s[1] < arr.shape[1]]
    return val 



# execute program
image_paths = [f'./images/img{i}.jpg' for i in range(1, 5)]

for idx, path in enumerate(image_paths):
    '''get binary img'''
    img = cv2.imread(path)
    grey_img = ImageConverter.convert_color_to_greyscale(img)
    binary_img = ImageConverter.convert_greyscale_to_binary(grey_img, threshold=128)
    cv2.imwrite(f'./binary_img/binary_image_{idx}.png', binary_img)

    '''distance transform'''
    distance4 = distance_transform(binary_img, 4)
    distance8 = distance_transform(binary_img, 8)
    ## save distance transform image
    dist_max = np.max(distance8)
    step = 255/dist_max
    distance_img = np.zeros_like(distance8, dtype=int)
    for i in range(distance8.shape[0]):
        for j in range(distance8.shape[1]):
            if distance8[i, j] == dist_max:
                distance_img[i, j] = 255
            else:
                distance_img[i, j] = int(step*distance8[i, j])
    cv2.imwrite(f'./results/img_{idx+1}_q1-1.png', distance_img)

    '''medial axis skeletonize'''
    skeleton = medial_axis_skeletonization(distance4, binary_img)
    skeleton = np.where(skeleton==0, 0, 255)
    cv2.imwrite(f'./results/img_{idx+1}_q1-2.png', skeleton)
