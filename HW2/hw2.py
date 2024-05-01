import cv2
import numpy as np
from typing import Tuple

# Preparation of needed tools
class ImageConverter:
    @staticmethod
    def convert_color_to_greyscale(src_img):
        height, width, _ = src_img.shape
        grey_img = np.zeros((height, width), dtype=np.uint8)

        # convert every pixel to grey
        for i in range(height):
            for j in range(width):
                greyLevel = int(0.3 * src_img[i, j, 0] + 0.59 * src_img[i, j, 1] + 0.11 * src_img[i, j, 2])
                grey_img[i, j] = greyLevel

        return grey_img

    @staticmethod
    def convert_greyscale_to_binary(grey_img, threshold=128):
        height, width = grey_img.shape
        binary_img = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                if grey_img[i, j] >= threshold:
                    binary_img[i, j] = 255
                else:
                    binary_img[i, j] = 0

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

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x
    def get_parent(self):
        return self.parent


def process_img(binary_img, connectivity):
    h, w = binary_img.shape
    marks = np.zeros((h, w), dtype=int)  # Initialize labels matrix
    # use union find the record the same label
    uf = UnionFind(10000) # initialize with size 10000
    color_map = dict() # store the (label: RGB) pair
    color_img = np.zeros((h, w, 3))
    current_label = 0  # Start labeling from 1

    def generate_random_color(existing_colors):
        while True:
            color = np.random.randint(1, 256, size=3)  # Generate random RGB color except of background color RGB(0, 0, 0)
            color_list = color.tolist()
            if color_list not in existing_colors:
                return color_list

    def get_4neighbors_marks(j, i)-> Tuple[int]:
        if i > 0 and j > 0:
            left = marks[j, i-1]
            top = marks[j-1, i]
        elif i > 0 and j == 0:
            left = marks[j, i-1]
            top = 0
        elif i == 0 and j > 0:
            left = 0
            top = marks[j-1, i]
        else:
            left = top = 0
        return top, left
    def get_8neighbors_marks(j, i)-> Tuple[int]:
        if i > 0 and j > 0:
            left = marks[j, i-1]
            left_top = marks[j-1, i-1]
            top = marks[j-1, i]
            right_top = marks[j-1, i+1] if i < len(marks[0])-1 else 0

        elif i > 0 and j == 0:
            left = marks[j, i-1]
            top = left_top = right_top = 0
        elif i == 0 and j > 0:
            left = left_top = 0
            top = marks[j-1, i]
            right_top = marks[j-1, i+1] if i < len(marks[0])-1 else 0
        else:
            left = top = left_top = right_top = 0

                    
        return left, left_top, top, right_top



    if connectivity == '4':
        for j in range(h):
            for i in range(w):
                # print(binary_img[j, i])
                if binary_img[j, i] == 0: # foreground: black
                    top, left = get_4neighbors_marks(j, i)
                    if top == 0 and left == 0:
                        current_label += 1
                        marks[j, i] = current_label
                    elif top and not left:
                        marks[j, i] = top
                    elif not top and left:
                        marks[j, i] = left
                    else: #top and left both labeled
                        if top == left:
                            marks[j, i] = top
                        else:
                            marks[j, i] = top
                            uf.union(top, left)
    elif connectivity == '8':
        for j in range(h):
            for i in range(w):
                if binary_img[j, i] == 0: # foreground: black
                    left, top_left, top, right_top = get_8neighbors_marks(j, i)
                    if left == 0 and top_left == 0 and top == 0 and right_top == 0:
                        current_label += 1
                        marks[j, i] = current_label
                    else:
                        neighbor_labels = [left, top_left, top, right_top]
                        neighbor_labels = [label for label in neighbor_labels if label != 0]
                        if not neighbor_labels:
                            current_label += 1
                            marks[j, i] = current_label
                        else:
                            min_label = min(neighbor_labels)
                            marks[j, i] = min_label
                            for label in neighbor_labels:
                                uf.union(min_label, label)
    else:
        raise TypeError
    
    for j in range(h):
        for i in range(w):
            if binary_img[j, i] == 0:
                label = uf.find(marks[j, i])
                if label not in color_map:
                    new_color = generate_random_color(color_map.values())
                    color_map[label] = new_color
                color_img[j, i] = color_map[label]
    # print(current_label)
    return color_img

# Start doing the task      
image_paths = [f'./images/img{i}.png' for i in range(1, 5)]
binary_images = []
thresholds = [120, 190, 210, 210]

i = 1
for path in image_paths:
    img = cv2.imread(path)
    grey_img = ImageConverter.convert_color_to_greyscale(img)
    binary_img = ImageConverter.convert_greyscale_to_binary(grey_img, threshold=thresholds[i-1])
    # cv2.imwrite(f'./binary_img/binary_image_{i}.jpg', binary_img)
    binary_images.append(binary_img)
    i += 1

for i, binary_img in enumerate(binary_images):
    label_img = process_img(binary_img, connectivity='4')
    cv2.imwrite(f'./results/img{i+1}_4.jpg', label_img)
    label_img = process_img(binary_img, connectivity='8')
    cv2.imwrite(f'./results/img{i+1}_8.jpg', label_img)

