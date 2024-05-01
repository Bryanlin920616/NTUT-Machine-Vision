import cv2
import numpy as np
import os
import math


# Part 1 definition
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
    def convert_greyscale_to_binary(grey_img):
        height, width = grey_img.shape
        binary_img = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                if grey_img[i, j] >= 128:
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

class HW1_1:
    def __init__(self, src_folder, dest_folder, show=False, save=True):
        self.src_folder = src_folder
        self.dest_folder = dest_folder
        self.img_file_list = os.listdir(src_folder)
        self.show = show
        self.save = save

    def process_images(self):
        all_grey_images = []
        for filename in self.img_file_list:
            src_path = os.path.join(self.src_folder, filename)
            dest_filename = filename.split('.')[0] + '_q1-1.png'
            dest_path = os.path.join(self.dest_folder, dest_filename)

            # 讀取彩色圖像
            img = cv2.imread(src_path)

            # 將彩色圖像轉換為灰階
            grey_img = ImageConverter.convert_color_to_greyscale(img)

            # 顯示灰階圖像
            if self.show:
                cv2.imshow(f'GreyLevel: {filename}', grey_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # 保存灰階圖像
            if self.save:
                cv2.imwrite(dest_path, grey_img)

            all_grey_images.append(grey_img)

        return all_grey_images

    def process_greyscale_images(self, grey_images):
        all_binary_images = []
        for i, grey_img in enumerate(grey_images):
            dest_filename = self.img_file_list[i].split('.')[0] + '_q1-2.png'
            dest_path = os.path.join(self.dest_folder, dest_filename)

            # 將灰階圖像二值化
            binary_img = ImageConverter.convert_greyscale_to_binary(grey_img)

            # 顯示二值化圖像
            if self.show:
                cv2.imshow(f'Binary: {self.img_file_list[i]}', binary_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # 保存二值化圖像
            if self.save:
                cv2.imwrite(dest_path, binary_img)

            all_binary_images.append(binary_img)

        return all_binary_images

    def process_index_color_images(self, color_map):
        all_index_color_images = []
        for filename in self.img_file_list:
            src_path = os.path.join(self.src_folder, filename)
            dest_filename = filename.split('.')[0] + '_q1-3.png'
            dest_path = os.path.join(self.dest_folder, dest_filename)

            # 讀取彩色圖像
            img = cv2.imread(src_path)

            # 將彩色圖像轉換為索引顏色圖像
            index_color_img = ImageConverter.convert_color_to_index_color(img, color_map)

            # 顯示索引顏色圖像
            if self.show:
                cv2.imshow(f'Index Color: {filename}', index_color_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # 保存索引顏色圖像
            if self.save:
                cv2.imwrite(dest_path, index_color_img)

            all_index_color_images.append(index_color_img)

        return all_index_color_images


# Part 2 definition


class ImageResizer:
    @staticmethod
    def resize_without_interpolation(image, scale_factor):
        # 假設scale_factor只能是>1的整數或是能整除總共的pixel
        new_height, new_width = (image.shape)[:2]
        new_height = int(new_height*scale_factor)
        new_width = int(new_width*scale_factor)


        newImg = np.zeros((int(new_height), int(new_width), 3))
        for row in range(len(newImg)):
            for col in range(len(newImg[row])):
                newImg[row, col] = image[int(row/scale_factor), int(col/scale_factor)]

        return newImg
    def resize_with_interpolation(image, scale_factor):
        step = 1/scale_factor
        oldH, oldW = image.shape[:2]
        newH, newW = int(oldH*scale_factor), int(oldW*scale_factor)
        newImg = np.zeros((newH, newW, 3))
        # print(f"old: {oldH, oldW}")
        # print(f"new: {newH, newW}")1
        for i in range(newH):
            for j in range(newW):
                # print('count', i, j)
                x = step*j #horizon
                y = step*i #vertical
                # cal the four vertex to do the interpolation
                # x_floor = math.floor(x)
                # x_ceil = math.ceil(x)
                # y_floor = math.floor(y)
                # y_ceil = math.ceil(y)
                
                x_floor = math.floor(x)
                x_ceil = min(oldW - 1, math.ceil(x))
                y_floor = math.floor(y)
                y_ceil = min(oldH - 1, math.ceil(y))

                # handling diff situation
                if (x_floor == x_ceil) and (y_floor == y_ceil):
                    q = image[int(y), int(x), :]
                elif (x_floor == x_ceil):
                    # keep the original x
                    q1 = image[int(y_floor), int(x), :]
                    q2 = image[int(y_ceil), int(x), :]
                    q = q1*(y_ceil-y) + q2*(y-y_floor)
                elif (y_floor == y_ceil):
                    # keep the original y
                    q1 = image[int(y), int(x_floor), :]
                    try:
                        q2 = image[int(y), int(x_ceil), :]
                    except:
                        print((y_ceil), (x_ceil))
                        print((y), (x))
                        print((y_floor), (x_floor))
                    q = q1*(x_ceil-x) + q2*(x-x_floor)
                else:
                    v1 = image[int(y_floor), int(x_floor), :] # top left
                    v2 = image[int(y_floor), int(x_ceil), :] # top right
                    v3 = image[int(y_ceil), int(x_floor), :] # buttom left
                    v4 = image[int(y_ceil), int(x_ceil), :] # buttom right

                    q1 = v1*(x_ceil-x) + v2*(x-x_floor) # top
                    q2 = v3*(x_ceil-x) + v4*(x-x_floor) # buttom
                    q = q1*(y_ceil-y) + q2*(y-y_floor)
                newImg[i, j, :] = q
        return newImg

            
class HW1_2:
    def __init__(self, src_folder, dest_folder, show=False, save=True):
        self.src_folder = src_folder
        self.dest_folder = dest_folder
        self.img_file_list = os.listdir(src_folder)
        self.show = show
        self.save = save

    def process_resized_image_without_interpolation(self, scale):
        all_images = []
        for filename in self.img_file_list:
            src_path = os.path.join(self.src_folder, filename)
            filePostfix = "_q2-1-half.png" if scale == 0.5 else "_q2-1-double.png"
            dest_filename = filename.split('.')[0] + filePostfix
            dest_path = os.path.join(self.dest_folder, dest_filename)

            # 讀取圖像
            img = cv2.imread(src_path)

            # 將圖像放大或縮小
            resized_img = ImageResizer.resize_without_interpolation(img, scale)

            # 顯示結果圖像
            if self.show:
                cv2.imshow(f'Resized_image : {filename}', resized_img)
                cv2.waitKey(2)
                cv2.destroyAllWindows()

            # 保存圖像
            if self.save:
                cv2.imwrite(dest_path, resized_img)

            all_images.append(resized_img)

        return all_images
    def process_resized_image_with_interpolation(self, scale):
        all_images = []
        for filename in self.img_file_list:
            src_path = os.path.join(self.src_folder, filename)
            filePostfix = "_q2-2-half.png" if scale == 0.5 else "_q2-2-double.png"
            dest_filename = filename.split('.')[0] + filePostfix
            dest_path = os.path.join(self.dest_folder, dest_filename)

            # 讀取圖像
            img = cv2.imread(src_path)

            # 將圖像放大或縮小
            resized_img = ImageResizer.resize_with_interpolation(img, scale)

            # 顯示結果圖像
            if self.show:
                cv2.imshow(f'Resized_image : {filename}', resized_img)
                cv2.waitKey(2)
                cv2.destroyAllWindows()

            # 保存圖像
            if self.save:
                cv2.imwrite(dest_path, resized_img)

            all_images.append(resized_img)
        return all_images



#======================running=====================
src_folder = "./images"
dest_folder = "./result"


# Part 1 : calling function
# 自己定義的顏色映射表，這裡以 BGR 格式表示顏色
color_map = np.array([[0, 0, 255],    # 藍色
                     [0, 255, 0],    # 綠色
                     [255, 0, 0],    # 紅色
                     [0, 255, 255],  # 黃色
                     [255, 0, 255],  # 洋紅
                     [255, 255, 0],  # 青色
                     [0, 0, 128],    # 深藍色
                     [0, 128, 0],    # 深綠色
                     [128, 0, 0],    # 深紅色
                     [128, 128, 0],  # 淺黃色
                     [128, 0, 128],  # 紫色
                     [0, 128, 128],  # 灰綠色
                     [192, 192, 192],# 銀色
                     [128, 128, 128],# 灰色
                     [255, 165, 0],  # 橙色
                     [255, 192, 203] # 粉紅色
                    ], dtype=np.uint8)

converter = HW1_1(src_folder, dest_folder)
grey_images = converter.process_images()
binary_images = converter.process_greyscale_images(grey_images)
index_color_images = converter.process_index_color_images(color_map)

# Part 2 : calling function
# 這邊的resize 只有處理題目需求，寬跟高都是同時縮放
resizer = HW1_2(src_folder, dest_folder)
largeer = resizer.process_resized_image_without_interpolation(2)
smaller = resizer.process_resized_image_without_interpolation(0.5)
largeer2 = resizer.process_resized_image_with_interpolation(2)
smaller2 = resizer.process_resized_image_with_interpolation(0.5)