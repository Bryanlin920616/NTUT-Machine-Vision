import cv2
import numpy as np
import os # create directory
import heapq # Priority queue

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

# dealing mouse event to mark different segment
def interactive_segmentation(image_path, colors):
    '''
    @params
    image_path->List(Str)): a array with src image paths
    colors->List(Tuple(b, g, r)): color map to represent every mark
    @Return
    labels->List(List(int)): 2d array of labels corresponding each pixel
    '''
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image, check the path: {}".format(image_path))

    # Create a window
    cv2.namedWindow('Interactive Segmentation')

    # Label array initialized to 0 (unmarked)
    labels = np.zeros(image.shape[:2], dtype=np.int32)

    # Color for marking, increment to change color for different regions
    current_label = 1
    is_drawing = False  # Variable to check if the mouse is being held down

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_label, labels, is_drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
            cv2.circle(image, (x, y), 3, colors[current_label % len(colors)], -1)
            cv2.circle(labels, (x, y), 3, current_label, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
        cv2.imshow('Interactive Segmentation', image)

    # Set the mouse callback function
    cv2.setMouseCallback('Interactive Segmentation', mouse_callback)

    # Display the image
    while True:
        cv2.imshow('Interactive Segmentation', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Press 'n' to switch to the next label
            current_label += 1
        elif key == ord('s'):  # Press 's' to save the marked image and label matrix
            marked_img_path = os.path.join('marked', os.path.basename(image_path).replace('.jpg', '_marked.png'))
            label_img_path = os.path.join('labels', os.path.basename(image_path).replace('.jpg', '_labels.png'))
            cv2.imwrite(marked_img_path, image)
            # Saving label image in a visible format by converting labels to colors
            label_img_colored = np.zeros_like(image)
            for i in range(1, current_label + 1):
                label_img_colored[labels == i] = colors[i % len(colors)]
            cv2.imwrite(label_img_path, label_img_colored)
            print("Images saved: {} (marked image), {} (label image)".format(marked_img_path, label_img_path))
        elif key == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()
    return labels
# calculate the gradient magnitude
def sobel_filters(image):
    # Define the Sobel operator kernels
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])

    # Extend the image borders to handle edge cases
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')

    # Prepare the output gradient images
    grad_x = np.zeros_like(image, dtype=float)
    grad_y = np.zeros_like(image, dtype=float)

    # Apply the kernels to the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the current region of interest
            region = padded_image[i:i+3, j:j+3]
            
            # Convolve the kernels with the region
            grad_x[i, j] = np.sum(Gx * region)
            grad_y[i, j] = np.sum(Gy * region)

    # Compute the gradient magnitude
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize the gradient to the range 0-255
    gradient = np.clip((gradient / gradient.max()) * 255, 0, 255).astype(np.uint8)

    return gradient, grad_x, grad_y
# doing the region growing(watershed)
def apply_watershed(gray_image, markers):
    # Compute gradient magnitude using Sobel operator

    norm_gradient, grad_x, grad_y = sobel_filters(gray_image)
    
    # Initialize the priority queue
    pq = []
    # Create a labeled image from the markers
    labeled_image = np.copy(markers).astype(np.int32)
    
    # Populate the priority queue with neighbors of marked pixels
    rows, cols = gray_image.shape
    for x in range(rows):
        for y in range(cols):
            if markers[x, y] > 0:  # This is a marked pixel
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and markers[nx, ny] == 0:
                        heapq.heappush(pq, (norm_gradient[nx, ny], nx, ny))
                        markers[nx, ny] = -2  # Mark as visited in the priority queue

    # Watershed algorithm using the priority queue
    while pq:
        priority, x, y = heapq.heappop(pq)
        # Check the labels of the 4-connected neighbors to determine the label of this pixel
        labels = set()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and labeled_image[nx, ny] > 0:
                label = labeled_image[nx, ny]
                labels.add(label)

        # Assign the label with the highest count to the current pixel
        if len(labels)==1:
            labeled_image[x, y] = labels.pop()
        else:
            labeled_image[x, y] = -1

        # Push non-labeled neighbors into the queue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and markers[nx, ny] == 0:
                heapq.heappush(pq, (norm_gradient[nx, ny], nx, ny))
                markers[nx, ny] = -2  # Mark as visited in the priority queue

    return labeled_image

def region_growing_tool(paths, colors):
    # Ensure directories exist, otherwise create it
    os.makedirs('marked', exist_ok=True)
    os.makedirs('labels', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    # os.makedirs('images_gray', exist_ok=True)

    # Process each image
    for path in paths:
        # get gray_level image
        image = cv2.imread(path)
        img_gray_level = ImageConverter.convert_color_to_greyscale(image)
        # cv2.imwrite(os.path.join('images_gray', os.path.basename(path).replace('.jpg', '_gray.png')), img_gray_level)
        # 1-1.mark area want to segment
        img_labels = interactive_segmentation(path, colors)
        # 1-2.Region growing
        img_labels = apply_watershed(img_gray_level, img_labels)

        # mix the color image and label color together
        colored_image = np.zeros((img_labels.shape[0], img_labels.shape[1], 3), dtype=np.uint8)
        for i in range(colored_image.shape[0]):
            for j in range(colored_image.shape[1]):
                mark = img_labels[i, j]
                b, g, r = colors[mark]
                # Blend the original and label color
                colored_image[i, j] = [
                    int(b * 0.3 + image[i, j][0] * 0.7),
                    int(g * 0.3 + image[i, j][1] * 0.7),
                    int(r * 0.3 + image[i, j][2] * 0.7)
                ]
        cv2.imwrite(os.path.join('./results', os.path.basename(path).replace('.jpg', '_q1.png')), colored_image)


if __name__ == '__main__':
    # define colors
    colors = [
        (0, 0, 255),     # Red
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 255, 255),   # Yellow
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (192, 192, 192), # Silver
        (0, 0, 128),     # Maroon
        (0, 128, 128),   # Olive
        (0, 128, 0),     # Dark Green
        (128, 0, 128),   # Purple
        (128, 128, 0),   # Teal
        (128, 0, 0),     # Navy
        (0, 165, 255),   # Orange
        (147, 20, 255),  # Deep Pink
        (0, 0, 0)        # Black for the edge
    ]

    # List of image paths
    image_paths = ['./images/img1.jpg', './images/img2.jpg', './images/img3.jpg']
    region_growing_tool(image_paths, colors)