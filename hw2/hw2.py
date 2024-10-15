import cv2
import numpy as np
import sys
import threading
from scipy.ndimage import map_coordinates
import math
# Initialize variables for storing points
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial positions
image_data_list=[]
# Structure to hold image data and window name
class ImageData:
    def __init__(self, img, window_name):
        self.img = img
        self.featureimg=img # record feature image
        self.window_name = window_name
        self.lines = [] # record points
        self.vector = []
        self.vector_length = []
        self.perp = []
        
def Perpendicular(start_point, end_point, param):
    vector = end_point-start_point
    param.vector.append(vector)
    v_length = np.sqrt(np.sum(vector**2))
    param.vector_length.append(v_length)
    
    # v_length = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    # v_homo = np.pad(v, ((0, 0), (0, 1)), mode="constant") # pad to R3, pad zeros
    # z_axis = np.zeros(v_homo.shape)
    # z_axis[:, -1] = 1
    # p = np.cross(v_homo, z_axis)
    # p = p[:, :-1] # ignore z axis
    # p_length = np.sqrt(np.sum(p**2, axis=1, keepdims=True))
    # p = p / (p_length + 1e-8) # now sum = 1
    # p *= v_length
    # return p

def PointInterpolation(v1, v2, ratio=0.5):
    # v1, v2 : N x 2
    # interpolation of points by finding the midpoint
    return v1 * (1. - ratio) + v2 * ratio

# Mouse callback function for drawing lines
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down
        drawing = True
        ix, iy = x, y  # Store the initial point
    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse move
        if drawing:  # Draw only when the mouse is pressed
            img_copy = param.featureimg.copy()  # Make a copy to avoid drawing permanent lines
            cv2.line(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(param.window_name, img_copy)  # Display the copy
    elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button released
        drawing = False  # End drawing
        # Draw the final line on the original image with an arrow
        cv2.arrowedLine(param.featureimg, (ix, iy), (x, y), (0, 255, 0), 2, tipLength=0.05)
        cv2.imshow(param.window_name, param.featureimg)  # Display the image with the final line

        param.lines.append(((ix, iy), (x, y)))
        # print(f"Line drawn in {param.window_name}: {param.lines}")
    

def show_point(param):
    print(f"Line drawn in {param.window_name}: {param.lines}")

def show_image(img_data):
    global alpha
    cv2.namedWindow(img_data.window_name)  # Create a window
    cv2.setMouseCallback(img_data.window_name, draw_line, param=img_data)  # Set the mouse callback function
    cv2.imshow(img_data.window_name, img_data.img)  # Show the image
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyWindow(img_data.window_name)  # Close the window when done
    cv2.imwrite(f"{img_data.window_name}.jpg",img_data.featureimg) # save the modified image
    # show_point(img_data)

def process_img(image_path, image_title):
    img = cv2.imread(image_path)  # Read the image file
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    img_data = ImageData(img, image_title)  # Create an ImageData object
    image_data_list.append(img_data)
    # Start a new thread to show the image
    threading.Thread(target=show_image, args=(img_data,)).start()

def bilinear(img, point, h, w):
    x, y = point[0], point[1]
    x1, y1 = math.floor(x), math.floor(y)
    x2, y2 = math.ceil(x), math.ceil(y)
    if x2>=h:
        x2 = h-1
    if y2>=w:
        y2 = w-1
    a, b = x-x1, y-y1
    # get the color
    val = (1-a)*(1-b)*img[x1, y1] + a*(1-b) * img[x2, y1] + (1 - a) * b * img[x1, y2] + a * b *img[x2, y2]
    return val
    
def warp(src, dst, P1, Q1, P2, Q2, alpha=0.4):
    assert len(P1)==len(Q1)==len(P2)==len(Q2)
    for i in range(len(P1)):
        Perpendicular(P1[i], Q1[i], src)
    for i in range(len(src.vector)):
        print(f"v{i} is {src.vector[i]}")
        print(f"v{i}_length is {src.vector_length[i]}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python hw2.py <image_path1> <image_path2>")
        sys.exit(1)

    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    process_img(file_path_1, 'Image 1')  # Process the first image
    process_img(file_path_2, 'Image 2')  # Process the second image
    
    threading.active_count()  # This will ensure we wait for all threads to finish
    input("Press Enter after closing the windows...\n")  # Wait for user input after closing the windows
    for img_data in image_data_list:
        show_point(img_data)
    alpha = float(input("enter the alpha value(0~1): \n"))
    if len(image_data_list)==2:
        # Initialize lists to store points
        P1, Q1, P2, Q2 = [], [], [], []
        
        lines1 = image_data_list[0].lines  # Points from the first image
        lines2 = image_data_list[1].lines  # Points from the second image
        # Make sure we have the same number of lines for morphing
        if len(lines1) != len(lines2):
            print("Error: The number of lines in both images must be the same.")
            sys.exit(1)

        for i in range(len(lines1)):
            p1 = lines1[i][0]  # Starting point of line in image 1
            q1 = lines1[i][1]  # Ending point of line in image 1
            p2 = lines2[i][0]  # Starting point of line in image 2
            q2 = lines2[i][1]  # Ending point of line in image 2
            
            P1.append(p1)
            Q1.append(q1)
            P2.append(p2)
            Q2.append(q2)

        # Convert to numpy arrays if necessary
        P1 = np.array(P1)
        Q1 = np.array(Q1)
        P2 = np.array(P2)
        Q2 = np.array(Q2)
        # print("P1 shape:", P1.shape)
        # print("Q1 shape:", Q1.shape)
        # print("P2 shape:", P2.shape)
        # print("Q2 shape:", Q2.shape)
        # Call the warp function with the points
        warp(image_data_list[0], image_data_list[1], P1, Q1, P2, Q2, alpha)
        
    elif len(image_data_list)<=0:
        print(f"Error image data size {len(image_data_list)}")
    # else:
    #     warp()