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
        self.vector = [] # record the vector
        self.vector_length = [] # store vector length
        self.perp = []  # store perpendicular vector
        
def Perpendicular(start_point, end_point, param):
    vector = end_point-start_point
    param.vector.append(vector)
    v_length = np.sqrt(np.sum(vector**2))
    param.vector_length.append(v_length)
    
    perp = np.empty_like(vector)
    perp[0] = -vector[1]
    perp[1] = vector[0]
    param.perp.append(perp)
    
def normalize(v):
    v = np.array(v)
    return v/np.linalg.norm(v)

def PointInterpolation(v1, v2, ratio=0.5):
    """
        Calculate interpolated image with different ratio
        v1 is t=0 vector, v1 is [x1, y1]
        v2 is t=1 vector, v2 is [x2, y2]
    """
    x = (1 - ratio) * v1[0] + ratio * v2[0]
    y = (1 - ratio) * v1[1] + ratio * v2[1]

    return (x, y)

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

def process_img(image_path, image_title):
    img = cv2.imread(image_path)  # Read the image file
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    img_data = ImageData(img, image_title)  # Create an ImageData object
    image_data_list.append(img_data)
    # Start a new thread to show the image
    threading.Thread(target=show_image, args=(img_data,)).start()

def bilinear_interpolate(image, x, y):
    h, w, _ = image.shape
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)

    if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
        return np.array([0, 0, 0])  # Return black for out-of-bounds

    # Calculate the weights for interpolation
    x_weight = x - x0
    y_weight = y - y0

    # Perform bilinear interpolation
    value = (
        image[y0, x0] * (1 - x_weight) * (1 - y_weight) +
        image[y0, x1] * x_weight * (1 - y_weight) +
        image[y1, x0] * (1 - x_weight) * y_weight +
        image[y1, x1] * x_weight * y_weight
    )

    return value
    
def calculate_warp_field(img_s, P_s, Q_s, P_d, Q_d):
    eps = 1e-8
    a = 1
    p = 0.5
    b = 1
    if P_s.ndim == 1:
        P_s = np.reshape(P_s, (1, 2))  # 将其变为 (1, 2)
    elif P_s.ndim != 2 or P_s.shape[1] != 2:
        raise ValueError("P_s 的形状不符合预期。")

    if Q_s.ndim == 1:
        Q_s = np.reshape(Q_s, (1, 2))  # 将其变为 (1, 2)
    elif Q_s.ndim != 2 or Q_s.shape[1] != 2:
        raise ValueError("Q_s 的形状不符合预期。")
    
    perp_d = np.array([-P_d[1], P_d[0]])  # 确保是方向正确
    perp_s = np.array([-P_s[1], P_s[0]])  # 同上

    # 确保 P_d 和 Q_d 是正确的形状
    dest_line_vec = Q_d - P_d
    source_line_vec = Q_s - P_s

    image_size = img_s.shape[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    X_d = np.dstack([x, y]).reshape(-1, 1, 2)  # 变形后的目标坐标

    to_p_vec = X_d - P_d
    to_q_vec = X_d - Q_d

    # 计算 u 和 v
    u = np.sum(to_p_vec * dest_line_vec, axis=-1) / (np.sum(dest_line_vec**2) + eps)  # 注意这里不再使用 axis=1
    v = np.sum(to_p_vec * perp_d, axis=-1) / (np.sqrt(np.sum(dest_line_vec**2)) + eps)  # 同样处理

    # Adjust dimensions for broadcasting
    u = np.expand_dims(u, -1)  # Shape: (N, 1)
    v = np.expand_dims(v, -1)  # Shape: (N, 1)

    # 确保 source_line_vec 和 P_s 的形状兼容
    source_line_vec = np.expand_dims(source_line_vec, 0)  # Shape: (1, 2)
    P_s = np.reshape(P_s, (1, 2))  # Shape: (1, 2)，假设 P_s 是 (2,) 形状的点

    X_s = np.expand_dims(P_s, 0) + \
        u * source_line_vec + \
        v * perp_s / (np.sqrt(np.sum(source_line_vec**2)) + eps)

    D = X_s - X_d

    # 计算权重
    to_p_mask = (u < 0).astype(np.float64)
    to_q_mask = (u > 1).astype(np.float64)
    to_line_mask = np.ones(to_p_mask.shape) - to_p_mask - to_q_mask

    to_p_dist = np.sqrt(np.sum(to_p_vec**2, axis=-1))
    to_q_dist = np.sqrt(np.sum(to_q_vec**2, axis=-1))
    to_line_dist = np.abs(v)

    dist = to_p_dist * to_p_mask + to_q_dist * to_q_mask + to_line_dist * to_line_mask
    dest_line_length = np.sqrt(np.sum(dest_line_vec**2))
    weight = (dest_line_length**p) / (((a + dist)**b) + eps)

    weighted_D = np.sum(D * np.expand_dims(weight, -1), axis=1) / (np.sum(weight, -1, keepdims=True) + eps)

    X_d = X_d.squeeze()
    X_s = X_d + weighted_D

    if len(img_s.shape) == 2:
        warped = map_coordinates(img_s, X_s[:, ::-1].T, mode="nearest")
    else:
        warped = np.zeros((image_size * image_size, img_s.shape[2]))
        for i in range(img_s.shape[2]):
            warped[:, i] = map_coordinates(img_s[:, :, i], X_s[:, ::-1].T, mode="nearest")
    
    warped = warped.reshape(image_size, image_size, -1).squeeze()
    return warped.astype(np.uint8)

def blend_images(warped_src, warped_dst, alpha=0.5):
    """
    混合兩張變形後的圖像。
    """
    blended_image = cv2.addWeighted(warped_src, 1 - alpha, warped_dst, alpha, 0)
    return blended_image

def warp(src, dst, P1, Q1, P2, Q2, alpha=0.4):
    assert len(P1)==len(Q1)==len(P2)==len(Q2)
    interpolate = []
   
    # Calculate interpolate point
    for i in range(len(P1)):
        interpolate_start = PointInterpolation(P1[i], P2[i], alpha)
        interpolate_end = PointInterpolation(Q1[i], Q2[i], alpha)
        interpolate.append([interpolate_start, interpolate_end])
    P1_np = np.array(P1)
    Q1_np = np.array(Q1)
    interpolate_start_np = np.array(interpolate[0][0])
    interpolate_end_np = np.array(interpolate[0][1])
    warped_image = calculate_warp_field(src.img, P1_np, Q1_np, interpolate_start_np, interpolate_end_np)
    return warped_image
    # cv2.imwrite("warped_1.jpg", warped_1)
    # cv2.imwrite("warped_2.jpg", warped_2)
    # merged = warped_1 * alpha + warped_2 * (1 - alpha)
    # merged = merged.astype(np.uint8)
    # return merged
    # 
    
    # check warp field
    # warped_image = cv2.remap(src.img, warp_field[..., 0], warp_field[..., 1], interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Warped Image", warped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
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
        result = warp(image_data_list[0], image_data_list[1], P1, Q1, P2, Q2, alpha)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif len(image_data_list)<=0:
        print(f"Error image data size {len(image_data_list)}")
    # else:
    #     warp()