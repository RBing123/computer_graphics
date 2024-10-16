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

def backward_mapping(image, x, y, a=1, b=1, p=0.5):
    h, w, _ = image.shape
    u1 = int(np.floor(x))
    u2 = min(u1 + 1, w - 1)
    v1 = int(np.floor(y))
    v2 = min(v1 + 1, h - 1)

    if u1 < 0 or v1 < 0 or u2 < 0 or v2 < 0:
        return np.array([0, 0, 0])  # Return black for out-of-bounds

    # Calculate the weights for interpolation
    x_weight = x - u1
    y_weight = y - v1

    # Perform bilinear interpolation
    value = (
        image[v1, u1] * (1 - x_weight) * (1 - y_weight) +
        image[v1, u2] * x_weight * (1 - y_weight) +
        image[v2, u1] * (1 - x_weight) * y_weight +
        image[v2, u2] * x_weight * y_weight
    )
    return value
    
def mapping(cur_point, P1, Q1, P2, Q2, p=0, a=1, b=2):
    # 计算源图像和目标图像的向量
    src_vector = Q1 - P1
    inter_vector = Q2 - P2

    # 计算垂直向量（垂直于源向量）
    src_perpen = np.array([-src_vector[1], src_vector[0]])  # 源图像的垂直向量
    PQ_perpen = np.array([-inter_vector[1], inter_vector[0]])  # 目标图像的垂直向量

    # 计算 PX 和 PQ
    PX = cur_point - P2  # 当前点与目标图像起点的向量
    PQ = inter_vector    # 目标图像的向量
    
    inter_len = np.sqrt(np.sum(PQ**2))  # 目标图像线段的长度

    # 计算 u 和 v
    u = np.inner(PX, PQ) / (inter_len**2)  # u 是在目标图像线段上的投影系数
    v = np.inner(PX, PQ_perpen) / inter_len  # v 是垂直于目标线段的偏移量

    # 根据 u 和 v 计算 Xt（在源图像上的点）
    PQt = src_vector  # 源图像的向量
    src_len = np.sqrt(np.sum(src_vector**2))  # 源图像线段的长度
    xt = P1 + u * PQt + (v * src_perpen) / src_len  # 在源图像中的映射点 xt

    # 计算权重的距离 dist，依赖于 u 的值
    if u < 0:
        dist = np.sqrt(np.sum(np.square(cur_point - P1)))  # 如果 u < 0，计算与起点的距离
    elif u > 1:
        dist = np.sqrt(np.sum(np.square(cur_point - Q1)))  # 如果 u > 1，计算与终点的距离
    else:
        dist = abs(v)  # 如果 0 <= u <= 1，距离为垂直距离

    # 计算权重 weight
    weight = pow((inter_len**p) / (a + dist), b)

    return xt, weight

def calculate_warp_field(img, src_feature_start, src_feature_end, target_feature_start, target_feature_end, a=1, b=1, p=0.5):
    h, w, _ = img.shape
    warp_img = np.empty_like(img)
    
    # Loop over each pixel in the image
    for x in range(w):
        for y in range(h):
            psum = np.array([0, 0], dtype=float)  # Initialize psum as float array
            wsum = 0
            for i in range(len(src_feature_start)):
                # print(f"src_feature_start is {src_feature_start[i]}, target_feature_start is {target_feature_start[i]}, src_feature_end is {src_feature_end[i]}, target_feature_end is {target_feature_end[i]}")
                
                # Get the transformed point and weight for each line
                xt, weight = mapping(np.array([x, y]), src_feature_start[i], src_feature_end[i], target_feature_start[i], target_feature_end[i], p, a, b)
                psum += xt * weight  # Accumulate psum
                wsum += weight  # Accumulate wsum
            
            # Compute the final point after weighting
            point = psum / wsum
            
            # Ensure point is within bounds
            point_x = point[0]
            point_y = point[1]

            # If point[0] or point[1] are arrays, flatten them or take their first element
            if isinstance(point_x, np.ndarray):
                point_x = point_x[0]
            if isinstance(point_y, np.ndarray):
                point_y = point_y[0]

            # Clamping point to image boundaries
            if point_x < 0:
                point_x = 0
            elif point_x >= h:
                point_x = h - 1
            
            if point_y < 0:
                point_y = 0
            elif point_y >= w:
                point_y = w - 1

            # Backward map the point to get the pixel value
            warp_img[y, x] = backward_mapping(img, point_x, point_y)
    
    return warp_img

def blend_images(warped_src, warped_dst, alpha=0.5):
    """
    混合兩張變形後的圖像。
    """
    blended_image = cv2.addWeighted(warped_src, 1 - alpha, warped_dst, alpha, 0)
    return blended_image

def warp(src, dst, P1, Q1, P2, Q2, alpha=0.4):
    assert len(P1)==len(Q1)==len(P2)==len(Q2)
    interpolate = []
   
    # Calculate interpolate point which stores every line with [(start_x, start_y), (end_x, end_y)]
    for i in range(len(P1)):
        interpolate_start = PointInterpolation(P1[i], P2[i], alpha)
        interpolate_end = PointInterpolation(Q1[i], Q2[i], alpha)
        interpolate.append([interpolate_start, interpolate_end])
    # v = backward_mapping(src.img, interpolate[0][0][0], interpolate[0][0][1])
    P1_np = np.array(P1)
    Q1_np = np.array(Q1)
    interpolate_start_np = np.array(interpolate[0][0])
    interpolate_end_np = np.array(interpolate[0][1])
    # print(P1[0], Q1[0], P2[0], Q2[0])
    # interpolate[i] is the i-th line point pair
    # print(P1[0], Q1[0])
    # print(interpolate[0][0], interpolate[0][1], interpolate[1][0])
    inter_start_points = np.array([pair[0] for pair in interpolate])  # 提取起点
    inter_end_points = np.array([pair[1] for pair in interpolate])
    # print(inter_end_points, inter_start_points)
    warped_image = calculate_warp_field(src.img, P1, Q1, inter_start_points, inter_end_points)
    # blend = blend_images(src.img, dst.img, alpha)
    return warped_image
    
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
        cv2.imwrite("res.jpg", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif len(image_data_list)<=0:
        print(f"Error image data size {len(image_data_list)}")
    # else:
    #     warp()