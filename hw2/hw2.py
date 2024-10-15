import cv2
import numpy as np
import sys
import threading
from scipy.ndimage import map_coordinates
# Initialize variables for storing points
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial positions
image_data_list=[]
# Structure to hold image data and window name
class ImageData:
    def __init__(self, img, window_name):
        self.img = img
        self.window_name = window_name
        self.lines = [] # record points
        self.featureimg=img
        
def Perpendicular(v):
    v_length = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    v_homo = np.pad(v, ((0, 0), (0, 1)), mode="constant") # pad to R3, pad zeros
    z_axis = np.zeros(v_homo.shape)
    z_axis[:, -1] = 1
    p = np.cross(v_homo, z_axis)
    p = p[:, :-1] # ignore z axis
    p_length = np.sqrt(np.sum(p**2, axis=1, keepdims=True))
    p = p / (p_length + 1e-8) # now sum = 1
    p *= v_length
    return p

def PointInterpolation(v1, v2, ratio=0.5):
    # v1, v2 : N x 2
    # interpolation of points by finding the midpoint
    return v1 * (1. - ratio) + v2 * ratio

# class Morphing:
#     def __init__(self, a=0.1, b=2.0, p=0.5):
#         self.a = a
#         self.b = b # recommended range: [0.5, 2]
#         self.p = p # recommended range: [0  , 1]

#     def MultiFeatMorphing(self, X, P, Q, Pp, Qp):
#         '''
#             Calculate morphing with multiple feature vectors on 
#             destination coordinate X. Using numpy as instead of 
#             for loops saves time.
#             X  : Destination coordinate         H x W
#             P  : Feature vector start (desti)   N x 2
#             Q  : Feature vector end (desti)     N x 2
#             Pp : Feature vector start (source)  N x 2
#             Qp : Feature vector end (source)    N x 2
#         '''
#         X      = np.repeat(np.expand_dims(X, 2), P.shape[0], axis=2) # HxWxNx2
#         u      = np.sum((X - P) * (Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2, axis=-1)**2 # HxWxN
#         v      = np.sum((X - P) * Perpendicular(Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2, axis=-1) # HxWxN
#         Xp     = Pp + np.repeat(np.expand_dims(u, axis=-1), 2, axis=-1) * (Qp - Pp) \
#                  + np.repeat(np.expand_dims(v, axis=-1), 2, axis=-1) * Perpendicular(Qp - Pp) \
#                  / np.repeat(np.linalg.norm(Qp - Pp, ord=2, axis=-1, keepdims=True), 2, axis=-1)
#         D      = Xp - X # HxWxNx2
#         dist = np.absolute(v) * (np.greater_equal(u, 0) & np.less_equal(u, 1)) \
#                 + np.linalg.norm(X - P, ord=2, axis=-1) * (np.less(u, 0)) \
#                 + np.linalg.norm(X - Q, ord=2, axis=-1) * (np.greater(u, 1))

#         weight = np.power(np.power(np.linalg.norm(Q - P, axis=-1), self.p) / (self.a + dist), self.b)
#         Xp     = X[:,:,0,:] + np.sum(D * np.repeat(np.expand_dims(weight, axis=3), 2, axis=-1), axis=2) \
#                  / np.repeat(np.expand_dims(np.sum(weight, axis=-1), axis=2), 2, axis=-1)
#         Xp = np.around(Xp).astype(int)
#         Xp[:,:,0] = np.clip(Xp[:,:,0], 0, X.shape[0] - 1)
#         Xp[:,:,1] = np.clip(Xp[:,:,1], 0, X.shape[1] - 1)
#         return Xp # HxWx2

#     def TwoImageMorphing(self, 
#             img_1, img_2, 
#             P1, Q1, P2, Q2, 
#             ratio=0.5):
#         assert img_1.shape == img_2.shape
#         H, W = img_1.shape[0], img_1.shape[1]
#         # Interpolate features (destination)
#         Pd, Qd = PointInterpolation(P1, P2, ratio), PointInterpolation(Q1, Q2, ratio)
#         # Calculate morphing
#         X = np.zeros((H, W, 2), dtype=float)
#         X[:,:,0] = np.expand_dims(np.arange(0, H, dtype=float), axis=1)
#         X[:,:,1] = np.expand_dims(np.arange(0, W, dtype=float), axis=0)
#         Xp_1 = self.MultiFeatMorphing(X.copy(), Pd, Qd, P1, Q1)
#         Xp_2 = self.MultiFeatMorphing(X.copy(), Pd, Qd, P2, Q2)
#         print("Xp_1 shape:", Xp_1.shape)
#         print("Xp_2 shape:", Xp_2.shape)
#         print(Xp_1[:,:,0].shape == Xp_2[:,:,0].shape)
#         result = img_1[Xp_1[:,:,0],Xp_1[:,:,1],:] * (1. - ratio)
#         # result = img_1[Xp_1[:,:,0],Xp_1[:,:,1],:] * (1. - ratio) + \
#         #          img_2[Xp_2[:,:,0],Xp_2[:,:,1],:] * ratio
        
#         return result, Pd, Qd
# Mouse callback function for drawing lines
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down
        drawing = True
        ix, iy = x, y  # Store the initial point
    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse move
        if drawing:  # Draw only when the mouse is pressed
            img_copy = param.img.copy()  # Make a copy to avoid drawing permanent lines
            cv2.line(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(param.window_name, img_copy)  # Display the copy
    elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button released
        drawing = False  # End drawing
        # Draw the final line on the original image with an arrow
        cv2.arrowedLine(param.img, (ix, iy), (x, y), (0, 255, 0), 2, tipLength=0.05)
        cv2.imshow(param.window_name, param.img)  # Display the image with the final line

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
    
    cv2.imwrite(f"{img_data.window_name}.jpg",img_data.img) # save the modified image
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

def MultiFeatMorphing(X, P, Q, Pp, Qp):
        '''
            Calculate morphing with multiple feature vectors on 
            destination coordinate X. Using numpy as instead of 
            for loops saves time.
            X  : Destination coordinate         H x W
            P  : Feature vector start (desti)   N x 2
            Q  : Feature vector end (desti)     N x 2
            Pp : Feature vector start (source)  N x 2
            Qp : Feature vector end (source)    N x 2
        '''
        p=0.5
        a=0.1
        b=2
        X      = np.repeat(np.expand_dims(X, 2), P.shape[0], axis=2) # HxWxNx2
        u      = np.sum((X - P) * (Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2, axis=-1)**2 # HxWxN
        v      = np.sum((X - P) * Perpendicular(Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2, axis=-1) # HxWxN
        Xp     = Pp + np.repeat(np.expand_dims(u, axis=-1), 2, axis=-1) * (Qp - Pp) \
                 + np.repeat(np.expand_dims(v, axis=-1), 2, axis=-1) * Perpendicular(Qp - Pp) \
                 / np.repeat(np.linalg.norm(Qp - Pp, ord=2, axis=-1, keepdims=True), 2, axis=-1)
        D      = Xp - X # HxWxNx2
        dist = np.absolute(v) * (np.greater_equal(u, 0) & np.less_equal(u, 1)) \
                + np.linalg.norm(X - P, ord=2, axis=-1) * (np.less(u, 0)) \
                + np.linalg.norm(X - Q, ord=2, axis=-1) * (np.greater(u, 1))

        weight = np.power(np.power(np.linalg.norm(Q - P, axis=-1), p) / (a + dist), b)
        Xp     = X[:,:,0,:] + np.sum(D * np.repeat(np.expand_dims(weight, axis=3), 2, axis=-1), axis=2) \
                 / np.repeat(np.expand_dims(np.sum(weight, axis=-1), axis=2), 2, axis=-1)
        Xp = np.around(Xp).astype(int)
        Xp[:,:,0] = np.clip(Xp[:,:,0], 0, X.shape[0] - 1)
        Xp[:,:,1] = np.clip(Xp[:,:,1], 0, X.shape[1] - 1)
        return Xp

def TwoImageMorphing(img_1, img_2, P1, Q1, P2, Q2, ratio=0.5):
        assert img_1.shape == img_2.shape
        b = 2
        p = 0.5
        a = 1
        eps = 1e-8
        perp_d = Perpendicular(Q2-P2)
        perp_s = Perpendicular(Q1-P1)
        dest_line_vec = Q2-P2
        source_line_vec = Q1-P1
        
        image_size = img_1.shape[0]
        x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
        X_d = np.dstack([x, y])
        X_d = X_d.reshape(-1, 1, 2)
        to_p_vec = X_d - P2
        to_q_vec = X_d - Q2
        u = np.sum(to_p_vec * dest_line_vec, axis=-1) / (np.sum(dest_line_vec**2, axis=1) + eps)
        v = np.sum(to_p_vec * perp_d, axis=-1) / (np.sqrt(np.sum(dest_line_vec**2, axis=1)) + eps)
        X_s = np.expand_dims(P1, 0) + \
        np.expand_dims(u, -1) * np.expand_dims(source_line_vec, 0) + \
        np.expand_dims(v, -1) * np.expand_dims(perp_s, 0) / (np.sqrt(np.sum(source_line_vec**2, axis=1)).reshape(1, -1, 1) + eps)
        D = X_s - X_d
        to_p_mask = (u < 0).astype(np.float64)
        to_q_mask = (u > 1).astype(np.float64)
        to_line_mask = np.ones(to_p_mask.shape) - to_p_mask - to_q_mask
        
        to_p_dist = np.sqrt(np.sum(to_p_vec**2, axis=-1))
        to_q_dist = np.sqrt(np.sum(to_q_vec**2, axis=-1))
        to_line_dist = np.abs(v)
        dist = to_p_dist * to_p_mask + to_q_dist * to_q_mask + to_line_dist * to_line_mask
        dest_line_length = np.sqrt(np.sum(dest_line_vec**2, axis=-1))
        weight = (dest_line_length**p) / (((a + dist))**b + eps)
        weighted_D = np.sum(D * np.expand_dims(weight, -1), axis=1) / (np.sum(weight, -1, keepdims=True) + eps)

        X_d = X_d.squeeze()
        X_s = X_d + weighted_D
        X_s_ij = X_s[:, ::-1]

        if len(img_1.shape) == 2:
            warped = map_coordinates(img_1, X_s_ij.T, mode="nearest")
        else:
            warped = np.zeros((image_size*image_size, img_1.shape[2]))
            for i in range(img_1.shape[2]):
                warped[:, i] = map_coordinates(img_1[:, :, i], X_s_ij.T, mode="nearest")
        warped = warped.reshape(image_size, image_size, -1).squeeze()
        return warped.astype(np.uint8)

def get_intermidate_lines(P_1, Q_1, P_2, Q_2, alpha = 0.5):
    P = P_1 * alpha + P_2 * (1 - alpha)
    Q = Q_1 * alpha + Q_2 * (1 - alpha)
    return P, Q

def get_intermediate_face_outline(face1, face2, alpha = 0.5):
    return face1 * alpha + face2 * (1 - alpha)

def warp(src, dst, P1, Q1, P2, Q2, alpha=0.4):
    
    # print(src.img.shape, dst.img.shape)
    result, Pd, Qd = TwoImageMorphing(src.img, dst.img, P1, Q1, P2, Q2, ratio=alpha)
    return result

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
        warp(image_data_list[0], image_data_list[1], alpha, P1, Q1, P2, Q2)
        
    elif len(image_data_list)<=0:
        print(f"Error image data size {len(image_data_list)}")
    # else:
    #     warp()