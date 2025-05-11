import cv2
import numpy as np
import glob
import tqdm
import os

def collect_calibration_points(image_path, pattern_size, square_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        return True, objp, corners2, gray.shape[::-1]
    return False, None, None, None

def calibrate_camera(objpoints, imgpoints, img_size):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return K, dist, rvecs, tvecs

def draw_cube_on_image(image_path, K, dist, rvec, tvec, pattern_size, square_size):
    axis = np.float32([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,-1], [1,0,-1], [1,1,-1], [0,1,-1]
    ]) * square_size

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), 5)  # 線を太く: 2→5
        for i in range(4):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i+4]), (255,0,0), 5)  # 線を太く: 2→5
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 5)  # 線を太く: 2→5
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(image_path), "..", "processed")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)
        print(f"Saved processed image to: {output_path}")

def main():
    image_path_list = glob.glob("CameraModel/images/src/*.jpg")
    pattern_size = (9, 7)
    square_size = 1.0
    objpoints, imgpoints = [], []
    img_size = None

    for image_path in tqdm.tqdm(image_path_list, desc="Collecting calibration points"):
        ret, objp, corners2, shape = collect_calibration_points(image_path, pattern_size, square_size)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners2)
            if img_size is None:
                img_size = shape

    K, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, img_size)
    print("Intrinsic Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist)
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rvec)
        print(f"\nImage {i+1} Extrinsic Parameters:")
        # Create a 3x4 extrinsic matrix [R|t]
        extrinsic_matrix = np.hstack((R, tvec.reshape(3, 1)))
        print("Extrinsic Matrix [R|t]:\n", np.array2string(extrinsic_matrix, precision=8, suppress_small=True))

    # Fix the IndexError by ensuring we only process images that have valid calibration data
    successful_images = []
    for i, image_path in enumerate(image_path_list):
        ret, _, _, _ = collect_calibration_points(image_path, pattern_size, square_size)
        if ret:
            successful_images.append((i, image_path))
    
    for idx, (i, image_path) in enumerate(tqdm.tqdm(successful_images, desc="Drawing cubes")):
        if idx < len(rvecs) and idx < len(tvecs):
            draw_cube_on_image(image_path, K, dist, rvecs[idx], tvecs[idx], pattern_size, square_size)

if __name__ == "__main__":
    main()