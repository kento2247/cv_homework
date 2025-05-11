#!/usr/bin/env python3
"""
Warp an image to a front view by selecting four points interactively.
"""
import argparse
import sys
import cv2
import numpy as np
import os

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]       # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]    # Bottom-left has largest difference
    return rect

def select_points(image: np.ndarray, num_points: int = 4, point_size: int = 50) -> np.ndarray:
    """
    Allow the user to select points on the image via mouse clicks.
    Returns:
        A numpy array of selected points of shape (num_points, 2).
    """
    points = []
    clone = image.copy()
    window_name = "Select Points (Press 'r' to reset, 'q' to quit)"
    cv2.namedWindow(window_name)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
            cv2.circle(clone, (x, y), point_size, (0, 0, 255), -1)
            cv2.imshow(window_name, clone)

    cv2.setMouseCallback(window_name, click_event)

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points.clear()
            clone[:] = image.copy()
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            sys.exit("User aborted point selection.")
        elif len(points) == num_points:
            break
    
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32), clone

def warp_image_to_front_view(
    image_path: str,
    output_path: str,
    output_size: tuple[int, int] = (300, 400),
    point_size: int = 50
) -> None:
    """
    Read an image, allow the user to select four points, compute the perspective transform,
    warp the image to the specified front view, and save the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"Error: Unable to load image '{image_path}'.")

    # Step 1: Select and order points
    pts, image_with_points = select_points(image, num_points=4, point_size=point_size)
    ordered_pts = order_points(pts)

    # Step 2: Define destination rectangle
    w, h = output_size
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # Step 3: Compute transform & warp
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h))

    # Step 4: Save result
    if not cv2.imwrite(output_path, warped):
        sys.exit(f"Error: Failed to save warped image to '{output_path}'.")
    print(f"Warped image saved to '{output_path}'.")

    # Save the image with selected points
    points_output_path = os.path.splitext(output_path)[0] + "_points.jpg"
    if not cv2.imwrite(points_output_path, image_with_points):
        sys.exit(f"Error: Failed to save image with points to '{points_output_path}'.")
    print(f"Image with selected points saved to '{points_output_path}'.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Warp an image to a front view by selecting four points interactively."
    )
    parser.add_argument(
        "--input", help="Path to input image", default="ImageProcessing/images/planer.jpg"
    )
    parser.add_argument(
        "--output", help="Path to save warped image", default="ImageProcessing/images/planer_warped.jpg"
    )
    parser.add_argument(
        "--width", type=int, default=297*4, help="Width of output image"
    )
    parser.add_argument(
        "--height", type=int, default=210*4, help="Height of output image"
    )
    parser.add_argument(
        "--point_size", type=int, default=50, help="Size of point marker"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    warp_image_to_front_view(
        args.input,
        args.output,
        output_size=(args.width, args.height),
        point_size=args.point_size
    )

if __name__ == "__main__":
    main()