#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 15:50:19 2021

@author: sajjad
"""

import cv2
import numpy as np

def load_images(image_paths):
    """
    Load images from the provided file paths.
    Args:
        image_paths (list of str): List of paths to the images.
    Returns:
        list of np.ndarray: Loaded images.
    """
    images = [cv2.imread(path) for path in image_paths]
    return images

def detect_and_match_features(img1, img2):
    """
    Detect and match features between two images using ORB.
    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
    Returns:
        keypoints1, keypoints2, matches: Keypoints and matched features.
    """
    orb = cv2.ORB_create()

    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return keypoints1, keypoints2, matches

def find_homography(kp1, kp2, matches):
    """
    Compute homography matrix using matched features.
    Args:
        kp1: Keypoints from the first image.
        kp2: Keypoints from the second image.
        matches: Matched features.
    Returns:
        Homography matrix and mask.
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def stitch_images(img1, img2, H):
    """
    Warp and stitch two images using the homography matrix.
    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        H (np.ndarray): Homography matrix.
    Returns:
        np.ndarray: Stitched panorama.
    """
    # Warp the second image
    height, width = img1.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (width * 2, height))

    # Place the first image onto the warped image
    img2_warped[0:height, 0:width] = img1

    return img2_warped

def main(image_paths):
    """
    Main function to perform panorama stitching.
    Args:
        image_paths (list of str): List of paths to input images.
    Returns:
        None
    """
    # Load images
    images = load_images(image_paths)

    # Assume only two images for simplicity
    img1, img2 = images[0], images[1]

    # Detect and match features
    keypoints1, keypoints2, matches = detect_and_match_features(img1, img2)

    # Find homography
    H, mask = find_homography(keypoints1, keypoints2, matches)

    # Stitch images
    panorama = stitch_images(img1, img2, H)

    # Save and display the result
    cv2.imwrite("panorama.jpg", panorama)
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage with two image paths
    image_paths = ["image1.jpg", "image2.jpg"]
    main(image_paths)
