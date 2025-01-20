# Panaroma_Stitching
Merge multiple overlapping images into a single seamless panorama.

###This code is part of a course assignment (Image Processing for Engineers), which I lectured in 2022. ###

**Code Description:**
This Python script automates the process of creating a panorama from two overlapping images. The program leverages OpenCV and NumPy to detect features in the input images, match those features, calculate the homography matrix for image alignment, and then stitch the images into a seamless panorama. The resulting panorama is saved and displayed to the user.

**Most Important Libraries/Functions Used:**
OpenCV:
1. cv2.ORB_create(): Detects and computes keypoints and descriptors using the ORB algorithm.
2. cv2.BFMatcher(): Matches descriptors between images using brute-force matching.
3. cv2.findHomography(): Computes the transformation matrix for image alignment.
4. cv2.warpPerspective(): Warps the perspective of one image to align it with the other.
5. cv2.imwrite() and cv2.imshow(): Save and display the stitched panorama.

**Custom Functions:**
1. load_images(): Loads the images for processing.
2. detect_and_match_features(): Detects and matches keypoints.
3. find_homography(): Computes the homography matrix.
4. stitch_images(): Aligns and merges the images.

**Real-Life Engineering Applications:**
1. Geospatial Mapping: Engineers can stitch aerial images captured by drones to create comprehensive maps for surveying and urban planning.
2. Robotics and Navigation: In robotics, panoramic vision systems can enhance spatial awareness for autonomous navigation.
3. Industrial Quality Control: Panorama stitching is useful for inspecting large surfaces, such as vehicle bodies or manufacturing lines, by combining multiple images into a single, detailed view.
