# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# %% import
import cv2
import numpy as np
from pose_estimation import PoseEstimation


# %% Read Image and set positions
im = cv2.imread("headPose.jpg")
size = im.shape

# 2D image points. If you change the image, you need to change vector
image_points = np.array(
    [(359, 391),     # Nose tip
     (399, 561),     # Chin
     (337, 297),     # Left eye left corner
     (513, 301),     # Right eye right corne
     (345, 465),     # Left Mouth corner
     (453, 469)      # Right mouth corner
     ], dtype="double")

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
    ])


image_points = np.array([
    (359, 391),     # Nose tip
    (399, 561),     # Chin
    (337, 297),     # Left eye left corner
    (513, 301),     # Right eye right corne
    ], dtype="double")

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    ])


# Camera internals
focal_length = size[1]  # approximate with the WIDTH of the image
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
    )

print("Camera Matrix :\n {0}".format(camera_matrix))

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion


# %% Estimation
success, rotation_vector, translation_vector, inliers = \
    cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs,
                       flags=cv2.SOLVEPNP_P3P)

print("Rotation Vector:\n {0}".format(rotation_vector))
print("Translation Vector:\n {0}".format(translation_vector))

rotation_mat, _ = cv2.Rodrigues(rotation_vector)
pose_mat = cv2.hconcat((rotation_mat, translation_vector))
dec_out = cv2.decomposeProjectionMatrix(pose_mat)
_, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)


pest = PoseEstimation(model_points, image_size=size)
euler_angles = pest.angle(image_points)


# %%
# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose

(nose_end_point2D, jacobian) = \
    cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                      translation_vector, camera_matrix, dist_coeffs)

for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(im, p1, p2, (255, 0, 0), 2)

# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
