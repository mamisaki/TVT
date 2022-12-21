# -*- coding: utf-8 -*-
"""
@author: mmisaki
"""

# %% import
import cv2
import numpy as np


# %%
class PoseEstimation():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, model_points, camera_matrix=None,
                 dist_coeffs=None, image_size=[480, 640]):
        # object model coordinates
        self.model_points = model_points

        # Camera model
        if camera_matrix is None:
            focal_length = image_size[1]  # approximate with the image WIDTH
            center = (image_size[1]/2, image_size[0]/2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float)
        else:
            self.camera_matrix = camera_matrix

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        else:
            self.dist_coeffs = dist_coeffs

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rot_trans(self, image_points):
        success, rot_vec, trans_vec, inliers = \
            cv2.solvePnPRansac(self.model_points, image_points,
                               self.camera_matrix, self.dist_coeffs,
                               flags=cv2.SOLVEPNP_P3P)
        return success, rot_vec, trans_vec

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def angle(self, image_points):
        """

        Parameters
        ----------
        image_points : n x 2 array
            x,y coordinates of n marker points.
            Number of points must corresponds to the number of model_points in
            self.model_points

        Returns
        -------
        euler_angles : Euler angles, alpha, beta, gamma
            alpha; rotation around z axis
            beta; rotation around x
            gamma; rotation around y
        """

        success, rot_vec, trans_vec = self.rot_trans(image_points)
        if success:
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            pose_mat = cv2.hconcat((rot_mat, trans_vec))
            dec_out = cv2.decomposeProjectionMatrix(pose_mat)
            euler_angles = dec_out[-1]

            return euler_angles
        else:
            return None


# %% __main__
if __name__ == '__main__':

    # test
    size = (675, 1200, 3)

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        (359, 391),     # Nose tip
        (399, 561),     # Chin
        (337, 297),     # Left eye left corner
        (513, 301),     # Right eye right corne
        ], dtype=np.float)

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        ])

    pest = PoseEstimation(model_points, image_size=size)
    euler_angles = pest.angle(image_points)
    assert np.allclose(euler_angles.ravel(),
                       [-175.40117516,   40.77472233,   -0.33483476])
