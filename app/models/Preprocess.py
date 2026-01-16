# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
import cv2
import numpy as np
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import time
# import os
# Removed MediaPipe imports as face cropping is not used
# import torchvision.models as models
# import torch.nn.functional as F


# Preprocessing Class
class DeepfakePreprocessor:
    def __init__(self, output_size=(150, 150), min_detection_confidence=0.5):
        """Initialize the preprocessor with the desired output size."""
        self.output_size = output_size
        # Face detection removed as it's not used

    def apply_gradient_transform(self, image):
        """
        Apply the gradient transformation as per the paper.

        Args:
            image: Input image in RGB format.

        Returns:
            Gradient-transformed image in uint8 format.
        """
        H, W, _ = image.shape
        gradient_img = np.zeros_like(image, dtype=np.float32)
        for c in range(3):  # Process each color channel
            channel = image[:, :, c].astype(np.float32)
            # Compute differences with neighboring pixels
            diff_y = channel[:-1, :-1] - channel[1:, :-1]  # Vertical difference
            diff_x = channel[:-1, :-1] - channel[:-1, 1:]  # Horizontal difference
            # Compute gradient magnitude
            magnitude = np.sqrt(diff_y**2 + diff_x**2)
            # Assign to output (last row and column remain zero)
            gradient_img[:-1, :-1, c] = magnitude
        # Normalize to 0-255 range
        gradient_img = cv2.normalize(gradient_img, None, 0, 255, cv2.NORM_MINMAX)
        return gradient_img.astype(np.uint8)

    def preprocess_image(self, image_path):
        """
        Preprocess a single image by cropping the face and applying the gradient transform.

        Args:
            image_path: Path to the input image.

        Returns:
            Preprocessed image, or None if preprocessing fails.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        # face = self.crop_face(image)
        # if face is None:
        #     return None
        face = image
        preprocessed = self.apply_gradient_transform(face)
        return preprocessed
