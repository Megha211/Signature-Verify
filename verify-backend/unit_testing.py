import unittest
import numpy as np
from io import BytesIO
import base64
from app2 import (
    resize_image,
    crop_center,
    crop_center_multiple,
    preprocess_signature,
)

class TestImagePreprocessing(unittest.TestCase):
    
        
    def test_resize_image(self):
        # Create a sample image
        img = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
        
        # Test that the output image has the expected size
        output = resize_image(img, (150, 220))
        self.assertEqual(output.shape, (150, 220))
        
    def test_crop_center(self):
        # Create a sample image
        img = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
        
        # Test that the output image has the expected size
        output = crop_center(img, (80, 120))
        self.assertEqual(output.shape, (80, 120))
        
    def test_crop_center_multiple(self):
        # Create a sample batch of images
        imgs = np.random.randint(0, 256, size=(10, 1, 100, 200), dtype=np.uint8)
        
        # Test that the output batch has the expected size
        output = crop_center_multiple(imgs, (80, 120))
        self.assertEqual(output.shape, (10, 1, 80, 120))
        
    def test_preprocess_signature(self):
        # Create a sample image
        img = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
        
        # Test that the output image has the expected size
        output = preprocess_signature(img, (952, 1360))
        self.assertEqual(output.shape, (150, 220))

if __name__ == '__main__':
    unittest.main()