import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
import pydicom

class CTDataset(Dataset):
    def __init__(self, image_dir, scale_factor=4, crop_size=128, mode='train'):
        """
        Args:
            image_dir: Path to directory containing HR images.
            scale_factor: Downsampling factor.
            crop_size: Size of the HR crop.
            mode: 'train' or 'test'.
        """
        self.image_dir = image_dir
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')) + 
                                  glob.glob(os.path.join(image_dir, '*.jpg')) +
                                  glob.glob(os.path.join(image_dir, '*.jpeg')) +
                                  glob.glob(os.path.join(image_dir, '*.tif')) +
                                  glob.glob(os.path.join(image_dir, '*.dcm')))
        
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.mode = mode

        # Transforms
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        # If no images found, return dummy length for testing purposes if in train mode
        if len(self.image_files) == 0 and self.mode == 'train':
            print(f"Warning: No images found in {self.image_dir}. Using dummy data.")
            return 100 
        return len(self.image_files)

    def _read_dicom(self, path):
        """Reads a DICOM file and converts it to a PIL Image."""
        try:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array.astype(float)
            
            # Rescale to 0-255
            image = (np.maximum(image, 0) / image.max()) * 255.0
            image = np.uint8(image)
            
            return Image.fromarray(image, mode='L')
        except Exception as e:
            print(f"Error reading DICOM {path}: {e}")
            # Return a blank image on error
            return Image.new('L', (512, 512))

    def __getitem__(self, idx):
        if len(self.image_files) == 0:
            # Generate dummy data: Random noise or gradient
            hr_image = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode='L')
        else:
            img_path = self.image_files[idx]
            
            if img_path.lower().endswith('.dcm'):
                hr_image = self._read_dicom(img_path)
            else:
                # Open as grayscale for CT
                hr_image = Image.open(img_path).convert('L')

        # Crop HR image
        w, h = hr_image.size
        
        if self.mode == 'train':
            # Random Crop
            if w < self.crop_size or h < self.crop_size:
                 hr_image = hr_image.resize((max(w, self.crop_size), max(h, self.crop_size)), Image.BICUBIC)
                 w, h = hr_image.size
            
            left = np.random.randint(0, w - self.crop_size + 1)
            top = np.random.randint(0, h - self.crop_size + 1)
            hr_crop = hr_image.crop((left, top, left + self.crop_size, top + self.crop_size))
        else:
            # Center Crop or Full Image (for simplicity using center crop or resizing to multiple of scale)
            # Ensure dimensions are multiples of scale_factor
            new_w = w - (w % self.scale_factor)
            new_h = h - (h % self.scale_factor)
            hr_crop = hr_image.crop((0, 0, new_w, new_h))

        # Downsample to create LR image
        lr_w = hr_crop.size[0] // self.scale_factor
        lr_h = hr_crop.size[1] // self.scale_factor
        lr_image = hr_crop.resize((lr_w, lr_h), Image.BICUBIC)

        return self.to_tensor(lr_image), self.to_tensor(hr_crop)
