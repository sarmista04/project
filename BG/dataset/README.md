# Dataset Structure

This directory should contain fingerprint images organized by blood groups:

```
dataset/
├── A+/
├── A-/
├── B+/
├── B-/
├── AB+/
├── AB-/
├── O+/
└── O-/
```

## Guidelines for Dataset:

1. Image Requirements:
   - Format: PNG, JPG, or JPEG
   - Resolution: Minimum 300 DPI recommended
   - Size: Images will be resized to 128x128 pixels
   - Type: Grayscale fingerprint images

2. Organization:
   - Place fingerprint images in their respective blood group folders
   - Ensure images are clear and well-captured
   - Recommended minimum of 100 images per class for better accuracy

3. Image Quality:
   - Clean, clear fingerprint impressions
   - Good contrast between ridges and valleys
   - Minimal noise or artifacts
   - Centered fingerprint in the image

4. File Naming:
   - Use descriptive names (e.g., 'fingerprint_001.jpg')
   - Avoid spaces in filenames
   - Use lowercase letters and numbers

Note: The more diverse and high-quality the dataset, the better the model's performance will be.