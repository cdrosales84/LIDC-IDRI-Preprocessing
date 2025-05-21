import argparse
import os
import numpy as np

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans

def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def segment_lung(img):
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    print(f"Debug - segment_lung input shape: {img.shape}, type: {img.dtype}")
    
    # Ensure input is 2D and float64
    if len(img.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    
    # Convert to float64 and normalize
    img = img.astype(np.float64)
    
    # Handle NaN and Inf values
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize the image
    mean = np.mean(img)
    std = np.std(img)
    if std > 0:
        img = (img - mean) / std
    else:
        img = img - mean
    
    # Process middle region
    middle = img[100:400, 100:400].copy()
    mean = np.mean(middle)
    
    # Handle outliers using percentile-based clipping
    min_val = np.percentile(img, 1)
    max_val = np.percentile(img, 99)
    img = np.clip(img, min_val, max_val)
    
    # Apply filters
    img = median_filter(img, size=3)
    
    # Apply anisotropic diffusion with error handling
    try:
        # Ensure the image is in the correct range for anisotropic diffusion
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = anisotropic_diffusion(img_normalized, niter=1)
    except Exception as e:
        print(f"Warning: Anisotropic diffusion failed: {str(e)}")
        # If anisotropic diffusion fails, use a simple Gaussian blur as fallback
        img = ndimage.gaussian_filter(img, sigma=1.0)
    
    # K-means clustering
    try:
        kmeans = KMeans(n_clusters=2, random_state=42).fit(np.reshape(middle, [-1, 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
    except Exception as e:
        print(f"Warning: K-means clustering failed: {str(e)}")
        threshold = np.mean(img)  # Fallback to mean
    
    # Create binary mask
    thresh_img = np.where(img < threshold, 1.0, 0.0)
    
    # Morphological operations
    try:
        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))
    except Exception as e:
        print(f"Warning: Morphological operations failed: {str(e)}")
        dilation = thresh_img  # Fallback to thresholded image
    
    # Label regions
    try:
        labels = measure.label(dilation)
        regions = measure.regionprops(labels)
        
        # Filter regions
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        
        # Create final mask
        mask = np.zeros_like(img, dtype=np.int8)
        for N in good_labels:
            mask = mask + (labels == N).astype(np.int8)
        
        mask = morphology.dilation(mask, np.ones([10, 10]))
    except Exception as e:
        print(f"Warning: Region labeling failed: {str(e)}")
        mask = dilation  # Fallback to dilated image
    
    # Apply mask to original image
    result = mask * img
    
    # Ensure output is in valid range
    result = np.clip(result, -1.0, 1.0)
    
    print(f"Debug - segment_lung output shape: {result.shape}, type: {result.dtype}")
    return result

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)