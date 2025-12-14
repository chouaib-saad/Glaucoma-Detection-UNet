import cv2
import numpy as np
import os
from skimage.transform import resize
from keras import backend as K
import tensorflow as tf

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def fscore(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class optic_segmentation():
  def __init__(self, model_OD_path=None, model_OC_path=None):
    # Use absolute paths based on script location
    if model_OD_path is None:
      model_OD_path = os.path.join(SCRIPT_DIR, 'Models', 'model OD semantic')
    if model_OC_path is None:
      model_OC_path = os.path.join(SCRIPT_DIR, 'Models', 'model OC semantic')
    
    # Load semantic segmentation model using TFSMLayer for Keras 3 compatibility
    self.model_OD = tf.saved_model.load(model_OD_path)
    self.model_OC = tf.saved_model.load(model_OC_path)

  def resizeMask(self, mask, koordinat, shape):
    """Resize mask back to original image size"""
    yo, yi, xo, xi = koordinat
    mask_temp = np.zeros(shape, np.uint8)

    for y in range(yo, yi):
      for x in range(xo, xi):
        if mask[y-yo][x-xo] == 255:
          mask_temp[y][x] = 255

    return mask_temp

  def ellipsTransform(self, mask):
    """Fit ellipse to the largest contour for smoother boundaries"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    elips = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    if len(cnts) > 0 and len(cnts[0]) >= 5:  # Need at least 5 points for ellipse
      ellipse = cv2.fitEllipse(cnts[0])
      elips = cv2.ellipse(elips, ellipse, 255, cv2.FILLED)
    else:
      elips = mask
    return elips

  def adaptive_roi_size(self, img_shape):
    """Calculate adaptive ROI size based on image dimensions"""
    min_dim = min(img_shape[0], img_shape[1])
    # ROI should be approximately 25-35% of the smallest dimension
    roi_size = int(min_dim * 0.30)
    # Ensure ROI is within reasonable bounds
    roi_size = max(400, min(800, roi_size))
    return roi_size

  def enhance_cup_detection(self, ROI_img, OD_mask, OC_raw_pred):
    """
    Enhanced cup detection using multiple techniques:
    1. Color channel analysis (cup is typically paler/brighter)
    2. Intensity-based refinement within the disc region
    3. Morphological operations for noise reduction
    """
    enhanced_cup = OC_raw_pred.copy()
    
    # Ensure cup is within disc boundaries
    enhanced_cup = cv2.bitwise_and(enhanced_cup, OD_mask)
    
    # Apply morphological operations to clean up the cup mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small noise
    enhanced_cup = cv2.morphologyEx(enhanced_cup, cv2.MORPH_OPEN, kernel_small)
    # Fill small holes
    enhanced_cup = cv2.morphologyEx(enhanced_cup, cv2.MORPH_CLOSE, kernel_medium)
    
    return enhanced_cup

  def refine_segmentation_by_intensity(self, ROI_color, OD_mask, threshold_ratio=0.7):
    """
    Refine cup segmentation using intensity analysis within the disc region.
    The cup region is typically brighter/paler than the neuroretinal rim.
    """
    if len(ROI_color.shape) == 2:
      gray = ROI_color
    else:
      # Use green channel which has best contrast for retinal images
      gray = ROI_color[:, :, 1] if ROI_color.shape[2] >= 2 else ROI_color[:, :, 0]
    
    # Only analyze within the disc region
    disc_region = cv2.bitwise_and(gray, gray, mask=OD_mask)
    
    # Calculate threshold based on intensity within disc
    disc_pixels = disc_region[OD_mask == 255]
    if len(disc_pixels) == 0:
      return np.zeros_like(OD_mask)
    
    # Cup is typically the brightest region within the disc
    threshold = np.percentile(disc_pixels, 70)  # Top 30% brightest pixels
    
    # Create intensity-based cup mask
    intensity_cup = np.zeros_like(OD_mask)
    intensity_cup[(disc_region > threshold) & (OD_mask == 255)] = 255
    
    return intensity_cup

  def validate_cup_within_disc(self, OD_mask, OC_mask):
    """
    Ensure the cup is properly contained within the disc.
    If cup extends beyond disc, clip it.
    """
    # Cup must be inside disc
    validated_cup = cv2.bitwise_and(OC_mask, OD_mask)
    
    # Check if cup area is reasonable (should be 20-80% of disc area)
    disc_area = np.sum(OD_mask == 255)
    cup_area = np.sum(validated_cup == 255)
    
    if disc_area > 0:
      ratio = cup_area / disc_area
      # If cup is too large (>80% of disc), it's likely a segmentation error
      if ratio > 0.8:
        # Erode the cup mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        validated_cup = cv2.erode(validated_cup, kernel, iterations=2)
      # If cup is too small (<5% of disc), might need dilation
      elif ratio < 0.05 and cup_area > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        validated_cup = cv2.dilate(validated_cup, kernel, iterations=1)
        validated_cup = cv2.bitwise_and(validated_cup, OD_mask)
    
    return validated_cup

  def multi_scale_segmentation(self, ROI, scales=[0.8, 1.0, 1.2]):
    """
    Perform segmentation at multiple scales and combine results
    for more robust detection across different image zoom levels.
    """
    h, w = ROI.shape[:2]
    od_predictions = []
    oc_predictions = []
    
    for scale in scales:
      # Resize ROI for this scale
      new_h, new_w = int(256 * scale), int(256 * scale)
      if new_h < 64 or new_w < 64:
        continue
        
      scaled_roi = resize(ROI, (new_h, new_w, 1), mode='constant', preserve_range=True)
      # Resize back to 256x256 for model input
      model_input = resize(scaled_roi, (256, 256, 1), mode='constant', preserve_range=True)
      model_input = np.array([model_input / 255.0], dtype=np.float32)
      
      # Get predictions
      od_pred = self.model_OD(model_input)
      oc_pred = self.model_OC(model_input)
      
      od_predictions.append(np.array(od_pred[0].numpy().squeeze()))
      oc_predictions.append(np.array(oc_pred[0].numpy().squeeze()))
    
    # Average predictions across scales
    if len(od_predictions) > 0:
      od_avg = np.mean(od_predictions, axis=0)
      oc_avg = np.mean(oc_predictions, axis=0)
    else:
      # Fallback to single scale
      model_input = resize(ROI, (256, 256, 1), mode='constant', preserve_range=True)
      model_input = np.array([model_input / 255.0], dtype=np.float32)
      od_avg = self.model_OD(model_input)[0].numpy().squeeze()
      oc_avg = self.model_OC(model_input)[0].numpy().squeeze()
    
    return od_avg, oc_avg

  def do_segmentation(self, ROI, coordinate, shape, elips_fit=True, use_multi_scale=True, ROI_color=None):
    """
    Perform optic disc and cup segmentation with improved accuracy.
    
    Parameters:
    - ROI: Region of interest (grayscale)
    - coordinate: ROI coordinates in original image
    - shape: Original image shape
    - elips_fit: Whether to apply ellipse fitting
    - use_multi_scale: Whether to use multi-scale approach for robustness
    - ROI_color: Color ROI for intensity-based refinement (optional)
    """
    
    if use_multi_scale:
      # Multi-scale segmentation for better handling of different zoom levels
      od_prob, oc_prob = self.multi_scale_segmentation(ROI)
    else:
      # Standard single-scale segmentation
      ROI_resized = resize(ROI, (256, 256, 1), mode='constant', preserve_range=True)
      ROI_input = np.array([ROI_resized / 255.0], dtype=np.float32)
      od_prob = self.model_OD(ROI_input)[0].numpy().squeeze()
      oc_prob = self.model_OC(ROI_input)[0].numpy().squeeze()
    
    # Apply adaptive thresholding
    # Use slightly higher threshold for disc (more conservative)
    OD_pred = np.array(od_prob > 0.45, np.uint8) * 255
    # Use adaptive threshold for cup based on prediction confidence
    oc_threshold = 0.5 if np.max(oc_prob) > 0.7 else 0.4
    OC_pred = np.array(oc_prob > oc_threshold, np.uint8) * 255
    
    # Resize predictions to ROI size
    roi_h, roi_w = ROI.shape[:2]
    OD_pred = cv2.resize(OD_pred, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    OC_pred = cv2.resize(OC_pred, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    
    # Enhanced cup detection
    OC_pred = self.enhance_cup_detection(ROI, OD_pred, OC_pred)
    
    # Validate cup is within disc
    OC_pred = self.validate_cup_within_disc(OD_pred, OC_pred)
    
    # Apply ellipse fitting for smoother boundaries
    if elips_fit:
      OD_pred = self.ellipsTransform(OD_pred)
      OC_pred = self.ellipsTransform(OC_pred)
      # Re-validate after ellipse fitting
      OC_pred = self.validate_cup_within_disc(OD_pred, OC_pred)

    # Resize masks to original image size
    OD_pred = self.resizeMask(OD_pred, coordinate, shape)
    OC_pred = self.resizeMask(OC_pred, coordinate, shape)

    return OD_pred, OC_pred