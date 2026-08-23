"""
Lightweight, Zero-Heavy-Dependency Leaf Isolation Stage.
Uses pure NumPy, SciPy (ndimage), and PIL to isolate leaf regions,
filter background clutter, and quantify detection confidence.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import scipy.ndimage as ndi


class LeafDetector:
    def __init__(self, min_area_fraction: float = 0.05, max_area_fraction: float = 0.98):
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction

    @staticmethod
    def _otsu_threshold(array_2d: np.ndarray) -> float:
        """Calculates Otsu threshold value for a 2D float/uint8 array."""
        flat = array_2d.flatten()
        flat = flat[~np.isnan(flat)]
        if len(flat) == 0:
            return 0.0
        
        # 256 bins histogram
        hist, bin_edges = np.histogram(flat, bins=256)
        total = flat.size
        
        current_max = 0.0
        threshold = bin_edges[0]
        sum_total = np.dot(np.arange(256), hist)
        
        sum_b = 0.0
        weight_b = 0
        
        for i in range(256):
            weight_b += hist[i]
            if weight_b == 0:
                continue
            weight_f = total - weight_b
            if weight_f == 0:
                break
            
            sum_b += i * hist[i]
            mean_b = sum_b / weight_b
            mean_f = (sum_total - sum_b) / weight_f
            
            # Between-class variance
            var_between = float(weight_b) * float(weight_f) * ((mean_b - mean_f) ** 2)
            if var_between > current_max:
                current_max = var_between
                threshold = (bin_edges[i] + bin_edges[i + 1]) / 2.0
                
        return float(threshold)

    def detect_and_isolate_leaf(
        self, image: Image.Image | np.ndarray
    ) -> Dict[str, Any]:
        """
        Detects primary leaf in image.
        Returns:
            - cropped_image: PIL Image of isolated leaf
            - bbox: [x, y, w, h] in pixels
            - normalized_bbox: [x_min, y_min, x_max, y_max] in [0, 1]
            - detection_confidence: float [0, 1]
            - fallback_used: bool
            - quality_assessment: 'good' | 'moderate' | 'poor'
            - leaf_area_ratio: float
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            np_img = np.array(pil_img)
        else:
            np_img = image
            pil_img = Image.fromarray(np_img)

        h, w = np_img.shape[:2]
        img_area = float(h * w)

        # 1. Multi-Space Foliar Saliency
        r = np_img[:, :, 0].astype(np.float32)
        g = np_img[:, :, 1].astype(np.float32)
        b = np_img[:, :, 2].astype(np.float32)

        # Excess Green Index: 2G - R - B
        exg = 2.0 * g - r - b
        
        # Color Saturation Index: max(R,G,B) - min(R,G,B) / max(R,G,B)
        rgb_max = np.maximum(np.maximum(r, g), b)
        rgb_min = np.minimum(np.minimum(r, g), b)
        saturation = np.where(rgb_max > 0, (rgb_max - rgb_min) / np.maximum(1.0, rgb_max), 0.0)

        # Foliar Score = ExG weighted by non-neutral saturation
        foliar_score = exg * (0.5 + 0.5 * saturation)

        # Compute Otsu Threshold
        thresh_val = self._otsu_threshold(foliar_score)
        binary_mask = foliar_score > max(0.0, thresh_val)

        # Morphological Closing & Hole Filling
        struct = ndi.generate_binary_structure(2, 2)
        radius = max(2, min(h, w) // 100)
        closed_mask = ndi.binary_closing(binary_mask, structure=struct, iterations=radius)
        filled_mask = ndi.binary_fill_holes(closed_mask)

        # Connected Components
        labeled_mask, num_features = ndi.label(filled_mask)
        
        best_bbox = None
        best_area = 0.0
        best_slice = None

        if num_features > 0:
            objects = ndi.find_objects(labeled_mask)
            for idx, slc in enumerate(objects):
                if slc is None:
                    continue
                # Calculate area of component
                comp_mask = labeled_mask[slc] == (idx + 1)
                area = float(np.sum(comp_mask))
                
                if area < (self.min_area_fraction * img_area):
                    continue
                
                y_slice, x_slice = slc
                ch = y_slice.stop - y_slice.start
                cw = x_slice.stop - x_slice.start
                aspect_ratio = float(cw) / max(1, ch)
                
                # Check valid biological leaf aspect ratio
                if 0.15 <= aspect_ratio <= 6.0 and area > best_area:
                    best_area = area
                    best_bbox = (x_slice.start, y_slice.start, cw, ch)
                    best_slice = slc

        leaf_area_ratio = best_area / img_area if img_area > 0 else 0.0

        if best_bbox is not None and leaf_area_ratio >= self.min_area_fraction:
            x, y, cw, ch = best_bbox
            fallback_used = False

            # Add 12% padding around leaf perimeter
            pad_w = int(cw * 0.12)
            pad_h = int(ch * 0.12)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + cw + pad_w)
            y2 = min(h, y + ch + pad_h)

            # Square aspect ratio adjustment
            crop_w = x2 - x1
            crop_h = y2 - y1
            max_side = max(crop_w, crop_h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            sq_x1 = max(0, cx - max_side // 2)
            sq_y1 = max(0, cy - max_side // 2)
            sq_x2 = min(w, sq_x1 + max_side)
            sq_y2 = min(h, sq_y1 + max_side)

            cropped_pil = pil_img.crop((sq_x1, sq_y1, sq_x2, sq_y2))

            # Quality calculation based on area coverage and density
            box_area = float(cw * ch)
            density = best_area / max(1.0, box_area)
            det_conf = min(0.98, max(0.45, 0.35 + 0.35 * min(1.0, leaf_area_ratio * 2.0) + 0.30 * density))
            
            if det_conf >= 0.70:
                quality = "good"
            elif det_conf >= 0.45:
                quality = "moderate"
            else:
                quality = "poor"

            bbox_out = [int(sq_x1), int(sq_y1), int(sq_x2 - sq_x1), int(sq_y2 - sq_y1)]
            norm_bbox = [sq_x1 / w, sq_y1 / h, sq_x2 / w, sq_y2 / h]
        else:
            # Fallback: Center Crop with explicit fallback flag
            fallback_used = True
            det_conf = 0.25
            quality = "poor"

            min_dim = min(w, h)
            cx, cy = w // 2, h // 2
            sq_x1 = max(0, cx - min_dim // 2)
            sq_y1 = max(0, cy - min_dim // 2)
            sq_x2 = min(w, sq_x1 + min_dim)
            sq_y2 = min(h, sq_y1 + min_dim)

            cropped_pil = pil_img.crop((sq_x1, sq_y1, sq_x2, sq_y2))
            bbox_out = [int(sq_x1), int(sq_y1), int(sq_x2 - sq_x1), int(sq_y2 - sq_y1)]
            norm_bbox = [sq_x1 / w, sq_y1 / h, sq_x2 / w, sq_y2 / h]

        return {
            "cropped_image": cropped_pil,
            "bbox": bbox_out,
            "normalized_bbox": [round(float(v), 4) for v in norm_bbox],
            "detection_confidence": float(round(det_conf, 3)),
            "fallback_used": fallback_used,
            "quality_assessment": quality,
            "leaf_area_ratio": float(round(leaf_area_ratio, 4)),
        }
