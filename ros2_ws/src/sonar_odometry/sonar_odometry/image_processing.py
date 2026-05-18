import numpy as np
import cv2


class SonarImageProcessor:
    def __init__(self):
        self.config = {
            # ── Active pipeline ───────────────────────────────────────────────
            "crop_row":             300,    # rows above this are water column and discarded
            "bilateral_d":          9,      # diameter of pixel neighbourhood
            "bilateral_sigma_color":50,     # filter sigma in colour space
            "bilateral_sigma_space":50,     # filter sigma in coordinate space
            "clahe_clip_limit":     2.0,    # contrast limit for CLAHE
            "clahe_tile_grid":      (8, 8), # tile grid size for CLAHE
            # ── Legacy filters (not used in process_image) ────────────────────
            "apply_denoise": True,
            "apply_unsharp": True,
            "unsharp_ksize": 5,
            "unsharp_strength": 1.5,
            "apply_gaussian": False,
            "apply_median": False,
            "apply_otsu": False,
            "apply_fuzzy": False,
            "apply_opening": False,
            "gamma_value": 1.6,
            "gaussian_ksize": (3, 3),
            "gaussian_sigma": 0,
            "median_ksize": 3,
            "morph_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            "final_normalization": False,
        }
    
    def apply_denoising(self, img: np.ndarray) -> np.ndarray:
        """
        Apply row-wise background subtraction denoising.
        
        Args:
            img: Input image
            
        Returns:
            Denoised image
        """
        if not self.config.get("apply_denoise", False):
            return img
        
        try:
            # Ensure img is float for processing
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)
            
            # Estimate background using quantile per row
            background = np.quantile(img, 0.1, axis=1, keepdims=True)
            img_denoised = np.clip(img - background, 0.0, None)
            
            # Renormalize if image has content
            if img_denoised.max() > 0:
                img_denoised = img_denoised / (img_denoised.max() + 1e-6)
            
            return img_denoised
        
        except Exception as e:
            #print(f"Warning: Denoising failed: {e}")
            return img
    
    def apply_smoothing(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian and median smoothing filters.
        
        Args:
            img: Input image
            
        Returns:
            Smoothed image
        """
        # Gaussian blur
        if self.config.get("apply_gaussian", True):
            img = cv2.GaussianBlur(
                img, 
                self.config["gaussian_ksize"], 
                self.config["gaussian_sigma"]
            )
        
        # Median filter
        if self.config.get("apply_median", True):
            img_u8 = (img * 255).astype(np.uint8)
            img_median = cv2.medianBlur(img_u8, self.config["median_ksize"])
            img = img_median.astype(np.float32) / 255.0
        
        return img
    
    def apply_thresholding(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Otsu thresholding for binary masking using OpenCV.
        
        Args:
            img: Input image
            
        Returns:
            Thresholded image
        """
        if not self.config.get("apply_otsu", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            # Use OpenCV's Otsu thresholding instead of skimage
            ret, binary_mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_mask = binary_mask.astype(np.float32) / 255.0
            return img * binary_mask
        
        except Exception as e:
            #print(f"Warning: Otsu thresholding failed: {e}")
            return img
    
    def apply_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction and fuzzy enhancement.
        
        Args:
            img: Input image
            
        Returns:
            Enhanced image
        """
        if not self.config.get("apply_fuzzy", False):
            return img
        
        try:
            epsilon = 1e-6
            img = np.clip(img, 0.0, 1.0)
            gamma = float(self.config.get("gamma_value", 2.0))
            
            # Fuzzy gamma enhancement
            img_enhanced = (img ** gamma) / (img ** gamma + (1.0 - img + epsilon) ** gamma)
            return img_enhanced
        
        except Exception as e:
            #print(f"Warning: Enhancement failed: {e}")
            return img
    
    def apply_morphological_operations(self, img: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening for noise removal.
        
        Args:
            img: Input image
            
        Returns:
            Morphologically processed image
        """
        if not self.config.get("apply_opening", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            img_opened = cv2.morphologyEx(
                img_u8, 
                cv2.MORPH_OPEN, 
                self.config["morph_kernel"]
            )
            return img_opened.astype(np.float32) / 255.0
        
        except Exception as e:
            #print(f"Warning: Morphological operations failed: {e}")
            return img
    
    def threshold_image(self, img: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        """
        Apply simple thresholding to create a binary mask.
        
        Args:
            img: Input image
            thresh: Threshold value
            
        Returns:
            Binary masked image
        """
        max = img.max()
        thresh = max * thresh
        #print(f"Image max value before thresholding: {max}")
        return (img > thresh).astype(np.float32) 
    
    def apply_final_normalization(self, img: np.ndarray) -> np.ndarray:
        """
        Apply final normalization to ensure [0,1] range.
        
        Args:
            img: Input image
            
        Returns:
            Normalized image
        """
        if not self.config.get("final_normalization", True):
            return img
        
        return cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX)
    
    def apply_log_compress(self, img: np.ndarray) -> np.ndarray:
        """
        Log compression followed by min-max stretch to full uint8 range.

        Sonar speckle is multiplicative: variance scales with local mean, so
        bright regions are noisier in absolute terms. Log compression converts
        that to approximately additive, uniform-variance noise — which is what
        bilateral filtering and CLAHE were designed to handle. It also expands
        the dark end more than a linear stretch, recovering faint structure that
        would otherwise stay below AKAZE's gradient threshold.

        Mapping: x → log(1 + 4x) / log(5), which sends [0,1] → [0,1] and
        expands the lower half of the intensity range.
        """
        u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        x = u8.astype(np.float32) / 255.0
        compressed = np.log1p(4.0 * x) / np.log(5.0)
        stretched = cv2.normalize(
            (compressed * 255).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX
        )
        return stretched.astype(np.float32) / 255.0

    def crop_water_column(self, img: np.ndarray) -> np.ndarray:
        """
        Discard the water column by cropping rows above crop_row.

        The top portion of the sonar image is the water column (no seafloor
        returns) and must be removed before processing and feature detection.
        crop_row is tunable via self.config["crop_row"].
        """
        return img[self.config["crop_row"]:, :]

    
    def apply_unsharp(self, img: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking to enhance edges and local contrast.

        result = img + strength * (img - GaussianBlur(img))

        Args:
            img: Input image — uint8 [0, 255] or float32 [0, 1]

        Returns:
            Sharpened float32 image in [0, 1]
        """
        if not self.config.get("apply_unsharp", True):
            return img

        # Normalise to float32 [0, 1] regardless of input dtype
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)

        ksize = self.config.get("unsharp_ksize", 5)
        strength = self.config.get("unsharp_strength", 1.5)

        # ksize must be odd
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        sharpened = img + strength * (img - blurred)
        return np.clip(sharpened, 0.0, 1.0).astype(np.float32)

    def apply_dog(self, img: np.ndarray) -> np.ndarray:
        """
        Difference of Gaussians bandpass filter.
        Highlights edges and blobs between sigma1 and sigma2 scales.
        Replaces CLAHE — no tile-boundary artifacts, no noise amplification.
        """
        f  = img if img.dtype == np.float32 else img.astype(np.float32) / 255.0
        s1 = self.config.get("dog_sigma1", 1.0)
        s2 = self.config.get("dog_sigma2", 3.0)
        k1 = int(6 * s1) | 1
        k2 = int(6 * s2) | 1
        g1 = cv2.GaussianBlur(f, (k1, k1), s1)
        g2 = cv2.GaussianBlur(f, (k2, k2), s2)
        d  = g1 - g2
        d  = d - d.min()
        if d.max() > 0:
            d /= d.max()
        return d.astype(np.float32)

    def apply_bilateral(self, img: np.ndarray) -> np.ndarray:
        """
        Edge-preserving bilateral filter.
        Smooths noise while keeping sonar target boundaries sharp.
        """
        u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        filtered = cv2.bilateralFilter(
            u8,
            self.config["bilateral_d"],
            self.config["bilateral_sigma_color"],
            self.config["bilateral_sigma_space"],
        )
        return filtered.astype(np.float32) / 255.0

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Contrast-Limited Adaptive Histogram Equalisation.
        Locally boosts contrast to make sonar targets more distinctive for AKAZE.
        """
        u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip_limit"],
            tileGridSize=self.config["clahe_tile_grid"],
        )
        return clahe.apply(u8).astype(np.float32) / 255.0

    def apply_row_background_subtraction(self, img: np.ndarray) -> np.ndarray:
        """
        Remove horizontal banding by subtracting the per-row 10th-percentile background.
        This eliminates the dominant seafloor reflection band that consumes CLAHE's
        contrast budget and prevents AKAZE from finding structural texture.
        """
        f = img.astype(np.float32) / 255.0 if img.dtype != np.float32 else img
        bg = np.quantile(f, 0.1, axis=1, keepdims=True)
        f  = np.clip(f - bg, 0.0, None)
        if f.max() > 0:
            f /= f.max() + 1e-6
        return f

    def process_image(self, img: np.ndarray) -> np.ndarray:
        """
        Active pipeline: crop → log-compress → bilateral → CLAHE.

        1. crop_water_column  — discard the water column (rows above crop_row).
        2. apply_log_compress — log-compress then min-max stretch to [0, 255].
                                Converts multiplicative sonar speckle to approximately
                                additive noise so bilateral and CLAHE operate correctly.
        3. apply_bilateral    — edge-preserving denoise before contrast enhancement
                                so CLAHE does not amplify speckle noise.
        4. apply_clahe        — local contrast boost to maximise AKAZE keypoints.
        """
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)

        img = self.crop_water_column(img)
        img = self.apply_log_compress(img)
        img = self.apply_bilateral(img)
        img = self.apply_clahe(img)

        return img
