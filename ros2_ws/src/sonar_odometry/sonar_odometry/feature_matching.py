import numpy as np
import cv2




class SonarFeatureMatcher:

    def __init__(self, lowe_ratio: float = 0.75, distance_threshold: float = 0.5,
                 crop_row: int = 0, pixel_ransac_threshold: float = 3.0,
                 use_pixel_ransac: bool = True):
        """
        Parameters
        ----------
        lowe_ratio             : Lowe's ratio-test threshold for match filtering.
        distance_threshold     : Stage-2 Cartesian LMEDS threshold (metres).
        crop_row               : Rows cropped from the top before feature detection.
        pixel_ransac_threshold : Stage-1 pixel-space RANSAC inlier threshold (pixels).
                                 Controls how far a match is allowed to deviate from the
                                 consensus translation in image (col, row) space.
                                 Too large → accepts false matches → inflated path.
                                 Too small → rejects real matches → compressed or noisy path.
        use_pixel_ransac       : If True (default), run Stage-1 pixel RANSAC before the
                                 Cartesian LMEDS. If False, all Lowe-ratio matches are
                                 passed directly to the Cartesian LMEDS (Stage 2 only).
        """
        self._lowe_ratio             = lowe_ratio
        self._distance_threshold     = distance_threshold
        self._crop_row               = crop_row
        self._pixel_ransac_threshold = pixel_ransac_threshold
        self._use_pixel_ransac       = use_pixel_ransac
        # Sonar geometry — populated from the first ROS message via update_sonar_params()
        self._beam_azimuths_rad: np.ndarray | None = None   # shape (beam_count,)
        self._ranges_m: np.ndarray | None = None             # shape (range_bins,)
        self._detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.001,
            nOctaves=8,
            nOctaveLayers=8,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )

    def set_sonar_params(self, beam_azimuths_rad: np.ndarray, ranges_m: np.ndarray) -> None:
        """
        Set sonar geometry directly from numpy arrays.
        Useful for notebooks and unit tests where no ROS message is available.

        Parameters
        ----------
        beam_azimuths_rad : azimuth angle (rad) for each beam column, shape (n_beams,)
        ranges_m          : range (m) at each bin centre, shape (n_bins,)
        """
        self._beam_azimuths_rad = np.asarray(beam_azimuths_rad, dtype=np.float32)
        self._ranges_m          = np.asarray(ranges_m,          dtype=np.float32)

    def update_sonar_params(self, msg) -> None:
        """
        Extract and cache sonar geometry from a ProjectedSonarImage message.
        Called once on the first message; ignored on subsequent calls.

        Geometry convention (from marine_acoustic_msgs):
          - Z-axis forward (boresight), beams in Y-Z plane
          - Azimuth = rotation about X  →  atan2(beam_direction.y, beam_direction.z)
          - msg.ranges contains the actual range (m) of each bin centre
        """
        if self._beam_azimuths_rad is not None:
            return  # already cached

        if not msg.beam_directions or not msg.ranges:
            return  # message not yet populated

        self._beam_azimuths_rad = np.array(
            [np.arctan2(bd.y, bd.z) for bd in msg.beam_directions],
            dtype=np.float32,
        )
        self._ranges_m = np.array(msg.ranges, dtype=np.float32)

    def polar_to_cartesian_coords(self, col: float, row: float) -> np.ndarray:
        """
        Convert polar sonar image coordinates to Cartesian (metres).

        Uses the geometry cached from the ROS message via update_sonar_params().
        Sub-pixel (col, row) values are handled with linear interpolation.

        Returns [x, y]:  x = forward (boresight),  y = right (positive starboard)
        """
        if self._beam_azimuths_rad is None or self._ranges_m is None:
            raise RuntimeError(
                "Sonar geometry not set. Call update_sonar_params(msg) before matching."
            )

        col_idx = np.arange(len(self._beam_azimuths_rad), dtype=np.float32)
        row_idx = np.arange(len(self._ranges_m),          dtype=np.float32)

        azimuth = float(np.interp(col, col_idx, self._beam_azimuths_rad))
        range_m = float(np.interp(row, row_idx, self._ranges_m))

        x = range_m * np.cos(azimuth)    # forward
        y = -range_m * np.sin(azimuth)  # right (positive starboard); negate because beam_directions.y is port-positive in marine_acoustic_msgs
        return np.array([x, y], dtype=np.float32)

######################## AKAZE ###############################
    

    def detect_sonar_features(self, image):
        """
        Detect features optimized for sonar images using AKAZE.
        """
        
        keypoints, descriptors = self._detector.detectAndCompute(image, None)
        
        return keypoints, descriptors

    def match_sonar_features(self, desc1, desc2, distance_threshold=None):
        """
        Match features using BFMatcher with Hamming distance for MLDB_UPRIGHT.
        """
        threshold = distance_threshold if distance_threshold is not None else self._lowe_ratio

        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        # Use BFMatcher with HAMMING distance for binary descriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Find 2 nearest neighbors for ratio test
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def estimate_transformation(self, kp1, kp2, matches):
        """
        Estimate transformation between matched keypoints using a two-stage pipeline.

        Stage 1 — Pixel-space RANSAC
        ─────────────────────────────
        RANSAC runs on raw (col, row) image coordinates.

        Why pixel space first: for a wide-FOV forward-looking sonar, pure forward
        surge shifts scatterers by d·cos(θ)/range_per_bin rows. cos(θ) varies from
        1.0 at boresight to ~0.42 at ±65°, so the row shift is not perfectly uniform
        — but it is far more uniform than in Cartesian space, where the same surge
        appears as a cos²(θ)-weighted x-shift that looks like a mix of translation
        and distortion to a Cartesian RANSAC. Single-stage Cartesian RANSAC on a
        wide-FOV sonar tends to fit different consensus groups frame-to-frame,
        causing forward/backward sign flips.

        Pixel RANSAC finds a stable inlier set in the space where the dominant
        (forward) motion has the strongest and most consistent signal.

        Threshold tuning: pixel_ransac_threshold ≈ 2× expected per-frame boresight
        row shift. See __init__ docstring for the formula and per-speed examples.

        Stage 2 — Cartesian LMEDS on pixel inliers
        ────────────────────────────────────────────
        The pixel-inlier subset is converted to Cartesian metres and a second
        robust fit (LMEDS) extracts the physical translation and heading change.
        No second RANSAC is needed because outliers are already removed.

        Returns
        -------
        dict with keys:
          transformation         : 2×3 Cartesian affine (T_scene: pts2 = T·pts1) or None
          transformation_pixel   : 2×3 pixel affine (T_scene in image coords) or None
          inliers                : bool mask over `matches`
          num_inliers            : int
          inlier_ratio           : float
        """
        _empty = {
            "transformation":       None,
            "transformation_pixel": None,
            "inliers":              None,
            "num_inliers":          0,
            "inlier_ratio":         0.0,
        }
        if len(matches) < 4:
            return _empty

        pts1_polar = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2_polar = np.float32([kp2[m.trainIdx].pt for m in matches])

        # ── Stage 1: pixel-space RANSAC (optional) ────────────────────────────
        if self._use_pixel_ransac:
            T_pixel, inliers_px = cv2.estimateAffinePartial2D(
                pts1_polar, pts2_polar,
                method=cv2.RANSAC,
                ransacReprojThreshold=self._pixel_ransac_threshold,
                maxIters=5000,
                confidence=0.99,
                refineIters=200,
            )
            if T_pixel is None or inliers_px is None:
                return _empty

            u, s, vt = np.linalg.svd(T_pixel[:2, :2])
            T_pixel[:2, :2] = u @ vt

            inlier_mask  = inliers_px.ravel().astype(bool)
            num_inliers  = int(inlier_mask.sum())
            inlier_ratio = num_inliers / len(matches)
        else:
            # Skip pixel RANSAC: pass all Lowe-ratio matches to Cartesian LMEDS
            T_pixel      = None
            inlier_mask  = np.ones(len(matches), dtype=bool)
            num_inliers  = len(matches)
            inlier_ratio = 1.0

        # ── Stage 2: Cartesian LMEDS on pixel inliers ────────────────────────
        # Row coordinates come from the cropped image — add _crop_row to recover
        # the original range-bin index before interpolating into ranges_m.
        T_cart: np.ndarray | None = None
        if num_inliers >= 4:
            pts1_cart = np.float32([
                self.polar_to_cartesian_coords(col, row + self._crop_row)
                for col, row in pts1_polar[inlier_mask]
            ])
            pts2_cart = np.float32([
                self.polar_to_cartesian_coords(col, row + self._crop_row)
                for col, row in pts2_polar[inlier_mask]
            ])
            T_cart, _ = cv2.estimateAffinePartial2D(
                pts1_cart, pts2_cart,
                method=cv2.LMEDS,
                ransacReprojThreshold=self._distance_threshold,
                maxIters=2000,
                confidence=0.99,
                refineIters=100,
            )
            if T_cart is not None:
                u, s, vt = np.linalg.svd(T_cart[:2, :2])
                T_cart[:2, :2] = u @ vt

        return {
            "transformation":       T_cart,
            "transformation_pixel": T_pixel,
            "inliers":              inlier_mask,
            "num_inliers":          num_inliers,
            "inlier_ratio":         inlier_ratio,
        }
    

    def process_sonar_image_pair(self, img1, img2, timestamp=None):
        """
        Process a pair of consecutive sonar images to estimate motion.
        
        Args:
            img1: Previous sonar image
            img2: Current sonar image
            timestamp: Current timestamp (optional)
            visualize: If True, display feature matches (default: False)
        
        Returns:
            Dictionary with processing results
        """

        # Detect sonar-optimized features
        kp1, desc1 = self.detect_sonar_features(img1)
        kp2, desc2 = self.detect_sonar_features(img2)
        matches = self.match_sonar_features(desc1, desc2)
        
        # Estimate transformation
        result = self.estimate_transformation(kp1, kp2, matches)
                
        # Update trajectory if transformation is valid and has sufficient inliers
        min_inliers = 6  # Minimum inliers for reliable sonar odometry
        if (result['transformation'] is not None and
            result['num_inliers'] >= min_inliers and
            result['inlier_ratio'] > 0.3):  # At least 30% inlier ratio
            # debug
            #print(f"number inliers: {result['num_inliers']}")
            pass
        
        return result, kp1,kp2,matches
    

