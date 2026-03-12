import math
import os

import cv2
import numpy as np
import pygame


class SharinganEffect:
    """Overlays Sharingan on detected eyes using the camera feed."""

    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.sharingan_img = self._load_or_generate_sharingan()
        self.rotation_angle = 0
        self.activation_alpha = 0.0

    def _load_or_generate_sharingan(self):
        """Load sharingan.png from assets, or generate one procedurally."""
        asset_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "sharingan.png"
        )

        if os.path.isfile(asset_path):
            img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img

        return self._generate_sharingan(256)

    def _generate_sharingan(self, size):
        """Generate a procedural Sharingan image with transparency."""
        img = np.zeros((size, size, 4), dtype=np.uint8)
        center = size // 2
        radius = size // 2 - 4

        # Outer red ring
        cv2.circle(img, (center, center), radius, (0, 0, 200, 255), 3)
        # Inner red fill
        cv2.circle(img, (center, center), radius - 2, (0, 0, 180, 200), -1)
        # Black pupil
        cv2.circle(img, (center, center), radius // 4, (0, 0, 0, 255), -1)

        # Three tomoe (comma-shaped marks)
        for i in range(3):
            angle = i * (2 * math.pi / 3)
            tomoe_r = radius * 0.55
            tx = int(center + tomoe_r * math.cos(angle))
            ty = int(center + tomoe_r * math.sin(angle))
            cv2.circle(img, (tx, ty), radius // 7, (0, 0, 0, 255), -1)
            # Tail of tomoe
            tail_angle = angle + 0.6
            tail_x = int(center + tomoe_r * 0.85 * math.cos(tail_angle))
            tail_y = int(center + tomoe_r * 0.85 * math.sin(tail_angle))
            cv2.line(img, (tx, ty), (tail_x, tail_y), (0, 0, 0, 255), 3)

        # Red ring border
        cv2.circle(img, (center, center), radius, (0, 0, 255, 255), 2)

        return img

    def _rotate_image(self, image, angle):
        """Rotate image around its center while preserving transparency."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
        )
        return rotated

    def _overlay_on_frame(self, frame, overlay, x, y, size):
        """Place a BGRA overlay onto a BGR frame at (x, y) with given size."""
        resized = cv2.resize(overlay, (size, size), interpolation=cv2.INTER_AREA)
        rotated = self._rotate_image(resized, self.rotation_angle)

        h, w = frame.shape[:2]
        x1 = max(0, x - size // 2)
        y1 = max(0, y - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Crop overlay to fit within frame bounds
        ox1 = max(0, -(x - size // 2))
        oy1 = max(0, -(y - size // 2))
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        overlay_crop = rotated[oy1:oy2, ox1:ox2]
        if overlay_crop.shape[2] == 4:
            alpha = overlay_crop[:, :, 3:4].astype(float) / 255.0
            alpha *= min(self.activation_alpha, 1.0)
            bg = frame[y1:y2, x1:x2].astype(float)
            fg = overlay_crop[:, :, :3].astype(float)
            frame[y1:y2, x1:x2] = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)

        return frame

    def render(self, camera_frame, eye_data):
        """Render Sharingan on eyes. eye_data = ((left_center, left_r), (right_center, right_r))."""
        self.rotation_angle = (self.rotation_angle + 2) % 360
        self.activation_alpha = min(self.activation_alpha + 0.05, 1.0)

        frame = camera_frame.copy()

        left_eye, right_eye = eye_data
        if left_eye:
            center, radius = left_eye
            # Size to pupil/iris only (not entire eye)
            self._overlay_on_frame(frame, self.sharingan_img, center[0], center[1], int(radius * 0.9))
        if right_eye:
            center, radius = right_eye
            self._overlay_on_frame(frame, self.sharingan_img, center[0], center[1], int(radius * 0.9))

        # Add red tint to the overall frame
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 40
        alpha = self.activation_alpha * 0.3
        frame = cv2.addWeighted(frame, 1 - alpha, red_overlay, alpha, 0)

        return frame

    def reset(self):
        self.activation_alpha = 0.0
        self.rotation_angle = 0
