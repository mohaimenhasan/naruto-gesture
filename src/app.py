import cv2
import numpy as np
import pygame

from src.capture import FaceCapture, HandCapture
from src.effects.rasengan import RasenganEffect
from src.effects.sharingan import SharinganEffect


def cv2_frame_to_pygame(frame, target_size):
    """Convert an OpenCV BGR frame to a Pygame surface."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size)
    return pygame.surfarray.make_surface(frame.swapaxes(0, 1))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    hand_capture = HandCapture()
    face_capture = FaceCapture()

    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Naruto Gesture")
    clock = pygame.time.Clock()

    sharingan = SharinganEffect(screen_w, screen_h)
    rasengan = RasenganEffect(screen_w, screen_h)

    sharingan_active = False
    rasengan_active = False

    # Blink detection state for Sharingan toggle
    eyes_closed_frames = 0
    BLINK_HOLD_FRAMES = 60  # ~2 seconds at 30fps
    eyes_were_held = False
    sharingan_cooldown = 0
    frame_count = 0

    # Auto-calibration for blink detection
    calibration_ears = []
    CALIBRATION_FRAMES = 150  # ~5 seconds at 30fps
    calibrated = False

    print("\n=== NARUTO GESTURE ===")
    print("Show open palm → Rasengan (keeps growing!)")
    print("Close eyes 2s+ then reopen → toggle Sharingan")
    print("Press [Q] or [ESC] to quit\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # --- Face tracking + blink detection ---
        left_eye, right_eye, eyes_closed = face_capture.get_eye_positions(frame)

        # Auto-calibration: silently collect EAR during first 5 seconds
        if not calibrated and left_eye is not None:
            calibration_ears.append(face_capture._last_ear)
            if frame_count >= CALIBRATION_FRAMES and len(calibration_ears) > 30:
                sorted_ears = sorted(calibration_ears)
                n = len(sorted_ears)
                avg_open = np.mean(sorted_ears[int(n * 0.7):])
                avg_closed = np.mean(sorted_ears[:max(1, int(n * 0.1))])
                face_capture.ear_threshold = (avg_open + avg_closed) / 2
                calibrated = True
                print(f"  Auto-calibrated! Threshold: {face_capture.ear_threshold:.3f}")

        if frame_count % 30 == 0 and left_eye is not None:
            status = "calibrating..." if not calibrated else f"threshold: {face_capture.ear_threshold:.3f}"
            print(f"  [DEBUG] EAR: {face_capture._last_ear:.4f} | Closed: {eyes_closed} | Hold: {eyes_closed_frames}/{BLINK_HOLD_FRAMES} | {status}")
        frame_count += 1

        # Blink-to-toggle Sharingan
        if sharingan_cooldown > 0:
            sharingan_cooldown -= 1

        if eyes_closed:
            eyes_closed_frames += 1
            if eyes_closed_frames >= BLINK_HOLD_FRAMES:
                eyes_were_held = True
        else:
            if eyes_were_held and sharingan_cooldown <= 0:
                sharingan_active = not sharingan_active
                sharingan_cooldown = 60
                if sharingan_active:
                    sharingan.reset()
                    print("  SHARINGAN ACTIVATED!")
                else:
                    print("  Sharingan deactivated")
            eyes_closed_frames = 0
            eyes_were_held = False

        # --- Hand detection: open palm triggers Rasengan ---
        hands_data, hand_results = hand_capture.extract_landmarks(frame)

        palm_detected = False
        rasengan_pos = None
        if hand_results.hand_landmarks:
            h_lm = hand_results.hand_landmarks[0]
            # Detect open palm: all fingertips above their base knuckles (y decreases upward)
            # This naturally detects an open hand facing the camera
            tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
            bases = [2, 5, 9, 13, 17]  # corresponding base joints
            fingers_open = sum(1 for t, b in zip(tips[1:], bases[1:]) if h_lm[t].y < h_lm[b].y)
            # Thumb: check x-distance (works for both hands)
            thumb_open = abs(h_lm[4].x - h_lm[2].x) > 0.05

            if fingers_open >= 3 and thumb_open:
                palm_detected = True
                palm_x = (h_lm[0].x + h_lm[9].x + h_lm[5].x + h_lm[17].x) / 4
                palm_y = (h_lm[0].y + h_lm[9].y + h_lm[5].y + h_lm[17].y) / 4
                rasengan_pos = (
                    int(palm_x * screen_w),
                    int(palm_y * screen_h),
                )

        if palm_detected:
            if not rasengan_active:
                rasengan.reset()
                rasengan_active = True
            rasengan.update(rasengan_pos)
            # Rasengan keeps expanding while palm is held
            rasengan.base_radius = min(rasengan.base_radius + 0.5, 400)
        else:
            if rasengan_active:
                rasengan_active = False
                rasengan.base_radius = 150
                rasengan.reset()

        # --- Render ---
        rendered_frame = frame.copy()

        # Sharingan eye overlay
        if sharingan_active and left_eye is not None:
            rendered_frame = sharingan.render(rendered_frame, (left_eye, right_eye))

        surface = cv2_frame_to_pygame(rendered_frame, (screen_w, screen_h))
        screen.blit(surface, (0, 0))

        # Rasengan on top
        if rasengan_active:
            dark = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 80))
            screen.blit(dark, (0, 0))
            rasengan.render(screen)

        # HUD
        hud_lines = []
        if eyes_closed and eyes_closed_frames > 0:
            progress = min(eyes_closed_frames / BLINK_HOLD_FRAMES * 100, 100)
            hud_lines.append(f"SHARINGAN CHARGING: {progress:.0f}%")
        if sharingan_active:
            hud_lines.append("SHARINGAN: ACTIVE")
        if rasengan_active:
            size_pct = int((rasengan.base_radius / 400) * 100)
            hud_lines.append(f"RASENGAN: {size_pct}% POWER")

        if hud_lines:
            hud = np.zeros((35 * len(hud_lines) + 10, 500, 3), dtype=np.uint8)
            for i, line in enumerate(hud_lines):
                if "CHARGING" in line:
                    color = (0, 165, 255)
                elif "SHARINGAN" in line:
                    color = (0, 0, 255)
                elif "RASENGAN" in line:
                    color = (255, 200, 0)
                else:
                    color = (0, 255, 255)
                cv2.putText(hud, line, (10, 30 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            hud_rgb = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)
            hud_surface = pygame.surfarray.make_surface(hud_rgb.swapaxes(0, 1))
            hud_surface.set_colorkey((0, 0, 0))
            screen.blit(hud_surface, (10, 10))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    hand_capture.release()
    face_capture.release()
    pygame.quit()


if __name__ == "__main__":
    main()
