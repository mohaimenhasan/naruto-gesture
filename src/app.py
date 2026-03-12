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
    BLINK_HOLD_FRAMES = 30  # ~1 second at 30fps
    eyes_were_held = False
    sharingan_cooldown = 0
    frame_count = 0

    # Auto-calibration for blink detection
    calibration_ears = []
    CALIBRATION_FRAMES = 150  # ~5 seconds at 30fps
    calibrated = False

    # Rasengan palm detection grace period
    palm_lost_frames = 0
    PALM_GRACE_FRAMES = 8

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
        fingers_open = 0
        thumb_open = False
        if hand_results.hand_landmarks:
            h_lm = hand_results.hand_landmarks[0]
            wrist = h_lm[0]

            def dist3d(a, b):
                return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

            # Orientation-independent: finger is open if tip is farther
            # from wrist than the DIP (second-to-last) joint
            tip_ids = [8, 12, 16, 20]
            dip_ids = [6, 10, 14, 18]
            fingers_open = sum(
                1 for t, d in zip(tip_ids, dip_ids)
                if dist3d(h_lm[t], wrist) > dist3d(h_lm[d], wrist)
            )
            # Thumb: tip farther from wrist than IP joint
            thumb_open = dist3d(h_lm[4], wrist) > dist3d(h_lm[3], wrist)

            if fingers_open >= 3 and thumb_open:
                palm_detected = True
                palm_x = (h_lm[9].x + h_lm[5].x + h_lm[17].x) / 3
                palm_y = (h_lm[9].y + h_lm[5].y + h_lm[17].y) / 3
                PALM_OFFSET_Y = -180
                rasengan_pos = (
                    int(palm_x * screen_w),
                    int(palm_y * screen_h) + PALM_OFFSET_Y,
                )

        if palm_detected and calibrated:
            palm_lost_frames = 0
            if not rasengan_active:
                rasengan.reset()
                rasengan.alive = True
                rasengan_active = True
            if rasengan.fading:
                rasengan.fading = False
            rasengan.update(rasengan_pos)
        else:
            if rasengan_active:
                palm_lost_frames += 1
                if palm_lost_frames >= PALM_GRACE_FRAMES and not rasengan.fading:
                    rasengan.start_fading()
                if rasengan.fading:
                    rasengan.update()
                if not rasengan.alive:
                    rasengan_active = False
                    rasengan.reset()
                    palm_lost_frames = 0

        # --- Render ---
        rendered_frame = frame.copy()

        # Debug overlay: draw hand detection info on camera feed
        if hand_results.hand_landmarks:
            h_lm = hand_results.hand_landmarks[0]
            fh, fw = rendered_frame.shape[:2]
            all_tips = [4, 8, 12, 16, 20]

            # Draw landmarks
            for i, lm in enumerate(h_lm):
                px, py = int(lm.x * fw), int(lm.y * fh)
                color = (0, 255, 0) if i in all_tips else (255, 255, 0)
                cv2.circle(rendered_frame, (px, py), 4, color, -1)

            # Draw palm center + rasengan target
            palm_cx = int(((h_lm[9].x + h_lm[5].x + h_lm[17].x) / 3) * fw)
            palm_cy = int(((h_lm[9].y + h_lm[5].y + h_lm[17].y) / 3) * fh)
            cv2.circle(rendered_frame, (palm_cx, palm_cy), 8, (0, 0, 255), -1)
            cv2.putText(rendered_frame, "PALM", (palm_cx + 10, palm_cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Detection info text
            det_text = f"Fingers: {fingers_open}/4  Thumb: {'Y' if thumb_open else 'N'}  Palm: {'YES' if palm_detected else 'NO'}"
            cv2.putText(rendered_frame, det_text, (10, fh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if rasengan_active:
                ras_text = f"Rasengan r={rasengan.base_radius:.0f}  fading={rasengan.fading}  lost={palm_lost_frames}"
                cv2.putText(rendered_frame, ras_text, (10, fh - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        else:
            fh = rendered_frame.shape[0]
            cv2.putText(rendered_frame, "No hand detected", (10, fh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw tracking outlines during calibration
        if not calibrated:
            hand_capture.draw_landmarks(rendered_frame, hand_results)
            if left_eye is not None:
                cv2.circle(rendered_frame, left_eye[0], left_eye[1], (0, 255, 255), 2)
                cv2.circle(rendered_frame, right_eye[0], right_eye[1], (0, 255, 255), 2)

        # Sharingan eye overlay
        if sharingan_active and left_eye is not None:
            rendered_frame = sharingan.render(rendered_frame, (left_eye, right_eye), eyes_closed=eyes_closed)

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
        if not calibrated:
            pct = min(frame_count / CALIBRATION_FRAMES * 100, 100)
            hud_lines.append(f"CALIBRATING: {pct:.0f}%")
        if eyes_closed and eyes_closed_frames > 0:
            progress = min(eyes_closed_frames / BLINK_HOLD_FRAMES * 100, 100)
            hud_lines.append(f"SHARINGAN CHARGING: {progress:.0f}%")
        if sharingan_active:
            hud_lines.append("SHARINGAN: ACTIVE")
        if rasengan_active:
            size_pct = int((rasengan.base_radius / RasenganEffect.MAX_RADIUS) * 100)
            hud_lines.append(f"RASENGAN: {size_pct}% POWER")

        if hud_lines:
            hud = np.zeros((35 * len(hud_lines) + 10, 500, 3), dtype=np.uint8)
            for i, line in enumerate(hud_lines):
                if "CALIBRATING" in line:
                    color = (0, 255, 255)
                elif "CHARGING" in line:
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
