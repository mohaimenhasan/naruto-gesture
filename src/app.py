import argparse
import sys
import time

import cv2
import numpy as np
import pygame

from src.capture import FaceCapture, HandCapture
from src.classifier import GestureClassifier
from src.effects.chidori import ChidoriEffect
from src.effects.rasengan import RasenganEffect
from src.effects.sharingan import SharinganEffect
from src.trainer import GESTURE_KEYS, save_sample, train_model


def run_training_mode():
    """Collect gesture samples and train the classifier."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    hand_capture = HandCapture()
    sample_counts = {}

    print("\n=== TRAINING MODE ===")
    print("Show your hand gesture to the camera, then press:")
    print("  [S] → Sharingan")
    print("  [C] → Chidori")
    print("  [R] → Rasengan")
    print("  [Q] → Quit and train model")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands_data, results = hand_capture.extract_landmarks(frame)
        frame = hand_capture.draw_landmarks(frame, results)

        # Display sample counts
        y_offset = 30
        cv2.putText(frame, "TRAINING MODE", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for label, count in sample_counts.items():
            y_offset += 30
            cv2.putText(frame, f"{label}: {count} samples", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if hands_data:
            cv2.putText(frame, "Hand detected! Press S/C/R to save", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "No hand detected", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow("Naruto Gesture Training", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            key = ord(chr(key).lower()) if key < 128 else key
        if key == ord("q"):
            break
        elif key in GESTURE_KEYS and hands_data:
            label = GESTURE_KEYS[key]
            save_sample(hands_data[0], label)
            sample_counts[label] = sample_counts.get(label, 0) + 1
            print(f"  Saved {label} sample (total: {sample_counts[label]})")

    cap.release()
    cv2.destroyAllWindows()
    hand_capture.release()

    print("\nTraining classifier...")
    train_model()


def cv2_frame_to_pygame(frame, target_size):
    """Convert an OpenCV BGR frame to a Pygame surface."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size)
    return pygame.surfarray.make_surface(frame.swapaxes(0, 1))


def run_battle_mode():
    """Live gesture detection with full-screen jutsu effects."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    hand_capture = HandCapture()
    face_capture = FaceCapture()
    classifier = GestureClassifier(confidence_threshold=0.80)

    if classifier.model is None:
        print("No trained model. Run training mode first: python -m src.app --mode train")
        cap.release()
        return

    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Naruto Gesture Battle")
    clock = pygame.time.Clock()

    sharingan = SharinganEffect(screen_w, screen_h)
    chidori = ChidoriEffect(screen_w, screen_h)
    rasengan = RasenganEffect(screen_w, screen_h)

    active_jutsu = None
    sharingan_active = False
    jutsu_cooldown = 0
    COOLDOWN_FRAMES = 15
    DEACTIVATE_FRAMES = 30
    no_gesture_count = 0

    # Blink detection state for Sharingan toggle
    eyes_closed_frames = 0
    BLINK_HOLD_FRAMES = 60  # ~2 seconds at 30fps
    eyes_were_held = False
    sharingan_cooldown = 0
    frame_count = 0

    print("\n=== BATTLE MODE ===")
    print("Perform gestures to activate jutsus!")
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

        # Face tracking (needed for both blink detection and Sharingan overlay)
        eye_data = face_capture.get_eye_positions(frame)
        left_eye, right_eye, eyes_closed = eye_data

        # Debug: print face/blink state every 30 frames (~1/sec)
        if frame_count % 30 == 0:
            if left_eye is None:
                print(f"  [DEBUG] No face detected")
            else:
                ear = getattr(face_capture, '_last_ear', -1)
                nlm = getattr(face_capture, '_num_landmarks', 0)
                print(f"  [DEBUG] Face OK ({nlm} landmarks) | EAR: {ear:.4f} | Closed: {eyes_closed} | Hold: {eyes_closed_frames}/{BLINK_HOLD_FRAMES}")
        frame_count += 1

        # Blink-to-toggle Sharingan: hold eyes closed 3s, then reopen
        if sharingan_cooldown > 0:
            sharingan_cooldown -= 1

        if eyes_closed:
            eyes_closed_frames += 1
            if eyes_closed_frames >= BLINK_HOLD_FRAMES:
                eyes_were_held = True
        else:
            if eyes_were_held and sharingan_cooldown <= 0:
                sharingan_active = not sharingan_active
                sharingan_cooldown = 60  # 2s cooldown to prevent rapid toggling
                if sharingan_active:
                    sharingan.reset()
                    print("  SHARINGAN ACTIVATED!")
                else:
                    print("  Sharingan deactivated")
            eyes_closed_frames = 0
            eyes_were_held = False

        # Detect hand gesture for Chidori/Rasengan
        hands_data, hand_results = hand_capture.extract_landmarks(frame)
        gesture = None
        confidence = 0.0

        if hands_data and jutsu_cooldown <= 0:
            gesture, confidence = classifier.predict(hands_data[0])

        if gesture:
            no_gesture_count = 0
            jutsu_cooldown = COOLDOWN_FRAMES
            if gesture != "sharingan":
                if gesture != active_jutsu:
                    active_jutsu = gesture
                    chidori.reset()
                    rasengan.reset()
        else:
            no_gesture_count += 1
            if no_gesture_count > DEACTIVATE_FRAMES and active_jutsu:
                active_jutsu = None
                chidori.reset()
                rasengan.reset()

        jutsu_cooldown = max(0, jutsu_cooldown - 1)

        # Always start with camera feed, then layer effects on top
        rendered_frame = frame.copy()

        # Sharingan persists independently — overlay on eyes
        if sharingan_active and left_eye is not None:
            rendered_frame = sharingan.render(rendered_frame, (left_eye, right_eye))

        # Convert camera frame (with possible sharingan) to pygame surface
        surface = cv2_frame_to_pygame(rendered_frame, (screen_w, screen_h))
        screen.blit(surface, (0, 0))

        # Layer Chidori/Rasengan on top
        if active_jutsu in ("chidori", "rasengan"):
            dark = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 120))
            screen.blit(dark, (0, 0))

            # Get hand landmark positions for effect placement
            chidori_pos = None
            rasengan_pos = None
            if hand_results.hand_landmarks:
                h_lm = hand_results.hand_landmarks[0]
                # Chidori at fingertip (landmark 12 = middle finger tip)
                chidori_pos = (
                    int(h_lm[12].x * screen_w),
                    int(h_lm[12].y * screen_h),
                )
                # Rasengan hovering above open palm
                # Average of wrist (0) and middle finger base (9) gives palm center,
                # then offset slightly toward fingertips
                palm_x = (h_lm[0].x + h_lm[9].x + h_lm[5].x + h_lm[17].x) / 4
                palm_y = (h_lm[0].y + h_lm[9].y + h_lm[5].y + h_lm[17].y) / 4
                rasengan_pos = (
                    int(palm_x * screen_w),
                    int(palm_y * screen_h),
                )

            if active_jutsu == "chidori":
                chidori.update(chidori_pos)
                chidori.render(screen)
            else:
                rasengan.update(rasengan_pos)
                rasengan.render(screen)

        # HUD overlay
        hud_lines = []
        if eyes_closed and eyes_closed_frames > 0:
            progress = min(eyes_closed_frames / BLINK_HOLD_FRAMES * 100, 100)
            hud_lines.append(f"SHARINGAN CHARGING: {progress:.0f}%")
        if sharingan_active:
            hud_lines.append("SHARINGAN: ACTIVE")
        if active_jutsu and active_jutsu != "sharingan":
            hud_lines.append(f"JUTSU: {active_jutsu.upper()}")
        if gesture:
            hud_lines.append(f"Confidence: {confidence:.0%}")

        if hud_lines:
            hud = np.zeros((35 * len(hud_lines) + 10, 500, 3), dtype=np.uint8)
            for i, line in enumerate(hud_lines):
                if "CHARGING" in line:
                    color = (0, 165, 255)
                elif "SHARINGAN" in line:
                    color = (0, 0, 255)
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


def main():
    parser = argparse.ArgumentParser(description="Naruto Gesture Recognition")
    parser.add_argument(
        "--mode",
        choices=["train", "battle"],
        required=True,
        help="'train' to collect gesture data, 'battle' to activate jutsus",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training_mode()
    elif args.mode == "battle":
        run_battle_mode()


if __name__ == "__main__":
    main()
