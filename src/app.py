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
    classifier = GestureClassifier(confidence_threshold=0.65)

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
    jutsu_cooldown = 0
    COOLDOWN_FRAMES = 10
    DEACTIVATE_FRAMES = 30
    no_gesture_count = 0

    font = pygame.font.Font(None, 36)

    print("\n=== BATTLE MODE ===")
    print("Perform gestures to activate jutsus!")
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

        # Detect gesture
        hands_data, hand_results = hand_capture.extract_landmarks(frame)
        gesture = None
        confidence = 0.0

        if hands_data and jutsu_cooldown <= 0:
            gesture, confidence = classifier.predict(hands_data[0])

        if gesture:
            no_gesture_count = 0
            if gesture != active_jutsu:
                active_jutsu = gesture
                jutsu_cooldown = COOLDOWN_FRAMES
                sharingan.reset()
                chidori.reset()
                rasengan.reset()
        else:
            no_gesture_count += 1
            if no_gesture_count > DEACTIVATE_FRAMES:
                active_jutsu = None
                sharingan.reset()
                chidori.reset()
                rasengan.reset()

        jutsu_cooldown = max(0, jutsu_cooldown - 1)

        # Render based on active jutsu
        if active_jutsu == "sharingan":
            eye_data = face_capture.get_eye_positions(frame)
            if eye_data[0] is not None:
                rendered = sharingan.render(frame, eye_data)
            else:
                rendered = frame
            # Convert OpenCV frame to Pygame surface
            surface = cv2_frame_to_pygame(rendered, (screen_w, screen_h))
            screen.blit(surface, (0, 0))

        elif active_jutsu in ("chidori", "rasengan"):
            # Show camera feed as background
            surface = cv2_frame_to_pygame(frame, (screen_w, screen_h))
            screen.blit(surface, (0, 0))

            # Dark overlay for effect visibility
            dark = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 120))
            screen.blit(dark, (0, 0))

            # Get hand position for effect centering
            hand_pos = None
            if hand_results.multi_hand_landmarks:
                h_lm = hand_results.multi_hand_landmarks[0]
                hand_pos = (
                    int(h_lm.landmark[9].x * screen_w),
                    int(h_lm.landmark[9].y * screen_h),
                )

            if active_jutsu == "chidori":
                chidori.update(hand_pos)
                chidori.render(screen)
            else:
                rasengan.update(hand_pos)
                rasengan.render(screen)
        else:
            # No active jutsu — just show camera feed
            surface = cv2_frame_to_pygame(frame, (screen_w, screen_h))
            screen.blit(surface, (0, 0))

        # HUD overlay
        if active_jutsu:
            jutsu_text = font.render(f"JUTSU: {active_jutsu.upper()}", True, (255, 255, 0))
            screen.blit(jutsu_text, (20, 20))
        if gesture:
            conf_text = font.render(f"Confidence: {confidence:.0%}", True, (200, 200, 200))
            screen.blit(conf_text, (20, 55))

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
