import math
import random

import numpy as np
import pygame


class LightningBolt:
    """A single lightning bolt segment."""

    def __init__(self, start, end, width, brightness):
        self.start = start
        self.end = end
        self.width = width
        self.brightness = brightness
        self.lifetime = random.randint(3, 8)
        self.age = 0

    @property
    def alive(self):
        return self.age < self.lifetime


class ChidoriEffect:
    """Full-screen lightning/electricity effect for Chidori activation."""

    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.bolts = []
        self.particles = []
        self.flash_intensity = 0
        self.activation_time = 0
        self.hand_pos = (screen_width // 2, screen_height // 2)

    def _spawn_bolt(self):
        """Create a new lightning bolt from the hand position."""
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(80, 300)
        end_x = self.hand_pos[0] + int(length * math.cos(angle))
        end_y = self.hand_pos[1] + int(length * math.sin(angle))

        points = [self.hand_pos]
        segments = random.randint(4, 10)
        for i in range(1, segments):
            t = i / segments
            x = int(self.hand_pos[0] + t * (end_x - self.hand_pos[0]) + random.randint(-30, 30))
            y = int(self.hand_pos[1] + t * (end_y - self.hand_pos[1]) + random.randint(-30, 30))
            points.append((x, y))
        points.append((end_x, end_y))

        width = random.randint(1, 4)
        brightness = random.randint(180, 255)

        for i in range(len(points) - 1):
            self.bolts.append(LightningBolt(points[i], points[i + 1], width, brightness))

    def _spawn_particles(self):
        """Spawn electric particles around the hand."""
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.particles.append({
                "x": float(self.hand_pos[0]),
                "y": float(self.hand_pos[1]),
                "vx": speed * math.cos(angle),
                "vy": speed * math.sin(angle),
                "life": random.randint(10, 25),
                "size": random.randint(2, 5),
            })

    def update(self, hand_position=None):
        """Update effect state each frame."""
        if hand_position:
            self.hand_pos = hand_position
        self.activation_time += 1

        # Spawn new bolts
        bolt_count = min(3 + self.activation_time // 10, 8)
        for _ in range(bolt_count):
            self._spawn_bolt()

        self._spawn_particles()

        # Update bolts
        for bolt in self.bolts:
            bolt.age += 1
        self.bolts = [b for b in self.bolts if b.alive]

        # Update particles
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

        # Flash effect
        self.flash_intensity = max(0, self.flash_intensity - 5)
        if random.random() < 0.15:
            self.flash_intensity = random.randint(30, 80)

    def render(self, surface):
        """Render the Chidori effect onto a Pygame surface."""
        # Dark blue background flash
        if self.flash_intensity > 0:
            flash = pygame.Surface((self.width, self.height))
            flash.fill((self.flash_intensity // 3, self.flash_intensity // 3, self.flash_intensity))
            flash.set_alpha(self.flash_intensity)
            surface.blit(flash, (0, 0))

        # Draw lightning bolts
        for bolt in self.bolts:
            color = (bolt.brightness // 2, bolt.brightness // 2, bolt.brightness)
            pygame.draw.line(surface, color, bolt.start, bolt.end, bolt.width)
            # Glow effect - wider dimmer line underneath
            glow_color = (bolt.brightness // 6, bolt.brightness // 6, bolt.brightness // 2)
            pygame.draw.line(surface, glow_color, bolt.start, bolt.end, bolt.width + 4)

        # Draw particles
        for p in self.particles:
            alpha_ratio = p["life"] / 25
            brightness = int(255 * alpha_ratio)
            color = (brightness // 2, brightness // 2, brightness)
            pygame.draw.circle(surface, color, (int(p["x"]), int(p["y"])), p["size"])

        # Core glow at hand position
        for radius in range(40, 5, -5):
            alpha = int(60 * (1 - radius / 40))
            glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (100, 150, 255, alpha), (radius, radius), radius)
            surface.blit(glow, (self.hand_pos[0] - radius, self.hand_pos[1] - radius))

    def reset(self):
        self.bolts.clear()
        self.particles.clear()
        self.flash_intensity = 0
        self.activation_time = 0
