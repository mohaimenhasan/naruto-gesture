import math
import random

import pygame


class RasenganEffect:
    """Palm-sized spinning Rasengan energy sphere that grows while held."""

    START_RADIUS = 75
    MAX_RADIUS = 150
    GROW_RATE = 0.4
    SHRINK_RATE = 2.0

    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.particles = []
        self.ring_angle = 0
        self.core_pulse = 0
        self.activation_time = 0
        self.hand_pos = (screen_width // 2, screen_height // 2)
        self.base_radius = self.START_RADIUS
        self.fading = False
        self.alive = False

    def update(self, hand_position=None):
        """Update effect state each frame."""
        if hand_position:
            self.hand_pos = hand_position

        if self.fading:
            # Shrink faster as it gets smaller (proportional to remaining size)
            shrink = max(self.SHRINK_RATE, self.base_radius * 0.08)
            self.base_radius -= shrink
            if self.base_radius <= 0:
                self.base_radius = 0
                self.alive = False
                return
        else:
            self.base_radius = min(self.base_radius + self.GROW_RATE, self.MAX_RADIUS)

        self.activation_time += 1
        self.ring_angle += 8
        self.core_pulse = math.sin(self.activation_time * 0.15) * (self.base_radius * 0.06)

        # Spawn spiral particles scaled to current radius
        count = max(2, int(6 * (self.base_radius / self.MAX_RADIUS)))
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, self.base_radius * 1.2)
            speed = random.uniform(0.5, 2)
            orbit_speed = random.uniform(3, 8)
            self.particles.append({
                "angle": angle,
                "dist": dist,
                "speed": speed,
                "orbit_speed": orbit_speed,
                "life": random.randint(10, 30),
                "size": max(1, random.randint(1, int(self.base_radius / 25) + 1)),
                "brightness": random.randint(150, 255),
            })

        # Update particles
        for p in self.particles:
            p["angle"] += math.radians(p["orbit_speed"])
            p["dist"] += p["speed"] * 0.2
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def start_fading(self):
        """Begin the shrink-and-disappear animation."""
        self.fading = True

    def render(self, surface):
        """Render the Rasengan effect onto a Pygame surface."""
        if self.base_radius <= 1:
            return
        cx, cy = self.hand_pos
        pulse_r = max(2, self.base_radius + self.core_pulse)

        # Outer glow
        for r in range(int(pulse_r * 2), int(pulse_r), -3):
            alpha = int(30 * (1 - (r - pulse_r) / pulse_r))
            glow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (50, 120, 255, max(0, alpha)), (r, r), r)
            surface.blit(glow, (cx - r, cy - r))

        # Spinning rings
        for i in range(3):
            ring_offset = self.ring_angle + i * 120
            ring_r = pulse_r * 0.8
            points = []
            for a in range(0, 360, 5):
                rad = math.radians(a + ring_offset)
                wobble = math.sin(rad * 3 + self.activation_time * 0.1) * 5
                px = cx + int((ring_r + wobble) * math.cos(rad))
                py = cy + int((ring_r + wobble) * math.sin(rad) * 0.4)
                points.append((px, py))
            if len(points) > 2:
                brightness = 180 + i * 25
                pygame.draw.lines(
                    surface, (brightness // 2, brightness, 255), True, points, 2
                )

        # Spiral particles
        for p in self.particles:
            px = cx + int(p["dist"] * math.cos(p["angle"]))
            py = cy + int(p["dist"] * math.sin(p["angle"]))
            b = p["brightness"]
            life_ratio = p["life"] / 40
            color = (int(b * 0.3 * life_ratio), int(b * 0.7 * life_ratio), int(b * life_ratio))
            pygame.draw.circle(surface, color, (px, py), p["size"])

        # Core sphere (bright center)
        core_r = max(2, int(pulse_r * 0.4))
        core_surface = pygame.Surface((core_r * 2, core_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(core_surface, (180, 220, 255, 200), (core_r, core_r), core_r)
        pygame.draw.circle(core_surface, (220, 240, 255, 255), (core_r, core_r), core_r // 2)
        surface.blit(core_surface, (cx - core_r, cy - core_r))

    def reset(self):
        self.particles.clear()
        self.ring_angle = 0
        self.core_pulse = 0
        self.activation_time = 0
        self.base_radius = self.START_RADIUS
        self.fading = False
        self.alive = False
