# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pygame",
#   "numpy",
#   "onnxruntime",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "tqdm",
# ]
# ///
"""Pygame real-time trajectory visualizer for the comma controls challenge."""

import argparse
import importlib
import logging
import math
import sys
import time

import numpy as np
import pygame
import pygame.gfxdraw

from tinyphysics import (
  ACC_G, CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH, DEL_T,
  LATACCEL_RANGE, STEER_RANGE, LAT_ACCEL_COST_MULTIPLIER,
  TinyPhysicsModel, TinyPhysicsSimulator, get_available_controllers,
)

log = logging.getLogger("viz_realtime")

# ── Window layout ──────────────────────────────────────────────────────────
WIN_W, WIN_H = 1400, 800
TRAJ_W = 960
PANEL_W = WIN_W - TRAJ_W
BG = (18, 18, 24)
PANEL_BG = (24, 26, 34)
WHITE = (230, 230, 235)
LIGHT = (180, 180, 190)
GRAY = (100, 104, 116)
DARK_GRAY = (42, 44, 54)
GRID_COL = (30, 32, 42)
GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)
ORANGE = (230, 126, 34)
RED = (231, 76, 60)
BLUE = (52, 152, 219)
CYAN = (26, 188, 156)
TEAL_GLOW = (26, 188, 156, 40)
TARGET_COL = (90, 95, 120)
ACCENT = (155, 89, 242)  # purple accent


def lerp_color(c1, c2, t):
  t = max(0.0, min(1.0, t))
  return (int(c1[0] + (c2[0] - c1[0]) * t),
          int(c1[1] + (c2[1] - c1[1]) * t),
          int(c1[2] + (c2[2] - c1[2]) * t))


def error_color(err: float, max_err: float = 2.0):
  t = min(abs(err) / max_err, 1.0)
  if t < 0.5:
    return lerp_color(GREEN, YELLOW, t * 2)
  return lerp_color(YELLOW, RED, (t - 0.5) * 2)


def aa_thick_line(surface, color, p1, p2, width):
  """Draw an anti-aliased thick line using a polygon."""
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  length = math.hypot(dx, dy)
  if length < 0.5:
    return
  nx = -dy / length * width / 2
  ny = dx / length * width / 2
  pts = [
    (p1[0] + nx, p1[1] + ny),
    (p2[0] + nx, p2[1] + ny),
    (p2[0] - nx, p2[1] - ny),
    (p1[0] - nx, p1[1] - ny),
  ]
  int_pts = [(int(round(x)), int(round(y))) for x, y in pts]
  try:
    pygame.gfxdraw.aapolygon(surface, int_pts, color)
    pygame.gfxdraw.filled_polygon(surface, int_pts, color)
  except Exception:
    pygame.draw.line(surface, color, p1, p2, max(1, int(width)))


def draw_aa_circle(surface, color, center, radius):
  cx, cy = int(round(center[0])), int(round(center[1]))
  r = int(round(radius))
  if r < 1:
    r = 1
  try:
    pygame.gfxdraw.aacircle(surface, cx, cy, r, color)
    pygame.gfxdraw.filled_circle(surface, cx, cy, r, color)
  except Exception:
    pygame.draw.circle(surface, color, (cx, cy), r)


def draw_rounded_rect(surface, color, rect, radius=6):
  x, y, w, h = rect
  r = min(radius, w // 2, h // 2)
  # Fill center
  pygame.draw.rect(surface, color, (x + r, y, w - 2 * r, h))
  pygame.draw.rect(surface, color, (x, y + r, w, h - 2 * r))
  # Corners
  for cx, cy in [(x + r, y + r), (x + w - r - 1, y + r),
                 (x + r, y + h - r - 1), (x + w - r - 1, y + h - r - 1)]:
    draw_aa_circle(surface, color, (cx, cy), r)


class PathGenerator:
  def __init__(self, lataccels, v_egos):
    log.debug("PathGenerator: integrating %d lataccel samples into 2D path", len(lataccels))
    t0 = time.perf_counter()
    self.xs = [0.0]
    self.ys = [0.0]
    heading = 0.0
    for i in range(len(lataccels) - 1):
      v = max(v_egos[i], 1.0)
      heading += (lataccels[i] / v) * DEL_T
      self.xs.append(self.xs[-1] + v * DEL_T * math.cos(heading))
      self.ys.append(self.ys[-1] + v * DEL_T * math.sin(heading))
    self.xs = np.array(self.xs)
    self.ys = np.array(self.ys)
    elapsed = time.perf_counter() - t0
    log.debug("PathGenerator: done — %d points, x [%.1f, %.1f], y [%.1f, %.1f] (%.3fs)",
              len(self.xs), self.xs.min(), self.xs.max(), self.ys.min(), self.ys.max(), elapsed)


def fit_path_to_rect(xs, ys, rect, margin=60):
  xmin, xmax = xs.min(), xs.max()
  ymin, ymax = ys.min(), ys.max()
  dx = xmax - xmin or 1.0
  dy = ymax - ymin or 1.0
  sx = (rect.width - 2 * margin) / dx
  sy = (rect.height - 2 * margin) / dy
  scale = min(sx, sy)
  cx = rect.x + rect.width / 2
  cy = rect.y + rect.height / 2
  ox = cx - (xmin + xmax) / 2 * scale
  oy = cy - (ymin + ymax) / 2 * scale
  return scale, ox, oy


def w2s(x, y, scale, ox, oy):
  return (x * scale + ox, y * scale + oy)


class TrajectoryVisualizer:
  def __init__(self, sim: TinyPhysicsSimulator):
    self.sim = sim
    log.info("Starting simulation rollout for %s …", sim.data_path)
    t0 = time.perf_counter()
    self.cost = sim.rollout()
    elapsed = time.perf_counter() - t0
    self.n_steps = len(sim.target_lataccel_history)
    log.info("Rollout complete: %d steps in %.2fs", self.n_steps, elapsed)
    log.info("Final costs — lataccel: %.4f, jerk: %.4f, total: %.4f",
             self.cost['lataccel_cost'], self.cost['jerk_cost'], self.cost['total_cost'])

    self.targets = np.array(sim.target_lataccel_history)
    self.actuals = np.array(sim.current_lataccel_history)
    self.actions = np.array(sim.action_history)
    self.states = sim.state_history
    self.v_egos = np.array([s.v_ego for s in self.states])
    log.debug("History arrays — targets: %s, actuals: %s, actions: %s, v_egos: %s",
              self.targets.shape, self.actuals.shape, self.actions.shape, self.v_egos.shape)

    log.info("Building target path …")
    self.target_path = PathGenerator(self.targets, self.v_egos)
    log.info("Building actual path …")
    self.actual_path = PathGenerator(self.actuals, self.v_egos)

    all_x = np.concatenate([self.target_path.xs, self.actual_path.xs])
    all_y = np.concatenate([self.target_path.ys, self.actual_path.ys])
    self.all_x = all_x
    self.all_y = all_y
    log.debug("Combined viewport — x: [%.1f, %.1f], y: [%.1f, %.1f]",
              all_x.min(), all_x.max(), all_y.min(), all_y.max())

    # Pre-compute screen coords for paths (done once in run() after we know the rect)
    self.screen_target = None
    self.screen_actual = None

    self.frame = 0
    self.playing = True
    self.speed = 1.0
    self.accumulator = 0.0

  def _precompute_screen_coords(self, rect):
    """Pre-compute screen positions for all path points."""
    scale, ox, oy = fit_path_to_rect(self.all_x, self.all_y, rect)
    self.traj_scale = scale
    self.traj_ox = ox
    self.traj_oy = oy
    self.screen_target = np.column_stack([
      self.target_path.xs * scale + ox,
      self.target_path.ys * scale + oy,
    ])
    self.screen_actual = np.column_stack([
      self.actual_path.xs * scale + ox,
      self.actual_path.ys * scale + oy,
    ])
    log.debug("Pre-computed %d target + %d actual screen coords",
              len(self.screen_target), len(self.screen_actual))

  def run(self):
    log.info("Initializing pygame (window %dx%d) …", WIN_W, WIN_H)
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("comma controls challenge — trajectory viewer")
    clock = pygame.time.Clock()

    # Fonts — try nicer system fonts, fall back to monospace
    for font_name in ["Menlo", "SF Mono", "Consolas", "DejaVu Sans Mono", "monospace"]:
      test = pygame.font.match_font(font_name)
      if test:
        break
    font = pygame.font.SysFont(font_name, 14)
    font_lg = pygame.font.SysFont(font_name, 20, bold=True)
    font_xl = pygame.font.SysFont(font_name, 28, bold=True)
    font_sm = pygame.font.SysFont(font_name, 11)
    font_cost = pygame.font.SysFont(font_name, 17, bold=True)

    traj_rect = pygame.Rect(0, 0, TRAJ_W, WIN_H)
    self._precompute_screen_coords(traj_rect)

    # Glow surface for car (pre-rendered)
    glow_size = 60
    self.glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
    for r in range(glow_size, 0, -1):
      alpha = int(30 * (1 - r / glow_size) ** 2)
      pygame.gfxdraw.filled_circle(self.glow_surf, glow_size, glow_size, r,
                                   (CYAN[0], CYAN[1], CYAN[2], alpha))

    log.info("Pygame initialized — entering main loop")

    frame_count = 0
    fps_log_interval = 300

    running = True
    while running:
      dt = clock.tick(60) / 1000.0
      frame_count += 1

      for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
          log.info("Quit event received")
          running = False
        elif ev.type == pygame.KEYDOWN:
          if ev.key == pygame.K_ESCAPE:
            log.info("ESC pressed — exiting")
            running = False
          elif ev.key == pygame.K_SPACE:
            self.playing = not self.playing
            log.info("Playback %s", "resumed" if self.playing else "paused")
          elif ev.key == pygame.K_r:
            self.frame = 0
            self.accumulator = 0.0
            log.info("Playback restarted")
          elif ev.key == pygame.K_RIGHT:
            self.frame = min(self.frame + 1, self.n_steps - 1)
            log.debug("Step forward → frame %d", self.frame)
          elif ev.key == pygame.K_LEFT:
            self.frame = max(self.frame - 1, 0)
            log.debug("Step backward → frame %d", self.frame)
          elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(self.speed + 0.5, 4.0)
            log.info("Speed increased to %.1fx", self.speed)
          elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(self.speed - 0.5, 0.5)
            log.info("Speed decreased to %.1fx", self.speed)

      if self.playing and self.frame < self.n_steps - 1:
        self.accumulator += dt * self.speed * 10
        while self.accumulator >= 1.0 and self.frame < self.n_steps - 1:
          self.frame += 1
          self.accumulator -= 1.0

      # ── Draw ──
      screen.fill(BG)
      self._draw_grid(screen, traj_rect)
      self._draw_trajectory(screen, traj_rect, font_sm)

      # Panel background with subtle gradient
      panel_rect = pygame.Rect(TRAJ_W, 0, PANEL_W, WIN_H)
      pygame.draw.rect(screen, PANEL_BG, panel_rect)
      # Left border accent line
      pygame.draw.line(screen, DARK_GRAY, (TRAJ_W, 0), (TRAJ_W, WIN_H), 1)
      pygame.draw.line(screen, (50, 52, 65), (TRAJ_W + 1, 0), (TRAJ_W + 1, WIN_H), 1)

      self._draw_panel(screen, panel_rect, font, font_lg, font_xl, font_sm, font_cost)

      # HUD bar at bottom
      hud_rect = pygame.Rect(0, WIN_H - 32, TRAJ_W, 32)
      hud_surf = pygame.Surface((TRAJ_W, 32), pygame.SRCALPHA)
      hud_surf.fill((18, 18, 24, 200))
      screen.blit(hud_surf, (0, WIN_H - 32))
      keys = [("SPACE", "play/pause"), ("\u2190\u2192", "step"), ("R", "restart"),
              ("+/-", "speed"), ("ESC", "quit")]
      kx = 16
      for key, desc in keys:
        # Key badge
        kw = font_sm.size(key)[0] + 10
        draw_rounded_rect(screen, DARK_GRAY, (kx, WIN_H - 26, kw, 20), 4)
        kt = font_sm.render(key, True, WHITE)
        screen.blit(kt, (kx + 5, WIN_H - 24))
        kx += kw + 4
        dt_surf = font_sm.render(desc, True, GRAY)
        screen.blit(dt_surf, (kx, WIN_H - 24))
        kx += dt_surf.get_width() + 16

      pygame.display.flip()

      if frame_count % fps_log_interval == 0:
        fps = clock.get_fps()
        log.debug("Render frame %d — sim frame %d/%d, FPS %.1f, speed %.1fx",
                  frame_count, self.frame, self.n_steps - 1, fps, self.speed)

    log.info("Main loop ended after %d render frames — shutting down pygame", frame_count)
    pygame.quit()

  def _draw_grid(self, screen, rect):
    """Subtle dot grid on trajectory pane."""
    spacing = 50
    for x in range(rect.x + spacing, rect.x + rect.width, spacing):
      for y in range(spacing, rect.height, spacing):
        screen.set_at((x, y), GRID_COL)

  def _draw_trajectory(self, screen, rect, font):
    f = self.frame
    st = self.screen_target
    sa = self.screen_actual

    # Target path — dotted line (draw dots every 3 steps)
    for i in range(0, min(f, len(st) - 1), 3):
      x, y = int(round(st[i, 0])), int(round(st[i, 1]))
      draw_aa_circle(screen, TARGET_COL, (x, y), 2)

    # Target path — thin connecting lines for context
    if f > 1:
      end = min(f, len(st))
      pts = [(int(round(st[i, 0])), int(round(st[i, 1]))) for i in range(0, end, 2)]
      if len(pts) > 1:
        pygame.draw.lines(screen, (50, 54, 72), False, pts, 1)

    # Actual path — thick anti-aliased color-coded segments
    trail_start = max(0, f - 40)  # fade-in trail
    for i in range(0, min(f, len(sa) - 1)):
      p1 = (sa[i, 0], sa[i, 1])
      p2 = (sa[i + 1, 0], sa[i + 1, 1])
      err = self.targets[i] - self.actuals[i] if i < len(self.targets) else 0
      col = error_color(err)
      # Fade older segments slightly
      if i < trail_start:
        col = lerp_color(col, BG, 0.4)
      width = 4.0 if i >= trail_start else 2.5
      aa_thick_line(screen, col, p1, p2, width)

    # Breadcrumb dots every 20 steps on actual path
    for i in range(0, min(f, len(sa)), 20):
      x, y = int(round(sa[i, 0])), int(round(sa[i, 1]))
      draw_aa_circle(screen, (80, 85, 100), (x, y), 2)

    # Car glow + sprite
    if f < len(sa):
      cx, cy = sa[f, 0], sa[f, 1]

      # Glow
      glow_r = self.glow_surf.get_width() // 2
      screen.blit(self.glow_surf, (int(cx) - glow_r, int(cy) - glow_r),
                  special_flags=pygame.BLEND_ADD)

      # Heading
      if f > 0:
        dx = sa[f, 0] - sa[f - 1, 0]
        dy = sa[f, 1] - sa[f - 1, 1]
        angle = math.atan2(dy, dx)
      else:
        angle = 0.0

      # Car body — elongated rounded shape
      sz = 14
      cos_a, sin_a = math.cos(angle), math.sin(angle)

      # Main body (filled polygon — elongated diamond/arrow)
      pts = [
        (cx + sz * 1.8 * cos_a, cy + sz * 1.8 * sin_a),                   # nose
        (cx + sz * 0.7 * cos_a + sz * 0.9 * (-sin_a),
         cy + sz * 0.7 * sin_a + sz * 0.9 * cos_a),                       # front-left
        (cx - sz * 1.2 * cos_a + sz * 0.7 * (-sin_a),
         cy - sz * 1.2 * sin_a + sz * 0.7 * cos_a),                       # rear-left
        (cx - sz * 1.2 * cos_a, cy - sz * 1.2 * sin_a),                   # tail
        (cx - sz * 1.2 * cos_a - sz * 0.7 * (-sin_a),
         cy - sz * 1.2 * sin_a - sz * 0.7 * cos_a),                       # rear-right
        (cx + sz * 0.7 * cos_a - sz * 0.9 * (-sin_a),
         cy + sz * 0.7 * sin_a - sz * 0.9 * cos_a),                       # front-right
      ]
      int_pts = [(int(round(x)), int(round(y))) for x, y in pts]
      pygame.gfxdraw.aapolygon(screen, int_pts, CYAN)
      pygame.gfxdraw.filled_polygon(screen, int_pts, CYAN)

      # Windshield highlight
      ws = sz * 0.4
      w_pts = [
        (cx + sz * 1.0 * cos_a, cy + sz * 1.0 * sin_a),
        (cx + sz * 0.3 * cos_a + ws * (-sin_a), cy + sz * 0.3 * sin_a + ws * cos_a),
        (cx + sz * 0.3 * cos_a - ws * (-sin_a), cy + sz * 0.3 * sin_a - ws * cos_a),
      ]
      w_int = [(int(round(x)), int(round(y))) for x, y in w_pts]
      pygame.gfxdraw.filled_polygon(screen, w_int, (60, 220, 180))

    # Step label near car
    if f < len(sa):
      step_txt = font.render(f"{f}", True, GRAY)
      screen.blit(step_txt, (int(sa[f, 0]) + 20, int(sa[f, 1]) - 8))

  def _draw_panel(self, screen, rect, font, font_lg, font_xl, font_sm, font_cost):
    x0 = rect.x + 20
    rw = PANEL_W - 40  # usable width
    y = 18
    f = self.frame

    # Title
    title = font_xl.render("Dashboard", True, WHITE)
    screen.blit(title, (x0, y))
    # Segment label
    seg_name = self.sim.data_path.split("/")[-1]
    seg_lbl = font_sm.render(seg_name, True, GRAY)
    screen.blit(seg_lbl, (x0 + title.get_width() + 10, y + 10))
    y += 42

    # Progress bar
    progress = f / max(self.n_steps - 1, 1)
    step_txt = font.render(f"Step {f} / {self.n_steps - 1}", True, LIGHT)
    screen.blit(step_txt, (x0, y))
    # Playback speed badge
    spd_txt = f"{self.speed:.1f}x {'▶' if self.playing else '⏸'}"
    spd_surf = font.render(spd_txt, True, CYAN if self.playing else ORANGE)
    screen.blit(spd_surf, (x0 + rw - spd_surf.get_width(), y))
    y += 22

    # Progress bar (rounded)
    bar_h = 8
    draw_rounded_rect(screen, DARK_GRAY, (x0, y, rw, bar_h), 4)
    fill_w = max(1, int(rw * progress))
    if fill_w > 2:
      draw_rounded_rect(screen, BLUE, (x0, y, fill_w, bar_h), 4)
    # Control start marker
    cs_x = x0 + int(rw * CONTROL_START_IDX / max(self.n_steps - 1, 1))
    pygame.draw.line(screen, ORANGE, (cs_x, y - 2), (cs_x, y + bar_h + 2), 2)
    y += 22

    self._draw_separator(screen, x0, y, rw); y += 12

    # ── Velocity ──
    v = self.v_egos[min(f, len(self.v_egos) - 1)]
    v_kmh = v * 3.6
    self._draw_label_value(screen, font_sm, font_lg, "VELOCITY", f"{v:.1f} m/s", LIGHT, x0, y)
    # km/h secondary
    kmh = font_sm.render(f"({v_kmh:.0f} km/h)", True, GRAY)
    screen.blit(kmh, (x0 + rw - kmh.get_width(), y + 4))
    y += 36

    # ── Steering gauge ──
    steer = self.actions[min(f, len(self.actions) - 1)] if f < len(self.actions) else 0
    self._draw_label_value(screen, font_sm, font, "STEERING", f"{steer:.3f}", BLUE, x0, y)
    y += 22
    self._draw_center_bar(screen, x0, y, rw, 14, steer, STEER_RANGE[0], STEER_RANGE[1], BLUE)
    y += 26

    # ── Lat Accel gauge ──
    target = self.targets[min(f, len(self.targets) - 1)]
    actual = self.actuals[min(f, len(self.actuals) - 1)]
    err = target - actual
    err_col = error_color(err)
    self._draw_label_value(screen, font_sm, font, "LAT ACCEL", f"{actual:.3f}", GREEN, x0, y)
    y += 22
    self._draw_center_bar(screen, x0, y, rw, 14, actual, LATACCEL_RANGE[0], LATACCEL_RANGE[1], GREEN)
    # Target marker (diamond)
    t_frac = (target - LATACCEL_RANGE[0]) / (LATACCEL_RANGE[1] - LATACCEL_RANGE[0])
    t_x = x0 + int(rw * np.clip(t_frac, 0, 1))
    diamond = [(t_x, y - 4), (t_x + 5, y + 7), (t_x, y + 18), (t_x - 5, y + 7)]
    pygame.gfxdraw.aapolygon(screen, diamond, RED)
    pygame.gfxdraw.filled_polygon(screen, diamond, RED)
    y += 24

    # Error readout
    err_txt = font_sm.render(f"target {target:+.3f}   actual {actual:+.3f}   error {err:+.3f}", True, GRAY)
    screen.blit(err_txt, (x0, y))
    y += 20

    self._draw_separator(screen, x0, y, rw); y += 12

    # ── Mini sparkline: recent lataccel ──
    spark_label = font_sm.render("LATACCEL HISTORY", True, GRAY)
    screen.blit(spark_label, (x0, y))
    y += 16
    self._draw_sparkline(screen, x0, y, rw, 50, self.targets, self.actuals, f)
    y += 60

    self._draw_separator(screen, x0, y, rw); y += 12

    # ── Cost cards ──
    costs_label = font_sm.render("COSTS", True, GRAY)
    screen.blit(costs_label, (x0, y))
    y += 18

    # Compute running cost
    end = min(f + 1, COST_END_IDX)
    start = CONTROL_START_IDX
    if end > start:
      t_arr = self.targets[start:end]
      a_arr = self.actuals[start:end]
      lat_cost = np.mean((t_arr - a_arr) ** 2) * 100
      jerk_cost = np.mean((np.diff(a_arr) / DEL_T) ** 2) * 100 if len(a_arr) > 1 else 0.0
      total = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    else:
      lat_cost = jerk_cost = total = 0.0

    card_w = (rw - 12) // 3
    for i, (name, val, col) in enumerate([("LAT", lat_cost, GREEN), ("JERK", jerk_cost, YELLOW), ("TOTAL", total, RED)]):
      cx = x0 + i * (card_w + 6)
      draw_rounded_rect(screen, DARK_GRAY, (cx, y, card_w, 52), 6)
      # Label
      nl = font_sm.render(name, True, GRAY)
      screen.blit(nl, (cx + (card_w - nl.get_width()) // 2, y + 4))
      # Value
      vl = font_cost.render(f"{val:.2f}", True, col)
      screen.blit(vl, (cx + (card_w - vl.get_width()) // 2, y + 22))
    y += 62

    # Final costs (dimmer)
    fin_label = font_sm.render("FINAL", True, (60, 62, 74))
    screen.blit(fin_label, (x0, y))
    y += 14
    for k, v in self.cost.items():
      short = k.replace("_cost", "")
      txt = font_sm.render(f"{short}: {v:.4f}", True, (60, 62, 74))
      screen.blit(txt, (x0, y))
      y += 14

    y += 8
    self._draw_separator(screen, x0, y, rw); y += 12

    # Region badge
    region = "CONTROL" if f >= CONTROL_START_IDX else "WARMUP"
    badge_col = GREEN if f >= CONTROL_START_IDX else ORANGE
    badge_bg = (badge_col[0] // 6, badge_col[1] // 6, badge_col[2] // 6)
    bw = font.size(f"  {region}  ")[0]
    draw_rounded_rect(screen, badge_bg, (x0, y, bw, 24), 5)
    bt = font.render(region, True, badge_col)
    screen.blit(bt, (x0 + (bw - bt.get_width()) // 2, y + 3))

  def _draw_separator(self, screen, x, y, w):
    pygame.draw.line(screen, DARK_GRAY, (x, y), (x + w, y), 1)

  def _draw_label_value(self, screen, font_sm, font_val, label, value, color, x, y):
    lbl = font_sm.render(label, True, GRAY)
    screen.blit(lbl, (x, y + 2))
    val = font_val.render(value, True, color)
    screen.blit(val, (x + lbl.get_width() + 10, y))

  def _draw_center_bar(self, screen, x, y, w, h, value, vmin, vmax, color):
    draw_rounded_rect(screen, DARK_GRAY, (x, y, w, h), 4)
    frac = (value - vmin) / (vmax - vmin)
    frac = np.clip(frac, 0, 1)
    mid = w // 2
    if frac >= 0.5:
      bar_x = x + mid
      bar_w = int((frac - 0.5) * 2 * mid)
    else:
      bar_w = int((0.5 - frac) * 2 * mid)
      bar_x = x + mid - bar_w
    if bar_w > 0:
      draw_rounded_rect(screen, color, (bar_x, y + 2, bar_w, h - 4), 3)
    # Center tick
    pygame.draw.line(screen, (80, 82, 95), (x + mid, y), (x + mid, y + h), 1)

  def _draw_sparkline(self, screen, x, y, w, h, targets, actuals, frame):
    """Mini dual-line chart of target vs actual lataccel."""
    draw_rounded_rect(screen, (28, 30, 38), (x, y, w, h), 4)

    n = min(frame + 1, len(targets))
    if n < 2:
      return

    # Show last 200 steps or all if fewer
    window = min(200, n)
    start = n - window
    t_slice = targets[start:n]
    a_slice = actuals[start:n]
    all_vals = np.concatenate([t_slice, a_slice])
    vmin, vmax = all_vals.min(), all_vals.max()
    vrange = vmax - vmin or 1.0

    margin = 4
    pw = w - 2 * margin
    ph = h - 2 * margin

    def val_to_y(v):
      return y + margin + ph - int((v - vmin) / vrange * ph)

    def idx_to_x(i):
      return x + margin + int(i / max(window - 1, 1) * pw)

    # Target line (thin, gray)
    t_pts = [(idx_to_x(i), val_to_y(t_slice[i])) for i in range(window)]
    if len(t_pts) > 1:
      pygame.draw.lines(screen, TARGET_COL, False, t_pts, 1)

    # Actual line (colored)
    for i in range(window - 1):
      p1 = (idx_to_x(i), val_to_y(a_slice[i]))
      p2 = (idx_to_x(i + 1), val_to_y(a_slice[i + 1]))
      err = t_slice[i] - a_slice[i]
      col = error_color(err)
      pygame.draw.line(screen, col, p1, p2, 2)


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser(description="Pygame real-time trajectory visualizer")
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--controller", default="pid", choices=available_controllers)
  parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
  args = parser.parse_args()

  logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
  )

  log.info("=== viz_realtime starting ===")
  log.info("Args: model_path=%s, data_path=%s, controller=%s", args.model_path, args.data_path, args.controller)

  log.info("Loading ONNX model from %s …", args.model_path)
  t0 = time.perf_counter()
  model = TinyPhysicsModel(args.model_path, debug=False)
  log.info("Model loaded in %.2fs", time.perf_counter() - t0)

  log.info("Instantiating controller: %s", args.controller)
  controller = importlib.import_module(f"controllers.{args.controller}").Controller()
  log.info("Controller ready: %s", type(controller).__name__)

  log.info("Creating simulator for %s …", args.data_path)
  sim = TinyPhysicsSimulator(model, args.data_path, controller=controller, debug=False)
  log.info("Simulator created — data has %d rows", len(sim.data))

  vis = TrajectoryVisualizer(sim)
  vis.run()
  log.info("=== viz_realtime exiting ===")
