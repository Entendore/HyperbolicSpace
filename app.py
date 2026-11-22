import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import math
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------- Constants ----------
N_STARS = 1500
MAX_TESS_POLYS = 400
DAMPING = 0.88
MOVE_ACCEL = 0.06
ROT_ACCEL = 0.06
MAX_CAM_R = 0.96
EPS = 1e-8


@dataclass
class Ripple:
    center: complex
    age: float = 0.0
    life: float = 2.5


@dataclass
class Circle:
    center: complex
    radius: float


class HyperbolicUniverse:
    """Interactive Poincaré disk / UHP / Klein universe explorer."""

    # ---------- Themes ----------
    THEMES = {
        'CYBERPUNK': {'bg': '#000005', 'boundary': '#00FFFF', 'user_geo': '#FFD700',
                      'tess': '#FF3366', 'tess_edge': '#FF66AA', 'star': '#FFFFFF',
                      'hud': '#00FFFF'},
        'MATRIX':    {'bg': '#000000', 'boundary': '#00FF00', 'user_geo': '#88FF88',
                      'tess': '#004400', 'tess_edge': '#00AA00', 'star': '#00FF00',
                      'hud': '#00FF00'},
        'VAPORWAVE': {'bg': '#0a001a', 'boundary': '#FF00FF', 'user_geo': '#00FFFF',
                      'tess': '#6600CC', 'tess_edge': '#AA66FF', 'star': '#FFCCFF',
                      'hud': '#FF00FF'},
    }

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.patch.set_facecolor('#000005')
        self.ax.set_facecolor('#000005')
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Minimap
        self.minimap_ax = self.fig.add_axes([0.78, 0.78, 0.2, 0.2], facecolor='#000005')
        self.minimap_ax.set_xlim(-1.1, 1.1); self.minimap_ax.set_ylim(-1.1, 1.1)
        self.minimap_ax.set_aspect('equal'); self.minimap_ax.axis('off')

        # Geometry state
        self.points: list[complex] = []
        self.circles: list[Circle] = []
        self.reflected_polys: list[list[complex]] = []
        self.ripples: list[Ripple] = []
        self.circumcircle: Optional[Circle] = None
        self.circle_radius = 1.0
        self.ideal_poly_sides = 3
        self.undo_stack: list[list[complex]] = []

        # Camera
        self.cam_a = 0.0 + 0.0j
        self.cam_theta = 0.0
        self.vel_fwd = 0.0
        self.vel_strafe = 0.0
        self.vel_rot = 0.0

        # Input
        self.keys_held: set[str] = set()
        self.mouse_pos = 0.0 + 0.0j
        self.dragging_idx = -1
        self.hover_idx = -1

        # View
        self.view_mode = 'DISK'
        self.focus_mode = False
        self.show_help = False

        # Tessellation
        self.current_p = 7
        self.current_q = 3
        self.tessellation = None
        self.tess_enabled = True
        self.time = 0.0

        # Theme
        self.theme_keys = list(self.THEMES.keys())
        self.theme_idx = 0
        self.theme = self.THEMES[self.theme_keys[0]]
        self._apply_theme()

        # Starfield (vectorized once)
        angles = np.random.uniform(0, 2 * np.pi, N_STARS)
        radii = np.sqrt(np.random.uniform(0, 0.98 ** 2, N_STARS))
        self.stars = radii * np.exp(1j * angles)
        self.star_sizes = np.clip(np.random.normal(1.5, 0.8, N_STARS), 0.5, 4.0)
        self.star_phases = np.random.uniform(0, 2 * np.pi, N_STARS)
        self._regenerate_star_colors()

        # Grid
        r_eucl_vals = [np.tanh(d / 2) for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
        self.base_grid_circles = [r * np.exp(1j * np.linspace(0, 2 * np.pi, 100)) for r in r_eucl_vals]
        self.base_grid_radials = [np.linspace(0, 0.99, 50) * np.exp(1j * a)
                                  for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]

        # Events
        for ev, fn in [('button_press_event', self.onclick),
                       ('button_release_event', self.on_release),
                       ('motion_notify_event', self.on_mouse_move),
                       ('key_press_event', self.on_key_press),
                       ('key_release_event', self.on_key_release),
                       ('scroll_event', self.on_scroll)]:
            self.fig.canvas.mpl_connect(ev, fn)

        # Animation
        self.timer = self.fig.canvas.new_timer(interval=16)
        self.timer.add_callback(self.anim_tick)
        self.timer.start()

        self.generate_tessellation()
        self.redraw()

    # ==========================================
    # THEME
    # ==========================================
    def _apply_theme(self):
        self.theme = self.THEMES[self.theme_keys[self.theme_idx]]
        self.fig.patch.set_facecolor(self.theme['bg'])
        self.ax.set_facecolor(self.theme['bg'])

    def _regenerate_star_colors(self):
        self.star_colors = np.random.choice(
            ['#FFFFFF', '#88CCFF', '#FFDDAA', '#FF9999', self.theme['star']],
            N_STARS, p=[0.6, 0.15, 0.1, 0.05, 0.1])

    # ==========================================
    # MATH CORE (vectorized where possible)
    # ==========================================
    @staticmethod
    def hyperbolic_translate(a, v):
        denom = 1 + np.conj(a) * v
        return (a + v) / denom

    @staticmethod
    def cayley_transform(z):
        return 1j * (1 - z) / (1 + z)

    @staticmethod
    def klein_transform(z):
        return 2 * z / (1 + np.abs(z) ** 2)

    @staticmethod
    def inverse_klein_transform(w):
        abs_w = np.abs(w)
        if abs_w >= 1:
            return 2.0
        return w / (1 + np.sqrt(1 - abs_w ** 2))

    def apply_view_transform(self, z):
        """Vectorized: works for scalars and ndarrays."""
        z_rot = z * np.exp(-1j * self.cam_theta)
        a = self.cam_a
        z_cam = (z_rot - a) / (1 - np.conj(a) * z_rot)
        if self.view_mode == 'UHP':
            return self.cayley_transform(z_cam)
        if self.view_mode == 'KLEIN':
            return self.klein_transform(z_cam)
        return z_cam

    def screen_to_disk(self, x, y):
        click_view = complex(x, y)
        if self.view_mode == 'UHP':
            click_disk = (1j - click_view) / (1j + click_view)
        elif self.view_mode == 'KLEIN':
            click_disk = self.inverse_klein_transform(click_view)
        else:
            click_disk = click_view
        a = self.cam_a
        z_rot = (click_disk + a) / (1 + np.conj(a) * click_disk)
        return z_rot * np.exp(1j * self.cam_theta)

    @staticmethod
    def hyperbolic_distance(z1, z2):
        if abs(abs(z1) - 1) < 1e-5 or abs(abs(z2) - 1) < 1e-5:
            return float('inf')
        num = abs(z1 - z2) ** 2
        denom = (1 - abs(z1) ** 2) * (1 - abs(z2) ** 2)
        if denom <= 0:
            return float('inf')
        return np.arccosh(1 + 2 * num / denom)

    def calc_midpoint(self, z1, z2):
        if abs(z1) < EPS and abs(z2) < EPS:
            return 0.0j
        v = -z1
        z2_trans = (z2 + v) / (1 + np.conj(v) * z2)
        d = self.hyperbolic_distance(0, z2_trans)
        m_trans = z2_trans * np.tanh(d / 4) / abs(z2_trans) if abs(z2_trans) > EPS else 0.0j
        return (m_trans - v) / (1 - np.conj(v) * m_trans)

    def calc_circumcircle(self, z1, z2, z3):
        w1, w2, w3 = self.klein_transform(z1), self.klein_transform(z2), self.klein_transform(z3)
        ax_, ay_ = w1.real, w1.imag
        bx_, by_ = w2.real, w2.imag
        cx_, cy_ = w3.real, w3.imag
        D = 2 * (ax_ * (by_ - cy_) + bx_ * (cy_ - ay_) + cx_ * (ay_ - by_))
        if abs(D) < EPS:
            return None, 0
        ux = ((ax_ ** 2 + ay_ ** 2) * (by_ - cy_) + (bx_ ** 2 + by_ ** 2) * (cy_ - ay_) +
              (cx_ ** 2 + cy_ ** 2) * (ay_ - by_)) / D
        uy = ((ax_ ** 2 + ay_ ** 2) * (cx_ - bx_) + (bx_ ** 2 + by_ ** 2) * (ax_ - cx_) +
              (cx_ ** 2 + cy_ ** 2) * (bx_ - ax_)) / D
        w_center = complex(ux, uy)
        if abs(w_center) >= 1:
            return None, 0
        z_center = self.inverse_klein_transform(w_center)
        r = self.hyperbolic_distance(z_center, z1)
        return z_center, r

    def get_geodesic_data(self, z1, z2):
        if abs(z1) < EPS and abs(z2) < EPS:
            return None, None, None
        x1, y1 = z1.real, z1.imag
        x2, y2 = z2.real, z2.imag
        A = np.array([[x1, y1], [x2, y2]])
        B = np.array([(x1 ** 2 + y1 ** 2 + 1) / 2, (x2 ** 2 + y2 ** 2 + 1) / 2])
        det = np.linalg.det(A)
        if abs(det) < EPS:
            return None, None, None
        xc, yc = np.linalg.solve(A, B)
        c = complex(xc, yc)
        v1, v2 = z1 - c, z2 - c
        cross = v1.real * v2.imag - v1.imag * v2.real
        r = abs(v1)
        return c, r, 1 if cross > 0 else -1

    def get_geodesic_points(self, z1, z2, n=50):
        c, r, direction = self.get_geodesic_data(z1, z2)
        if c is None:
            return np.linspace(z1, z2, n)
        a1, a2 = np.angle(z1 - c), np.angle(z2 - c)
        if direction == 1:
            while a2 < a1: a2 += 2 * np.pi
        else:
            while a2 > a1: a2 -= 2 * np.pi
        return c + r * np.exp(1j * np.linspace(a1, a2, n))

    def reflect_point(self, z, c, r, v1, v2):
        if c is None:
            angle = np.angle(v2 - v1)
            return np.exp(2j * angle) * np.conj(z)
        return c + r ** 2 / np.conj(z - c)

    def calculate_hyperbolic_area(self):
        n = len(self.points)
        if n < 3:
            return 0, []
        angles = []
        for i in range(n):
            z_prev = self.points[(i - 1) % n]
            z_curr = self.points[i]
            z_next = self.points[(i + 1) % n]
            if abs(abs(z_curr) - 1) < 1e-5:
                angles.append(0.0); continue
            c_in, _, dir_in = self.get_geodesic_data(z_prev, z_curr)
            c_out, _, dir_out = self.get_geodesic_data(z_curr, z_next)
            t_in = (z_curr - z_prev) if c_in is None else (
                1j * (z_curr - c_in) if dir_in == 1 else -1j * (z_curr - c_in))
            t_out = (z_next - z_curr) if c_out is None else (
                1j * (z_curr - c_out) if dir_out == 1 else -1j * (z_curr - c_out))
            ang = np.angle(t_out) - np.angle(-t_in)
            ang = ang % (2 * np.pi)
            if ang > np.pi:
                ang = 2 * np.pi - ang
            angles.append(ang)
        return (n - 2) * np.pi - sum(angles), angles

    # ==========================================
    # TESSELLATION
    # ==========================================
    def generate_tessellation(self):
        p, q = self.current_p, self.current_q
        if 1 / p + 1 / q >= 0.5:
            self.tessellation = None
            return
        cosh_R = np.cos(np.pi / p) / np.sin(np.pi / q)
        R = np.arccosh(cosh_R)
        r_eucl = np.tanh(R / 2)
        verts = [r_eucl * np.exp(1j * 2 * np.pi * k / p) for k in range(p)]

        polygons = []
        queue = [(verts, 0)]
        visited = set()
        central = sum(verts) / p
        visited.add(round(central.real, 5) + 1j * round(central.imag, 5))

        while queue and len(polygons) < MAX_TESS_POLYS:
            curr, depth = queue.pop(0)
            polygons.append((curr, depth))
            for i in range(p):
                v1, v2 = curr[i], curr[(i + 1) % p]
                c, r, _ = self.get_geodesic_data(v1, v2)
                if c is not None and abs(c) < 5:
                    refl = [self.reflect_point(v, c, r, v1, v2) for v in curr]
                else:
                    refl = [self.reflect_point(v, None, 0, v1, v2) for v in curr]
                rc = sum(refl) / p
                key = round(rc.real, 4) + 1j * round(rc.imag, 4)
                if key not in visited and abs(rc) < 0.99:
                    visited.add(key)
                    queue.append((refl, depth + 1))

        self.tessellation = {'p': p, 'q': q, 'polygons': polygons}

    def toggle_tessellation(self):
        """Proper toggle — fixes the original bug."""
        if self.tessellation is None or not self.tess_enabled:
            self.tess_enabled = True
            self.generate_tessellation()
        else:
            self.tess_enabled = False

    def generate_ideal_polygon(self):
        n = self.ideal_poly_sides
        self._push_undo()
        self.points = [np.exp(1j * 2 * np.pi * k / n) for k in range(n)]
        self.circumcircle = None
        self.reflected_polys = []
        self.ripples.append(Ripple(center=0.0j))

    # ==========================================
    # UNDO
    # ==========================================
    def _push_undo(self):
        self.undo_stack.append(list(self.points))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.points = self.undo_stack.pop()

    # ==========================================
    # ANIMATION & INPUT
    # ==========================================
    def _update_movement_from_held_keys(self):
        """Continuous control while keys are held."""
        fwd = 0.0
        strafe = 0.0
        rot = 0.0
        for k in self.keys_held:
            if k in ('up', 'w'): fwd += MOVE_ACCEL
            elif k in ('down', 's'): fwd -= MOVE_ACCEL
            elif k in ('left', 'a'): strafe += MOVE_ACCEL
            elif k in ('right', 'd'): strafe -= MOVE_ACCEL
            elif k == 'q': rot += ROT_ACCEL
            elif k == 'e': rot -= ROT_ACCEL
        # Blend: if a key is held, snap to that velocity; otherwise damp.
        if fwd != 0: self.vel_fwd = fwd
        if strafe != 0: self.vel_strafe = strafe
        if rot != 0: self.vel_rot = rot

    def anim_tick(self):
        self.time += 0.03
        self._update_movement_from_held_keys()

        if any(abs(v) > 1e-4 for v in (self.vel_fwd, self.vel_strafe, self.vel_rot)):
            self.cam_theta += self.vel_rot
            v = complex(self.vel_strafe, self.vel_fwd)
            self.cam_a = self.hyperbolic_translate(self.cam_a, v)
            # Damping only for axes with no held key
            held = self.keys_held
            if not any(k in held for k in ('up', 'w', 'down', 's')): self.vel_fwd *= DAMPING
            if not any(k in held for k in ('left', 'a', 'right', 'd')): self.vel_strafe *= DAMPING
            if not any(k in held for k in ('q', 'e')): self.vel_rot *= DAMPING
            if abs(self.cam_a) > MAX_CAM_R:
                self.cam_a *= MAX_CAM_R / abs(self.cam_a)

        # Ripples
        for r in self.ripples:
            r.age += 0.05
        self.ripples = [r for r in self.ripples if r.age < r.life]
        self.redraw()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        click_disk = self.screen_to_disk(event.xdata, event.ydata)
        if abs(abs(click_disk) - 1) < 0.02:
            click_disk = click_disk / abs(click_disk)
        if abs(click_disk) > 1:
            return
        if event.button == 1:
            if self.hover_idx != -1:
                self.dragging_idx = self.hover_idx
            else:
                self._push_undo()
                self.points.append(click_disk)
                self.ripples.append(Ripple(center=click_disk))
        elif event.button == 3:
            if self.points:
                self._push_undo()
                self.points.pop()

    def on_release(self, event):
        self.dragging_idx = -1

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        self.mouse_pos = complex(event.xdata, event.ydata)
        if self.dragging_idx != -1:
            click_disk = self.screen_to_disk(event.xdata, event.ydata)
            if abs(abs(click_disk) - 1) < 0.02:
                click_disk = click_disk / abs(click_disk)
            if abs(click_disk) < 1:
                self.points[self.dragging_idx] = click_disk
        else:
            self.hover_idx = -1
            for i, pt in enumerate(self.points):
                t_pt = self.apply_view_transform(pt)
                if abs(self.mouse_pos - t_pt) < 0.05:
                    self.hover_idx = i
                    break

    def on_scroll(self, event):
        # Zoom toward the cursor: translate *screen* cursor into disk, then
        # nudge camera toward that point.
        click_disk = self.screen_to_disk(event.xdata, event.ydata)
        if abs(click_disk) < 1e-5:
            direction = 1.0
        else:
            # Move toward the disk-space cursor, but rotated into camera frame
            # so it visually flies toward cursor.
            direction = click_disk / abs(click_disk)
        step = 0.12 * event.step
        v = direction * step
        self.cam_a = self.hyperbolic_translate(self.cam_a, v)
        if abs(self.cam_a) > MAX_CAM_R:
            self.cam_a *= MAX_CAM_R / abs(self.cam_a)

    def on_key_press(self, event):
        self.keys_held.add(event.key)
        k = event.key
        if k == ' ':
            self.vel_fwd = self.vel_strafe = self.vel_rot = 0
            self.keys_held.clear()
        elif k == 'c':
            self._push_undo()
            self.points = []; self.reflected_polys = []; self.circumcircle = None
        elif k == 'x':
            self.circles = []
        elif k == 'h':
            self.view_mode = {'DISK': 'UHP', 'UHP': 'KLEIN', 'KLEIN': 'DISK'}[self.view_mode]
        elif k == 'f':
            self.focus_mode = not self.focus_mode
        elif k == '?':
            self.show_help = not self.show_help
        elif k == 'tab':
            self.theme_idx = (self.theme_idx + 1) % len(self.theme_keys)
            self._apply_theme()
            self._regenerate_star_colors()
        elif k == 'p':
            fname = f"hyperbolic_capture_{int(time.time())}.png"
            self.fig.savefig(fname, dpi=300, facecolor=self.fig.get_facecolor())
            print(f"Saved: {fname}")
        elif k == 'm':
            if len(self.points) >= 2:
                self._push_undo()
                z1, z2 = self.points[-2], self.points[-1]
                mid = self.calc_midpoint(z1, z2)
                self.points.insert(-1, mid)
                self.ripples.append(Ripple(center=mid))
        elif k == 'i':
            self.ideal_poly_sides = self.ideal_poly_sides + 1 if self.ideal_poly_sides < 10 else 3
            self.generate_ideal_polygon()
        elif k == '1':
            self.current_p = max(3, self.current_p - 1); self.generate_tessellation()
        elif k == '2':
            self.current_p += 1; self.generate_tessellation()
        elif k == '3':
            self.current_q = max(3, self.current_q - 1); self.generate_tessellation()
        elif k == '4':
            self.current_q += 1; self.generate_tessellation()
        elif k == 't':
            self.toggle_tessellation()
        elif k == 'r':
            if len(self.points) >= 3:
                v1, v2 = self.points[-2], self.points[-1]
                c, r, _ = self.get_geodesic_data(v1, v2)
                if c is not None and abs(c) < 5:
                    refl = [self.reflect_point(v, c, r, v1, v2) for v in self.points]
                else:
                    refl = [self.reflect_point(v, None, 0, v1, v2) for v in self.points]
                self.reflected_polys.append(refl)
        elif k == 'v':
            if len(self.points) >= 3:
                z_c, r = self.calc_circumcircle(self.points[0], self.points[1], self.points[2])
                if z_c is not None:
                    self.circles.append(Circle(z_c, r))
        elif k == 'o':
            click_disk = self.screen_to_disk(self.mouse_pos.real, self.mouse_pos.imag)
            if abs(click_disk) < 1:
                self.circles.append(Circle(click_disk, self.circle_radius))
        elif k in ('+', '='):
            self.circle_radius += 0.2
        elif k == '-':
            self.circle_radius = max(0.2, self.circle_radius - 0.2)
        elif k == 'ctrl+z':
            self.undo()

    def on_key_release(self, event):
        self.keys_held.discard(event.key)
        # Properly zero velocities by axis (fixes the original bug)
        k = event.key
        if k in ('up', 'w', 'down', 's'):
            self.vel_fwd = 0
        elif k in ('left', 'a', 'right', 'd'):
            self.vel_strafe = 0
        elif k in ('q', 'e'):
            self.vel_rot = 0

    # ==========================================
    # RENDERING
    # ==========================================
    def draw_glow_line(self, xs, ys, color, base_width=2, zorder=5, alpha=1.0, linestyle='solid'):
        self.ax.plot(xs, ys, color=color, linewidth=base_width * 4, alpha=0.05 * alpha,
                     zorder=zorder - 1, solid_capstyle='round', solid_joinstyle='round', linestyle=linestyle)
        self.ax.plot(xs, ys, color=color, linewidth=base_width * 2, alpha=0.2 * alpha,
                     zorder=zorder, solid_capstyle='round', solid_joinstyle='round', linestyle=linestyle)
        self.ax.plot(xs, ys, color=color, linewidth=base_width, alpha=1.0 * alpha,
                     zorder=zorder + 1, solid_capstyle='round', solid_joinstyle='round', linestyle=linestyle)

    def draw_glow_point(self, x, y, color, size=8, is_ideal=False, is_hover=False):
        self.ax.plot(x, y, 'o', color=color, markersize=size * 4, alpha=0.1, zorder=9)
        self.ax.plot(x, y, 'o', color=color, markersize=size * 2, alpha=0.3, zorder=10)
        if is_ideal:
            self.ax.plot(x, y, 'o', color=color, markersize=size,
                         markerfacecolor='none', markeredgewidth=2, zorder=11)
        else:
            self.ax.plot(x, y, 'o', color=color, markersize=size, zorder=11)
        if is_hover:
            self.ax.plot(x, y, 'o', color=color, markersize=size * 3,
                         markerfacecolor='none', markeredgewidth=1.5, alpha=0.6, zorder=12)

    def draw_background_gradient(self):
        x = np.linspace(-1.1, 1.1, 50)
        y = np.linspace(-1.1, 1.1, 50)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)
        R_norm = np.clip(R / 1.1, 0, 1)
        img = np.zeros((50, 50, 4))
        img[..., 0] = 0.01 + (1 - R_norm) * 0.03
        img[..., 1] = 0.01 + (1 - R_norm) * 0.02
        img[..., 2] = 0.05 + (1 - R_norm) * 0.06
        img[..., 3] = 1.0
        self.ax.imshow(img, extent=[-1.1, 1.1, -1.1, 1.1], origin='lower',
                       zorder=0, aspect='equal')

    def draw_minimap(self):
        self.minimap_ax.clear()
        self.minimap_ax.set_xlim(-1.1, 1.1); self.minimap_ax.set_ylim(-1.1, 1.1)
        self.minimap_ax.set_aspect('equal'); self.minimap_ax.axis('off')
        theta = np.linspace(0, 2 * np.pi, 50)
        self.minimap_ax.plot(np.cos(theta), np.sin(theta),
                             color=self.theme['boundary'], linewidth=1.5, alpha=0.8)
        cam_x, cam_y = self.cam_a.real, self.cam_a.imag
        self.minimap_ax.plot(cam_x, cam_y, 'o',
                             color=self.theme['user_geo'], markersize=6, alpha=0.9)
        # Fixed direction sign: camera looks along +cam_theta in disk frame
        dx, dy = np.cos(self.cam_theta) * 0.2, np.sin(self.cam_theta) * 0.2
        self.minimap_ax.plot([cam_x, cam_x + dx], [cam_y, cam_y + dy],
                             color=self.theme['user_geo'], linewidth=2, alpha=0.9)

    def redraw(self):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        if not self.focus_mode:
            self.draw_background_gradient()

        # Starfield (vectorized)
        if not self.focus_mode:
            t_stars = self.apply_view_transform(self.stars)
            valid = np.abs(t_stars) < 0.99
            t_stars_v = t_stars[valid]
            sizes_v = self.star_sizes[valid]
            phases_v = self.star_phases[valid]
            colors_v = self.star_colors[valid]
            twinkle = 0.7 + 0.3 * np.sin(self.time + phases_v)
            self.ax.scatter(t_stars_v.real, t_stars_v.imag,
                            s=sizes_v * 2 * twinkle, c=colors_v,
                            alpha=0.8, marker='.', zorder=1)

        # Grid + tessellation in ONE LineCollection (BUGFIX: column_stack)
        grid_segments = []
        for circle in self.base_grid_circles:
            t_circle = self.apply_view_transform(circle)
            grid_segments.append(np.column_stack((t_circle.real, t_circle.imag)))
        for radial in self.base_grid_radials:
            t_radial = self.apply_view_transform(radial)
            grid_segments.append(np.column_stack((t_radial.real, t_radial.imag)))

        tess_colors, tess_widths = [], []
        if self.tessellation and self.tess_enabled:
            max_depth = max(d for _, d in self.tessellation['polygons']) or 1
            for poly, depth in self.tessellation['polygons']:
                t_poly = self.apply_view_transform(np.array(poly))
                t_poly = np.concatenate([t_poly, [t_poly[0]]])
                grid_segments.append(np.column_stack((t_poly.real, t_poly.imag)))
                # Fade by depth
                fade = 1.0 - depth / (max_depth + 1)
                tess_colors.append(self.theme['tess_edge'])
                tess_widths.append(0.8 + 0.6 * fade)

        # Build a color/width list matching grid_segments
        line_colors = [self.theme['tess_edge']] * (len(grid_segments) - len(tess_colors)) + tess_colors
        line_widths = [1.0] * (len(grid_segments) - len(tess_widths)) + tess_widths
        lc = mcoll.LineCollection(grid_segments, colors=line_colors,
                                  linewidths=line_widths, alpha=0.4, zorder=2)
        self.ax.add_collection(lc)

        # Reflected polygons
        for poly in self.reflected_polys:
            t_poly = self.apply_view_transform(np.array(poly))
            t_poly = np.concatenate([t_poly, [t_poly[0]]])
            self.ax.fill(t_poly.real, t_poly.imag, color=self.theme['user_geo'],
                         alpha=0.1, zorder=4)
            self.ax.plot(t_poly.real, t_poly.imag, color=self.theme['user_geo'],
                         linewidth=1.5, alpha=0.6, zorder=5, solid_joinstyle='round')

        # Ripples
        for r in self.ripples:
            if r.age <= 0:
                continue
            t_eucl = np.tanh(r.age / 2)
            angles = np.linspace(0, 2 * np.pi, 50)
            abs_zc = abs(r.center)
            if abs_zc > EPS:
                c_e = (1 - t_eucl ** 2) * r.center / (1 - abs_zc ** 2 * t_eucl ** 2)
                r_e = (1 - abs_zc ** 2) * t_eucl / (1 - abs_zc ** 2 * t_eucl ** 2)
                circle_z = c_e + r_e * np.exp(1j * angles)
            else:
                circle_z = t_eucl * np.exp(1j * angles)
            t_circle_z = self.apply_view_transform(circle_z)
            self.draw_glow_line(t_circle_z.real, t_circle_z.imag,
                                self.theme['star'], base_width=1.5,
                                zorder=5, alpha=1.0 - r.age / r.life)

        # User circles
        for circ in self.circles:
            t = np.tanh(circ.radius / 2)
            abs_zc = abs(circ.center)
            if abs_zc > EPS:
                c_e = (1 - t ** 2) * circ.center / (1 - abs_zc ** 2 * t ** 2)
                r_e = (1 - abs_zc ** 2) * t / (1 - abs_zc ** 2 * t ** 2)
            else:
                c_e = 0.0j; r_e = t
            angles = np.linspace(0, 2 * np.pi, 50)
            circle_z = c_e + r_e * np.exp(1j * angles)
            t_circle_z = self.apply_view_transform(circle_z)
            self.draw_glow_line(t_circle_z.real, t_circle_z.imag,
                                '#FF00FF', base_width=2, zorder=5)
            t_center = self.apply_view_transform(circ.center)
            self.ax.plot(t_center.real, t_center.imag, '+', color='#FF00FF',
                         markersize=10, markeredgewidth=2, zorder=6)

        # Polygon metrics
        perimeter = (sum(self.hyperbolic_distance(self.points[i], self.points[(i + 1) % len(self.points)])
                         for i in range(len(self.points)))
                     if len(self.points) >= 2 else 0)
        area, angles = self.calculate_hyperbolic_area() if len(self.points) >= 3 else (0, [])

        # User polygon fill + edges
        t_points = self.apply_view_transform(np.array(self.points)) if self.points else np.array([])
        if len(self.points) >= 3:
            all_z = []
            for i in range(len(self.points)):
                if self.view_mode == 'KLEIN':
                    arc_z = np.linspace(self.points[i], self.points[(i + 1) % len(self.points)], 2)
                else:
                    arc_z = self.get_geodesic_points(self.points[i], self.points[(i + 1) % len(self.points)], n=30)
                all_z.extend(self.apply_view_transform(arc_z))
            all_z = np.array(all_z)
            self.ax.fill(all_z.real, all_z.imag, color=self.theme['user_geo'],
                         alpha=0.15, zorder=7)

        if len(self.points) >= 2:
            n_pts = len(self.points)
            close_loop = n_pts >= 3
            edge_count = n_pts if close_loop else n_pts - 1
            for i in range(edge_count):
                z1 = self.points[i]
                z2 = self.points[(i + 1) % n_pts]
                if self.view_mode == 'KLEIN':
                    arc_z = np.linspace(z1, z2, 2)
                else:
                    arc_z = self.get_geodesic_points(z1, z2)
                t_arc_z = self.apply_view_transform(arc_z)
                self.draw_glow_line(t_arc_z.real, t_arc_z.imag,
                                    self.theme['user_geo'], base_width=2.5, zorder=8)
                dist = self.hyperbolic_distance(z1, z2)
                mid_idx = len(t_arc_z) // 2
                txt = "∞" if dist == float('inf') else f"{dist:.2f}"
                self.ax.text(t_arc_z[mid_idx].real, t_arc_z[mid_idx].imag, txt,
                             color='white', fontsize=9, ha='center', va='center', weight='bold',
                             bbox=dict(facecolor=self.theme['bg'], edgecolor=self.theme['user_geo'],
                                       alpha=0.9, pad=2, boxstyle='round'), zorder=9)

        # Hover preview geodesic
        if self.points and self.dragging_idx == -1 and self.hover_idx == -1:
            mouse_disk = self.screen_to_disk(self.mouse_pos.real, self.mouse_pos.imag)
            if abs(mouse_disk) < 0.99:
                z1 = self.points[-1]
                z2 = mouse_disk
                if self.view_mode == 'KLEIN':
                    arc_z = np.linspace(z1, z2, 2)
                else:
                    arc_z = self.get_geodesic_points(z1, z2, n=50)
                t_arc_z = self.apply_view_transform(arc_z)
                self.draw_glow_line(t_arc_z.real, t_arc_z.imag, self.theme['boundary'],
                                    base_width=1.5, zorder=7, alpha=0.7, linestyle='--')

        # Glowing points
        for i, tz in enumerate(t_points):
            is_ideal = abs(abs(self.points[i]) - 1) < 1e-5
            color = '#00FFCC' if is_ideal else self.theme['tess_edge']
            self.draw_glow_point(tz.real, tz.imag, color, size=8,
                                 is_ideal=is_ideal,
                                 is_hover=(i == self.hover_idx or i == self.dragging_idx))
            self.ax.text(tz.real, tz.imag + 0.05, f"P{i + 1}", color='white',
                         fontsize=9, weight='bold', zorder=12,
                         bbox=dict(facecolor=self.theme['bg'], edgecolor='none', alpha=0.7, pad=1))
            if i < len(angles):
                self.ax.text(tz.real, tz.imag - 0.06, f"{np.degrees(angles[i]):.0f}°",
                             color='#00FFCC', fontsize=9, ha='center', va='top', zorder=12)

        # Crosshair
        if self.hover_idx == -1:
            mx, my = self.mouse_pos.real, self.mouse_pos.imag
            self.draw_glow_point(mx, my, self.theme['star'], size=4, is_ideal=False)
            self.ax.plot([mx, mx], [my - 0.03, my + 0.03], color=self.theme['star'], alpha=0.7, zorder=15)
            self.ax.plot([mx - 0.03, mx + 0.03], [my, my], color=self.theme['star'], alpha=0.7, zorder=15)

        # Pulsing boundary
        pulse_alpha = 0.8 + 0.2 * math.sin(self.time * 2)
        if self.view_mode == 'UHP':
            self.ax.set_xlim(-3, 3); self.ax.set_ylim(0, 3)
            self.draw_glow_line(np.linspace(-3, 3, 200), np.zeros(200),
                                self.theme['boundary'], base_width=3, zorder=2, alpha=pulse_alpha)
        else:
            self.ax.set_xlim(-1.1, 1.1); self.ax.set_ylim(-1.1, 1.1)
            theta = np.linspace(0, 2 * np.pi, 50)
            self.draw_glow_line(np.cos(theta), np.sin(theta), self.theme['boundary'],
                                base_width=3, zorder=2, alpha=pulse_alpha)

        # HUD
        if not self.focus_mode:
            self.draw_minimap()
            self._draw_hud(perimeter, area)
        if self.show_help:
            self._draw_help()

        self.fig.canvas.draw()

    def _draw_hud(self, perimeter, area):
        metric_text = ("ds² = |dz|² / Im(z)²" if self.view_mode == 'UHP'
                       else ("ds² = 4|dz|² / (1-|z|²)²" if self.view_mode == 'DISK'
                             else "Klein (chordal)"))
        model_name = {"UHP": "Upper Half-Plane (ℍ)",
                      "DISK": "Poincaré Disk (𝔻)",
                      "KLEIN": "Beltrami-Klein (𝕂)"}[self.view_mode]
        tess_name = "None" if not (self.tessellation and self.tess_enabled) \
            else f"{{{self.tessellation['p']},{self.tessellation['q']}}}"

        self.ax.text(0, 1.13, f"{model_name}  |  Tiling: {tess_name}  |  Theme: {self.theme_keys[self.theme_idx]}",
                     color=self.theme['hud'], fontsize=12, ha='center', weight='bold',
                     bbox=dict(facecolor=self.theme['bg'], edgecolor=self.theme['hud'],
                               alpha=0.8, pad=5, boxstyle='round'),
                     transform=self.ax.transAxes)
        self.ax.text(0, 1.08, f"Riemannian Metric: {metric_text}",
                     color='#AAAAAA', fontsize=10, ha='center', style='italic',
                     transform=self.ax.transAxes)

        cam_r = abs(self.cam_a)
        cam_theta_deg = np.degrees(np.angle(self.cam_a))
        mouse_disk = self.screen_to_disk(self.mouse_pos.real, self.mouse_pos.imag)
        cursor_dist = self.hyperbolic_distance(0, mouse_disk) if abs(mouse_disk) < 1 else float('inf')

        info_text = (f"Cam Pos: r={cam_r:.3f}, θ={cam_theta_deg:.1f}°\n"
                     f"Cursor Dist: {cursor_dist:.3f}\n"
                     f"Perimeter: {perimeter:.3f}\n"
                     f"Area: {area:.3f}\n"
                     f"Circle r: {self.circle_radius:.2f}")
        self.ax.text(-1.08, -1.08, info_text, color=self.theme['user_geo'],
                     fontsize=10, family='monospace', verticalalignment='bottom',
                     bbox=dict(facecolor=self.theme['bg'], edgecolor=self.theme['user_geo'],
                               alpha=0.8, pad=5, boxstyle='round'))

        keys_text = ("WASD/Arrows: Fly | Space: Brake | Q/E: Rotate\n"
                     "LMB: Add/Drag | RMB: Undo | 'O': Circle | +/-: Size\n"
                     "'M': Mid | 'V': Circumcircle | 'R': Reflect | 'I': Ideal Poly\n"
                     "1-4: Tiling | 'H': Model | Tab: Theme | F: Focus | ?: Help | P: Pic")
        self.ax.text(1.08, -1.08, keys_text, color='#888888', fontsize=9,
                     ha='right', va='bottom', family='monospace',
                     bbox=dict(facecolor=self.theme['bg'], edgecolor='#444444',
                               alpha=0.8, pad=5, boxstyle='round'))

    def _draw_help(self):
        help_text = (
            "HYPERBOLIC UNIVERSE — HELP\n"
            "─────────────────────────\n"
            "Navigation\n"
            "  WASD / Arrows : Fly through ℍ²\n"
            "  Q / E         : Rotate camera\n"
            "  Space         : Brake to stop\n"
            "  Scroll        : Zoom toward cursor\n\n"
            "Geometry\n"
            "  LMB           : Add point (or drag existing)\n"
            "  RMB           : Remove last point\n"
            "  M             : Insert hyperbolic midpoint\n"
            "  V             : Circumcircle of first 3 pts\n"
            "  O             : Drop circle at cursor\n"
            "  R             : Reflect polygon across last edge\n"
            "  I             : Cycle ideal {3,4,…,10}-gon\n"
            "  C / X         : Clear points / circles\n"
            "  Ctrl+Z        : Undo\n\n"
            "Visualization\n"
            "  H             : Cycle Disk ↔ UHP ↔ Klein\n"
            "  T             : Toggle tessellation\n"
            "  1 / 2         : p − / p +\n"
            "  3 / 4         : q − / q +\n"
            "  Tab           : Cycle theme\n"
            "  F             : Focus (hide HUD)\n"
            "  ?             : Toggle this help\n"
            "  P             : Save PNG\n"
        )
        self.ax.text(0, 0, help_text, transform=self.ax.transAxes,
                     color=self.theme['hud'], fontsize=10, family='monospace',
                     ha='center', va='center',
                     bbox=dict(facecolor=self.theme['bg'], edgecolor=self.theme['hud'],
                               alpha=0.95, pad=10, boxstyle='round'))


if __name__ == "__main__":
    app = HyperbolicUniverse()
    plt.show()