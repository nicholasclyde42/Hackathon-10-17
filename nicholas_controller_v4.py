"""EcoCAR V4: baseline pure pursuit with cautious, perception-only obstacle avoidance.

Experimental: benchmark across all public courses before use. No track/map access.
"""
from __future__ import annotations

import math
import numpy as np
from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams


def _ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class NicholasControllerV4(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c = self.config
        self.k_ld = float(c.get("lookahead_gain", 0.5))
        self.ld_min = float(c.get("lookahead_min", 4.0))
        self.a_lat = float(c.get("lat_accel_frac", 0.65)) * vehicle.mu * 9.81
        self.a_dec = float(c.get("decel_frac", 0.45)) * vehicle.max_decel
        self.a_acc = float(c.get("accel_frac", 0.5)) * vehicle.max_accel
        self.v_max = float(c.get("v_max", 24.0))
        self.kp = float(c.get("kp", 0.5))
        self.ki = float(c.get("ki", 0.1))
        self.i_err = 0.0
        self.v_ref = None
        self.maneuver = None  # (station, chosen lateral offset); station from live progress

    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid:
            return Command(brake=1.0)
        s = np.asarray(road.preview_s, dtype=float)
        cl = np.asarray(road.preview_centerline, dtype=float)
        curvature = np.asarray(road.preview_curvature, dtype=float)
        if (len(s) < 3 or cl.shape != (len(s), 2) or
                len(curvature) != len(s) or not np.all(np.isfinite(cl)) or
                not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0)):
            return Command(brake=1.0)
        v = max(float(obs.speed), 0.0)
        half_w = 0.5 * vp.width
        margin = 0.45
        wl = np.asarray(road.preview_width_left, dtype=float)
        wr = np.asarray(road.preview_width_right, dtype=float)
        have_widths = wl.shape == s.shape and wr.shape == s.shape and np.all(np.isfinite(wl)) and np.all(np.isfinite(wr))

        # Keep side selection stable from step to step; expire after passing the obstacle.
        if self.maneuver is not None and obs.progress > self.maneuver[0] + 13.0:
            self.maneuver = None
        if self.maneuver is None and have_widths:
            for ob in sorted(road.obstacles, key=lambda o: o.distance):
                if not (np.isfinite(ob.distance) and 3.0 < ob.distance < s[-1] - 1.0):
                    continue
                if not (np.isfinite(ob.lateral_min) and np.isfinite(ob.lateral_max)):
                    continue
                j = int(np.clip(np.searchsorted(s, ob.distance), 0, len(s) - 1))
                # Robot centre must lie outside obstacle lateral extent by half vehicle width + safety buffer.
                left = float(ob.lateral_max + half_w + margin)
                right = float(ob.lateral_min - half_w - margin)
                left_limit = float(wl[j] - half_w - margin)
                right_limit = float(-wr[j] + half_w + margin)
                options = []
                if left <= left_limit:
                    options.append(left)
                if right >= right_limit:
                    options.append(right)
                # If both sides unsafe, leave path unchanged and slow down instead.
                shift = min(options, key=abs) if options else 0.0
                self.maneuver = (float(obs.progress + ob.distance), float(shift))
                break

        path = cl.copy()
        remaining = float('inf')
        avoiding = 0.0
        blocked = False
        if self.maneuver is not None and have_widths:
            station, shift = self.maneuver
            remaining = float(station - obs.progress)
            blocked = shift == 0.0
            if not blocked:
                # Start 27 m before; fully displaced 5 m before; hold through obstacle;
                # return to centerline smoothly by 20 m after.
                entry = _ease((s - (remaining - 27.0)) / 22.0)
                exit_ = 1.0 - _ease((s - (remaining + 6.0)) / 14.0)
                offset = shift * entry * exit_
                lower = -wr + half_w + margin
                upper = wl - half_w - margin
                offset = np.minimum(np.maximum(offset, lower), upper)
                # Normal to previewed centerline in vehicle coordinates.
                tangent = np.gradient(cl, axis=0)
                norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-6)
                normals = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
                path += offset[:, None] * normals
                avoiding = 1.0

        # Original Pure Pursuit steering, applied to our published path.
        ld = max(self.ld_min, self.k_ld * v + 2.0)
        tgt = np.array([np.interp(ld, s, path[:, 0]), np.interp(ld, s, path[:, 1])])
        alpha = math.atan2(tgt[1], max(tgt[0], 0.1))
        delta = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(math.hypot(*tgt), 1.0))
        steer = float(np.clip(delta / vp.max_steer, -1.0, 1.0))

        # Original predictive curvature speed limit; V2's tested cornering/deceleration fractions.
        v_curve = np.minimum(np.sqrt(self.a_lat / np.maximum(np.abs(curvature), 1e-4)), self.v_max)
        v_curve[-1] = min(v_curve[-1], float(self.config.get('v_horizon', 6.0)))
        v_tgt = float(np.min(np.sqrt(v_curve ** 2 + 2.0 * self.a_dec * s)))
        if np.isfinite(remaining) and remaining < 35.0:
            # Approach obstacles using a braking-distance speed envelope.
            # Aim for at most 10 m/s by 12 m before the obstacle, and hold
            # that limit until 7 m past it. Maintain lower cornering limits.
            if blocked:
                v_tgt = min(v_tgt, 3.0)
            elif remaining > -7.0:
                distance_to_slow = max(remaining - 12.0, 0.0)
                approach_limit = math.sqrt(10.0 ** 2 + 2.0 * self.a_dec * distance_to_slow)
                v_tgt = min(v_tgt, approach_limit)
        if self.v_ref is None:
            self.v_ref = v
        self.v_ref = min(v_tgt, self.v_ref + self.a_acc * vp.control_dt)
        v_tgt = self.v_ref
        err = v_tgt - v
        self.i_err = float(np.clip(self.i_err + err * vp.control_dt, -5.0, 5.0))
        u = self.kp * err + self.ki * self.i_err
        lat_use = min(abs(obs.ay) / (vp.mu * 9.81), 1.0)
        throttle_cap = max(1.0 - lat_use ** 2, 0.15)
        throttle, brake = (min(u, throttle_cap), 0.0) if u >= 0 else (0.0, min(-u * 0.6, 1.0))
        return Command(steer=steer, throttle=float(throttle), brake=float(brake),
                       path=path, path_frame="vehicle", target_speed=float(v_tgt),
                       debug={"obstacle_distance": remaining if np.isfinite(remaining) else -1.0,
                              "avoiding": avoiding, "lookahead": ld})
