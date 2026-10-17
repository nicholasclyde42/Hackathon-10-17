"""EcoCAR V4: baseline pure pursuit with cautious, perception-only obstacle avoidance.

Experimental: benchmark across all public courses before use. No track/map access.
"""
from __future__ import annotations

import math
import numpy as np
from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams


def _ease(x):
    """Quintic C2 minimum-jerk S-curve (zero slope and zero curvature at 0 and 1)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


class NicholasControllerV4(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c = self.config
        self.k_ld = float(c.get("lookahead_gain", 0.46))
        self.ld_min = float(c.get("lookahead_min", 3.8))
        self.mu_g = vehicle.mu * 9.81
        self.a_lat_fast = float(c.get("lat_accel_frac_fast", 0.71)) * self.mu_g
        self.a_lat_tight = float(c.get("lat_accel_frac_tight", 0.61)) * self.mu_g
        self.a_dec = float(c.get("decel_frac", 0.55)) * vehicle.max_decel
        self.a_acc = float(c.get("accel_frac", 0.78)) * vehicle.max_accel
        self.v_max = float(c.get("v_max", 25.0))
        self.kp = float(c.get("kp", 0.72))
        self.ki = float(c.get("ki", 0.12))
        self.i_err = 0.0
        self.v_ref = None
        self.maneuvers: list[tuple[float, float, bool]] = []  # [(station, shift, blocked), ...]
        # Warm up NumPy ufuncs during untimed on_start to avoid cold-start page faults on step 0
        _d = np.linspace(0.0, 40.0, 41)
        _ = np.interp(5.0, _d, _d) + float(np.min(np.sqrt(_d)))

    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid:
            return Command(brake=1.0)
        s = road.preview_s
        cl = road.preview_centerline
        curvature = road.preview_curvature
        n_pts = len(s)
        if (n_pts < 3 or cl.shape != (n_pts, 2) or len(curvature) != n_pts
                or not (np.isfinite(s).all() and np.isfinite(cl).all())
                or (np.diff(s) <= 0.0).any()):
            return Command(brake=1.0)
        v = max(float(obs.speed), 0.0)
        half_w = 0.5 * vp.width
        margin = 0.45
        wl = road.preview_width_left
        wr = road.preview_width_right
        have_widths = (wl.shape == s.shape and wr.shape == s.shape
                       and np.isfinite(wl).all() and np.isfinite(wr).all())

        # Expire maneuvers only after their exit ramp (station + 20 m) is behind the car.
        if self.maneuvers:
            self.maneuvers = [m for m in self.maneuvers if obs.progress <= m[0] + 20.0]

        if have_widths and road.obstacles:
            for ob in sorted(road.obstacles, key=lambda o: o.distance):
                if not (math.isfinite(ob.distance) and 2.0 < ob.distance < s[-1] - 1.0):
                    continue
                if not (math.isfinite(ob.lateral_min) and math.isfinite(ob.lateral_max)):
                    continue
                st_ob = float(obs.progress + ob.distance)
                if any(abs(st_ob - m[0]) < 6.0 for m in self.maneuvers):
                    continue
                j = int(np.clip(np.searchsorted(s, ob.distance), 0, n_pts - 1))
                # Safe corridor bounds keeping >= margin from both the obstacle and outer boundary cones.
                left_min = float(ob.lateral_max + half_w + margin)
                left_max = float(wl[j] - half_w - margin)
                right_max = float(ob.lateral_min - half_w - margin)
                right_min = float(-wr[j] + half_w + margin)
                # Pure Pursuit cuts inside corners; request extra clearance on the inside of a turn,
                # clamped so we never push closer than `margin` to the outer boundary cone row.
                k_local = float(curvature[j])
                extra_obs = 0.14
                left_des = left_min + extra_obs + max(-k_local, 0.0) * 5.5
                right_des = right_max - extra_obs - max(k_local, 0.0) * 5.5
                options = []
                if left_min <= left_max:
                    options.append(min(left_des, left_max))
                if right_min <= right_max:
                    options.append(max(right_des, right_min))
                if options:
                    shift = min(options, key=abs)
                    self.maneuvers.append((st_ob, float(shift), False))
                else:
                    self.maneuvers.append((st_ob, 0.0, True))
            self.maneuvers.sort(key=lambda m: m[0])

        ref_path = cl
        remaining = float('inf')
        avoiding = 0.0
        blocked = False
        active_shift = 0.0
        if self.maneuvers and have_widths:
            offset = np.zeros_like(s)
            for station, shift, is_blocked in self.maneuvers:
                rem = float(station - obs.progress)
                if rem > -7.0 and rem < remaining:
                    remaining = rem
                    blocked = is_blocked
                    active_shift = abs(shift)
                if not is_blocked:
                    # Start 30 m before; reach 100% lane shift 8 m before the cone so the car is
                    # already settled straight in the open lane; smoothly merge back 5..20 m after.
                    entry = _ease((s - (rem - 30.0)) / 22.0)
                    exit_ = 1.0 - _ease((s - (rem + 5.0)) / 15.0)
                    offset += shift * entry * exit_
            if np.any(np.abs(offset) > 1e-4):
                offset = np.clip(offset, -wr + half_w + margin, wl - half_w - margin)
                dx = np.empty_like(s)
                dy = np.empty_like(s)
                dx[0] = cl[1, 0] - cl[0, 0]
                dx[1:-1] = 0.5 * (cl[2:, 0] - cl[:-2, 0])
                dx[-1] = cl[-1, 0] - cl[-2, 0]
                dy[0] = cl[1, 1] - cl[0, 1]
                dy[1:-1] = 0.5 * (cl[2:, 1] - cl[:-2, 1])
                dy[-1] = cl[-1, 1] - cl[-2, 1]
                scale = offset / np.maximum(np.hypot(dx, dy), 1e-6)
                ref_path = np.column_stack((cl[:, 0] - dy * scale, cl[:, 1] + dx * scale))
                avoiding = 1.0

        # Pure Pursuit steering on ref_path.
        ld = max(self.ld_min, self.k_ld * v + 1.8)
        rx, ry = ref_path[:, 0], ref_path[:, 1]
        tx = float(np.interp(ld, s, rx))
        ty = float(np.interp(ld, s, ry))
        alpha = math.atan2(ty, max(tx, 0.1))
        delta = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(math.hypot(tx, ty), 1.0))
        steer = float(np.clip(delta / vp.max_steer, -1.0, 1.0))

        # Roll out the actual trajectory Pure Pursuit will follow from (0, 0) over the preview
        # horizon so CommittedPath (evaluated 0.5 s later) matches our true corner/slalom line.
        path = ref_path.copy()
        px, py = 0.0, 0.0
        pth = math.atan2(float(obs.vy), max(float(obs.vx), 1.0)) if v > 1.5 else 0.0
        L = vp.wheelbase
        max_st = vp.max_steer
        cur_st = float(obs.steer) * max_st
        slew_ds = vp.max_steer_rate * max_st / max(v, 3.0)
        n_roll = min(n_pts, 20)
        path[0, 0], path[0, 1] = 0.0, 0.0
        for i in range(1, n_roll):
            ds_i = float(s[i] - s[i - 1])
            s_la = float(s[i - 1]) + ld
            j = min(int(s_la), n_pts - 1)
            frac = min(max(s_la - j, 0.0), 1.0) if j + 1 < n_pts else 0.0
            j2 = min(j + 1, n_pts - 1)
            lx = float(rx[j] + frac * (rx[j2] - rx[j])) - px
            ly = float(ry[j] + frac * (ry[j2] - ry[j])) - py
            c_th, s_th = math.cos(pth), math.sin(pth)
            bx = c_th * lx + s_th * ly
            by = -s_th * lx + c_th * ly
            ang = math.atan2(by, max(bx, 0.1))
            tgt_st = min(max(math.atan2(2.0 * L * math.sin(ang), max(math.hypot(bx, by), 1.0)), -max_st), max_st)
            cur_st = min(max(tgt_st, cur_st - slew_ds * ds_i), cur_st + slew_ds * ds_i)
            kap = math.tan(cur_st) / L
            pth_mid = pth + 0.5 * kap * ds_i
            px += ds_i * math.cos(pth_mid)
            py += ds_i * math.sin(pth_mid)
            pth += kap * ds_i
            # Blend smoothly into ref_path near the end of the rollout window
            w_b = 1.0 if i <= 12 else (n_roll - i) / (n_roll - 12.0)
            path[i, 0] = w_b * px + (1.0 - w_b) * rx[i]
            path[i, 1] = w_b * py + (1.0 - w_b) * ry[i]

        # Predictive curvature speed limit: look from the front axle (2 m ahead) to 4 m ahead
        # so braking finishes before turn-in while corner exit accelerates as soon as the nose clears the apex.
        k_abs = np.abs(curvature)
        k_lead = np.maximum(np.pad(k_abs[2:], (0, 2), mode="edge"),
                            np.pad(k_abs[4:], (0, 4), mode="edge"))
        tight = np.clip((k_lead - 0.050) / 0.022, 0.0, 1.0)
        a_lat_prof = self.a_lat_fast + tight * (self.a_lat_tight - self.a_lat_fast)
        v_curve = np.minimum(np.sqrt(a_lat_prof / np.maximum(k_lead, 1e-4)), self.v_max)
        # Only enforce end-of-horizon caution if the finish line is not yet in sight.
        if not (math.isfinite(road.finish_distance) and 0.0 <= road.finish_distance <= s[-1]):
            v_curve[-1] = min(v_curve[-1], float(self.config.get("v_horizon", 6.7)))
        v_tgt = float(np.min(np.sqrt(v_curve ** 2 + 2.0 * self.a_dec * s)))
        if math.isfinite(remaining) and remaining < 35.0:
            if blocked:
                v_tgt = min(v_tgt, 3.0)
            elif remaining > -5.0:
                pass_speed = max(11.5, 14.5 - 1.5 * active_shift)
                distance_to_slow = max(remaining - 10.0, 0.0)
                approach_limit = math.sqrt(pass_speed ** 2 + 2.0 * self.a_dec * distance_to_slow)
                v_tgt = min(v_tgt, approach_limit)
        if self.v_ref is None:
            self.v_ref = v
        self.v_ref = min(v_tgt, self.v_ref + self.a_acc * vp.control_dt)
        v_tgt = self.v_ref
        err = v_tgt - v
        # Anti-windup: clear positive integral windup immediately when braking is required.
        if err < -0.1 and self.i_err > 0.0:
            self.i_err = 0.0
        self.i_err = float(np.clip(self.i_err + err * vp.control_dt, -5.0, 5.0))
        u = self.kp * err + self.ki * self.i_err
        lat_use = min(abs(obs.ay) / self.mu_g, 1.0)
        throttle_cap = max(1.0 - lat_use ** 2, 0.15)
        throttle, brake = (min(u, throttle_cap), 0.0) if u >= 0 else (0.0, min(-u * 0.85, 1.0))
        return Command(steer=steer, throttle=float(throttle), brake=float(brake),
                       path=path, path_frame="vehicle", target_speed=float(v_tgt),
                       debug={"obstacle_distance": remaining if math.isfinite(remaining) else -1.0,
                              "avoiding": avoiding, "lookahead": ld})
