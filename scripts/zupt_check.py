# -*- coding: utf-8 -*-
"""
zupt_check.py - standalone validation for the ZUPT module (src/zupt.hpp)

Run with any Python 3 + numpy:
    python zupt_check.py

Checks:
  1. zero-velocity detector (GLRT/MV/MAG/ARE + hysteresis) classification on
     synthetic IMU data (static / moving / static segments)
  2. analytic Jacobian of the gravity-constraint residual vs finite differences
  3. Joseph-form Kalman update math (velocity ZUPT + angular-rate ZUPT) on a
     linear subset of the error state

The detector code here mirrors src/zupt.hpp 1:1; if you change one, change both.
"""
import math
import numpy as np

G = 9.81
STATE_DIM = 23
IDX_ROT, IDX_VEL, IDX_BG, IDX_BA = 3, 12, 15, 18

# --------------------------------------------------------------------------
# 1. Detector (mirror of zupt.hpp)
# --------------------------------------------------------------------------
class Params:
    def __init__(self):
        self.window_sec = 0.15
        self.detector_type = 0      # 0=GLRT 1=MV 2=MAG 3=ARE
        self.sigma_acc = 0.15
        self.sigma_gyr = 0.02
        self.gamma = 10.0
        self.max_gyro_mean = 0.06
        self.max_acc_dev = 0.5
        self.enter_cnt = 2
        self.exit_cnt = 1


class Detector:
    def __init__(self, p):
        self.p = p
        self.buf = []               # (t, acc(3,), gyr(3,))
        self.stationary = False
        self.enter_cnt = 0
        self.exit_cnt = 0
        self.stat = 0.0

    def push(self, acc, gyr, t):
        self.buf.append((t, np.asarray(acc, float), np.asarray(gyr, float)))
        while self.buf and (self.buf[-1][0] - self.buf[0][0]) > self.p.window_sec:
            self.buf.pop(0)
        self._evaluate()

    def mean_acc(self):
        return np.mean([b[1] for b in self.buf], axis=0)

    def mean_gyr(self):
        return np.mean([b[2] for b in self.buf], axis=0)

    def _evaluate(self):
        self.stat = 0.0
        if len(self.buf) < 5:
            return
        W = float(len(self.buf))
        a_bar = self.mean_acc()
        g_bar = self.mean_gyr()
        acc_dev = abs(np.linalg.norm(a_bar) - G)
        gyro_norm = np.linalg.norm(g_bar)

        if self.p.detector_type == 1:        # MV
            self.stat = sum(np.sum((b[1] - a_bar) ** 2) for b in self.buf) / W
        elif self.p.detector_type == 2:      # MAG
            self.stat = sum((np.linalg.norm(b[1]) - G) ** 2 for b in self.buf) / W
        elif self.p.detector_type == 3:      # ARE
            self.stat = sum(np.sum(b[2] ** 2) for b in self.buf) / W
        else:                                # GLRT / SHOE
            sa2 = max(1e-8, self.p.sigma_acc ** 2)
            sg2 = max(1e-8, self.p.sigma_gyr ** 2)
            d = a_bar / np.linalg.norm(a_bar) if np.linalg.norm(a_bar) > 1e-6 else np.array([0, 0, 1.0])
            T = 0.0
            for _, a, w in self.buf:
                tmp = a - G * d
                T += np.sum(w ** 2) / sg2 + np.sum(tmp ** 2) / sa2
            self.stat = T / W

        hit = self.stat <= self.p.gamma
        if self.p.max_gyro_mean > 0 and gyro_norm > self.p.max_gyro_mean:
            hit = False
        if self.p.max_acc_dev > 0 and acc_dev > self.p.max_acc_dev:
            hit = False

        if hit:
            self.enter_cnt += 1
            self.exit_cnt = 0
            if not self.stationary and self.enter_cnt >= max(1, self.p.enter_cnt):
                self.stationary = True
        else:
            self.exit_cnt += 1
            self.enter_cnt = 0
            if self.stationary and self.exit_cnt >= max(1, self.p.exit_cnt):
                self.stationary = False


def synth_imu(dt=0.005, seed=7):
    """static 3 s -> moving 3 s -> static 3 s, 200 Hz"""
    rng = np.random.default_rng(seed)
    sa, sg, bias_g = 0.15, 0.02, np.array([0.01, -0.008, 0.012])
    ts, accs, gyrs, truth = [], [], [], []
    t = 0.0
    T_move_begin, T_move_end = 3.0, 6.0
    while t < 9.0:
        moving = T_move_begin <= t < T_move_end
        if moving:
            w = 2 * math.pi * 0.5                     # 0.5 Hz turn
            gyr = np.array([0.3 * math.sin(w * t), 0.2 * math.cos(w * t), 0.6]) + bias_g + rng.normal(0, sg, 3)
            acc = np.array([2.0 * math.sin(w * t), 1.5 * math.cos(w * t), G]) + rng.normal(0, sa, 3)
        else:
            gyr = bias_g + rng.normal(0, sg, 3)
            acc = np.array([0.0, 0.0, G]) + rng.normal(0, sa, 3)
        ts.append(t); accs.append(acc); gyrs.append(gyr); truth.append(moving)
        t += dt
    return np.array(ts), accs, gyrs, np.array(truth)


def test_detector():
    print("== test 1: zero-velocity detector ==")
    ts, accs, gyrs, truth = synth_imu()   # truth[i] = True means MOVING
    p = Params()
    det = Detector(p)
    flags = []
    for t, a, w in zip(ts, accs, gyrs):
        det.push(a, w, t)
        flags.append(det.stationary)
    flags = np.array(flags)               # True = reported stationary

    static = ~truth
    moving = truth
    # exclude detector transition zones (0.5 s after each state change) from the
    # static accuracy metric - enter latency there is by design
    settle = np.ones_like(static)
    for t0 in (0.0, 3.0, 6.0):
        settle &= ~(np.abs(ts - t0) < 0.5)
    settled_static = static & settle & (ts >= 0.5)
    false_static_in_motion = int(np.sum(flags[moving]))
    false_moving_in_static = int(np.sum(~flags[settled_static]))
    static_ok = 1.0 - false_moving_in_static / max(1, int(np.sum(settled_static)))
    moving_ok = 1.0 - false_static_in_motion / max(1, int(np.sum(moving)))

    # latency after each transition
    def latency(t0, want):
        idx = np.searchsorted(ts, t0)
        seg = flags[idx:]
        hit = np.flatnonzero(seg == want)
        return ts[idx + hit[0]] - t0 if hit.size else float("nan")

    enter_delay = latency(6.0, True)     # motion stops at 6.0 s -> stationary True
    exit_delay = latency(3.0, False)     # motion starts at 3.0 s -> stationary False

    print(f"  static segment accuracy (after 0.5s warmup): {static_ok*100:.1f}%")
    print(f"  moving segment accuracy: {moving_ok*100:.1f}%")
    print(f"  false 'stationary' during motion: {false_static_in_motion}")
    print(f"  false 'moving' during static (settled): {false_moving_in_static}")
    print(f"  enter latency after stop @6.0s: {enter_delay:.3f}s")
    print(f"  exit latency after motion start @3.0s: {exit_delay:.3f}s")
    ok = (false_static_in_motion == 0 and false_moving_in_static == 0
          and enter_delay <= 0.5 and exit_delay <= 0.5)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def exp_so3(dtheta):
    th = np.linalg.norm(dtheta)
    if th < 1e-12:
        return np.eye(3)
    k = dtheta / th
    K = skew(k)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def test_jacobian():
    print("== test 2: gravity residual Jacobian vs finite differences ==")
    rng = np.random.default_rng(1)
    R = exp_so3(rng.normal(0, 0.3, 3))
    grav = np.array([0.3, -0.2, -9.78])
    ba = rng.normal(0, 0.05, 3)
    a_mean = rng.normal(0, 0.1, 3)

    def residual(dtheta):
        R_new = R @ exp_so3(dtheta)
        return a_mean - ba + R_new.T @ grav

    H_analytic = skew(R.T @ grav)          # dr/d(dtheta), see zupt.hpp
    H_num = np.zeros((3, 3))
    eps = 1e-6
    for i in range(3):
        dp = np.zeros(3); dp[i] = eps
        dm = np.zeros(3); dm[i] = -eps
        H_num[:, i] = (residual(dp) - residual(dm)) / (2 * eps)
    err = np.max(np.abs(H_analytic - H_num))
    print(f"  max |analytic - numeric| = {err:.2e}")
    ok = err < 1e-6
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_kf_update():
    print("== test 3: Joseph-form ZUPT update (velocity + gyro bias) ==")
    rng = np.random.default_rng(2)
    n = 6                       # [v(3), bg(3)] subset
    v_true = np.zeros(3)
    bg_true = np.array([0.01, -0.008, 0.012])
    v_est = np.array([0.35, -0.22, 0.11])       # wrong velocity after drift
    bg_est = np.zeros(3)                        # wrong bias
    x = np.concatenate([v_est, bg_est])
    P = np.diag([0.25, 0.25, 0.25, 0.01, 0.01, 0.01])

    sigma_vel, sigma_bg = 0.03, 0.02
    H = np.zeros((6, n))
    H[0:3, 0:3] = np.eye(3)          # velocity ZUPT
    H[3:6, 3:6] = np.eye(3)          # angular-rate ZUPT
    var = np.array([sigma_vel**2]*3 + [sigma_bg**2]*3)

    errs = []
    for k in range(30):
        # measurement: v = 0 + noise, bg = gyro_mean + noise (gyro_mean ~ bg_true)
        z_v = v_true + rng.normal(0, sigma_vel, 3)
        z_b = bg_true + rng.normal(0, sigma_bg, 3)
        z = np.concatenate([z_v, z_b])
        r = z - H @ x
        S = H @ P @ H.T + np.diag(var)
        K = P @ H.T @ np.linalg.inv(S)
        dx = K @ r
        x = x + dx
        IKH = np.eye(n) - K @ H
        P = IKH @ P @ IKH.T + K @ np.diag(var) @ K.T
        P = 0.5 * (P + P.T)
        errs.append((np.linalg.norm(x[0:3] - v_true), np.linalg.norm(x[3:6] - bg_true)))

    v_err, bg_err = errs[-1]
    eig_min = np.min(np.linalg.eigvalsh(P))
    print(f"  after 30 updates: |v_err|={v_err:.4f} m/s  |bg_err|={bg_err:.4f} rad/s  min eig(P)={eig_min:.2e}")
    print(f"  P symmetric: {np.allclose(P, P.T)}")
    ok = v_err < 0.02 and bg_err < 0.01 and eig_min > -1e-12
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = test_detector()
    r2 = test_jacobian()
    r3 = test_kf_update()
    print()
    print("ALL:", "PASS" if (r1 and r2 and r3) else "FAIL")
