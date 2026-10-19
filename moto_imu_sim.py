#!/usr/bin/env python3
"""
moto_imu_sim.py - Simulated motorcycle IMU dataset for a riding-style classifier.

Purpose
    Generate physically plausible, noisy, class-overlapping IMU data (MPU6050-class
    sensor on a ~200 cc single-cylinder motorcycle). Use it to validate an ML
    pipeline (windowing, feature engineering, GroupKFold by ride) before real
    ESP32 data exists. Dependencies: numpy, scipy, pandas, matplotlib.

Conventions
    Body frame: x forward, y left, z up (right-handed).
    Roll phi > 0 is a right-hand rotation about +x, so the top of the bike tilts
    RIGHT. A bike leaning right turns right, so the yaw rate about +z is negative:
        psi_dot = -g * tan(phi) / v        (|psi_dot| = g * tan|phi| / v)
    Pitch theta > 0 is a right-hand rotation about +y (left), i.e. nose DOWN.
    Accelerometer: specific force in g (reads +1 g on z when level and still).
    Gyroscope: deg/s.

Pipeline per ride
    1. Rider, vehicle, mount and sensor parameters drawn per ride.
    2. Manoeuvre sequence -> longitudinal accel a(t), speed v(t), lean phi(t),
       surface per segment. Physics runs at CONFIG['fs_sim'] (1 kHz).
    3. Drivetrain: 5-speed gearbox state machine -> gear(t), rpm(t), shift transients.
    4. Rigid-body specific force and angular rate in the body frame.
    5. Additive dynamics: fork dive/squat, shift jolts, rider micro-motion,
       road/suspension noise, potholes and bumps, engine vibration with frame
       resonances and mount transfer.
    6. Sensor model: mount tilt, scale, cross-axis, bias, thermal drift, gyro
       random walk, ADC saturation, DLPF, sampling with timing jitter (no
       anti-alias filter beyond the DLPF, so vibration aliases), white noise,
       16-bit quantisation, dropped and duplicated samples.
    7. Outputs and validation.

Outputs (in CONFIG['out_dir'])
    imu_data.csv          ride_id, t, ax, ay, az, gx, gy, gz, label
    ride_metadata.csv     per-ride parameters and realised statistics
    ground_truth.csv      speed [m/s], rpm, gear, lean_deg, surface for ONE ride per
                          class. Debug only - never use as features.
    validation_spectrogram.png, validation_histograms.png, validation_steady_lean.png
"""

import math
import os
import time

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.80665
SURFACES = ("smooth", "rough", "cobble_gravel")
AXES = ("ax", "ay", "az", "gx", "gy", "gz")

# =============================================================================
# CONFIG - every tunable value lives here
# =============================================================================
CONFIG = {
    # ---- general ------------------------------------------------------------
    "seed": 20261011,
    "out_dir": "moto_imu_sim",
    "rides_per_class": {"calm": 12, "normal": 16, "aggressive": 8},
    "ride_duration_s": 300.0,
    "fs_sim": 1000.0,              # physics rate [Hz]
    "fs_out": 200.0,               # output (logger) rate [Hz]
    "csv_float_format": "%.6f",

    # ---- rider classes (ranges overlap on purpose) --------------------------
    # A ride draws one rider_intensity r from its class range. Its position inside
    # that range (plus jitter) sets where the ride's centre values sit inside each
    # class range below. Individual events are lognormal around those centres.
    "classes": {
        "calm": {
            "intensity": (0.05, 0.45),
            "brake_g": (0.10, 0.30),           # peak braking centre
            "lean_deg": (10.0, 25.0),          # peak lean centre
            "accel_g": (0.08, 0.20),           # peak throttle accel centre
            "lean_omega": (1.6, 2.6),          # rad/s, critically damped lean response
            "jerk_rise_s": (0.8, 1.3),         # accel/brake ramp time (style-dependent jerk)
            "upshift_rpm": (4200, 5600),
            "downshift_rpm": (2000, 2600),
            "brake_downshift_rpm": (2600, 3400),
            "cruise_kmh": (30, 55),
            "cruise_s": (8, 25),
            "stop_s": (4, 20),
            "alt_mood_prob": 0.08,             # share of manoeuvres in the "other" mood
            "alt_mood_factor": 1.30,           # calm rider occasionally in a hurry
        },
        "normal": {
            "intensity": (0.30, 0.70),
            "brake_g": (0.20, 0.50),
            "lean_deg": (20.0, 32.0),
            "accel_g": (0.15, 0.30),
            "lean_omega": (2.4, 3.6),
            "jerk_rise_s": (0.5, 0.9),
            "upshift_rpm": (5400, 7000),
            "downshift_rpm": (2400, 3200),
            "brake_downshift_rpm": (3200, 4300),
            "cruise_kmh": (40, 70),
            "cruise_s": (5, 18),
            "stop_s": (3, 12),
            "alt_mood_prob": 0.15,
            "alt_mood_factor": 0.70,
        },
        "aggressive": {
            "intensity": (0.55, 0.95),
            "brake_g": (0.40, 0.90),
            "lean_deg": (30.0, 45.0),
            "accel_g": (0.25, 0.48),
            "lean_omega": (3.4, 5.5),
            "jerk_rise_s": (0.25, 0.5),
            "upshift_rpm": (7300, 9200),
            "downshift_rpm": (3200, 4200),
            "brake_downshift_rpm": (4500, 6000),
            "cruise_kmh": (50, 90),
            "cruise_s": (3, 12),
            "stop_s": (2, 8),
            "alt_mood_prob": 0.25,             # real aggressive riders have calm stretches
            "alt_mood_factor": 0.55,
        },
    },
    "within_class_jitter": 0.15,       # jitter of a ride's position inside class ranges
    "event_lognormal_sigma": 0.18,     # spread of per-event brake/lean/throttle intensity
    "event_timing_sigma": 0.15,        # spread of per-event lean omega / jerk rise time
    "mood_stay_prob": 0.70,            # mood persistence between manoeuvres
    "brake_g_limits": (0.05, 0.95),
    "lean_deg_limits": (3.0, 48.0),
    "accel_g_limits": (0.04, 0.55),

    # ---- manoeuvre generator -----------------------------------------------
    "maneuver_probs": {"cruise": 0.28, "corner": 0.34, "brake": 0.18, "accel": 0.20},
    "first_stop_s": (2.0, 5.0),
    "brake_to_stop_prob": 0.35,
    "brake_partial_frac": (0.30, 0.75),
    "corner_entry_speed_frac": (0.55, 0.92),
    "corner_hold_s": (1.5, 6.0),
    "corner_exit_accel_prob": 0.75,
    "corner_min_speed_mps": 3.0,
    "corner_min_radius_m": 6.0,
    "lean_ramp_omegas": 1.5,           # lean target ramp duration = this / omega
    "lean_settle_omegas": 10.0,        # superposition window after lean release
    "cruise_wobble_mps": (0.3, 1.0),    # peak speed swing while cruising
    "cruise_speed_sigma": 0.20,
    "min_cruise_kmh": 15.0,
    "max_speed_kmh": 115.0,
    "min_ramp_s": 0.15,

    # ---- vehicle ------------------------------------------------------------
    "vehicle": {
        "mass_kg": 185.0,              # bike + rider
        "power_w": 17500.0,
        "wheel_radius_m": 0.30,
        "wheelbase_m": 1.35,
        "primary_ratio": 3.00,
        "final_ratio": 3.07,
        "gear_ratios": (2.83, 1.88, 1.40, 1.13, 0.96),
        "idle_rpm": 1300.0,
        "redline_rpm": 9500.0,
        "power_cap_frac": 0.90,
        "sensor_height_m": {"frame": (0.60, 0.80), "handlebar": (0.95, 1.10)},
    },

    # ---- chassis pitch (fork dive / squat) ----------------------------------
    "pitch": {
        "dive_deg_per_g": (2.5, 3.5),
        "squat_deg_per_g": (0.6, 1.0),
        "freq_hz": (1.5, 2.2),
        "zeta": (0.30, 0.45),
    },

    # ---- drivetrain ---------------------------------------------------------
    "drivetrain": {
        "logic_rate_hz": 100.0,
        "launch_rpm": (2300.0, 5200.0),          # interpolated by rider intensity
        "min_shift_gap_s": (0.9, 0.4),
        "cruise_upshift_rpm": (3800.0, 6200.0),
        "upshift_jitter_sigma": 0.05,
        "shift_dur_s": 0.30,
        "shift_blend_tau_s": 0.07,
        "shift_dip_rpm": (300.0, 750.0),
        "downshift_dip_scale": 0.5,
        "shift_jolt_g": (0.04, 0.12),
        "shift_rebound_hz": 6.0,
        "shift_rebound_tau_s": 0.08,
        "shift_jolt_pitch_dps_per_g": 6.0,
        "shift_load": 0.15,
        "load_smooth_s": 0.10,
    },

    # ---- engine vibration ---------------------------------------------------
    "engine": {
        "rpm_ref": 6000.0,
        "vib_amp_g": (0.10, 0.20),               # frame-mount az amplitude at rpm_ref, full load
        "coast_frac": 0.40,                      # amplitude fraction at zero load
        "orders": (0.5, 1.0, 2.0, 3.0, 4.0),     # 0.5 = combustion (4-stroke single)
        "order_amp": (0.55, 1.00, 0.45, 0.20, 0.10),
        "axis_gain": (0.70, 0.35, 1.00, 3.0, 5.0, 2.0),  # ax, ay, az [g/g]; gx, gy, gz [deg/s per g]
        "amp_jitter": (0.05, 0.10),              # per combustion cycle, relative
        "phase_jitter_rad": (0.05, 0.10),        # per combustion cycle, crank angle
        "n_resonances": (1, 2),
        "res_freq_hz": (50.0, 200.0),
        "res_q": (5.0, 20.0),
        "res_gain": (1.0, 3.0),
        "handlebar_prob": 1.0,
        "handlebar_gain": (1.6, 2.4),
        "rubber_prob": 0.4,
        "rubber_cutoff_hz": (35.0, 80.0),
        "sim_taper": (0.40, 0.48),               # fade orders above these fractions of fs_sim
    },

    # ---- road and suspension -----------------------------------------------
    "road": {
        "roughness": (0.6, 1.6),                 # per-ride road roughness multiplier
        "surface_base_probs": (0.62, 0.28, 0.10),
        "surface_stay_prob": 0.75,
        "surface_level": (1.0, 2.2, 4.0),
        "v_ref_mps": 12.0,
        "speed_exp": 0.9,
        "speed_factor_max": 3.0,
        "brown_pole": 0.995,
        "white_mix": 0.4,
        "body_freq_hz": (1.2, 1.8),
        "body_q": 1.7,
        "hop_freq_hz": (10.0, 14.0),
        "hop_q": 3.0,
        "body_rms_g": 0.025,
        "hop_rms_g": 0.035,
        "cobble_band_hz": (15.0, 45.0),
        "cobble_rms_g": 0.05,
        "ax_frac": 0.30, "ay_frac": 0.15,
        "gx_dps": 1.0, "gy_dps": 2.0, "gz_dps": 0.5,
        "event_rate_per_km": (0.4, 2.0, 5.0),
        "speed_bump_frac": 0.25,
        "event_amp_g": (0.5, 1.6),
        "event_amp_limits_g": (0.5, 2.0),
        "event_min_speed_mps": 1.5,
        "event_len_s": 1.5,
        "hop_tau_s": 0.06,
        "body_tau_s": 0.35,
        "rear_scale": (0.7, 0.9),
        "event_pitch_dps_per_g": 15.0,
    },

    # ---- rider micro-motion -------------------------------------------------
    "rider": {
        "band_hz": (0.5, 2.0),
        "acc_g": (0.008, 0.025),       # ax, ay; az uses half
        "gx_dps": (0.8, 2.5),
        "gy_dps": (0.4, 1.2),
        "gz_dps": (0.5, 1.5),
        "handlebar_scale": 1.5,
    },

    # ---- sensor (MPU6050-class) ---------------------------------------------
    "sensor": {
        "tilt_deg": 3.0,
        "accel_bias_mg": (20.0, 50.0),
        "gyro_bias_dps": (1.0, 3.0),
        "scale_err": (0.01, 0.03),
        "cross_axis": (0.01, 0.02),
        "accel_noise_mg": (4.0, 10.0),           # interpolated by per-ride noise_scale
        "gyro_noise_dps": (0.05, 0.30),
        "gyro_rw_dps_rt_s": (0.005, 0.02),
        "ambient_c": (6.0, 28.0),
        "heat_rise_c": (8.0, 25.0),
        "heat_tau_s": (60.0, 240.0),
        "accel_tc_mg_per_c": 0.8,
        "gyro_tc_dps_per_c": 0.04,
        "dlpf_accel_hz": 94.0,
        "dlpf_gyro_hz": 98.0,
        "dlpf_order": 2,
        "accel_range_g": 8.0,
        "gyro_range_dps": 1000.0,
        "adc_bits": 16,
        "jitter_s": 0.00015,
        "jitter_spike_prob": 0.001,
        "jitter_spike_s": (0.001, 0.003),
        "drop_prob": 0.002,
        "drop_burst_prob": 0.0002,
        "drop_burst_len": (2, 10),
        "dup_prob": 0.001,
    },

    # ---- validation ---------------------------------------------------------
    "validation": {
        "window_s": 10.0,
        "steady_min_lean_deg": 12.0,
        "steady_max_rate_dps": 1.0,
        "steady_max_roll_acc_dps2": 5.0,
        "steady_max_along_g": 0.03,
        "steady_min_speed_mps": 3.0,
        "steady_min_run_s": 0.5,
        "rigid_tol_g": 0.01,
        "measured_median_tol_g": 0.10,
        "spectrogram_class": "aggressive",
    },
}


# =============================================================================
# Helpers
# =============================================================================
def U(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def lerp(lo_hi, u):
    return float(lo_hi[0] + (lo_hi[1] - lo_hi[0]) * u)


def rand_sign(rng, n=None):
    return rng.choice([-1.0, 1.0], size=n)


def raised_cos_ramp(n):
    k = (np.arange(n) + 0.5) / n
    return 0.5 * (1.0 - np.cos(np.pi * k))


def unit_std(x, axis=-1):
    s = x.std(axis=axis, keepdims=True)
    s[s == 0] = 1.0
    return x / s


def butter_sos(order, fc, fs, btype="low"):
    return signal.butter(order, fc, btype=btype, fs=fs, output="sos")


def peak_sos(f0, q, fs):
    b, a = signal.iirpeak(f0, q, fs=fs)
    return signal.tf2sos(b, a)


def filt_dc(sos, x):
    """Causal filter started in steady state at x[0] (no start-up transient)."""
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


def rot_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def runs_of_true(mask):
    """Return (start, end_exclusive) index pairs of contiguous True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# =============================================================================
# 1. Per-ride parameters
# =============================================================================
def draw_ride_params(ride_id, label, rng, cfg):
    cc = cfg["classes"][label]
    lo, hi = cc["intensity"]
    r = U(rng, cc["intensity"])
    u0 = (r - lo) / (hi - lo)

    def pos():
        return float(np.clip(u0 + rng.normal(0, cfg["within_class_jitter"]), 0, 1))

    dt_cfg, eng, road, rider, sen, veh = (cfg["drivetrain"], cfg["engine"], cfg["road"],
                                          cfg["rider"], cfg["sensor"], cfg["vehicle"])
    rp = {"ride_id": ride_id, "label": label, "rider_intensity": r}
    for key in ("brake_g", "lean_deg", "accel_g", "lean_omega", "jerk_rise_s",
                "upshift_rpm", "downshift_rpm", "brake_downshift_rpm", "cruise_kmh"):
        rp[key + "_center"] = lerp(cc[key], pos())
    # jerk_rise is "slower" for lower u: flip so higher intensity -> faster ramps
    rp["jerk_rise_s_center"] = lerp(cc["jerk_rise_s"][::-1], pos())
    rp["cruise_s"] = cc["cruise_s"]
    rp["stop_s"] = cc["stop_s"]
    rp["alt_mood_prob"] = cc["alt_mood_prob"]
    rp["alt_mood_factor"] = cc["alt_mood_factor"]

    # style-dependent drivetrain behaviour, interpolated by global intensity r
    rp["launch_rpm"] = lerp(dt_cfg["launch_rpm"], r)
    rp["min_shift_gap_s"] = lerp(dt_cfg["min_shift_gap_s"], r)
    rp["cruise_upshift_rpm"] = lerp(dt_cfg["cruise_upshift_rpm"], r)
    rp["shift_dip_rpm"] = lerp(dt_cfg["shift_dip_rpm"], r)
    rp["shift_jolt_g"] = lerp(dt_cfg["shift_jolt_g"], r)

    # chassis pitch
    pc = cfg["pitch"]
    rp["dive_deg_per_g"] = U(rng, pc["dive_deg_per_g"])
    rp["squat_deg_per_g"] = U(rng, pc["squat_deg_per_g"])
    rp["pitch_freq_hz"] = U(rng, pc["freq_hz"])
    rp["pitch_zeta"] = U(rng, pc["zeta"])

    # mount and engine vibration
    rp["mount_type"] = "handlebar" if rng.random() < eng["handlebar_prob"] else "frame"
    rp["rubber_mount"] = bool(rng.random() < eng["rubber_prob"])
    rp["rubber_cutoff_hz"] = U(rng, eng["rubber_cutoff_hz"]) if rp["rubber_mount"] else np.nan
    rp["handlebar_gain"] = U(rng, eng["handlebar_gain"]) if rp["mount_type"] == "handlebar" else 1.0
    rp["sensor_height_m"] = U(rng, veh["sensor_height_m"][rp["mount_type"]])
    rp["vib_amp_g"] = U(rng, eng["vib_amp_g"])
    rp["vib_amp_jitter"] = U(rng, eng["amp_jitter"])
    rp["vib_phase_jitter_rad"] = U(rng, eng["phase_jitter_rad"])
    n_res = int(rng.integers(eng["n_resonances"][0], eng["n_resonances"][1] + 1))
    rp["resonances"] = [(U(rng, eng["res_freq_hz"]), U(rng, eng["res_q"]), U(rng, eng["res_gain"]))
                        for _ in range(n_res)]

    # road
    rp["road_roughness"] = U(rng, road["roughness"])
    rr = rp["road_roughness"]
    p = np.array(road["surface_base_probs"]) * np.array([1.0, rr, rr ** 2])
    rp["surface_probs"] = p / p.sum()
    rp["body_freq_hz"] = U(rng, road["body_freq_hz"])
    rp["hop_freq_hz"] = U(rng, road["hop_freq_hz"])

    # rider micro-motion
    hs = rider["handlebar_scale"] if rp["mount_type"] == "handlebar" else 1.0
    a_mm = U(rng, rider["acc_g"])
    rp["micro_amp"] = np.array([a_mm, a_mm, 0.5 * a_mm,
                                U(rng, rider["gx_dps"]) * hs,
                                U(rng, rider["gy_dps"]),
                                U(rng, rider["gz_dps"]) * hs])

    # sensor
    t = math.radians(sen["tilt_deg"])
    rp["tilt_rad"] = rng.uniform(-t, t, 3)
    rp["accel_bias_g"] = rand_sign(rng, 3) * rng.uniform(*sen["accel_bias_mg"], 3) / 1000.0
    rp["gyro_bias_dps"] = rand_sign(rng, 3) * rng.uniform(*sen["gyro_bias_dps"], 3)
    rp["accel_scale_err"] = rand_sign(rng, 3) * rng.uniform(*sen["scale_err"], 3)
    rp["gyro_scale_err"] = rand_sign(rng, 3) * rng.uniform(*sen["scale_err"], 3)
    rp["accel_cross"] = rand_sign(rng, 6) * rng.uniform(*sen["cross_axis"], 6)
    rp["gyro_cross"] = rand_sign(rng, 6) * rng.uniform(*sen["cross_axis"], 6)
    rp["noise_scale"] = float(rng.random())
    rp["accel_noise_g"] = lerp(sen["accel_noise_mg"], rp["noise_scale"]) / 1000.0
    rp["gyro_noise_dps"] = lerp(sen["gyro_noise_dps"], rp["noise_scale"])
    rp["gyro_rw"] = U(rng, sen["gyro_rw_dps_rt_s"])
    rp["ambient_c"] = U(rng, sen["ambient_c"])
    rp["heat_rise_c"] = U(rng, sen["heat_rise_c"])
    rp["heat_tau_s"] = U(rng, sen["heat_tau_s"])
    rp["accel_tc_g_per_c"] = rng.uniform(-1, 1, 3) * sen["accel_tc_mg_per_c"] / 1000.0
    rp["gyro_tc_dps_per_c"] = rng.uniform(-1, 1, 3) * sen["gyro_tc_dps_per_c"]
    return rp


# =============================================================================
# 2. Manoeuvres -> a(t), v(t), lean(t), surface(t)
# =============================================================================
def _S1(tau, om):
    """Integral of the critically damped unit-step response (0 for tau<0)."""
    tau = np.maximum(tau, 0.0)
    return tau - 2.0 / om + (2.0 / om + tau) * np.exp(-om * tau)


def _S(tau, om):
    tau = np.maximum(tau, 0.0)
    return 1.0 - (1.0 + om * tau) * np.exp(-om * tau)


def _H(tau, om):
    tau = np.maximum(tau, 0.0)
    return om * om * tau * np.exp(-om * tau)


def generate_profile(rp, rng, cfg):
    fs = cfg["fs_sim"]
    dt = 1.0 / fs
    n_total = int(round(cfg["ride_duration_s"] * fs))
    sig = cfg["event_lognormal_sigma"]
    tsig = cfg["event_timing_sigma"]
    veh = cfg["vehicle"]
    vmax = cfg["max_speed_kmh"] / 3.6
    vmin_cruise = cfg["min_cruise_kmh"] / 3.6
    road = cfg["road"]

    st = {"n": 0, "v": 0.0, "alt": False, "surf": int(rng.choice(3, p=rp["surface_probs"]))}
    acc_chunks, surf_chunks, lean_events = [], [], []
    counts = {"stop": 0, "accel": 0, "brake": 0, "cruise": 0, "corner": 0}
    p_stay = cfg["mood_stay_prob"]
    pa = rp["alt_mood_prob"]
    p_in = pa * (1 - p_stay) / max(1e-6, 1 - pa)

    def mood():
        return rp["alt_mood_factor"] if st["alt"] else 1.0

    def push(a_arr, kind):
        acc_chunks.append(a_arr)
        surf_chunks.append(np.full(len(a_arr), st["surf"], np.int8))
        st["v"] = max(st["v"] + float(a_arr.sum()) * dt, 0.0)
        st["n"] += len(a_arr)
        counts[kind] += 1

    def speed_change(v1, peak_g, rise_s):
        dv = v1 - st["v"]
        if abs(dv) < 0.05:
            return np.zeros(int(0.5 * fs))
        a_p = peak_g * G
        tr = max(rise_s, cfg["min_ramp_s"])
        if abs(dv) / a_p < tr:            # small change: shorter ramps, lower peak
            tr = max(cfg["min_ramp_s"], abs(dv) / a_p)
            a_p = abs(dv) / tr
        tp = abs(dv) / a_p - tr
        nr = max(2, int(round(tr * fs)))
        npl = max(0, int(round(tp * fs)))
        ramp = raised_cos_ramp(nr)
        prof = np.concatenate([ramp, np.ones(npl), ramp[::-1]])
        return prof * (dv / (prof.sum() * dt))

    def lognorm():
        return math.exp(rng.normal(0, sig))

    def rise():
        return float(np.clip(rp["jerk_rise_s_center"] / math.sqrt(mood())
                             * math.exp(rng.normal(0, tsig)), 0.15, 2.0))

    def accel_peak(v0, v1):
        a = np.clip(rp["accel_g_center"] * mood() * lognorm(), *cfg["accel_g_limits"])
        vm = max(0.5 * (v0 + v1), 4.0)
        cap = veh["power_cap_frac"] * veh["power_w"] / (veh["mass_kg"] * vm) / G
        return float(min(a, cap))

    def brake_peak():
        return float(np.clip(rp["brake_g_center"] * mood() * lognorm(), *cfg["brake_g_limits"]))

    def cruise_target():
        v1 = rp["cruise_kmh_center"] / 3.6 * math.exp(rng.normal(0, cfg["cruise_speed_sigma"]))
        v1 *= mood() ** 0.4
        return float(np.clip(v1, vmin_cruise, vmax))

    first = True
    while st["n"] < n_total:
        st["alt"] = (rng.random() < p_stay) if st["alt"] else (rng.random() < p_in)
        if rng.random() > road["surface_stay_prob"]:
            st["surf"] = int(rng.choice(3, p=rp["surface_probs"]))
        v = st["v"]

        if v < 0.2:
            st["v"] = 0.0
            dur = U(rng, cfg["first_stop_s"]) if first else U(rng, rp["stop_s"]) / mood() ** 0.5
            first = False
            push(np.zeros(int(dur * fs)), "stop")
            v1 = cruise_target()
            push(speed_change(v1, accel_peak(0.0, v1), rise()), "accel")
            continue

        kinds = list(cfg["maneuver_probs"].keys())
        probs = np.array(list(cfg["maneuver_probs"].values()))
        kind = str(rng.choice(kinds, p=probs / probs.sum()))
        if kind == "corner" and v < cfg["corner_min_speed_mps"]:
            kind = "accel"
        if kind == "accel" and v > 0.9 * vmax:
            kind = "cruise"

        if kind == "cruise":
            dur = U(rng, rp["cruise_s"])
            n = int(dur * fs)
            cycles = int(rng.integers(1, 3))
            # a = A*sin(2*pi*c*t/T) swings speed by A*T/(pi*c) peak; size A from that swing
            amp = U(rng, cfg["cruise_wobble_mps"]) * np.pi * cycles / dur
            push(amp * np.sin(2 * np.pi * cycles * np.arange(n) / n), "cruise")

        elif kind == "accel":
            v1 = cruise_target()
            if v1 < v + 1.5:
                v1 = min(vmax, v * U(rng, (1.2, 1.6)))
            if v1 <= v + 0.5:
                push(np.zeros(int(2 * fs)), "cruise")
            else:
                push(speed_change(v1, accel_peak(v, v1), rise()), "accel")

        elif kind == "brake":
            v1 = 0.0 if rng.random() < cfg["brake_to_stop_prob"] else v * U(rng, cfg["brake_partial_frac"])
            push(speed_change(v1, brake_peak(), rise()), "brake")

        else:  # corner: optional braking to entry speed, lean event, optional drive out
            v_entry = v * U(rng, cfg["corner_entry_speed_frac"])
            v_entry = max(v_entry, cfg["corner_min_speed_mps"])
            if v_entry < v - 0.5:
                push(speed_change(v_entry, brake_peak(), rise()), "brake")
            vc = st["v"]
            lean = np.clip(rp["lean_deg_center"] * mood() * lognorm(), *cfg["lean_deg_limits"])
            lean_cap = math.degrees(math.atan(vc ** 2 / (G * cfg["corner_min_radius_m"])))
            lean = min(lean, lean_cap)
            om = float(np.clip(rp["lean_omega_center"] * math.sqrt(mood())
                               * math.exp(rng.normal(0, tsig)), 1.0, 8.0))
            tr = cfg["lean_ramp_omegas"] / om
            hold = U(rng, cfg["corner_hold_s"])
            t_off = tr + 4.0 / om + hold
            lean_events.append((st["n"], st["n"] + int(t_off * fs),
                                float(rand_sign(rng)) * math.radians(lean), om, tr))
            push(np.zeros(int((t_off + 2.0 / om) * fs)), "corner")
            if rng.random() < cfg["corner_exit_accel_prob"]:
                v1 = max(cruise_target(), min(vmax, vc + 2.0))
                push(speed_change(v1, accel_peak(vc, v1), rise()), "accel")

    a = np.concatenate(acc_chunks)[:n_total]
    surf = np.concatenate(surf_chunks)[:n_total]
    v = np.clip(np.cumsum(a) * dt, 0.0, None)
    a[(v <= 0.0) & (a < 0)] = 0.0

    # lean: superposition of ramp-into-critically-damped responses (analytic derivatives)
    phi = np.zeros(n_total)
    phid = np.zeros(n_total)
    phidd = np.zeros(n_total)
    for n_on, n_off, php, om, tr in lean_events:
        if n_on >= n_total:
            continue
        n_end = min(n_total, n_off + int((tr + cfg["lean_settle_omegas"] / om) * fs))
        idx = np.arange(n_on, n_end)
        k = php / tr
        for n0, sgn in ((n_on, 1.0), (n_off, -1.0)):
            tau = (idx - n0) * dt
            phi[idx] += sgn * k * (_S1(tau, om) - _S1(tau - tr, om))
            phid[idx] += sgn * k * (_S(tau, om) - _S(tau - tr, om))
            phidd[idx] += sgn * k * (_H(tau, om) - _H(tau - tr, om))

    return {"a": a, "v": v, "phi": phi, "phid": phid, "phidd": phidd, "surf": surf,
            "counts": counts, "n_corners": len(lean_events)}


# =============================================================================
# 3. Drivetrain: gear, rpm, shift transients, engine load
# =============================================================================
def simulate_drivetrain(v, a, rp, cfg, rng):
    fs = cfg["fs_sim"]
    veh, dtc = cfg["vehicle"], cfg["drivetrain"]
    n = len(v)
    k_rpm = np.array([veh["primary_ratio"] * g * veh["final_ratio"] * 60.0
                      / (2 * np.pi * veh["wheel_radius_m"]) for g in veh["gear_ratios"]])
    idle, red = veh["idle_rpm"], veh["redline_rpm"]
    n_gears = len(k_rpm)
    dec = int(round(fs / dtc["logic_rate_hz"]))
    vd, ad = v[::dec].tolist(), a[::dec].tolist()
    nd = len(vd)
    gear_d = np.empty(nd, dtype=np.int16)
    rpm_d = np.empty(nd)
    gear, last = 1, -1e9
    shifts = []
    up_cur = rp["upshift_rpm_center"] * math.exp(rng.normal(0, dtc["upshift_jitter_sigma"]))
    for i in range(nd):
        t = i * dec / fs
        vi, ai = vd[i], ad[i]
        if vi < 0.3:
            gear = 1                                   # clutch in, first selected at rest
            rpm = rp["launch_rpm"] if ai > 0.03 * G else idle
        else:
            r = vi * k_rpm[gear - 1]
            if t - last >= rp["min_shift_gap_s"]:
                new = gear
                if gear < n_gears:
                    r_up = vi * k_rpm[gear]
                    if r > red - 150:
                        new = gear + 1
                    elif ai > 0.02 * G and r >= up_cur:
                        new = gear + 1
                    elif abs(ai) <= 0.02 * G and r >= rp["cruise_upshift_rpm"] \
                            and r_up > rp["downshift_rpm_center"] + 600:
                        new = gear + 1
                if new == gear and gear > 1:
                    thr = rp["brake_downshift_rpm_center"] if ai < -0.08 * G else rp["downshift_rpm_center"]
                    if r < thr and vi * k_rpm[gear - 2] < red - 400:
                        new = gear - 1
                if new != gear:
                    shifts.append((i * dec, gear, new, ai))
                    gear, last = new, t
                    up_cur = rp["upshift_rpm_center"] * math.exp(rng.normal(0, dtc["upshift_jitter_sigma"]))
            r = vi * k_rpm[gear - 1]
            if gear == 1 and ai > 0.03 * G:
                r = max(r, rp["launch_rpm"])           # clutch slip on launch
            rpm = max(r, idle)
        gear_d[i], rpm_d[i] = gear, rpm

    t_sim = np.arange(n) / fs
    rpm = np.interp(t_sim, np.arange(nd) * dec / fs, rpm_d)
    gear_full = gear_d[np.minimum(np.arange(n) // dec, nd - 1)]

    # shift transients: rpm blend + ~0.3 s dip, longitudinal jolt, pitch, load cut
    T = dtc["shift_dur_s"]
    tau = dtc["shift_blend_tau_s"]
    n_t = int(T * fs)
    jolt_x = np.zeros(n)
    jolt_gy = np.zeros(n)
    load_cut = np.ones(n)
    for n0, g_old, g_new, ai in shifts:
        n1 = min(n, n0 + n_t + int(0.4 * fs))
        k = np.arange(n0, n1)
        tt = (k - n0) / fs
        win = tt < T
        v0 = v[n0]
        r_old = max(v0 * k_rpm[g_old - 1], idle)
        r_new = max(v0 * k_rpm[g_new - 1], idle)
        up = g_new > g_old
        dip = rp["shift_dip_rpm"] * math.exp(rng.normal(0, 0.15)) * (1.0 if up else dtc["downshift_dip_scale"])
        rpm[k] += (r_old - r_new) * np.exp(-tt / tau) - dip * np.sin(np.pi * tt / T) * win
        J = rp["shift_jolt_g"] * math.exp(rng.normal(0, 0.2))
        if up:
            J *= float(np.clip(0.4 + ai / (0.25 * G), 0.3, 1.5))
            after = ~win
            jx = (-J * np.sin(np.pi * tt / T) * win
                  + 0.5 * J * np.exp(-(tt - T) / dtc["shift_rebound_tau_s"])
                  * np.sin(2 * np.pi * dtc["shift_rebound_hz"] * (tt - T)) * after)
        else:
            jx = -0.6 * J * np.sin(np.pi * tt / T) * win
        jolt_x[k] += jx
        jolt_gy[k] += -jx * dtc["shift_jolt_pitch_dps_per_g"]
        load_cut[n0:min(n, n0 + n_t)] = dtc["shift_load"]
    rpm = np.maximum(rpm, 0.85 * idle)

    load = np.clip(0.3 + a / (0.35 * G) + 0.3 * (v / 28.0) ** 2, 0.1, 1.0)
    load[(v < 0.3) & (a <= 0.03 * G)] = 0.3
    load = np.minimum(load, load_cut)
    alpha = 1.0 / (dtc["load_smooth_s"] * fs)
    load = signal.lfilter([alpha], [1, alpha - 1], load, zi=[load[0] * (1 - alpha)])[0]
    return {"rpm": rpm, "gear": gear_full, "jolt_x": jolt_x, "jolt_gy": jolt_gy,
            "load": load, "n_shifts": len(shifts)}


# =============================================================================
# 4. Engine vibration (phase-integrated, harmonics, resonances, mount)
# =============================================================================
def engine_vibration(rpm, load, rp, cfg, rng):
    fs = cfg["fs_sim"]
    ec = cfg["engine"]
    n = len(rpm)
    f1 = rpm / 60.0
    phase = 2 * np.pi * np.cumsum(f1) / fs              # phase = 2*pi*integral(f dt)
    cyc = phase / (4 * np.pi)                           # combustion cycle = 2 revs
    nc = int(cyc[-1]) + 3
    grid = np.arange(nc)
    amp_t = np.interp(cyc, grid, 1.0 + rp["vib_amp_jitter"] * rng.standard_normal(nc))
    crank = phase + np.interp(cyc, grid, rp["vib_phase_jitter_rad"] * rng.standard_normal(nc))
    env = (rp["vib_amp_g"] * (rpm / ec["rpm_ref"]) ** 2
           * (ec["coast_frac"] + (1 - ec["coast_frac"]) * load) * amp_t)
    gains = np.array(ec["axis_gain"])[:, None]
    lo, hi = ec["sim_taper"]
    out = np.zeros((6, n))
    for order, rel in zip(ec["orders"], ec["order_amp"]):
        f = order * f1
        taper = np.clip((hi * fs - f) / ((hi - lo) * fs), 0.0, 1.0)  # avoid sim-rate aliasing
        s, c = np.sin(order * crank), np.cos(order * crank)
        offs = rng.uniform(0, 2 * np.pi, 6)[:, None]
        out += gains * (rel * taper) * (s * np.cos(offs) + c * np.sin(offs))
    out *= env
    base = out.copy()
    for f0, q, g in rp["resonances"]:                   # lightly damped frame modes
        out += g * signal.sosfilt(peak_sos(f0, q, fs), base, axis=1)
    out *= rp["handlebar_gain"]
    if rp["rubber_mount"]:
        out = signal.sosfilt(butter_sos(2, rp["rubber_cutoff_hz"], fs), out, axis=1)
    return out


# =============================================================================
# 5. Road, suspension, potholes / bumps, rider micro-motion
# =============================================================================
def road_and_suspension(v, surf, rp, cfg, rng):
    fs = cfg["fs_sim"]
    rc = cfg["road"]
    n = len(v)
    white = rng.standard_normal((5, n))
    brown = unit_std(signal.lfilter([1.0], [1.0, -rc["brown_pole"]], white[:4], axis=1))
    col = unit_std(brown + rc["white_mix"] * white[:4])
    body = unit_std(signal.sosfilt(peak_sos(rp["body_freq_hz"], rc["body_q"], fs), col[0]))
    hop = unit_std(signal.sosfilt(peak_sos(rp["hop_freq_hz"], rc["hop_q"], fs), col[1]))
    hop2 = unit_std(signal.sosfilt(peak_sos(rp["hop_freq_hz"], rc["hop_q"], fs), col[2]))
    pitchn = unit_std(signal.sosfilt(peak_sos(rp["body_freq_hz"], rc["body_q"], fs), col[3]))
    hf = unit_std(signal.sosfilt(butter_sos(2, rc["cobble_band_hz"], fs, "band"), white[4]))

    level = np.array(rc["surface_level"])[surf]
    spd = np.minimum((v / rc["v_ref_mps"]) ** rc["speed_exp"], rc["speed_factor_max"])
    env = rp["road_roughness"] * level * spd
    cob = (surf == 2).astype(float)
    out = np.zeros((6, n))
    out[2] = env * (rc["body_rms_g"] * body + rc["hop_rms_g"] * hop + cob * rc["cobble_rms_g"] * hf)
    out[0] = env * rc["ax_frac"] * rc["hop_rms_g"] * hop2
    out[1] = env * rc["ay_frac"] * rc["hop_rms_g"] * hop2[::-1]
    out[3] = env * rc["gx_dps"] * hop2[::-1]
    out[4] = env * rc["gy_dps"] * pitchn
    out[5] = env * rc["gz_dps"] * hop2

    # potholes and speed bumps placed by distance; rear wheel repeats after L/v
    s = np.cumsum(v) / fs
    lam = np.array(rc["event_rate_per_km"]) / 1000.0 * rp["road_roughness"]
    lam_max = lam.max()
    n_c = rng.poisson(lam_max * s[-1]) if s[-1] > 0 else 0
    n_events = 0
    L = int(rc["event_len_s"] * fs)
    for p in np.sort(rng.uniform(0, s[-1], n_c)):
        i0 = int(np.searchsorted(s, p))
        if i0 >= n or rng.random() > lam[surf[i0]] / lam_max:
            continue
        vi = v[i0]
        if vi < rc["event_min_speed_mps"]:
            continue
        n_events += 1
        bump = rng.random() < rc["speed_bump_frac"]
        A = float(np.clip(U(rng, rc["event_amp_g"]) * (vi / 10.0) ** 0.5, *rc["event_amp_limits_g"]))
        side = float(rand_sign(rng))
        i_rear = i0 + int(cfg["vehicle"]["wheelbase_m"] / vi * fs)
        for start, scale, sgn in ((i0, 1.0, -1.0), (i_rear, U(rng, rc["rear_scale"]), 1.0)):
            if start >= n:
                continue
            k = np.arange(start, min(n, start + L))
            tt = (k - start) / fs
            eh = np.exp(-tt / rc["hop_tau_s"])
            eb = np.exp(-tt / rc["body_tau_s"])
            h = eh * np.sin(2 * np.pi * rp["hop_freq_hz"] * tt)
            b = eb * np.sin(2 * np.pi * rp["body_freq_hz"] * tt)
            wz = (0.5 * h + 0.8 * b) if bump else (h + 0.25 * b)
            wz /= np.abs(wz).max()
            amp = scale * A
            out[2, k] += amp * wz
            out[0, k] += amp * 0.3 * eh * np.cos(2 * np.pi * rp["hop_freq_hz"] * tt)
            out[1, k] += amp * 0.08 * side * h
            out[3, k] += amp * 1.0 * side * h
            out[4, k] += sgn * amp * rc["event_pitch_dps_per_g"] * (b + 0.3 * h)
    return out, n_events


def rider_micro_motion(n, rp, cfg, rng):
    fs = cfg["fs_sim"]
    sos = butter_sos(2, cfg["rider"]["band_hz"], fs, "band")
    m = unit_std(signal.sosfilt(sos, rng.standard_normal((6, n)), axis=1))
    return m * rp["micro_amp"][:, None]


# =============================================================================
# Full physics for one ride
# =============================================================================
def simulate_ride(rp, cfg, rng):
    fs = cfg["fs_sim"]
    prof = generate_profile(rp, rng, cfg)
    v, a, phi, phid, phidd = prof["v"], prof["a"], prof["phi"], prof["phid"], prof["phidd"]
    n = len(v)
    dt = 1.0 / fs
    t = np.arange(n) * dt
    dtr = simulate_drivetrain(v, a, rp, cfg, rng)

    # chassis pitch (fork dive under braking, squat under throttle)
    th_tgt = np.radians(rp["dive_deg_per_g"] * np.clip(-a / G, 0, None)
                        - rp["squat_deg_per_g"] * np.clip(a / G, 0, None))
    wn, z = 2 * np.pi * rp["pitch_freq_hz"], rp["pitch_zeta"]
    b, aa = signal.bilinear([wn ** 2], [1.0, 2 * z * wn, wn ** 2], fs)
    theta = signal.sosfilt(signal.tf2sos(b, aa), th_tgt)
    thetad = np.gradient(theta, dt)

    # yaw rate (steady turn), faded to zero at standstill
    psid = -G * np.tan(phi) / np.maximum(v, 1.0) * np.clip(v / 1.0, 0.0, 1.0)

    h = rp["sensor_height_m"]
    # rigid-body specific force [g] and angular rate [deg/s] in the body frame
    rig_acc = np.vstack([
        a / G - np.sin(theta),
        -h * phidd / G,                                   # roll acceleration only
        np.cos(theta) / np.cos(phi) - h * phid ** 2 / G,  # g/cos(phi) in steady lean
    ])
    rig_gyr = np.degrees(np.vstack([phid, psid * np.sin(phi) + thetad, psid * np.cos(phi)]))
    rig_acc[0] += dtr["jolt_x"]
    rig_gyr[1] += dtr["jolt_gy"]

    road, n_events = road_and_suspension(v, prof["surf"], rp, cfg, rng)
    micro = rider_micro_motion(n, rp, cfg, rng)
    vib = engine_vibration(dtr["rpm"], dtr["load"], rp, cfg, rng)

    acc = rig_acc + road[:3] + micro[:3] + vib[:3]
    gyr = rig_gyr + road[3:] + micro[3:] + vib[3:]
    return {"t": t, "v": v, "a": a, "phi": phi, "phid": phid, "phidd": phidd, "surf": prof["surf"],
            "rpm": dtr["rpm"], "gear": dtr["gear"], "acc": acc, "gyr": gyr,
            "rig_acc": rig_acc, "counts": prof["counts"], "n_corners": prof["n_corners"],
            "n_shifts": dtr["n_shifts"], "n_road_events": n_events,
            "vib_az_rms": float(vib[2][v > 0.3].std()) if np.any(v > 0.3) else 0.0}


# =============================================================================
# 6. Sensor model and sampling
# =============================================================================
def cross_matrix(scale, cross):
    m = np.eye(3) + np.diag(scale)
    m[0, 1], m[0, 2], m[1, 0], m[1, 2], m[2, 0], m[2, 1] = cross
    return m


def sensor_model(sim, rp, cfg, rng):
    fs, fo = cfg["fs_sim"], cfg["fs_out"]
    sc = cfg["sensor"]
    t = sim["t"]
    n = len(t)
    R = rot_matrix(*rp["tilt_rad"])
    acc = cross_matrix(rp["accel_scale_err"], rp["accel_cross"]) @ (R @ sim["acc"])
    gyr = cross_matrix(rp["gyro_scale_err"], rp["gyro_cross"]) @ (R @ sim["gyr"])

    d_temp = rp["heat_rise_c"] * (1.0 - np.exp(-t / rp["heat_tau_s"]))
    acc += rp["accel_bias_g"][:, None] + rp["accel_tc_g_per_c"][:, None] * d_temp
    rw = np.cumsum(rng.normal(0, rp["gyro_rw"] / math.sqrt(fs), (3, n)), axis=1)
    gyr += rp["gyro_bias_dps"][:, None] + rp["gyro_tc_dps_per_c"][:, None] * d_temp + rw

    ar, gr = sc["accel_range_g"], sc["gyro_range_dps"]
    adc_sat = int((np.abs(acc) >= ar).sum() + (np.abs(gyr) >= gr).sum())
    acc = np.clip(acc, -ar, ar)
    gyr = np.clip(gyr, -gr, gr)

    sos_a = butter_sos(sc["dlpf_order"], sc["dlpf_accel_hz"], fs)
    sos_g = butter_sos(sc["dlpf_order"], sc["dlpf_gyro_hz"], fs)
    acc = np.vstack([filt_dc(sos_a, x) for x in acc])
    gyr = np.vstack([filt_dc(sos_g, x) for x in gyr])

    # sample instants with jitter; no anti-alias filter beyond the DLPF
    n_out = int(round(cfg["ride_duration_s"] * fo))
    ts = np.arange(n_out) / fo + rng.normal(0, sc["jitter_s"], n_out)
    spikes = rng.random(n_out) < sc["jitter_spike_prob"]
    ts[spikes] += rng.uniform(*sc["jitter_spike_s"], spikes.sum())
    ts = np.maximum.accumulate(np.clip(ts, 0.0, t[-1]))
    data = np.vstack([np.interp(ts, t, x) for x in np.vstack([acc, gyr])])
    data[:3] += rng.normal(0, rp["accel_noise_g"], (3, n_out))
    data[3:] += rng.normal(0, rp["gyro_noise_dps"], (3, n_out))

    # 16-bit quantisation and range clipping
    full = 2 ** (sc["adc_bits"] - 1)
    lsb = np.array([ar] * 3 + [gr] * 3)[:, None] / full
    counts = np.clip(np.round(data / lsb), -full, full - 1)
    out_sat = int(((counts <= -full) | (counts >= full - 1)).sum())
    data = counts * lsb

    # duplicated (stale) and dropped samples
    dup = np.where(rng.random(n_out) < sc["dup_prob"])[0]
    dup = dup[dup > 0]
    data[:, dup] = data[:, dup - 1]
    keep = rng.random(n_out) >= sc["drop_prob"]
    for start in np.where(rng.random(n_out) < sc["drop_burst_prob"])[0]:
        keep[start:start + int(rng.integers(sc["drop_burst_len"][0], sc["drop_burst_len"][1] + 1))] = False
    return {"t": np.round(ts[keep], 6), "data": data[:, keep], "adc_sat": adc_sat,
            "adc_total": 6 * n, "out_sat": int(((counts[:, keep] <= -full) | (counts[:, keep] >= full - 1)).sum()),
            "n_out": int(keep.sum()), "n_dropped": int((~keep).sum()), "n_dup": int(len(dup))}


# =============================================================================
# 7. Validation
# =============================================================================
def window_features(ride_id, label, t, data, win_s):
    idx = np.floor(t / win_s).astype(int)
    df = pd.DataFrame({"w": idx, "az": data[2], "gz": np.abs(data[5])})
    g = df.groupby("w")
    f = pd.DataFrame({"peak_az": g["az"].max(), "peak_abs_gz": g["gz"].max(),
                      "az_rms_ac": g["az"].std()})
    f["ride_id"], f["label"] = ride_id, label
    return f.reset_index(drop=True)


def overlap_coef(a, b, bins):
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return float(np.minimum(ha, hb).sum())


def plot_spectrogram(dbg, cfg, path):
    fs, fo = cfg["fs_sim"], cfg["fs_out"]
    fig, axs = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    f, tt, S = signal.spectrogram(dbg["az_phys"] - dbg["az_phys"].mean(), fs=fs, nperseg=1024, noverlap=768)
    axs[0].pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma", vmin=-80)
    rpm_s = np.interp(tt, dbg["t_sim"], dbg["rpm"])
    for o in cfg["engine"]["orders"]:
        axs[0].plot(tt, o * rpm_s / 60, "c--", lw=0.7, alpha=0.7)
    axs[0].set_ylim(0, fs / 2)
    axs[0].set_ylabel("Hz")
    axs[0].set_title(f"{dbg['ride_id']} ({dbg['label']}): physical az at {fs:.0f} Hz, "
                     f"dashed = engine orders {cfg['engine']['orders']} x crank")

    tu = np.arange(0, dbg["t_out"][-1], 1 / fo)
    azu = np.interp(tu, dbg["t_out"], dbg["az_out"])
    f2, tt2, S2 = signal.spectrogram(azu - azu.mean(), fs=fo, nperseg=256, noverlap=192)
    axs[1].pcolormesh(tt2, f2, 10 * np.log10(S2 + 1e-12), shading="auto", cmap="magma", vmin=-80)
    rpm_s2 = np.interp(tt2, dbg["t_sim"], dbg["rpm"])
    for o in (0.5, 1.0, 2.0):
        fa = np.abs(((o * rpm_s2 / 60 + fo / 2) % fo) - fo / 2)
        axs[1].plot(tt2, fa, "c:", lw=0.7, alpha=0.7)
    axs[1].set_ylim(0, fo / 2)
    axs[1].set_ylabel("Hz")
    axs[1].set_title(f"Logged az at {fo:.0f} Hz (dotted = predicted alias of 0.5x, 1x, 2x crank)")

    axs[2].plot(dbg["t_sim"][::10], dbg["rpm"][::10], lw=0.8, label="rpm")
    axs[2].set_ylabel("rpm")
    ax2 = axs[2].twinx()
    ax2.step(dbg["t_sim"][::10], dbg["gear"][::10], "r", lw=0.8, where="post", label="gear")
    ax2.plot(dbg["t_sim"][::10], dbg["v"][::10] * 3.6 / 20, "g", lw=0.6, alpha=0.7, label="speed/20 [km/h]")
    ax2.set_ylabel("gear | speed/20")
    axs[2].set_xlabel("t [s]")
    axs[2].legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_histograms(wf, cfg, path):
    feats = [("peak_az", "peak az per window [g]"), ("peak_abs_gz", "peak |gz| per window [deg/s]"),
             ("az_rms_ac", "az RMS per window, mean removed [g]")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    print(f"\n[validation] Per-class window features ({cfg['validation']['window_s']:.0f} s windows) "
          f"- histogram overlap coefficient (1 = identical, 0 = separable):")
    for ax, (col, title) in zip(axs, feats):
        lo, hi = np.percentile(wf[col], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for lab, c in zip(("calm", "normal", "aggressive"), ("tab:green", "tab:blue", "tab:red")):
            ax.hist(wf.loc[wf.label == lab, col], bins=bins, density=True, histtype="stepfilled",
                    alpha=0.35, color=c, label=lab)
        ax.set_title(title)
        ax.legend()
        x = {lab: wf.loc[wf.label == lab, col].values for lab in ("calm", "normal", "aggressive")}
        oc_cn = overlap_coef(x["calm"], x["normal"], bins)
        oc_na = overlap_coef(x["normal"], x["aggressive"], bins)
        oc_ca = overlap_coef(x["calm"], x["aggressive"], bins)
        print(f"  {col:12s} calm-normal {oc_cn:.2f} | normal-aggr {oc_na:.2f} | calm-aggr {oc_ca:.2f} | "
              f"medians c/n/a = {np.median(x['calm']):.3f} / {np.median(x['normal']):.3f} / "
              f"{np.median(x['aggressive']):.3f}")
        if oc_cn < 0.05 or oc_na < 0.05:
            print(f"  WARNING: adjacent classes barely overlap on {col}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def steady_lean_check(debug_rides, cfg, path):
    vc = cfg["validation"]
    fs = cfg["fs_sim"]
    print("\n[validation] Steady-lean check: az ~ g/cos(phi)")
    pts = []
    for d in debug_rides:
        mask = ((np.abs(d["phi"]) > np.radians(vc["steady_min_lean_deg"]))
                & (np.abs(np.degrees(d["phid"])) < vc["steady_max_rate_dps"])
                & (np.abs(np.degrees(d["phidd"])) < vc["steady_max_roll_acc_dps2"])
                & (np.abs(d["a"]) < vc["steady_max_along_g"] * G)
                & (d["v"] > vc["steady_min_speed_mps"]))
        if not mask.any():
            print(f"  {d['ride_id']} ({d['label']}): no steady-lean samples")
            continue
        expect = 1.0 / np.cos(d["phi"][mask])
        err_z = np.abs(d["rig_acc"][2, mask] - expect).max()
        err_y = np.abs(d["rig_acc"][1, mask]).max()
        assert err_z < vc["rigid_tol_g"], f"rigid az != 1/cos(phi): max err {err_z:.4f} g"
        assert err_y < 2 * vc["rigid_tol_g"], f"rigid ay != 0 in steady lean: max {err_y:.4f} g"
        n_run = 0
        for s, e in runs_of_true(mask):
            if (e - s) / fs < vc["steady_min_run_s"]:
                continue
            t0, t1 = s / fs, e / fs
            sel = (d["t_out"] >= t0) & (d["t_out"] < t1)
            if sel.sum() < 5:
                continue
            pts.append((float(np.mean(1.0 / np.cos(d["phi"][s:e]))), float(d["az_out"][sel].mean()),
                        float(np.degrees(np.abs(d["phi"][s:e]).mean())), d["label"]))
            n_run += 1
        print(f"  {d['ride_id']} ({d['label']}): rigid-body max|az - 1/cos(phi)| = {err_z:.2e} g, "
              f"max|ay| = {err_y:.2e} g over {mask.sum() / fs:.1f} s; {n_run} steady runs")
    assert pts, "no steady-lean runs found in debug rides"
    p = np.array([(a, b, c) for a, b, c, _ in pts])
    err = p[:, 1] - p[:, 0]
    med = float(np.median(np.abs(err)))
    print(f"  measured (sensor output, run mean) vs g/cos(phi): median |err| = {med:.3f} g, "
          f"90th pct = {np.percentile(np.abs(err), 90):.3f} g over {len(err)} runs "
          f"(bias, scale, tilt, thermal drift and aliased vibration included)")
    assert med < vc["measured_median_tol_g"], f"measured az deviates from g/cos(phi): median {med:.3f} g"
    print("  PASS: az ~ g/cos(phi) during steady lean")

    fig, ax = plt.subplots(figsize=(6, 5))
    cols = {"calm": "tab:green", "normal": "tab:blue", "aggressive": "tab:red"}
    for a, b, c, lab in pts:
        ax.scatter(a, b, c=cols[lab], s=18)
    lim = [1.0, max(1.5, p[:, 0].max() + 0.05)]
    ax.plot(lim, lim, "k--", lw=1, label="az = 1/cos(phi)")
    ax.set_xlabel("1/cos(phi) [g]")
    ax.set_ylabel("mean logged az over steady-lean run [g]")
    ax.set_title("Steady lean: logged az vs g/cos(phi)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def flatten_meta(rp, sim, sens, cfg):
    row = {"ride_id": rp["ride_id"], "label": rp["label"], "rider_intensity": rp["rider_intensity"]}
    for k in ("brake_g_center", "lean_deg_center", "accel_g_center", "lean_omega_center",
              "jerk_rise_s_center", "upshift_rpm_center", "downshift_rpm_center",
              "brake_downshift_rpm_center", "cruise_kmh_center", "launch_rpm", "min_shift_gap_s",
              "cruise_upshift_rpm", "shift_dip_rpm", "shift_jolt_g", "mount_type", "rubber_mount",
              "rubber_cutoff_hz", "handlebar_gain", "sensor_height_m", "vib_amp_g",
              "vib_amp_jitter", "vib_phase_jitter_rad", "road_roughness", "body_freq_hz",
              "hop_freq_hz", "dive_deg_per_g", "noise_scale", "accel_noise_g", "gyro_noise_dps",
              "gyro_rw", "ambient_c", "heat_rise_c", "heat_tau_s"):
        row[k] = rp[k]
    for i, (f0, q, g) in enumerate(rp["resonances"]):
        row[f"res{i + 1}_hz"], row[f"res{i + 1}_q"], row[f"res{i + 1}_gain"] = f0, q, g
    for name, arr in (("tilt_deg", np.degrees(rp["tilt_rad"])), ("accel_bias_g", rp["accel_bias_g"]),
                      ("gyro_bias_dps", rp["gyro_bias_dps"]), ("accel_scale_err", rp["accel_scale_err"]),
                      ("gyro_scale_err", rp["gyro_scale_err"])):
        for ax, val in zip("xyz", arr):
            row[f"{name}_{ax}"] = float(val)
    row["accel_cross_max"] = float(np.abs(rp["accel_cross"]).max())
    row["gyro_cross_max"] = float(np.abs(rp["gyro_cross"]).max())
    # realised statistics (truth; for analysis only)
    row["real_max_lean_deg"] = float(np.degrees(np.abs(sim["phi"]).max()))
    row["real_max_brake_g"] = float(np.clip(-sim["a"], 0, None).max() / G)
    row["real_max_accel_g"] = float(np.clip(sim["a"], 0, None).max() / G)
    row["real_mean_speed_kmh"] = float(sim["v"].mean() * 3.6)
    row["real_distance_km"] = float(sim["v"].sum() / cfg["fs_sim"] / 1000)
    row["real_max_rpm"] = float(sim["rpm"].max())
    row["n_corners"] = sim["n_corners"]
    row["n_shifts"] = sim["n_shifts"]
    row["n_road_events"] = sim["n_road_events"]
    row["vib_az_rms_g"] = sim["vib_az_rms"]
    row["n_samples"] = sens["n_out"]
    row["n_dropped"] = sens["n_dropped"]
    row["n_duplicated"] = sens["n_dup"]
    row["adc_sat_share"] = sens["adc_sat"] / sens["adc_total"]
    row["out_fullscale_share"] = sens["out_sat"] / (6 * sens["n_out"])
    return row


def main(cfg=CONFIG):
    t_start = time.time()
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    p_imu = os.path.join(cfg["out_dir"], "imu_data.csv")
    if os.path.exists(p_imu):
        os.remove(p_imu)

    labels = [lab for lab, k in cfg["rides_per_class"].items() for _ in range(k)]
    rng.shuffle(labels)                         # ride_id carries no label information
    debug_idx = {lab: labels.index(lab) for lab in cfg["rides_per_class"]}
    print(f"Simulating {len(labels)} rides x {cfg['ride_duration_s']:.0f} s "
          f"(physics {cfg['fs_sim']:.0f} Hz -> output {cfg['fs_out']:.0f} Hz)")

    meta_rows, wf_list, gt_list, debug_rides = [], [], [], []
    sat_adc = sat_adc_tot = sat_out = sat_out_tot = 0
    for i, label in enumerate(labels):
        ride_id = f"R{i + 1:03d}"
        rrng = np.random.default_rng(rng.integers(0, 2 ** 63))
        rp = draw_ride_params(ride_id, label, rrng, cfg)
        sim = simulate_ride(rp, cfg, rrng)
        sens = sensor_model(sim, rp, cfg, rrng)

        df = pd.DataFrame({"ride_id": ride_id, "t": sens["t"]})
        for k, name in enumerate(AXES):
            df[name] = sens["data"][k]
        df["label"] = label
        df.to_csv(p_imu, mode="a", header=(i == 0), index=False, float_format=cfg["csv_float_format"])

        meta_rows.append(flatten_meta(rp, sim, sens, cfg))
        wf_list.append(window_features(ride_id, label, sens["t"], sens["data"], cfg["validation"]["window_s"]))
        sat_adc += sens["adc_sat"]
        sat_adc_tot += sens["adc_total"]
        sat_out += sens["out_sat"]
        sat_out_tot += 6 * sens["n_out"]

        if i == debug_idx[label]:
            to = sens["t"]
            ni = np.clip(np.round(to * cfg["fs_sim"]).astype(int), 0, len(sim["t"]) - 1)
            gt_list.append(pd.DataFrame({
                "ride_id": ride_id, "t": to,
                "speed": np.round(np.interp(to, sim["t"], sim["v"]), 4),
                "rpm": np.round(np.interp(to, sim["t"], sim["rpm"]), 1),
                "gear": sim["gear"][ni],
                "lean_deg": np.round(np.degrees(np.interp(to, sim["t"], sim["phi"])), 3),
                "surface": np.array(SURFACES)[sim["surf"][ni]],
            }))
            debug_rides.append({"ride_id": ride_id, "label": label, "t_sim": sim["t"], "v": sim["v"],
                                "a": sim["a"], "phi": sim["phi"], "phid": sim["phid"], "phidd": sim["phidd"],
                                "rig_acc": sim["rig_acc"], "rpm": sim["rpm"], "gear": sim["gear"],
                                "az_phys": sim["acc"][2].copy(), "t_out": to, "az_out": sens["data"][2]})
        r = meta_rows[-1]
        print(f"  {ride_id} {label:10s} r={rp['rider_intensity']:.2f} mount={rp['mount_type']:9s} "
              f"maxlean={r['real_max_lean_deg']:4.1f}deg maxbrake={r['real_max_brake_g']:.2f}g "
              f"maxrpm={r['real_max_rpm']:5.0f} v_mean={r['real_mean_speed_kmh']:4.1f}km/h "
              f"fullscale={100 * r['out_fullscale_share']:.3f}%")

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(os.path.join(cfg["out_dir"], "ride_metadata.csv"), index=False)
    pd.concat(gt_list).to_csv(os.path.join(cfg["out_dir"], "ground_truth.csv"), index=False)
    wf = pd.concat(wf_list, ignore_index=True)

    # ---------------- validation ----------------
    print("\n[validation] Realised per-class ranges (truth):")
    print(meta.groupby("label")[["rider_intensity", "real_max_lean_deg", "real_max_brake_g",
                                 "real_max_rpm", "real_mean_speed_kmh"]].agg(["min", "max"]).round(2).to_string())

    dbg_spec = next(d for d in debug_rides if d["label"] == cfg["validation"]["spectrogram_class"])
    plot_spectrogram(dbg_spec, cfg, os.path.join(cfg["out_dir"], "validation_spectrogram.png"))
    moving = dbg_spec["v"] > 0.3
    f_crank = dbg_spec["rpm"][moving] / 60
    print(f"\n[validation] Spectrogram saved for {dbg_spec['ride_id']}. Crank 1x above fs_out/2.5 "
          f"({cfg['fs_out'] / 2.5:.0f} Hz) on {100 * np.mean(f_crank > cfg['fs_out'] / 2.5):.0f}% of moving "
          f"samples -> aliased into the logged band (not filtered out).")

    plot_histograms(wf, cfg, os.path.join(cfg["out_dir"], "validation_histograms.png"))
    steady_lean_check(debug_rides, cfg, os.path.join(cfg["out_dir"], "validation_steady_lean.png"))

    print("\n[validation] Clipping:")
    print(f"  ADC saturation at {cfg['fs_sim']:.0f} Hz (pre-DLPF): {100 * sat_adc / sat_adc_tot:.4f}% of samples")
    print(f"  Logged samples at full scale:            {100 * sat_out / sat_out_tot:.4f}% of samples")
    print("  By class (logged, full-scale share):")
    print((meta.groupby("label")["out_fullscale_share"].mean() * 100).round(4).to_string())

    n_rows = int(meta["n_samples"].sum())
    print(f"\nWrote {n_rows:,} rows to {p_imu}")
    print(f"Done in {time.time() - t_start:.0f} s. Outputs in ./{cfg['out_dir']}/")


if __name__ == "__main__":
    main()
