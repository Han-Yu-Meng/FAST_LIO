/*******************************************************************************
 * zupt.hpp - Zero-Velocity Update (ZUPT) for FAST-LIO2
 *
 * Adds IMU zero-velocity constraints to the FAST-LIO iterated ESKF:
 *   1. velocity ZUPT      : pseudo-measurement v = 0 when stationary
 *   2. angular-rate ZUPT  : gyro measurement observes the gyro bias bg
 *   3. gravity constraint : stationary accelerometer observes attitude
 *                           (roll/pitch) and accel bias ba
 *
 * Zero-velocity detection implements the four classic detectors from
 *   I. Skog, P. Handel, J.-O. Nilsson, "Zero-Velocity Detection - An
 *   Algorithm Evaluation", IEEE TBME 2010, and the OpenShoe MATLAB
 *   implementation (GLRT.m / MV.m / MAG.m / ARE.m):
 *     GLRT (SHOE): T = (1/W) * sum [ |w|^2/sig_g^2 + |a - g*a_bar/|a_bar||^2/sig_a^2 ]
 *     MV         : T = (1/W) * sum |a - a_bar|^2
 *     MAG        : T = (1/W) * sum (|a| - g)^2
 *     ARE        : T = (1/W) * sum |w|^2
 *
 * The update runs as a plain sequential error-state Kalman update on the
 * full 23-dim tangent state, right after IMU propagation and before the
 * iterated LiDAR update. (The LiDAR measurement Jacobian in this fork of
 * esekfom is hard-coded to 12 columns - pos/rot/extrinsics only - so ZUPT
 * rows which observe vel/bg cannot be appended there. A sequential update
 * is mathematically equivalent to a joint update in the EKF.)
 *
 * Error-state tangent layout (state_ikfom):
 *   [0-2] pos, [3-5] rot, [6-8] offset_R_L_I, [9-11] offset_T_L_I,
 *   [12-14] vel, [15-17] bg, [18-20] ba, [21-22] grav (S2)
 *
 * Copyright (c) 2026. Added to mid360_localization / FAST_LIO.
 ******************************************************************************/
#ifndef ZUPT_HPP
#define ZUPT_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <Eigen/Eigen>
#include "common_lib.h"
#include "use-ikfom.hpp"

namespace zupt {

// error-state tangent indices of state_ikfom
constexpr int IDX_POS = 0;
constexpr int IDX_ROT = 3;
constexpr int IDX_VEL = 12;
constexpr int IDX_BG  = 15;
constexpr int IDX_BA  = 18;
constexpr int IDX_GRAV = 21;
constexpr int STATE_DIM = 23;

enum DetectorType {
    DET_GLRT = 0,   // SHOE detector (generalized likelihood ratio test)
    DET_MV   = 1,   // acceleration moving variance
    DET_MAG  = 2,   // acceleration magnitude
    DET_ARE  = 3    // angular rate energy
};

enum UpdateMode {
    MODE_NONE     = 0,  // detect only, never update
    MODE_COV_ONLY = 1,  // damp covariance only (no state injection)
    MODE_FULL     = 2   // full Kalman update (state + covariance)
};

struct Params {
    bool   enable            = false;
    double window_sec        = 0.15;   // detector window length [s]
    int    detector_type     = DET_GLRT;
    double sigma_acc         = 0.15;   // accel white-noise std [m/s^2], GLRT weight
    double sigma_gyr         = 0.02;   // gyro white-noise std [rad/s], GLRT weight
    double gamma             = 10.0;   // detector threshold (stat <= gamma -> stationary)
    double max_gyro_mean     = 0.06;   // hard gate: |mean gyro| [rad/s], 0 = disabled
    double max_acc_dev       = 0.5;    // hard gate: ||a|-g| [m/s^2], 0 = disabled
    int    enter_cnt         = 2;      // consecutive hits before entering stationary
    int    exit_cnt          = 1;      // consecutive misses before leaving stationary
    bool   vel_zupt          = true;   // velocity ZUPT rows
    bool   gyro_zupt         = true;   // angular-rate ZUPT rows (observes bg)
    bool   gravity_constraint = true;  // gravity/attitude rows (observes rot + ba)
    double sigma_vel         = 0.03;   // velocity ZUPT noise std [m/s]
    double sigma_bg          = 0.02;   // angular-rate ZUPT noise std [rad/s]
    double sigma_grav        = 0.10;   // gravity constraint noise std [m/s^2]
    int    update_mode       = MODE_FULL;
};

struct Status {
    bool   stationary = false;
    double stat       = 0.0;   // detector test statistic
    double acc_dev    = 0.0;   // | |mean acc| - g |
    double gyro_norm  = 0.0;   // |mean gyro|
    double mean_acc_norm = 0.0;
    int    win_size   = 0;
};

inline Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m <<    0.0, -v(2),  v(1),
         v(2),    0.0, -v(0),
        -v(1),  v(0),    0.0;
    return m;
}

class Detector
{
public:
    void set_params(const Params &p) { params_ = p; reset(); }
    const Params &params() const { return params_; }

    void reset()
    {
        buf_.clear();
        stationary_ = false;
        enter_cnt_ = 0;
        exit_cnt_ = 0;
        status_ = Status();
    }

    // acc: normalized acceleration [m/s^2] (specific force, ~9.81 magnitude)
    // gyr: raw angular velocity [rad/s]
    void push(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, double t)
    {
        Sample s;
        s.t = t;
        s.acc = acc;
        s.gyr = gyr;
        buf_.push_back(s);

        while (!buf_.empty() && (buf_.back().t - buf_.front().t) > params_.window_sec)
        {
            buf_.pop_front();
        }

        evaluate();
    }

    bool stationary() const { return stationary_; }
    const Status &status() const { return status_; }

    Eigen::Vector3d mean_acc() const
    {
        Eigen::Vector3d m = Eigen::Vector3d::Zero();
        if (buf_.empty()) return m;
        for (const auto &s : buf_) m += s.acc;
        return m / static_cast<double>(buf_.size());
    }

    Eigen::Vector3d mean_gyr() const
    {
        Eigen::Vector3d m = Eigen::Vector3d::Zero();
        if (buf_.empty()) return m;
        for (const auto &s : buf_) m += s.gyr;
        return m / static_cast<double>(buf_.size());
    }

private:
    struct Sample
    {
        double t;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyr;
    };

    void evaluate()
    {
        status_ = Status();
        status_.win_size = static_cast<int>(buf_.size());

        // need a few samples before deciding
        if (buf_.size() < 5)
        {
            return;
        }

        const double W = static_cast<double>(buf_.size());
        const Eigen::Vector3d a_bar = mean_acc();
        const Eigen::Vector3d g_bar = mean_gyr();
        const double g_ref = G_m_s2;

        status_.mean_acc_norm = a_bar.norm();
        status_.acc_dev = std::fabs(a_bar.norm() - g_ref);
        status_.gyro_norm = g_bar.norm();

        double stat = 0.0;
        switch (params_.detector_type)
        {
        case DET_MV:
        {
            for (const auto &s : buf_) stat += (s.acc - a_bar).squaredNorm();
            stat /= W;
            break;
        }
        case DET_MAG:
        {
            for (const auto &s : buf_)
            {
                const double d = s.acc.norm() - g_ref;
                stat += d * d;
            }
            stat /= W;
            break;
        }
        case DET_ARE:
        {
            for (const auto &s : buf_) stat += s.gyr.squaredNorm();
            stat /= W;
            break;
        }
        case DET_GLRT:
        default:
        {
            // SHOE / GLRT (OpenShoe GLRT.m):
            //   T = (1/W) sum [ |w|^2/sig_g^2 + |a - g * a_bar/|a_bar||^2 / sig_a^2 ]
            const double sa2 = std::max(1e-8, params_.sigma_acc * params_.sigma_acc);
            const double sg2 = std::max(1e-8, params_.sigma_gyr * params_.sigma_gyr);
            Eigen::Vector3d dir = a_bar;
            if (dir.norm() > 1e-6) dir.normalize();
            else dir = Eigen::Vector3d::UnitZ();
            for (const auto &s : buf_)
            {
                const Eigen::Vector3d tmp = s.acc - g_ref * dir;
                stat += s.gyr.squaredNorm() / sg2 + tmp.squaredNorm() / sa2;
            }
            stat /= W;
            break;
        }
        }
        status_.stat = stat;

        // primary decision: statistic threshold
        bool hit = (stat <= params_.gamma);

        // optional hard gates (0 disables the gate)
        if (params_.max_gyro_mean > 0.0 && status_.gyro_norm > params_.max_gyro_mean) hit = false;
        if (params_.max_acc_dev > 0.0 && status_.acc_dev > params_.max_acc_dev) hit = false;

        // hysteresis (Schmitt trigger)
        if (hit)
        {
            enter_cnt_++;
            exit_cnt_ = 0;
            if (!stationary_ && enter_cnt_ >= std::max(1, params_.enter_cnt))
            {
                stationary_ = true;
            }
        }
        else
        {
            exit_cnt_++;
            enter_cnt_ = 0;
            if (stationary_ && exit_cnt_ >= std::max(1, params_.exit_cnt))
            {
                stationary_ = false;
            }
        }
        status_.stationary = stationary_;
    }

    Params params_;
    std::deque<Sample> buf_;
    bool stationary_ = false;
    int enter_cnt_ = 0;
    int exit_cnt_ = 0;
    Status status_;
};

/**
 * Sequential error-state Kalman update with the ZUPT pseudo-measurements.
 * kf must be esekfom::esekf<state_ikfom, 12, input_ikfom>.
 * Returns true if the filter state was changed; *why logs the outcome.
 */
template <typename ESEKF>
bool apply_update(ESEKF &kf, const Detector &det, const Params &p, std::string *why = nullptr)
{
    if (p.update_mode == MODE_NONE)
    {
        if (why) *why = "update disabled (MODE_NONE)";
        return false;
    }

    int m = 0;
    if (p.vel_zupt) m += 3;
    if (p.gyro_zupt) m += 3;
    if (p.gravity_constraint) m += 3;
    if (m == 0)
    {
        if (why) *why = "no ZUPT row enabled";
        return false;
    }

    state_ikfom x = kf.get_x();
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> P = kf.get_P();

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, STATE_DIM);
    Eigen::VectorXd r(m);
    Eigen::VectorXd var(m);
    int row = 0;

    // --- velocity ZUPT: z = 0 (world frame), h(x) = vel ---
    if (p.vel_zupt)
    {
        H.block<3, 3>(row, IDX_VEL) = Eigen::Matrix3d::Identity();
        r.segment<3>(row) << -x.vel[0], -x.vel[1], -x.vel[2];
        var.segment<3>(row).setConstant(p.sigma_vel * p.sigma_vel);
        row += 3;
    }

    // --- angular-rate ZUPT: stationary => gyro measures the bias: z = w_mean, h(x) = bg ---
    if (p.gyro_zupt)
    {
        H.block<3, 3>(row, IDX_BG) = Eigen::Matrix3d::Identity();
        const Eigen::Vector3d bg(x.bg[0], x.bg[1], x.bg[2]);
        r.segment<3>(row) = det.mean_gyr() - bg;
        var.segment<3>(row).setConstant(p.sigma_bg * p.sigma_bg);
        row += 3;
    }

    // --- gravity constraint: stationary => z = a_mean = ba - R^T * grav ---
    // residual  r = z - h(x) = a_mean - ba + R^T grav
    // dr/d(dtheta) = skew(R^T grav)   (right perturbation: R <- R * Exp(dtheta))
    // dr/d(db)     = -I  (ba block)
    if (p.gravity_constraint)
    {
        const Eigen::Vector3d grav(x.grav[0], x.grav[1], x.grav[2]);
        const Eigen::Vector3d g_body = x.rot.conjugate() * grav;   // R^T * grav
        const Eigen::Vector3d ba(x.ba[0], x.ba[1], x.ba[2]);
        H.block<3, 3>(row, IDX_ROT) = skew(g_body);
        H.block<3, 3>(row, IDX_BA) = -Eigen::Matrix3d::Identity();
        r.segment<3>(row) = det.mean_acc() - ba + g_body;
        var.segment<3>(row).setConstant(p.sigma_grav * p.sigma_grav);
        row += 3;
    }

    if (!r.allFinite() || !H.allFinite())
    {
        if (why) *why = "non-finite residual/jacobian, skipped";
        return false;
    }

    // Kalman gain with diagonal R:  S = H P H^T + R
    Eigen::MatrixXd S = H * P * H.transpose();
    S.diagonal().array() += var.array();
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success)
    {
        if (why) *why = "innovation covariance not solvable, skipped";
        return false;
    }
    const Eigen::MatrixXd K = P * H.transpose() * ldlt.solve(
        Eigen::MatrixXd::Identity(m, m));
    Eigen::Matrix<double, STATE_DIM, 1> dx = K * r;

    // safety clamps: a correct stationary detection must yield small corrections
    if (!dx.allFinite())
    {
        if (why) *why = "non-finite update, skipped";
        return false;
    }
    const double dtheta = dx.segment<3>(IDX_ROT).norm();
    const double dvel = dx.segment<3>(IDX_VEL).norm();
    if (dtheta > 0.087 || dvel > 0.5)   // 5 deg / 0.5 m/s
    {
        if (why)
        {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "update too large (dtheta=%.3f rad, dvel=%.3f m/s), skipped",
                          dtheta, dvel);
            *why = buf;
        }
        return false;
    }

    if (p.update_mode == MODE_FULL)
    {
        x.boxplus(dx);
        if (!x.vel.allFinite() || !x.rot.coeffs().allFinite())
        {
            if (why) *why = "state became non-finite, skipped";
            return false;
        }
        kf.change_x(x);
    }

    // Joseph-form covariance update (works for both FULL and COV_ONLY)
    const Eigen::Matrix<double, STATE_DIM, STATE_DIM> IKH =
        Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Identity() - K * H;
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> P_new =
        IKH * P * IKH.transpose() + K * var.asDiagonal() * K.transpose();
    // enforce symmetry
    P_new = 0.5 * (P_new + P_new.transpose());
    kf.change_P(P_new);

    if (why)
    {
        char buf[168];
        std::snprintf(buf, sizeof(buf),
                      "applied rows=%d stat=%.2f |v|=%.3f dtheta=%.4f dvel=%.4f mode=%d",
                      m, det.status().stat, x.vel.norm(), dtheta, dvel, p.update_mode);
        *why = buf;
    }
    return true;
}

} // namespace zupt

#endif // ZUPT_HPP
