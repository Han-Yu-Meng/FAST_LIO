#ifndef CT_SGAL_REFINER_HPP
#define CT_SGAL_REFINER_HPP

// CT-SGAL-LIO style causal trajectory post-processing.
//
// The discrete LIO publishes one posterior pose per IMU propagation; scan
// corrections reset the predictor at ~10 Hz, so the output stream alternates
// between IMU-propagated states and corrected states and shows short-horizon
// oscillation even when the robot is stationary.
//
// This layer treats the published poses as observations. Over a bounded window
// of recent poses it fits a cubic polynomial in the SE(3) local tangent chart,
// with Huber-robust IRLS, covariance weighting and curvature regularization,
// and evaluates the fit at a fixed lag so the query time lies INSIDE the
// observed interval (interpolation, never extrapolation). Only past/current
// measurements are used, so the output stays causal.
//
// Reference: arXiv:2611.16918, CT-SGAL-LIO: Causal Continuous-Time Refinement
// of Discrete LiDAR-Inertial Odometry.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <deque>

class CTSGALRefiner {
 public:
  struct Params {
    bool enabled = false;
    double window = 0.3;        // W: history kept BEFORE the query time (s)
    double lag = 0.2;           // ell: fixed output lag (s)
    double huber_delta = 0.05;  // Huber threshold in Mahalanobis units
    double lambda_p = 1.0;      // curvature weight, translation
    double lambda_R = 1.0;      // curvature weight, rotation
    double lambda_d = 1e-6;     // damping (keeps the normal system SPD)
    double sigma_p = 1.0;       // covariance floor scale, translation (m)
    double sigma_R = 1.0;       // covariance floor scale, rotation (rad)
    int irls_iters = 3;
  };

  void configure(const Params& p) {
    p_ = p;
    p_.window = std::max(p_.window, 0.05);
    p_.lag = std::max(p_.lag, 0.0);
    p_.irls_iters = std::max(p_.irls_iters, 1);
    p_.sigma_p = std::max(p_.sigma_p, 1e-6);
    p_.sigma_R = std::max(p_.sigma_R, 1e-6);
    p_.huber_delta = std::max(p_.huber_delta, 1e-6);
    // Keep enough history that the active window is always fully covered.
    memory_ = std::max(1.0, 2.0 * (p_.window + p_.lag));
    buffer_.clear();
  }

  bool enabled() const { return p_.enabled; }
  void reset() { buffer_.clear(); }

  // t must be a monotonic timestamp in seconds. Returns true when the pose
  // was refined; false means the caller should publish the raw pose.
  bool update(double t, const Eigen::Vector3d& p,
              const Eigen::Quaterniond& q_in,
              Eigen::Vector3d& out_p, Eigen::Quaterniond& out_q) {
    if (!p_.enabled || !std::isfinite(t) || !p.allFinite() ||
        !q_in.coeffs().allFinite() || q_in.norm() < 1e-9) {
      return false;
    }
    Eigen::Quaterniond q = q_in.normalized();
    if (!buffer_.empty()) {
      const Sample& last = buffer_.back();
      if (t <= last.t) {
        if (t == last.t) {
          buffer_.back().p = p;
          buffer_.back().q = q;
        }
        return false;  // out-of-order: pass through
      }
      // A large jump means a discontinuity (re-initialization / mode switch).
      // Do not fit a polynomial across it.
      if ((p - last.p).norm() > 1.0) {
        buffer_.clear();
        return false;
      }
    }
    buffer_.push_back({t, p, q});
    while (buffer_.size() > 2 && t - buffer_.front().t > memory_) {
      buffer_.pop_front();
    }

    const double t_star = t - p_.lag;
    // Startup: wait until the window [t_star - W, t] is fully covered.
    if (buffer_.front().t > t_star - p_.window + 0.02) return false;
    if (t_star <= buffer_.front().t + 1e-6) return false;

    const double c = (t_star - p_.window + t) * 0.5;
    const double h = (p_.window + p_.lag) * 0.5;
    if (h < 1e-9) return false;

    // Local tangent chart around the first sample in the active window.
    const Sample& ref = buffer_.front();
    int n = 0;
    for (const Sample& s : buffer_) {
      if (s.t < t_star - p_.window - 1e-9 || s.t > t + 1e-9) continue;
      ++n;
    }
    if (n < 6) return false;

    Eigen::Matrix<double, 6, Eigen::Dynamic> Z(6, n);
    Eigen::Matrix<double, 4, Eigen::Dynamic> B(4, n);
    int i = 0;
    for (const Sample& s : buffer_) {
      if (s.t < t_star - p_.window - 1e-9 || s.t > t + 1e-9) continue;
      Z.col(i).head<3>() = s.p - ref.p;
      Z.col(i).tail<3>() = so3_log(ref.q.inverse() * s.q);
      const double u = (s.t - c) / h;
      B.col(i) << 1.0, u, u * u, u * u * u;
      ++i;
    }

    // Curvature regularization evaluated at q in {-1, -0.5, 0, 0.5, 1}.
    // b''(u) = [0, 0, 2, 6u]^T; only the quadratic/cubic coefficients are
    // penalized, which is what suppresses high-frequency oscillation.
    Eigen::Matrix<double, 24, 24> H_reg =
        Eigen::Matrix<double, 24, 24>::Zero();
    const double qpoints[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    for (double qv : qpoints) {
      Eigen::Matrix<double, 4, 1> bdd;
      bdd << 0.0, 0.0, 2.0, 6.0 * qv;
      const Eigen::Matrix<double, 4, 4> m = bdd * bdd.transpose();
      for (int cidx = 0; cidx < 6; ++cidx) {
        const double w = (cidx < 3) ? p_.lambda_p : p_.lambda_R;
        H_reg.block<4, 4>(4 * cidx, 4 * cidx).noalias() += w * m;
      }
    }

    const double wp = 1.0 / (p_.sigma_p * p_.sigma_p);
    const double wr = 1.0 / (p_.sigma_R * p_.sigma_R);
    Eigen::Matrix<double, 24, 1> a = Eigen::Matrix<double, 24, 1>::Zero();

    for (int iter = 0; iter < p_.irls_iters; ++iter) {
      Eigen::Matrix<double, 24, 24> H = H_reg;
      H.diagonal().array() += p_.lambda_d;
      Eigen::Matrix<double, 24, 1> g =
          Eigen::Matrix<double, 24, 1>::Zero();
      for (int k = 0; k < n; ++k) {
        // a = vec(A^T): component c of A*b is a.segment<4>(6c).dot(b)
        Eigen::Matrix<double, 6, 1> fit;
        for (int cidx = 0; cidx < 6; ++cidx) {
          fit(cidx) = a.segment<4>(4 * cidx).dot(B.col(k));
        }
        const Eigen::Matrix<double, 6, 1> r = Z.col(k) - fit;
        const double e2 = wp * r.head<3>().squaredNorm() +
                          wr * r.tail<3>().squaredNorm();
        const double e = std::sqrt(std::max(e2, 0.0));
        const double w = (e > p_.huber_delta) ? p_.huber_delta / e : 1.0;
        for (int cidx = 0; cidx < 6; ++cidx) {
          const double sw = w * ((cidx < 3) ? wp : wr);
          H.block<4, 4>(4 * cidx, 4 * cidx).noalias() +=
              sw * B.col(k) * B.col(k).transpose();
          g.segment<4>(4 * cidx).noalias() += sw * Z.col(k)(cidx) * B.col(k);
        }
      }
      Eigen::LDLT<Eigen::Matrix<double, 24, 24>> ldlt(H);
      if (ldlt.info() != Eigen::Success) return false;
      a = ldlt.solve(g);
      if (!a.allFinite()) return false;
    }

    const double u_star = (p_.window - p_.lag) / (p_.window + p_.lag);
    Eigen::Matrix<double, 4, 1> b_star;
    b_star << 1.0, u_star, u_star * u_star, u_star * u_star * u_star;
    Eigen::Matrix<double, 6, 1> fit_star;
    for (int cidx = 0; cidx < 6; ++cidx) {
      fit_star(cidx) = a.segment<4>(4 * cidx).dot(b_star);
    }
    if (!fit_star.allFinite()) return false;

    out_p = ref.p + fit_star.head<3>();
    out_q = ref.q * so3_exp(fit_star.tail<3>());
    out_q.normalize();
    return true;
  }

 private:
  struct Sample {
    double t;
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
  };

  static Eigen::Vector3d so3_log(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond qn = q.normalized();
    if (qn.w() < 0.0) qn.coeffs() = -qn.coeffs();  // shortest arc
    Eigen::Vector3d v(qn.x(), qn.y(), qn.z());
    const double s = v.norm();
    if (s < 1e-9) return 2.0 * v;
    const double angle = 2.0 * std::atan2(s, qn.w());
    return (angle / s) * v;
  }

  static Eigen::Quaterniond so3_exp(const Eigen::Vector3d& w) {
    const double theta = w.norm();
    if (theta < 1e-9) {
      Eigen::Quaterniond q;
      q.coeffs() << w.x(), w.y(), w.z(), 1.0;
      return q.normalized();
    }
    return Eigen::Quaterniond(Eigen::AngleAxisd(theta, w / theta));
  }

  Params p_;
  double memory_ = 1.0;
  std::deque<Sample> buffer_;
};

#endif  // CT_SGAL_REFINER_HPP
