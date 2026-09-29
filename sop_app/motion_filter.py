"""Adaptive constant-velocity Kalman filter for measured workpiece corners.

Predictions are internal only: a missing image measurement never produces a
transform usable by the completion engine.
"""
import numpy as np


class AdaptiveMotionFilter:
    def __init__(self, corners, correction_ratio=.02):
        self.corners = np.asarray(corners, dtype=float)
        self.design = np.column_stack((self.corners, np.ones(len(self.corners))))
        self.correction_ratio = correction_ratio
        self.reset()

    def reset(self):
        self.state = None
        self.covariance = None
        self.timestamp = None
        self.size = None

    def update(self, matrix, timestamp, width, height, quality=1., verified=False):
        if matrix is None:
            # No stale velocity after an occlusion; reacquire without a waiting period.
            self.reset()
            return None
        measured = (self.corners @ matrix[:, :2].T + matrix[:, 2]) * [width, height]
        z = measured.ravel()
        n = len(z)
        diagonal = max(float(np.linalg.norm(np.ptp(measured, axis=0))), 1.)
        dt = timestamp - self.timestamp if self.timestamp is not None else 0.
        if self.state is None or self.size != (width, height) or not 0 < dt <= .5:
            self.state = np.concatenate((z, np.zeros(n)))
            self.covariance = np.diag([4.] * n + [diagonal ** 2] * n)
            self.timestamp, self.size = timestamp, (width, height)
            return matrix.copy()

        transition = np.eye(2 * n)
        transition[:n, n:] = np.eye(n) * dt
        predicted = transition @ self.state
        deviation = float(np.max(np.linalg.norm((z - predicted[:n]).reshape(-1, 2), axis=1)) / diagonal)
        if verified and deviation > self.correction_ratio:
            self.reset()
            return self.update(matrix, timestamp, width, height, quality, verified)

        # More acceleration uncertainty when motion departs from the prediction.
        acceleration = diagonal * (.15 + 3. * min(deviation / self.correction_ratio, 1.))
        gain_acceleration = np.vstack((np.eye(n) * dt ** 2 / 2, np.eye(n) * dt))
        process_noise = gain_acceleration @ gain_acceleration.T * acceleration ** 2
        prior = transition @ self.covariance @ transition.T + process_noise
        sigma = max(1., diagonal * .002) / np.sqrt(np.clip(quality, .15, 1.))
        noise = np.eye(n) * sigma ** 2
        gain = np.linalg.solve(prior[:n, :n] + noise, prior[:, :n].T).T
        self.state = predicted + gain @ (z - predicted[:n])
        # Joseph form preserves positive covariance under floating-point roundoff.
        residual = np.eye(2 * n)
        residual[:, :n] -= gain
        self.covariance = residual @ prior @ residual.T + gain @ noise @ gain.T
        self.timestamp = timestamp
        points = self.state[:n].reshape(-1, 2) / [width, height]
        return np.linalg.lstsq(self.design, points, rcond=None)[0].T
