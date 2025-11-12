#!/usr/bin/env python3

from typing import Tuple
import numpy as np

class PIDController():
    def __init__(self):

        # We will initialize some variables that might be useful
        self.prev_e_heading = 0.0
        self.prev_e_offset = 0.0
        self.prev_int_heading = 0.0
        self.prev_int_offset = 0.0

        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0


    def HeadingControl(self,
                       v_ref: float,
                       theta_ref: float,
                       theta_curr: float,
                       delta_t: float
    ) -> Tuple[float, float]:
        """
        PID performing heading control.
        Args:
            v_ref:      reference velocity.
            theta_ref:  reference heading pose.
            theta_curr: the current estimated heading.
            delta_t:    time interval since last call.
        Returns:
            v:          linear velocity of the Duckiebot
            omega:      angular velocity of the Duckiebot
        """
        
         # 1) Heading error (wrap to [-pi, pi] so we turn the shortest way)
        e = theta_ref - theta_curr
        e = (e + np.pi) % (2.0 * np.pi) - np.pi

        # Guard tiny/zero dt for numerical stability
        dt = max(delta_t, 1e-6)

        # 2) Integral term: accumulate error over time
        self.prev_int_heading += e * dt

        # 3) Derivative term: rate of change of error (backward difference)
        de_dt = (e - self.prev_e_heading) / dt

        # 4) PID combination -> angular velocity command
        omega = self.kp * e + self.ki * self.prev_int_heading + self.kd * de_dt

        # 5) Update memory for next call
        self.prev_e_heading = e

        # Linear velocity is passed through from UI
        v = v_ref
        return v, omega


    def OffsetControl(self,
                      v_ref: float,
                      y_ref: float,
                      y_curr: float,
                      delta_t: float
                      ) -> Tuple[float, float]:
        """
        PID performing lateral offset control.
        Args:
            v_ref:      linear Duckiebot speed.
            y_ref:      reference heading pose.
            y_curr:     the current estimated "y" coordinate (offset)
            delta_t:    time interval since last call.
        Returns:
            v:          linear velocity of the Duckiebot
            omega:      angular velocity of the Duckiebot
        """
        
        # 1) Error: how far we are from desired lateral position
        e = y_ref - y_curr

        # 2) Guard tiny dt for numerical stability
        dt = max(delta_t, 1e-6)

        # 3) Integral: accumulate error over time
        self.prev_int_offset += e * dt

        # 4) Derivative: how fast the error is changing
        de_dt = (e - self.prev_e_offset) / dt

        # 5) PID: turn rate to reduce lateral error
        omega = self.kp * e + self.ki * self.prev_int_offset + self.kd * de_dt

        # 6) Bookkeeping for next call
        self.prev_e_offset = e

        v = v_ref
        return v, omega

    def SetGains(self, kp: float, ki: float, kd: float) -> None:
        # Set the PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd