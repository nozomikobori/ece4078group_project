import numpy as np
import math

class Robot:
    def __init__(self, wheels_width, wheels_scale, camera_matrix, camera_dist):
        # State is a vector of [x,y,theta]'
        self.state = np.zeros((3,1))
        
        # Wheel parameters
        self.wheels_width = wheels_width  # The distance between the left and right wheels
        self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s

        # Camera parameters
        self.camera_matrix = camera_matrix  # Matrix of the focal lengths and camera centre
        self.camera_dist = camera_dist  # Distortion coefficients
    
    def drive(self, drive_meas):
        # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity
        linear_velocity, angular_velocity = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        # Apply the velocities
        dt = drive_meas.dt
        dt_turn = drive_meas.dt/6.2
        # TODO: compute state (x,y,theta) from linear and angular velocity
        # ------------------------------------------
        # robot is turning

        #if angular_velocity != 0:
        #    R = linear_velocity / angular_velocity

            # change in theta
        #    current_theta = self.state[2] # store current theta	
        #    self.state[2] = self.state[2] + angular_velocity * dt # replace with new theta
        #    next_theta = self.state[2] # store next theta

            # next position (x, y)
        #    self.state[0] = self.state[0] + R * (-np.sin(current_theta) + np.sin(next_theta))
        #    self.state[1] = self.state[1] + R * (np.cos(current_theta) - np.cos(next_theta))

        #else:# robot is moving straight (angular_velocity == 0)
            # next position (x,y)
        #    self.state[0] = self.state[0] + np.cos(self.state[2]) * linear_velocity * dt
        #    self.state[1] = self.state[1] + np.sin(self.state[2]) * linear_velocity * dt
	
        #print(self.state)
        #return self.state
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        R = 0

        if math.isclose(angular_velocity, 0):
            next_x = x + np.cos(theta) * linear_velocity * dt
            next_y = y + np.sin(theta) * linear_velocity * dt
            next_theta = theta

        else:
            R = linear_velocity / angular_velocity
            next_theta = theta + angular_velocity * dt_turn
            next_x = x + R * (-np.sin(theta) * np.sin(next_theta))
            next_y = y + R * (np.cos(theta) - np.cos(next_theta))

        self.state[0] = next_x
        self.state[1] = next_y
        self.state[2] = next_theta
        #print(self.state)
        return self.state

    def measure(self, markers, idx_list):
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        robot_xy = self.state[0:2,:]

        measurements = []
        for idx in idx_list:
            marker = markers[:,idx:idx+1]
            marker_bff = Rot_theta.T @ (marker - robot_xy)
            measurements.append(marker_bff)

        # Stack the measurements in a 2xm structure.
        markers_bff = np.concatenate(measurements, axis=1)
        return markers_bff
    
    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.wheels_scale
        right_speed_m = right_speed * self.wheels_scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
        return linear_velocity, angular_velocity

    # Derivatives and Covariance
    # --------------------------

    def derivative_drive(self, drive_meas):
        # Compute the differential of drive w.r.t. the robot state
        DFx = np.zeros((3,3))
        DFx[0,0] = 1
        DFx[1,1] = 1
        DFx[2,2] = 1

        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        dt = drive_meas.dt
        th = self.state[2]
        if ang_vel == 0:
            DFx[0,2] = -np.sin(th) * lin_vel * dt
            DFx[1,2] = np.cos(th) * lin_vel * dt
        else:
            DFx[0,2] = lin_vel / ang_vel * (np.cos(th+dt*ang_vel) - np.cos(th))
            DFx[1,2] = -lin_vel / ang_vel * (-np.sin(th+dt*ang_vel) + np.sin(th))

        return DFx

    def derivative_measure(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        n = 2*len(idx_list)
        m = 3 + 2*markers.shape[1]

        DH = np.zeros((n,m))

        robot_xy = self.state[0:2,:]
        th = self.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.

            lmj_inertial = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)

            # robot xy DH
            DH[2*i:2*i+2,0:2] = - Rot_theta.T
            # robot theta DH
            DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
            # lm xy DH
            DH[2*i:2*i+2, 3+2*j:3+2*j+2] = Rot_theta.T

            # print(DH[i:i+2,:])

        return DH
    
    def covariance_drive(self, drive_meas):
        # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed
        Jac1 = np.array([[self.wheels_scale/2, self.wheels_scale/2],
                [-self.wheels_scale/self.wheels_width, self.wheels_scale/self.wheels_width]])
        
        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        th = self.state[2]
        dt = drive_meas.dt
        th2 = th + dt*ang_vel

        # Derivative of x,y,theta w.r.t. lin_vel, ang_vel
        Jac2 = np.zeros((3,2))
        if ang_vel == 0:
            Jac2[0,0] = np.cos(th)
            Jac2[1,0] = np.sin(th)
        else:
            Jac2[0,0] = 1/ang_vel * (np.sin(th2) - np.sin(th))
            Jac2[0,1] = -1/(ang_vel**2) * (np.sin(th2) - np.sin(th)) + \
                            lin_vel / ang_vel * (dt * np.cos(th2))

            Jac2[1,0] = -1/ang_vel * (np.cos(th2) - np.cos(th))
            Jac2[1,1] = 1/(ang_vel**2) * (np.cos(th2) - np.cos(th)) + \
                            -lin_vel / ang_vel * (-dt * np.sin(th2))
        Jac2[2,1] = dt

        # Derivative of x,y,theta w.r.t. left_speed, right_speed
        Jac = Jac2 @ Jac1

        # Compute covariance
        cov = np.diag((drive_meas.left_cov, drive_meas.right_cov))
        cov = Jac @ cov @ Jac.T
        
        return cov
