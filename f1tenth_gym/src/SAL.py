import numpy as np
import cv2
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import cv2
import numpy as np
import cvxpy as cp
from scipy.interpolate import CubicSpline

import os
import random
from collections import deque
import time
from typing import List, Tuple, Union
import pyglet
from pyglet.gl import GL_LINES

###########################################
##   LIDAR TO BITMAP, COURTESY OF ALY    ##
###########################################

def _lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str='CCW',          
        starting_angle: float=-np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool=True,  
        output_image_dims: tuple[int]=(256, 256),
        target_beam_count: int=600,
        fov: float=2*np.pi,
        draw_mode: str="FILL"
    ) -> np.ndarray:  
    """
    Creates a bitmap image based on lidar input.
    Assumes rays are equally spaced within the FOV.
    Internal only DO NOT USE. Use lidar_to_bitmap instead.

    Args:
        scan (list[float]): A list of lidar measurements.

        winding_dir (str): The direction that the rays wind. Must either be CW or CCW in a right handed coord system.
        
        starting_angle (float): The offset from the pos-x axis that points "up" or "forward.
        
        max_scan_radius (float | None): The maximum range expected from the scans. Used to scale the value into the image if given.
        
        scaling_factor (float | None): Scaling factor for the ranges from the scan.
        
        bg_color (str): Either \'white\' or \'black\'. The accent color is always the opposite.
        
        draw_center (bool): Should this function draw a square in the center of the bitmap?
        
        output_image_dims (tuple[int]): The dimensions of the output image. Should be square but not enforced.
        
        target_beam_count (int): The target number of beams (rays) cast into the environment.
        
        fov (float): The field of view of the car measured in radians. Note: the output will look pinched if this is setup incorrectly.

        draw_mode (str): How should the final image be drawn. Can be \'RAYS\' (view the ray casts - keep beam count low), \'POLYGON\' (draws the outline of the rays), or \'FILL\' (filled in driveable and nondriveable boundary). 

    Returns:
        np.ndarray: A single-channel, grayscale image with a birds-eye-view of the lidar scan.
    """
    assert winding_dir in ['CW', 'CCW'], "winding_dir must be either clockwise or counterclockwise"
    assert bg_color in ['black', 'white']
    assert draw_mode in ['RAYS', 'POLYGON', 'FILL']
    assert len(output_image_dims) == 2
    assert all([x > 0 for x in output_image_dims]), "output_image_dims must be at least 1x1"
    assert 0 < target_beam_count < len(scan)
    assert 0 < fov <= 2*np.pi, "FOV must be between 0 and 2pi"

    if max_scan_radius is not None:
        scaling_factor = min(output_image_dims) / max_scan_radius
    elif scaling_factor is None:
        raise ValueError("Must provide either max_scan_radius or scaling_factor")

    BG_COLOR, DRAW_COLOR = (0, 255) if bg_color == 'black' else (255, 0)

    # Initialize a blank grayscale image for the output
    image = np.ones((output_image_dims[0], output_image_dims[1]), dtype=np.uint8) * BG_COLOR

    # Direction factor
    dir = 1 if winding_dir == 'CCW' else -1

    # Select target beam count using linspace for accurate downsampling
    indices = np.linspace(0, len(scan) - 1, target_beam_count, dtype=int)
    data = np.array(scan)[indices]

    # Precompute angles
    angles = starting_angle + dir * fov * np.linspace(0, 1, target_beam_count)

    # Compute (x, y) positions
    center = np.array([output_image_dims[0] // 2, output_image_dims[1] // 2])
    points = np.column_stack((
        np.rint(center[0] + scaling_factor * data * np.cos(angles)).astype(int),
        np.rint(center[1] + scaling_factor * data * np.sin(angles)).astype(int)
    ))

    # draw according to the correct mode
    if draw_mode == 'FILL':
        cv2.fillPoly(image, [points], DRAW_COLOR)
    elif draw_mode == 'POLYGON':
        cv2.polylines(image, [points], isClosed=True, color=DRAW_COLOR, thickness=1)
    elif draw_mode == 'RAYS':
        for p in points:
            cv2.line(image, tuple(center), tuple(p), color=DRAW_COLOR, thickness=1)
            cv2.rectangle(image, tuple(p - 2), tuple(p + 2), color=DRAW_COLOR, thickness=-1)

    # Draw center point if needed
    if draw_center:
        cv2.rectangle(image, tuple(center - 2), tuple(center + 2), color=BG_COLOR if draw_mode == "FILL" else DRAW_COLOR, thickness=-1)
    
    return image

def lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str='CCW',          
        starting_angle: float=-np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool=True,  
        output_image_dims: tuple[int]=(256, 256),
        target_beam_count: int=600,
        fov: float=2*np.pi,
        draw_mode: str="POLYGON",
        channels: int=1
    ) -> np.ndarray:  
    """
    Creates a bitmap image based on lidar input.
    Assumes rays are equally spaced within the FOV.

    Args:
        scan (list[float]): A list of lidar measurements.

        winding_dir (str): The direction that the rays wind. Must either be CW or CCW in a right handed coord system.
        
        starting_angle (float): The offset from the pos-x axis that points "up" or "forward.
        
        max_scan_radius (float | None): The maximum range expected from the scans. Used to scale the value into the image if given.
        
        scaling_factor (float | None): Scaling factor for the ranges from the scan.
        
        bg_color (str): Either \'white\' or \'black\'. The accent color is always the opposite.
        
        draw_center (bool): Should this function draw a square in the center of the bitmap?
        
        output_image_dims (tuple[int]): The dimensions of the output image. Should be square but not enforced.
        
        target_beam_count (int): The target number of beams (rays) cast into the environment.

        fov (float): The field of view of the car measured in radians. Note: the output will look pinched if this is setup incorrectly.

        draw_mode (str): How should the final image be drawn. Can be \'RAYS\' (view the ray casts - keep beam count low), \'POLYGON\' (draws the outline of the rays), or \'FILL\' (filled in driveable and nondriveable boundary). 

        channels (int): The number of channels in the output. Must be 1 (grayscale), 3 (RGB), or 4 (RGBA). Default is 1.
    Returns:
        np.ndarray: An image with a birds-eye-view of the lidar scan.
    """
    assert channels in [1, 3, 4], "channels must 1, 3, or 4"

    grayscale_img = _lidar_to_bitmap(scan, winding_dir, starting_angle, max_scan_radius, scaling_factor, bg_color, draw_center, output_image_dims, target_beam_count, fov, draw_mode)
    if channels == 1:
        return grayscale_img  # Shape: (256, 256)
    elif channels == 3:
        return np.stack([grayscale_img] * 3, axis=-1)  # Shape: (256, 256, 3)
    elif channels == 4:
        alpha_channel = np.full_like(grayscale_img, 255)  # Alpha is fully opaque (255)
        return np.stack([grayscale_img, grayscale_img, grayscale_img, alpha_channel], axis=-1)  # Shape: (256, 256, 4)
    else:
        raise ValueError("Invalid number of channels. Supported: 1 (grayscale), 3 (RGB), 4 (RGBA)")
##############################
##        OPIUM MODEL       ##
##############################

class Actor(nn.Module):
    def __init__(self, action_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*28*28, 512)
        self.fc_mean = nn.Linear(512, action_dim)
        self.fc_log_std = nn.Linear(512, action_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return y_t, log_prob

class Critic(nn.Module):
    def __init__(self, action_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*28*28 + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x, action):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=3) # Output: (batch_size, 32, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: (batch_size, 64, 32, 32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: (batch_size, 128, 16, 16)
        
        self.flatten = nn.Flatten() # Flatten the output into a 1D vector.

        # Define a fully connected layer to output probability.
        self.fc1 = nn.Linear(128 * 16 * 16+action_dim, 256) #Evaluates value of state and action pair 
        self.fc2 = nn.Linear(256,256)
        self.q = nn.Linear(256,1)

        self.optimizer = optim.Adam(self.parameters(),lr=beta)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network, estimating Q-value.
        
        :param x: A (batch, 1, 256, 256) input tensor (the bitmap observation).
        :param action: A (batch, action_dim) tensor of actions.
        :return: A (batch, 1) tensor representing Q-values for state-action pairs.
        """
        
class Sample:
    """
    Wraps a transition for prioritized replay.
    """
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.weight = 1.0
        self.cumulative_weight = 1.0

    def is_interesting(self):
        return self.done or self.reward != 0

    def __lt__(self, other):
        return self.cumulative_weight < other.cumulative_weight
    
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s,a,r,ns,d):
        self.buffer.append((s,a,r,ns,d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = map(np.stack, zip(*batch))
        return s,a,r,ns,d
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, device, action_dim=32, gamma=0.99, tau=0.005, alpha=0.2,
                 actor_lr=3e-4, critic_lr=3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = Actor(action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic1 = Critic(action_dim).to(device)
        self.critic2 = Critic(action_dim).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.critic1_target = Critic(action_dim).to(device)
        self.critic2_target = Critic(action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
    
    def select_action(self, state, evaluate=False):
        st = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor.forward(st)
                act = torch.tanh(mean)
                return act.cpu().numpy().flatten()
        else:
            with torch.no_grad():
                act, _ = self.actor.sample(st)
                return act.cpu().numpy().flatten()
    
    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return 0, 0, 0
        s,a,r,ns,d = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        if len(s.shape)==3: s = s.unsqueeze(1)
        ns = torch.FloatTensor(ns).to(self.device)
        if len(ns.shape)==3: ns = ns.unsqueeze(1)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(np.float32(d)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(ns)
            tq1 = self.critic1_target(ns, next_a)
            tq2 = self.critic2_target(ns, next_a)
            tq = torch.min(tq1, tq2) - self.alpha * next_logp
            tv = r + (1-d)*self.gamma*tq
        
        cq1 = self.critic1(s, a)
        cq2 = self.critic2(s, a)
        c1_loss = F.mse_loss(cq1, tv)
        c2_loss = F.mse_loss(cq2, tv)
        
        self.critic1_optimizer.zero_grad()
        c1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        c2_loss.backward()
        self.critic2_optimizer.step()
        
        new_a, logp = self.actor.sample(s)
        q1n = self.critic1(s, new_a)
        q2n = self.critic2(s, new_a)
        qn = torch.min(q1n, q2n)
        a_loss = (self.alpha * logp - qn).mean()
        
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        
        for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        
        return a_loss.item(), c1_loss.item(), c2_loss.item()

#######################################
## PATH CLAMP & MPC HELPER FUNCTIONS ##
#######################################

def clamp_angle_diff(prev_angle: float, desired_angle: float, max_diff_deg: float=10.0) -> float:
    """Ensures desired_angle doesn't deviate from prev_angle by more than max_diff_deg (in degrees)."""
    delta = desired_angle - prev_angle
    delta = (delta + np.pi) % (2*np.pi) - np.pi
    max_diff_rad = np.deg2rad(max_diff_deg)
    if delta > max_diff_rad:
        delta = max_diff_rad
    elif delta < -max_diff_rad:
        delta = -max_diff_rad
    return prev_angle + delta

def compute_vectors_with_angle_clamp(raw_action: np.ndarray) -> np.ndarray:
    """
    Interpret raw_action (shape=(32,)) as 16 increments in [-1,1]^2,
    but clamp each vector's angle so it cannot deviate more than ±10° from the previous.
    The first vector is forced to be (1,0) in local heading space (straight forward).
    Returns an array of shape (16,2) in [-1,1], but with angle constraints.
    """
    increments = raw_action.reshape(16, 2)
    # Force the first vector to be purely forward => angle=0, magnitude=1
    increments[0] = np.array([1.0, 0.0], dtype=np.float32)

    clamped = np.zeros_like(increments)
    clamped[0] = increments[0]

    prev_angle = 0.0  # first vector angle

    for i in range(1, 16):
        dx, dy = increments[i]
        mag = np.sqrt(dx*dx + dy*dy) + 1e-8
        angle = np.arctan2(dy, dx)
        # Clamp angle relative to the previous angle
        angle = clamp_angle_diff(prev_angle, angle, max_diff_deg=10.0)
        # Keep the same magnitude
        new_dx = mag * np.cos(angle)
        new_dy = mag * np.sin(angle)
        clamped[i] = np.array([new_dx, new_dy], dtype=np.float32)
        prev_angle = angle

    return clamped

def rotate_local_to_global(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    """Rotate the local vector (dx, dy) by 'heading' radians into the global frame."""
    global_dx = dx * np.cos(heading) - dy * np.sin(heading)
    global_dy = dx * np.sin(heading) + dy * np.cos(heading)
    return global_dx, global_dy

############################
##     MPC CONTROLLER     ##
############################

def MPC_controller(path: np.ndarray, desiredVelocity: float, timeStep: float, totalSteps: float, horizonLength: float, stateCost: np.ndarray, inputCost: np.ndarray, terminalCost: np.ndarray) -> np.ndarray:
    """
    Computes control input(x and y acceleration) at each timeStep along path
    
    :param path: Array of vectors of projected path car should follow. Should start at current x, y coordinates of car
    :param desiredVelocity: Desired constant speed along the track
    :param timeStep: Time step (seconds), how often our simulation will update
    :param totalSteps: Total simulation steps, how long the simulation will run
    :param horizonLength: MPC horizon (number of steps), how far ahead the controller plans
    :param stateCost: MPC cost weights, penalizes deviations from the reference trajectory (the vectorized path)
    :param inputCost: MPC cost weights, penalizes deviations from the reference trajectory (the vectorized path)
    :param terminalCost: MPC cost weights, penalizes deviations from the reference trajectory (the vectorized path)

    :return: An array of [x_acceleration, y_acceleration] for the converter
    """
    # Calculates the distance between each pair of points along path, and adds them all to one cumulative arc length
    dists = [0]
    for i in range(1, len(path)):
        dists.append(dists[-1] + np.linalg.norm(path[i] - path[i-1]))
    dists = np.array(dists)

    '''
    Cublic Splines are cubic functions used to interpolate between points, maintaining smoothness
    between the points. This is useful for creating the paths for this project.
    '''
    # Uses the cubic spline function to interpolate between the points on the track
    cs_x = CubicSpline(dists, path[:, 0])
    cs_y = CubicSpline(dists, path[:, 1])

    # 2D double-integrator model:
    # State: [x, y, vx, vy]; Control: [ax, ay]
    A = np.array([[1, 0, timeStep, 0],
                [0, 1, 0, timeStep],
                [0, 0, 1,  0],
                [0, 0, 0,  1]])
    B = np.array([[0.5*timeStep**2, 0],
                [0, 0.5*timeStep**2],
                [timeStep, 0],
                [0, timeStep]])

    # Precompute the reference trajectory along the drawn track.
    # For each simulation time (plus horizon), compute the reference state.
    # We use s = v_des * t (i.e., the distance along the track increases at constant speed).
    ref_traj = np.zeros((totalSteps + horizonLength + 1, 4)) # 4x4 Array to store the reference trajectory at each time step (+ the horizon)
    for i in range(totalSteps + horizonLength + 1):
        t = i * timeStep # Current time
        s = desiredVelocity * t  # arc-length traveled along the track

        # If s exceeds the maximum distance of the drawn track, hold the last point.
        if s > dists[-1]:
            s = dists[-1]
        
        # Compute the reference position from the spline.
        x_ref = cs_x(s)
        y_ref = cs_y(s)
        
        # Compute the derivative (velocity components) from the spline derivatives.
        vx_ref = cs_x.derivative()(s)
        vy_ref = cs_y.derivative()(s)
        
        # Optionally normalize the velocity to the desired speed.
        speed = np.hypot(vx_ref, vy_ref) # Calculates magnitue of the velocity vector
        if speed > 1e-3:
            vx_ref = desiredVelocity * vx_ref / speed
            vy_ref = desiredVelocity * vy_ref / speed
        else:
            vx_ref = 0
            vy_ref = 0
        
        ref_traj[i, :] = np.array([x_ref, y_ref, vx_ref, vy_ref])

    u_history = [] # Record of control inputs at each timeStep
    state_history = [] # Record of car state at each timeStep

    # Set the initial state.
    # Here we start at the first point of the drawn track, with zero velocity.
    x_current = np.array([path[0, 0], path[0, 1], 0, 0])
    state_history.append(x_current)

    # Iterates through the simulation steps
    for t in range(totalSteps):
        # Define cvxpy variables for the state and control over the horizon.
        x = cp.Variable((4, horizonLength+1)) # Array to store the state at each time step
        u = cp.Variable((2, horizonLength)) # Array to store the control input at each time step
        
        cost = 0
        constraints = []
        
        # Initial condition for the horizon. ensuring the first state in the horizon = the current state
        constraints += [x[:, 0] == x_current]
        
        # Build the cost function and dynamics constraints over the horizon.
        for k in range(horizonLength):
            ref_state = ref_traj[t + k] # The reference state at the current step in the horizon
            cost += cp.quad_form(x[:, k] - ref_state, stateCost) + cp.quad_form(u[:, k], inputCost) # Adds a penalty to any deviation from the reference state and control input
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]] # Constraints on the state dynamics
            constraints += [u[:, k] <= np.array([1.0, 1.0]),
                            u[:, k] >= np.array([-1.0, -1.0])] # Constraints on the control inputs (between -1 & 1)
        
        # Terminal cost for the final state in the horizon.
        ref_state_terminal = ref_traj[t + horizonLength]
        cost += cp.quad_form(x[:, horizonLength] - ref_state_terminal, terminalCost)
        
        # Solve the MPC optimization problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        # Extract the first control input from the optimal sequence.
        u_apply = u[:, 0].value
        if u_apply is None:
            u_apply = np.zeros(2)
        u_history.append(u_apply)

        # Update the current state using the system dynamics.
        x_current = A @ x_current + B @ u_apply

        state_history.append(x_current, u_apply)
    
    # u_history and state_history can be combined into state_history. Optional
    u_history = np.array(u_history)
    state_history = np.array(state_history)

    return u.history

def MPC_converter(x_accel: float, y_accel: float, current_speed: float, current_steer: float, max_steer: float, max_accel: float, max_velo: float, min_velo: float) -> np.ndarray:
    """
    Takes MPC Controller control inputs(x and y accelration) and convertes them into a 1D Array of [steering, thrust]
    
    :param x_accel: MPC calculated x-acceleration of car
    :param y_accel: MPC calculated y-acceleration of car
    :param current_speed: Current speed of car
    :param current_steer: Current steering angle of car
    :param max_steer: Maximum possible steering angle of car
    :param max_accel: Maximum possible acceleration of car
    :param max_velo: Maximum possible velocity of car (forwards)
    :param min_velo: Minimum possible velocity of car (backwards)

    :return: A 1D array [steering, thrust] for the simulator step.
    """
    # Calculate total acceleration
    total_accel = np.sqrt(x_accel**2 + y_accel**2)
    
    # Normalize the acceleration to within given limits
    thrust = min(total_accel, max_accel)
    
    # Calculate the desired angle from acceleration (direction of desired velocity)
    desired_steer = np.arctan2(y_accel, x_accel)  # Angle of the desired velocity vector
    
    # Calculate the steering angle difference
    steer_diff = desired_steer - current_steer
    
    # Normalize steering
    if np.fabs(steer_diff) > 1e-4:
        final_steer = (steer_diff / np.fabs(steer_diff)) * max_steer
    else:
        final_steer = 0.0
    
    return np.array([final_steer, thrust])

##################
##     MAIN     ##
##################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f110_env = gym.make('f110_gym:f110-v0', map='example_map', map_ext='.png', 
                        num_agents=1, timestep=0.015)
    f110_env.add_render_callback(render_callback)
    
    env = SACF110Env(f110_env)
    agent = SACAgent(device, action_dim=32)
    replay_buffer = ReplayBuffer()
    
    max_episodes = 1000
    max_steps = 2000
    batch_size = 64
    update_after = 1000
    update_every = 50
    
    total_steps = 0
    for ep in range(max_episodes):
        obs = env.reset()
        ep_reward = 0
        for st in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            
            f110_env.render("human")
            cv2.imshow("LiDAR Bitmap", obs)
            cv2.waitKey(1)
            
            if total_steps > update_after and total_steps % update_every == 0:
                a_loss, c1_loss, c2_loss = agent.update(replay_buffer, batch_size)
                print(f"Step {total_steps}: Actor={a_loss:.4f}, Critic1={c1_loss:.4f}, Critic2={c2_loss:.4f}")
            
            if done:
                break
        print(f"Episode {ep} Reward={ep_reward:.2f}")
    
    torch.save(agent.actor.state_dict(), "sac_actor.pth")
    cv2.destroyAllWindows()  # Close the bitmap window when done
    print("Training complete, model saved as sac_actor.pth")

if __name__=="__main__":
    main()
