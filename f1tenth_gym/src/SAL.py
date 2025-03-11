import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import cv2
import numpy as np
import cvxpy as cp
from scipy.interpolate import CubicSpline

import os
import random
import bisect
import pickle
import math
from collections import deque
from typing import List, Tuple, Union
import time
import pyglet
from pyglet.gl import GL_LINES

##############################
##     GYM ENVIRONOMENT     ##
##############################
class SACF110Env(gym.Env):
    """
    This environment builds a new path only once the car has physically reached
    the previous path’s final point (i.e. within DIST_THRESHOLD).
    
    - The 32D action is interpreted as 16 local (x,y) increments.
    - Angles between increments are clamped (±10°) to ensure a smooth path.
    - A sub-index (0..15) tracks which waypoint is being pursued.
    - If a new action is provided before the path is finished, it is stored as pending.
    """
    DIST_THRESHOLD = 0.2  # [meters] threshold to consider a waypoint reached

    def __init__(self, f110_env: gym.Env):
        super().__init__()
        self.f110_env = f110_env
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(256,256), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
        
        self.car_length = 0.3
        self.vector_length = 0.5
        
        self.path_points = None    # List of 16 (x,y) points (global coordinates)
        self.sub_index = 16        # Forces a new path parse on first step
        self.pending_action = None # Latest agent action waiting to be used

        self.last_obs = None
        self.prev_x = None
        self.prev_y = None

    def reset(self):
        # Example starting pose: (0, 0) with 90° heading
        default_pose = np.array([[0.0, 0.0, 1.57]])
        obs, _, _, _ = self.f110_env.reset(default_pose)

        lidar_scan = obs['scans'][0]
        # Use FILL mode with a black background for the lidar bitmap
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256,256),
                                 bg_color='black', draw_mode='FILL')
        self.last_obs = obs
        
        self.prev_x = obs['poses_x'][0]
        self.prev_y = obs['poses_y'][0]

        # Force new path
        self.path_points = None
        self.sub_index = 16
        self.pending_action = None

        return bitmap

    def step(self, raw_action: np.ndarray):
        """
        1) If the current path is finished and the car is near its final point,
           parse pending_action (or raw_action) to build a new path.
        2) If mid-path, store the latest action as pending without re-parsing.
        3) Compute a steering & speed command (via MPC) to drive toward the current waypoint.
        4) Advance the sub_index if the car is within DIST_THRESHOLD of the waypoint.
        """
        car_x = self.last_obs['poses_x'][0]
        car_y = self.last_obs['poses_y'][0]
        
        if self.path_points is None:
            self._parse_new_path(raw_action)
        else:
            if self.sub_index >= 16:
                final_x, final_y = self.path_points[-1]
                dist_to_final = np.hypot(final_x - car_x, final_y - car_y)
                if dist_to_final < self.DIST_THRESHOLD:
                    self._parse_new_path(raw_action)
                else:
                    self.pending_action = raw_action
            else:
                self.pending_action = raw_action

        target_x, target_y = self.path_points[self.sub_index]
        # Use the MPC controller (which computes steering and speed) for this step.
        action_out = MPC_controller(
            target_x, target_y,
            car_x, car_y,
            self.last_obs['poses_theta'][0]
        )

        obs, base_reward, done, info = self.f110_env.step(np.array([action_out]))

        lidar_scan = obs['scans'][0]
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256,256),
                                 bg_color='black', draw_mode='FILL')
        
        collision_penalty = -100.0 if done else 0.0
        old_x, old_y = self.prev_x, self.prev_y
        new_x = obs['poses_x'][0]
        new_y = obs['poses_y'][0]
        dist_traveled = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
        self.prev_x, self.prev_y = new_x, new_y
        
        not_moving_penalty = -2.0 if dist_traveled < 0.001 else 0.0
        progress_reward = dist_traveled * 10.0
        
        lap_completion_bonus = 0.0
        if 'lap_time' in info and info['lap_time'] > 0:
            lap_t = info['lap_time']
            lap_completion_bonus = 500.0 - 10.0 * lap_t
            done = True
        
        total_reward = (base_reward + progress_reward + lap_completion_bonus +
                        not_moving_penalty - collision_penalty)

        self.last_obs = obs

        car_x2 = obs['poses_x'][0]
        car_y2 = obs['poses_y'][0]
        target_x2, target_y2 = self.path_points[self.sub_index]
        dist_to_waypoint = np.hypot(target_x2 - car_x2, target_y2 - car_y2)
        if dist_to_waypoint < self.DIST_THRESHOLD:
            self.sub_index += 1

        global current_planned_path
        flattened = []
        for px, py in self.path_points:
            flattened.extend([px, py])
        current_planned_path = np.array(flattened, dtype=np.float32)

        return bitmap, total_reward, done, info

    def _parse_new_path(self, raw_action: np.ndarray):
        """
        Parse the provided (or pending) 32D action into 16 local increments,
        then compute a new global path based on the car's current pose.
        """
        if self.pending_action is not None:
            action_to_use = self.pending_action
            self.pending_action = None
        else:
            action_to_use = raw_action
        
        # Compute clamped vectors (each normalized to have unit length)
        increments = compute_vectors_with_angle_clamp(action_to_use)

        car_x = self.last_obs['poses_x'][0]
        car_y = self.last_obs['poses_y'][0]
        car_theta = self.last_obs['poses_theta'][0]

        front_x = car_x + self.car_length * np.cos(car_theta)
        front_y = car_y + self.car_length * np.sin(car_theta)

        new_points = [(front_x, front_y)]
        for i in range(16):
            dx, dy = increments[i]
            mag = np.sqrt(dx*dx + dy*dy) + 1e-8
            dx_norm, dy_norm = dx/mag, dy/mag
            dx_scaled = dx_norm * self.vector_length
            dy_scaled = dy_norm * self.vector_length
            
            # Rotate the increment from local to global frame
            global_dx = dx_scaled * np.cos(car_theta) - dy_scaled * np.sin(car_theta)
            global_dy = dx_scaled * np.sin(car_theta) + dy_scaled * np.cos(car_theta)
            
            px, py = new_points[-1]
            new_x = px + global_dx
            new_y = py + global_dy
            new_points.append((new_x, new_y))
        
        self.path_points = new_points[1:]
        self.sub_index = 0

###########################################
##   LIDAR TO BITMAP, COURTESY OF ALY    ##
###########################################
def _lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str = 'CCW',          
        starting_angle: float = -np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool = True,  
        output_image_dims: tuple[int] = (256, 256),
        target_beam_count: int = 600,
        fov: float = 2*np.pi,
        draw_mode: str = "FILL"
    ) -> np.ndarray:  
    """
    Creates a bitmap image from lidar scan data.
    Assumes rays are equally spaced over the field of view.
    
    Args:
        scan (list[float]): List of lidar measurements.
        winding_dir (str): Direction for the rays ('CW' or 'CCW').
        starting_angle (float): Offset from the positive x-axis.
        max_scan_radius (float | None): Maximum expected range; if provided, used for scaling.
        scaling_factor (float | None): Scaling factor if max_scan_radius is not given.
        bg_color (str): Background color, either 'white' or 'black'.
        draw_center (bool): Whether to draw a center marker.
        output_image_dims (tuple[int]): Dimensions (height, width) of the output image.
        target_beam_count (int): Number of beams (rays) to be used.
        fov (float): Field of view in radians.
        draw_mode (str): 'RAYS', 'POLYGON', or 'FILL' mode for drawing.
        
    Returns:
        np.ndarray: A single-channel (grayscale) bitmap image.
    """
    assert winding_dir in ['CW', 'CCW'], "winding_dir must be either CW or CCW"
    assert bg_color in ['black', 'white']
    assert draw_mode in ['RAYS', 'POLYGON', 'FILL']
    assert len(output_image_dims) == 2 and all(x > 0 for x in output_image_dims)
    assert 0 < target_beam_count < len(scan)
    assert 0 < fov <= 2*np.pi, "FOV must be between 0 and 2pi"

    if max_scan_radius is not None:
        scaling_factor = min(output_image_dims) / max_scan_radius
    elif scaling_factor is None:
        raise ValueError("Provide either max_scan_radius or scaling_factor")
    
    BG_COLOR, DRAW_COLOR = (0, 255) if bg_color == 'black' else (255, 0)
    image = np.ones(output_image_dims, dtype=np.uint8) * BG_COLOR
    direction = 1 if winding_dir == 'CCW' else -1

    indices = np.linspace(0, len(scan) - 1, target_beam_count, dtype=int)
    data = np.array(scan)[indices]
    angles = starting_angle + direction * fov * np.linspace(0, 1, target_beam_count)
    center = np.array([output_image_dims[0] // 2, output_image_dims[1] // 2])
    points = np.column_stack((
        np.rint(center[0] + scaling_factor * data * np.cos(angles)).astype(int),
        np.rint(center[1] + scaling_factor * data * np.sin(angles)).astype(int)
    ))

    if draw_mode == 'FILL':
        cv2.fillPoly(image, [points], DRAW_COLOR)
    elif draw_mode == 'POLYGON':
        cv2.polylines(image, [points], isClosed=True, color=DRAW_COLOR, thickness=1)
    elif draw_mode == 'RAYS':
        for p in points:
            cv2.line(image, tuple(center), tuple(p), color=DRAW_COLOR, thickness=1)
            cv2.rectangle(image, tuple(p - 2), tuple(p + 2), color=DRAW_COLOR, thickness=-1)

    if draw_center:
        cv2.rectangle(image, tuple(center - 2), tuple(center + 2),
                      color=BG_COLOR if draw_mode == "FILL" else DRAW_COLOR, thickness=-1)
    
    return image

def lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str = 'CCW',          
        starting_angle: float = -np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool = True,  
        output_image_dims: tuple[int] = (256, 256),
        target_beam_count: int = 600,
        fov: float = 2*np.pi,
        draw_mode: str = "POLYGON",
        channels: int = 1
    ) -> np.ndarray:  
    """
    Wraps _lidar_to_bitmap to optionally convert the grayscale image
    into a multi-channel image.
    
    Args:
        (see _lidar_to_bitmap for other parameters)
        channels (int): 1 (grayscale), 3 (RGB), or 4 (RGBA).
    
    Returns:
        np.ndarray: The lidar bitmap image.
    """
    assert channels in [1, 3, 4], "channels must be 1, 3, or 4"
    grayscale_img = _lidar_to_bitmap(scan, winding_dir, starting_angle,
                                     max_scan_radius, scaling_factor, bg_color,
                                     draw_center, output_image_dims,
                                     target_beam_count, fov, draw_mode)
    if channels == 1:
        return grayscale_img
    elif channels == 3:
        return np.stack([grayscale_img] * 3, axis=-1)
    elif channels == 4:
        alpha_channel = np.full_like(grayscale_img, 255)
        return np.stack([grayscale_img, grayscale_img, grayscale_img, alpha_channel], axis=-1)
    else:
        raise ValueError("Invalid number of channels. Supported: 1, 3, or 4.")

##############################
##        OPIUM MODEL       ##
##############################
class Actor(nn.Module):
    """
    The Actor outputs a 32D continuous action (in [-1,1]) representing 16 local (x,y) increments.
    Processes the 256x256 lidar bitmap through convolutional layers.
    """
    def __init__(self, action_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc_mean = nn.Linear(512, action_dim)
        self.fc_log_std = nn.Linear(512, action_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = torch.clamp(self.fc_log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = (dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)).sum(1, keepdim=True)
        return y_t, log_prob

class Critic(nn.Module):
    """
    The Critic estimates the Q-value for a given state (bitmap) and action (32D vector).
    """
    def __init__(self, action_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 28 * 28 + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
    """
    Stores (state, action, reward, next_state, done) tuples for off-policy RL.
    """
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: bool):
        self.buffer.append((s, a, r, ns, d))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d
    
    def __len__(self) -> int:
        return len(self.buffer)

##############################
##        SAC AGENT         ##
##############################
class SACAgent:
    """
    Soft Actor-Critic agent for continuous control.
    
    Attributes:
        device: Torch device.
        actor: The policy network.
        critic1, critic2: The Q-value estimation networks.
        Target networks for critics (for soft updates).
    """
    def __init__(self, device: torch.device, action_dim: int = 32, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2, actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4):
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
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state (np.ndarray): Observation (expected shape: 256x256 or similar).
            evaluate (bool): If True, choose the mean (deterministic) action.
            
        Returns:
            np.ndarray: A 1D action vector (length 32).
        """
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
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> Tuple[float, float, float]:
        """
        Performs a SAC update (both actor and critics).
        
        Args:
            replay_buffer (ReplayBuffer): Buffer to sample transitions from.
            batch_size (int): Number of samples per update.
            
        Returns:
            Tuple containing (actor_loss, critic1_loss, critic2_loss).
        """
        if len(replay_buffer) < batch_size:
            return 0, 0, 0
        
        s, a, r, ns, d = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        if len(s.shape) == 3: s = s.unsqueeze(1)
        ns = torch.FloatTensor(ns).to(self.device)
        if len(ns.shape) == 3: ns = ns.unsqueeze(1)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(np.float32(d)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(ns)
            tq1 = self.critic1_target(ns, next_a)
            tq2 = self.critic2_target(ns, next_a)
            tq = torch.min(tq1, tq2) - self.alpha * next_logp
            tv = r + (1 - d) * self.gamma * tq
        
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
        
        # Soft update target networks
        for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        return a_loss.item(), c1_loss.item(), c2_loss.item()

#######################################
## PATH CLAMP & MPC HELPER FUNCTIONS ##
#######################################
def clamp_vector_angle_diff(prev_angle: float, desired_angle: float, max_diff_deg: float = 10.0) -> float:
    """
    Ensures consecutive vectors differ by at most ±10° (or the given max_diff_deg).
    
    Args:
        prev_angle (float): Previous vector’s angle (radians).
        desired_angle (float): Desired current angle (radians).
        max_diff_deg (float): Maximum allowed deviation in degrees.
        
    Returns:
        float: The clamped angle (radians).
    """
    max_diff_rad = np.radians(max_diff_deg)
    angle_diff = (desired_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi
    if angle_diff > max_diff_rad:
        return prev_angle + max_diff_rad
    elif angle_diff < -max_diff_rad:
        return prev_angle - max_diff_rad
    return desired_angle

def compute_vectors_with_angle_clamp(raw_action: np.ndarray, max_diff_deg: float = 10.0) -> np.ndarray:
    """
    Interprets a 32D raw action as 16 local (x,y) increments,
    forcing the first vector to be (1,0) and clamping subsequent angles.
    
    Args:
        raw_action (np.ndarray): 1D array of length 32.
        max_diff_deg (float): Maximum angle change between successive vectors.
        
    Returns:
        np.ndarray: (16, 2) array of clamped, normalized increments.
    """
    assert raw_action.shape == (32,), "Raw action must be a 32D vector (16 x 2D movements)."
    vectors = raw_action.reshape(16, 2)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    clamped_vectors = np.zeros_like(vectors)
    clamped_vectors[0] = [1, 0]
    prev_angle = np.arctan2(clamped_vectors[0][1], clamped_vectors[0][0])
    for i in range(1, 16):
        desired_angle = np.arctan2(vectors[i][1], vectors[i][0])
        clamped_angle = clamp_vector_angle_diff(prev_angle, desired_angle, max_diff_deg)
        clamped_vectors[i] = [np.cos(clamped_angle), np.sin(clamped_angle)]
        prev_angle = clamped_angle
    return clamped_vectors

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
    cv2.destroyAllWindows()
    print("Training complete, model saved as sac_actor.pth")

if __name__ == "__main__":
    main()