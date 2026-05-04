"""
Optimized DQN Agent for 2048 Game
===================================
Full-featured agent with:
- Dueling CNN architecture (multi-scale convolutions)
- Double DQN (reduces overestimation)
- Prioritized Experience Replay (PER) — learns from important experiences
- N-step returns — propagates rewards faster
- Gradient clipping — training stability

Architecture (default: dueling_cnn):
    Input: (batch, 18, 4, 4) one-hot encoded board
       ↓
    ┌─ Conv 2x2 (adjacent tiles)   ─┐
    ├─ Conv 1x4 (row patterns)      ├→ Concat → FC
    ├─ Conv 4x1 (column patterns)   │
    └─ Conv 3x3 (area patterns)     ┘      ↓
                                      ┌─ Value V(s)      ─┐
                                      └─ Advantage A(s,a) ┘→ Q(s,a)
"""

import numpy as np
import random
import os
import pickle
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Compact Board Storage Helpers
# =============================================================================
def compact_board(board_flat_or_2d):
    """Convert board → (4,4) int8 log2 indices. 72x smaller than one-hot."""
    board = np.asarray(board_flat_or_2d).reshape(4, 4)
    result = np.zeros((4, 4), dtype=np.int8)
    mask = board > 0
    result[mask] = np.log2(board[mask]).astype(np.int8)
    return result


def expand_states_gpu(compact_tensor):
    """(B, 4, 4) int/float tensor → (B, 18, 4, 4) one-hot float32 on same device."""
    idx = compact_tensor.long().unsqueeze(1)       # (B, 1, 4, 4)
    out = torch.zeros(compact_tensor.size(0), 18, 4, 4,
                      device=compact_tensor.device, dtype=torch.float32)
    out.scatter_(1, idx, 1.0)
    return out


# =============================================================================
# NoisyLinear Layer (Fortunato et al., 2017)
# =============================================================================
class NoisyLinear(nn.Module):
    """Factored Gaussian Noisy Linear layer for learned exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))

    def reset_noise(self):
        device = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scale_noise(size, device):
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

    def decay_sigma(self, factor=0.999):
        """Decay sigma parameters to reduce exploration over time."""
        with torch.no_grad():
            self.weight_sigma.mul_(factor)
            self.bias_sigma.mul_(factor)


# =============================================================================
# Network 1: Dueling CNN (recommended)
# =============================================================================
class DuelingCNNNetwork(nn.Module):
    """
    Dueling CNN Q-Network for 2048.
    Multi-scale convolutions + Value/Advantage streams.
    Input:  (batch, 18, 4, 4)
    Output: (batch, action_size)
    """

    def __init__(self, action_size, seed, n_filters=64):
        super(DuelingCNNNetwork, self).__init__()
        torch.manual_seed(seed)
        nf = n_filters

        # === Multi-scale CNN ===
        # Branch 1: 2x2 → 2x2 (local tile patterns)
        self.conv_2x2_a = nn.Conv2d(18, nf * 2, kernel_size=2, padding=0)
        self.conv_2x2_b = nn.Conv2d(nf * 2, nf * 2, kernel_size=2, padding=0)
        # → (nf*2, 2, 2)

        # Branch 2: 1x4 (row patterns)
        self.conv_row = nn.Conv2d(18, nf, kernel_size=(1, 4), padding=0)
        # → (nf, 4, 1)

        # Branch 3: 4x1 (column patterns)
        self.conv_col = nn.Conv2d(18, nf, kernel_size=(4, 1), padding=0)
        # → (nf, 1, 4)

        # Branch 4: 3x3 (larger spatial)
        self.conv_3x3 = nn.Conv2d(18, nf, kernel_size=3, padding=0)
        # → (nf, 2, 2)

        # Feature sizes: nf*2*4 + nf*4 + nf*4 + nf*4 = nf*20
        flat_size = nf * 20

        # Shared FC
        self.fc_shared = nn.Linear(flat_size, 256)

        # Dueling streams
        self.value_fc = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage_out = nn.Linear(128, action_size)

    def forward(self, state):
        x1 = F.relu(self.conv_2x2_a(state))
        x1 = F.relu(self.conv_2x2_b(x1))
        x1 = x1.reshape(x1.size(0), -1)

        x2 = F.relu(self.conv_row(state))
        x2 = x2.reshape(x2.size(0), -1)

        x3 = F.relu(self.conv_col(state))
        x3 = x3.reshape(x3.size(0), -1)

        x4 = F.relu(self.conv_3x3(state))
        x4 = x4.reshape(x4.size(0), -1)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = F.relu(self.fc_shared(x))

        v = F.relu(self.value_fc(x))
        v = self.value_out(v)

        a = F.relu(self.advantage_fc(x))
        a = self.advantage_out(a)

        return v + (a - a.mean(dim=1, keepdim=True))


# =============================================================================
# Network 3: Afterstate Value Network (V-learning)
# =============================================================================
class AfterstateValueNetwork(nn.Module):
    """
    V(afterstate) network for afterstate learning.
    Same multi-scale CNN backbone as DuelingCNN.
    Output: single scalar V (value of afterstate).
    """
    def __init__(self, seed, n_filters=64, noisy=False, sigma_init=0.5):
        super().__init__()
        torch.manual_seed(seed)
        nf = n_filters
        self.noisy = noisy

        # CNN backbone (identical to DuelingCNNNetwork)
        self.conv_2x2_a = nn.Conv2d(18, nf * 2, kernel_size=2)
        self.conv_2x2_b = nn.Conv2d(nf * 2, nf * 2, kernel_size=2)
        self.conv_row = nn.Conv2d(18, nf, kernel_size=(1, 4))
        self.conv_col = nn.Conv2d(18, nf, kernel_size=(4, 1))
        self.conv_3x3 = nn.Conv2d(18, nf, kernel_size=3)

        flat_size = nf * 20
        self.fc_shared = nn.Linear(flat_size, 256)

        # Value head only (NoisyLinear if enabled)
        Lin = lambda i, o: NoisyLinear(i, o, sigma_init) if noisy else nn.Linear(i, o)
        self.value_fc = Lin(256, 128)
        self.value_out = Lin(128, 1)

    def reset_noise(self):
        if self.noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

    def forward(self, state):
        x1 = F.relu(self.conv_2x2_a(state))
        x1 = F.relu(self.conv_2x2_b(x1))
        x1 = x1.reshape(x1.size(0), -1)
        x2 = F.relu(self.conv_row(state)).reshape(state.size(0), -1)
        x3 = F.relu(self.conv_col(state)).reshape(state.size(0), -1)
        x4 = F.relu(self.conv_3x3(state)).reshape(state.size(0), -1)
        x = F.relu(self.fc_shared(torch.cat([x1, x2, x3, x4], dim=1)))
        return self.value_out(F.relu(self.value_fc(x)))

    def decay_noise(self, factor=0.999):
        """Decay sigma in all NoisyLinear layers to reduce exploration over time."""
        if self.noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.decay_sigma(factor)


# =============================================================================
# Network 2: Simple MLP (for comparison)
# =============================================================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed,
                 fc1_units=512, fc2_units=512, fc3_units=256):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# =============================================================================
# SumTree for Prioritized Experience Replay
# =============================================================================
class SumTree:
    """
    Binary sum tree for O(log n) priority-based sampling.
    Each leaf stores a priority value; parent nodes store the sum of children.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self._update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # Propagate change up to root (iterative)
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, s):
        """Find leaf index for cumulative sum s."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1

    def get(self, s):
        """Get (tree_idx, priority, data) for cumulative sum s."""
        leaf_idx = self._retrieve(s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def get_batch(self, s_array):
        """Vectorized retrieval of multiple samples."""
        idx = np.zeros(len(s_array), dtype=int)
        
        # Array must be copied to avoid mutating the original input array view
        s_array = np.array(s_array, copy=True)
        
        left_children = 2 * idx + 1
        is_leaf = left_children >= len(self.tree)
        
        while not np.all(is_leaf):
            not_leaf = ~is_leaf
            curr_idx = idx[not_leaf]
            curr_s = s_array[not_leaf]
            
            left = 2 * curr_idx + 1
            left_vals = self.tree[left]
            
            go_left = curr_s <= left_vals
            
            new_idx = np.where(go_left, left, left + 1)
            new_s = np.where(go_left, curr_s, curr_s - left_vals)
            
            idx[not_leaf] = new_idx
            s_array[not_leaf] = new_s
            
            left_children = 2 * idx + 1
            is_leaf = left_children >= len(self.tree)
            
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], [self.data[i] for i in data_idx]

    def update(self, tree_idx, priority):
        self._update(tree_idx, priority)


# =============================================================================
# Prioritized Experience Replay Buffer
# =============================================================================
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2015).

    Experiences with higher TD-error get sampled more often.
    Uses importance sampling weights to correct for the bias.

    Parameters:
        alpha: prioritization exponent (0=uniform, 1=full prioritization)
        beta: importance sampling correction (annealed from beta_start to 1.0)
    """

    def __init__(self, capacity, batch_size, alpha=0.6,
                 beta_start=0.4, beta_end=1.0, beta_frames=200000):
        self.tree = SumTree(capacity)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self):
        """Sample a batch with priorities. Returns (experiences, indices, weights)."""
        if self.tree.n_entries < self.batch_size:
            return None

        # Anneal beta
        self.beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * self.frame / self.beta_frames
        )
        self.frame += 1

        segment = self.tree.total() / self.batch_size

        # Vectorized sampling
        a = segment * np.arange(self.batch_size)
        b = segment * (np.arange(self.batch_size) + 1)
        s_array = np.random.uniform(a, b)
        
        indices, priorities, batch = self.tree.get_batch(s_array)
        
        # Filter None
        valid = [i for i, data in enumerate(batch) if data is not None]
        batch = [batch[i] for i in valid]
        indices = indices[valid]
        priorities = priorities[valid]
        
        priorities = np.maximum(priorities, self.epsilon)

        if len(batch) < self.batch_size:
            return None

        # Importance sampling weights
        probs = np.array(priorities, dtype=np.float64) / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / weights.max()

        states = torch.from_numpy(np.array([e[0] for e in batch])).float().to(device)
        actions = torch.from_numpy(np.array([e[1] for e in batch]).reshape(-1, 1)).long().to(device)
        rewards = torch.from_numpy(np.array([e[2] for e in batch]).reshape(-1, 1)).float().to(device)
        next_states = torch.from_numpy(np.array([e[3] for e in batch])).float().to(device)
        dones = torch.from_numpy(
            np.array([e[4] for e in batch]).astype(np.uint8).reshape(-1, 1)
        ).float().to(device)
        weights = torch.from_numpy(weights.reshape(-1, 1).astype(np.float32)).to(device)

        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors. No hard clip — max_priority cap handles extremes."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = min(max(self.max_priority, priority), 100.0)  # Cap at 100
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# =============================================================================
# Uniform Replay Buffer (for comparison / non-PER mode)
# =============================================================================
class UniformReplayBuffer:
    """Standard replay buffer with uniform sampling."""

    def __init__(self, capacity, batch_size, seed=42):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.array([e[0] for e in batch])).float().to(device)
        actions = torch.from_numpy(np.array([e[1] for e in batch]).reshape(-1, 1)).long().to(device)
        rewards = torch.from_numpy(np.array([e[2] for e in batch]).reshape(-1, 1)).float().to(device)
        next_states = torch.from_numpy(np.array([e[3] for e in batch])).float().to(device)
        dones = torch.from_numpy(
            np.array([e[4] for e in batch]).astype(np.uint8).reshape(-1, 1)
        ).float().to(device)
        return (states, actions, rewards, next_states, dones), None, None

    def __len__(self):
        return len(self.memory)


# =============================================================================
# N-Step Return Buffer
# =============================================================================
class NStepBuffer:
    """
    Accumulates transitions and computes n-step discounted returns.

    Instead of: Q = r + γ * Q(s')
    Computes:   Q = r₀ + γr₁ + γ²r₂ + ... + γⁿQ(sₙ)

    This propagates reward signals faster through the value function.
    """

    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque()

    def reset(self):
        self.buffer.clear()

    def add(self, state, action, reward, next_state, done):
        """
        Add transition. Returns list of ready n-step transitions.
        When done=True, flushes all remaining transitions.
        """
        self.buffer.append((state, action, reward, next_state, done))
        transitions = []

        if done:
            # Episode ended: flush all remaining transitions
            while len(self.buffer) > 0:
                transitions.append(self._compute_return())
                self.buffer.popleft()
        elif len(self.buffer) >= self.n_step:
            # Buffer full: compute and pop oldest
            transitions.append(self._compute_return())
            self.buffer.popleft()

        return transitions

    def _compute_return(self):
        """Compute n-step return from current buffer contents."""
        R = 0.0
        last_next_state = self.buffer[-1][3]
        last_done = self.buffer[-1][4]

        for i, (s, a, r, ns, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:
                last_next_state = ns
                last_done = True
                break

        return (self.buffer[0][0], self.buffer[0][1],
                R, last_next_state, last_done)


# =============================================================================
# DQN Agent (Full-featured + Afterstate V-Learning)
# =============================================================================
class DQNAgent:
    """
    Optimized DQN Agent for 2048.

    Modes:
    - Q-learning (default): Q(s,a) with Dueling CNN
    - Afterstate V-learning: V(afterstate) — learns board evaluation after merge
    
    Enhancements: Double DQN, PER, N-step, NoisyNet, AMP, gradient clipping.
    """

    def __init__(self, state_size=288, action_size=4, seed=42,
                 # Network
                 network_type='dueling_cnn', n_filters=64,
                 fc1_units=512, fc2_units=512, fc3_units=256,
                 # Core hyperparameters
                 lr=1e-4, gamma=0.99, tau=1e-3,
                 buffer_size=200000, batch_size=256, update_every=4,
                 # Enhancements
                 double_dqn=True,
                 use_per=True, per_alpha=0.5, per_beta_start=0.5, per_beta_frames=100000,
                 n_step=3,
                 # New features
                 afterstate=False, noisy_net=False, sigma_init=0.5):
                 
        if afterstate:
            n_step = 1  # Force n-step to 1 for afterstate to avoid mixing state spaces

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.network_type = network_type
        self.afterstate = afterstate
        self.noisy_net = noisy_net
        random.seed(seed)
        np.random.seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.double_dqn = double_dqn
        self.use_per = use_per
        self.n_step = n_step

        # === Networks ===
        if afterstate:
            self.qnetwork_local = AfterstateValueNetwork(
                seed, n_filters, noisy=noisy_net, sigma_init=sigma_init
            ).to(device)
            self.qnetwork_target = AfterstateValueNetwork(
                seed, n_filters, noisy=noisy_net, sigma_init=sigma_init
            ).to(device)
        elif network_type == 'dueling_cnn':
            self.qnetwork_local = DuelingCNNNetwork(action_size, seed, n_filters).to(device)
            self.qnetwork_target = DuelingCNNNetwork(action_size, seed, n_filters).to(device)
        else:
            self.qnetwork_local = QNetwork(
                state_size, action_size, seed, fc1_units, fc2_units, fc3_units
            ).to(device)
            self.qnetwork_target = QNetwork(
                state_size, action_size, seed, fc1_units, fc2_units, fc3_units
            ).to(device)

        # Removed torch.compile because it conflicts with CUDAGraphs and NoisyLinear,
        # and it causes massive recompilation overhead for dynamic batch sizes (1-4).

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self._hard_update()

        # === AMP ===
        self.use_amp = (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # === Replay Buffer ===
        if use_per:
            self.memory = PrioritizedReplayBuffer(
                buffer_size, batch_size,
                alpha=per_alpha, beta_start=per_beta_start, beta_frames=per_beta_frames
            )
        else:
            self.memory = UniformReplayBuffer(buffer_size, batch_size, seed)

        # === N-Step Buffer ===
        self.n_step_buffer = NStepBuffer(n_step, gamma) if n_step > 1 else None

        # === Metrics ===
        self.t_step = 0
        self.learn_step = 0
        self.losses = []

        # === Print summary ===
        total_params = sum(p.numel() for p in self.qnetwork_local.parameters())
        features = []
        if afterstate:
            features.append("Afterstate-V")
        if double_dqn:
            features.append("Double")
        if not afterstate:
            features.append("Dueling+CNN" if network_type == 'dueling_cnn' else "MLP")
        else:
            features.append("CNN")
        if noisy_net:
            features.append("NoisyNet")
        if use_per:
            features.append("PER")
        if n_step > 1:
            features.append(f"{n_step}-step")
        print(f"  Agent          : {' + '.join(features)}")
        print(f"  Parameters     : {total_params:,}")
        print(f"  Gamma^n_step   : {gamma}^{n_step} = {gamma**n_step:.6f}")
        print(f"  AMP            : {'ON' if self.use_amp else 'OFF'}")

    # -----------------------------------------------------------------
    def step(self, state, action, reward, next_state, done):
        """Add experience (with n-step processing) and learn periodically."""
        if self.n_step_buffer is not None:
            transitions = self.n_step_buffer.add(
                state, action, reward, next_state, done
            )
            for t in transitions:
                self.memory.add(*t)
        else:
            self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1
        if self.t_step % self.update_every == 0:
            if len(self.memory) >= self.batch_size:
                self._sample_and_learn()

    # -----------------------------------------------------------------
    def act(self, state):
        """Return Q-values (or V-value for afterstate mode)."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.qnetwork_local(state_t)
        return q.cpu().data.numpy()

    # -----------------------------------------------------------------
    def evaluate_batch(self, states_np):
        """Evaluate a batch of states. Returns numpy array of values."""
        states_t = torch.from_numpy(states_np).float().to(device)
        with torch.no_grad():
            v = self.qnetwork_local(states_t)
        return v.cpu().numpy()

    # -----------------------------------------------------------------
    def _sample_and_learn(self):
        """Sample from buffer and learn. For afterstate: expand compact states + GPU augmentation."""
        if self.noisy_net:
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()
        result = self.memory.sample()
        if result is None:
            return
        experiences, indices, weights = result
        if self.afterstate:
            states, actions, rewards, next_states, dones = experiences
            # Expand compact (B, 4, 4) → (B, 18, 4, 4) on GPU
            states = expand_states_gpu(states)
            # Expand compact (B, 4, 4, 4) → (B, 4, 18, 4, 4)
            B = states.size(0)
            next_flat = next_states.reshape(B * 4, 4, 4)
            next_expanded = expand_states_gpu(next_flat)
            next_states = next_expanded.reshape(B, 4, 18, 4, 4)
            # GPU augmentation: 4x batch via rotational symmetry
            states, next_states, rewards, dones = self._augment_batch_gpu(
                states, next_states, rewards, dones)
            if weights is not None:
                weights = weights.repeat(4, 1)
            experiences = (states, actions, rewards, next_states, dones)
            self._learn_afterstate(experiences, indices, weights, orig_batch_size=B)
        else:
            self._learn(experiences, indices, weights)

    # -----------------------------------------------------------------
    def _augment_batch_gpu(self, states, next_4, rewards, dones):
        """Augment batch with 3 rotations on GPU. Returns 4x batch size."""
        _PERM = [[1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        all_s, all_n = [states], [next_4]
        for k, perm in enumerate(_PERM, 1):
            all_s.append(torch.rot90(states, k=k, dims=[-2, -1]))
            rot_n = next_4[:, perm]                        # permute action dim
            rot_n = torch.rot90(rot_n, k=k, dims=[-2, -1]) # rotate spatial dims
            all_n.append(rot_n)
        return (torch.cat(all_s), torch.cat(all_n),
                rewards.repeat(4, 1), dones.repeat(4, 1))

    # -----------------------------------------------------------------
    def _learn(self, experiences, indices=None, weights=None):
        """Standard Q-learning update (unchanged from original)."""
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.double_dqn:
                    best_actions = self.qnetwork_local(next_states).argmax(dim=1).unsqueeze(1)
                    Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
                else:
                    Q_targets_next = self.qnetwork_target(next_states).max(dim=1)[0].unsqueeze(1)

        gamma_n = self.gamma ** self.n_step
        Q_targets = rewards + (gamma_n * Q_targets_next * (1.0 - dones))

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            td_errors = (Q_expected - Q_targets).detach().float()
            if weights is not None:
                loss = (F.mse_loss(Q_expected, Q_targets, reduction='none') * weights).mean()
            else:
                loss = F.mse_loss(Q_expected, Q_targets)

        self._backward(loss)

        if indices is not None:
            self.memory.update_priorities(indices, td_errors.cpu().numpy().flatten())

        self._soft_update()
        self.learn_step += 1
        self.losses.append(loss.item())

    # -----------------------------------------------------------------
    def _learn_afterstate(self, experiences, indices=None, weights=None, orig_batch_size=None):
        """
        Afterstate V-learning update.
        Batch may be 4x augmented; orig_batch_size tracks the real PER entries.
        """
        afterstates, _, rewards, next_states_4, dones = experiences
        batch_size = afterstates.size(0)
        
        # Valid mask: one-hot all-zero = invalid afterstate
        valid_mask = (next_states_4.sum(dim=(-3, -2, -1)) > 0)  # (batch_size, 4)
        
        flat_next_states = next_states_4.reshape(batch_size * 4, 18, 4, 4)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.double_dqn:
                    local_vals = self.qnetwork_local(flat_next_states).view(batch_size, 4)
                    target_vals = self.qnetwork_target(flat_next_states).view(batch_size, 4)
                    local_vals = local_vals.masked_fill(~valid_mask, float('-inf'))
                    best_actions = local_vals.argmax(dim=1, keepdim=True)
                    V_next_max = target_vals.gather(1, best_actions)
                else:
                    target_vals = self.qnetwork_target(flat_next_states).view(batch_size, 4)
                    target_vals = target_vals.masked_fill(~valid_mask, float('-inf'))
                    V_next_max = target_vals.max(dim=1, keepdim=True)[0]
                    
        V_next_max = V_next_max.masked_fill(dones.bool(), 0.0)

        gamma_n = self.gamma ** self.n_step
        V_targets = rewards + (gamma_n * V_next_max)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            V_expected = self.qnetwork_local(afterstates)
            td_errors = (V_expected - V_targets).detach().float()
            if weights is not None:
                loss = (F.mse_loss(V_expected, V_targets, reduction='none') * weights).mean()
            else:
                loss = F.mse_loss(V_expected, V_targets)

        self._backward(loss)

        # Only update PER priorities for original (non-augmented) samples
        if indices is not None:
            n = orig_batch_size if orig_batch_size else batch_size
            self.memory.update_priorities(indices, td_errors[:n].cpu().numpy().flatten())

        self._soft_update()
        self.learn_step += 1
        self.losses.append(loss.item())

    # -----------------------------------------------------------------
    def _backward(self, loss):
        """Backward pass with AMP support."""
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # -----------------------------------------------------------------
    def _soft_update(self):
        for tp, lp in zip(self.qnetwork_target.parameters(),
                          self.qnetwork_local.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    def _hard_update(self):
        for tp, lp in zip(self.qnetwork_target.parameters(),
                          self.qnetwork_local.parameters()):
            tp.data.copy_(lp.data)

    # -----------------------------------------------------------------
    def save(self, name, save_dir='./data/'):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.qnetwork_local.state_dict(),
                    os.path.join(save_dir, f'dqn_local_{name}.pth'))
        torch.save(self.qnetwork_target.state_dict(),
                    os.path.join(save_dir, f'dqn_target_{name}.pth'))
        torch.save(self.optimizer.state_dict(),
                    os.path.join(save_dir, f'dqn_optimizer_{name}.pth'))
        state = {
            'losses': self.losses, 't_step': self.t_step,
            'learn_step': self.learn_step, 'network_type': self.network_type,
            'double_dqn': self.double_dqn, 'use_per': self.use_per,
            'n_step': self.n_step, 'gamma': self.gamma, 'tau': self.tau,
            'afterstate': self.afterstate, 'noisy_net': self.noisy_net,
        }
        with open(os.path.join(save_dir, f'dqn_state_{name}.pkl'), 'wb') as f:
            pickle.dump(state, f)

    def load(self, name, save_dir='./data/'):
        self.qnetwork_local.load_state_dict(
            torch.load(os.path.join(save_dir, f'dqn_local_{name}.pth'), map_location=device))
        self.qnetwork_target.load_state_dict(
            torch.load(os.path.join(save_dir, f'dqn_target_{name}.pth'), map_location=device))
        self.optimizer.load_state_dict(
            torch.load(os.path.join(save_dir, f'dqn_optimizer_{name}.pth'), map_location=device))
        with open(os.path.join(save_dir, f'dqn_state_{name}.pkl'), 'rb') as f:
            state = pickle.load(f)
        self.losses = state['losses']
        self.t_step = state['t_step']
        self.learn_step = state['learn_step']

