# 2048 Deep Reinforcement Learning Agent 🧠🎮

An advanced Deep Reinforcement Learning project that trains an AI agent to play the classic **2048 puzzle game**. 

This project evolved from a basic MLP into a **State-of-the-Art (SOTA) Dueling Double DQN architecture**, and finally integrated **Afterstate V-Learning, NoisyNets, and AMP GPU Acceleration** to maximize sample efficiency and performance.

---

## 🚀 Key Features & Architecture

To overcome the immense stochasticity (random tile spawns) and sparse rewards of 2048, this project implements a highly optimized RL pipeline:

### 1. Afterstate V-Learning (Core Breakthrough)
*   **Variance Reduction (~10x):** Instead of learning standard $Q(s, a)$—which forces the agent to guess the average outcome of random tile spawns—the agent learns $V(afterstate)$. 
*   **How it works:** The agent evaluates the board *immediately after* a merge action but *before* the random tile spawns. By isolating the deterministic part of the game (the swipe) from the stochastic part (the spawn), the agent learns the true value of board layouts incredibly fast.

### 2. NoisyNet Exploration
*   **Learned Exploration:** Traditional $\epsilon$-greedy exploration relies on random actions, which frequently destroys carefully built combos in 2048. 
*   **Factored Gaussian Noise:** We implement `NoisyLinear` layers in the network's value head (Fortunato et al., 2017). The agent injects learned noise directly into its weights, enabling state-dependent, "intelligent" exploration rather than blind random moves.

### 3. Dueling Double DQN + Multi-scale CNN
*   **Multi-scale CNN:** The 4x4 board is one-hot encoded into an `(18, 4, 4)` 3D tensor. Multi-branch convolutions (`2x2`, `1x4`, `4x1`, `3x3`) capture spatial relationships (e.g., adjacent tiles, monotonic rows).
*   **Double Q-Learning:** Prevents value overestimation by decoupling action selection from action evaluation.
*   **Prioritized Experience Replay (PER) & N-Step Returns ($N=3$):** Uses a `SumTree` for O(log N) prioritized sampling of high TD-error experiences, combined with 3-step returns for faster reward propagation.

### 4. GPU Acceleration & Optimization
*   **AMP (Automatic Mixed Precision):** Uses `torch.amp` (FP16) to halve VRAM usage and maximize Tensor Core performance on modern GPUs (e.g., RTX 40-series).
*   **PyTorch 2.0 Compile:** Uses `torch.compile(mode='reduce-overhead')` for further graph optimization.
*   **Cosine Annealing LR:** Smoothly decays the learning rate from $10^{-4}$ down to $10^{-6}$ for stable late-game convergence when the agent reaches high tiles (1024/2048).

---

## 🛠 Installation

Requirements: Python 3.8+, PyTorch 2.0+ (CUDA recommended)

```bash
# Clone the repository
git clone <repository_url>
cd 2048_RL-master

# Install required dependencies
pip install numpy torch matplotlib
```

---

## 🕹 Usage & Training

The main training script is `training_dqn.py`. It is highly configurable via command-line arguments. 

### 1. The Recommended Run (SOTA Configuration)
Train using Afterstate V-learning, NoisyNet exploration, and Cosine Annealing LR. This configuration is optimized to hit the 2048 tile within ~4 hours on a modern GPU.

```bash
python3 training_dqn.py --episodes 200000 --afterstate --noisy-net --lr-schedule --batch-size 512 --tau 0.005 --save-name 2048_sota
```

### 2. Standard Dueling Q-Learning (Legacy Comparison)
If you want to train using the classic action-value $Q(s,a)$ method with Epsilon-Greedy exploration (slower convergence):

```bash
python3 training_dqn.py --episodes 50000 --eps-decay 0.9995
```

### 3. Ablation Studies (Vanilla DQN)
You can test the impact of the enhancements by turning them off to run a Vanilla DQN comparison:

```bash
# Pure Vanilla DQN with MLP (Flat Network)
python3 training_dqn.py --network-type mlp --no-double-dqn --no-per --n-step 1
```

---

## 📊 Outputs & Monitoring

During training, the console will output progress every `100` episodes, showing the Average Score, Average Max Tile, Win Rates ($\ge512$, $\ge1024$), and Loss.

**Artifacts Saved (in `./data/` folder):**
1.  **Network Weights:** `dqn_local_*.pth`, `dqn_target_*.pth`. The system automatically saves a `_best.pth` checkpoint whenever the agent achieves a new record tile!
2.  **Training State:** `dqn_optimizer_*.pth`, `dqn_state_*.pkl` (Includes replay data)
3.  **Visualization:** `*_results.png` (A 6-panel graph showing Game Score, Max Tile progression, Loss curve (log scale), and Tile Distribution charts).

---

## 🧠 Project Structure

*   `game.py`: The 2048 environment logic, including optimized pure-numpy functions for Afterstate calculations.
*   `agent_dqn.py`: Contains the `DQNAgent`, `AfterstateValueNetwork`, `NoisyLinear` layers, and Replay Buffers (`SumTree`, `PER`, `NStepBuffer`). Handles all PyTorch/GPU logic.
*   `training_dqn.py`: The primary training loop, state encoding, PBRS (Potential-Based Reward Shaping), and Matplotlib plotting.
