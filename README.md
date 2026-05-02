# 2048 Deep Reinforcement Learning Agent 

An advanced Deep Reinforcement Learning project that trains an AI agent to play the classic **2048 puzzle game**. 

This project evolved from a basic MLP into a **high-performance Afterstate V-Learning pipeline** with Dueling CNN, NoisyNet, Rotational Data Augmentation, and GPU Acceleration — designed to consistently reach the **2048 tile**.

---

##  Key Features & Architecture

To overcome the immense stochasticity (random tile spawns) and sparse rewards of 2048, this project implements a highly optimized RL pipeline:

### 1. Afterstate V-Learning (Core Breakthrough)
*   **Variance Reduction (~10x):** Instead of learning standard $Q(s, a)$—which forces the agent to guess the average outcome of random tile spawns—the agent learns $V(afterstate)$. 
*   **How it works:** The agent evaluates the board *immediately after* a merge action but *before* the random tile spawns. By isolating the deterministic part of the game (the swipe) from the stochastic part (the spawn), the agent learns the true value of board layouts incredibly fast.

### 2. NoisyNet Exploration with Sigma Decay
*   **Learned Exploration:** Traditional $\epsilon$-greedy exploration relies on random actions, which frequently destroys carefully built combos in 2048. 
*   **Factored Gaussian Noise:** We implement `NoisyLinear` layers in the network's value head (Fortunato et al., 2017). The agent injects learned noise directly into its weights, enabling state-dependent, "intelligent" exploration rather than blind random moves.
*   **Sigma Decay (0.9999/ep):** Noise intensity gradually decreases over training. By ~50K episodes, exploration is near-zero and the agent plays almost purely greedy — critical for late-game precision.

### 3. Dueling Double DQN + Multi-scale CNN
*   **Multi-scale CNN:** The 4x4 board is one-hot encoded into an `(18, 4, 4)` 3D tensor. Multi-branch convolutions (`2x2`, `1x4`, `4x1`, `3x3`) capture spatial relationships (e.g., adjacent tiles, monotonic rows).
*   **Double Q-Learning:** Prevents value overestimation by decoupling action selection from action evaluation.
*   **Prioritized Experience Replay (PER):** Uses a vectorized `SumTree` for O(log N) prioritized sampling. TD-errors are **uncapped** (no hard clip) to let critical experiences (deaths, big merges) receive maximum replay priority. `max_priority` is capped at 100 for stability.

### 4. Rotational Data Augmentation (4x Sample Efficiency)
*   **Board Symmetry:** The value of a 2048 board is rotationally invariant — $V(board) = V(rot_{90}(board))$.
*   **Implementation:** Every experience is stored 4 times (original + 3 rotations: 90°, 180°, 270°). Action indices in the `next_states_4` tensor are correctly permuted to match the rotated orientation.
*   **Impact:** The agent learns generalized board patterns **4x faster** with zero additional environment interaction.

### 5. Reward Shaping & Survival Training
*   **Potential-Based Reward Shaping (PBRS):** Mathematically guarantees optimal policies remain unchanged. The potential function (reduced to 50% intensity to avoid heuristic overfitting) provides bonuses for:
    *   **Corner placement** (+0.25 × log2(max_tile))
    *   **Monotonicity on ALL rows/columns** (+0.4 per sorted line)
    *   **Adjacent same-tile bonus** (Snake Strategy, +0.1 × log2)
    *   **Smoothness penalty** (-0.05 × |log2 difference| between adjacent tiles)
    *   **Empty cell bonus** (+0.04 per empty cell)
*   **Death Penalty (-10):** Applied when the game ends in afterstate mode. Forces the agent to prioritize survival in late-game scenarios where the board is nearly full.

### 6. Performance Optimizations
*   **Precomputed Afterstate Buffers:** Caches 4D `(4, 18, 4, 4)` afterstates into the Replay Buffer during environment steps, allowing 100% GPU-vectorized target evaluation.
*   **Vectorized PER SumTree:** Batch sampling via vectorized NumPy masking (`get_batch()`), replacing slow Python loops.
*   **AMP (Automatic Mixed Precision):** Uses `torch.amp` (FP16) on CUDA to halve VRAM usage.
*   **MPS Support:** Automatic detection of Apple Metal GPU (M-series chips) for Mac users.
*   **Dynamic Memory Scaling:** Automatically scales down `buffer_size` to 100K when using afterstate mode to prevent OOM.
*   **Cosine Annealing LR:** Decays learning rate from $10^{-4}$ to $10^{-5}$ (raised floor to maintain learning capacity in late training).

---

##  Installation

Requirements: Python 3.8+, PyTorch 2.0+ (CUDA or MPS recommended)

```bash
# Clone the repository
git clone https://github.com/CaoDuong204/RL-2048.git
cd RL-2048

# Install required dependencies
pip install numpy torch matplotlib
```

---

##  Usage & Training

The main training script is `training_dqn.py`. It is highly configurable via command-line arguments. 

### 1. Recommended Run (Full Pipeline)
Train using Afterstate V-learning, NoisyNet + sigma decay, data augmentation, and Cosine Annealing LR:

```bash
# 50K episodes (~1.5-3 hours on CUDA GPU)
python training_dqn.py \
    --episodes 50000 \
    --afterstate \
    --noisy-net \
    --lr-schedule \
    --buffer-size 100000 \
    --batch-size 512 \
    --save-name afterstate_v7_50k
```

```bash
# 100K episodes (~3-6 hours, higher ≥2048 rate)
python training_dqn.py \
    --episodes 100000 \
    --afterstate \
    --noisy-net \
    --lr-schedule \
    --buffer-size 100000 \
    --batch-size 512 \
    --save-name afterstate_v7_100k
```

### 2. Standard Dueling Q-Learning (Baseline Comparison)
Classic action-value $Q(s,a)$ with Epsilon-Greedy (slower convergence):

```bash
python training_dqn.py --episodes 50000 --eps-decay 0.9995
```

### 3. Ablation Studies (Vanilla DQN)
Test the impact of enhancements by disabling them:

```bash
python training_dqn.py --network-type mlp --no-double-dqn --no-per --n-step 1
```

---

##  Outputs & Monitoring

### Training Log (every 100 episodes)
```
Ep  3,000 | Score:  1285 | AvgTile:  684 | Med:  512 | Std:  210 | ≥256:100% ≥512: 87% ≥1K: 40% ≥2K:  3% | Fail<256:  0% | Loss:4.65 | LR:9.99e-05 | 4.87s
```

### Greedy Evaluation (every 1000 episodes, 200 games, no noise)
```
    ┌─ [EVAL - 200 games, greedy]
    │  Score:  1350 | AvgTile:  720 | Med:  512 | Std:  180
    │  ≥512: 91% | ≥1024: 48% | ≥2048:  6%
    └─ Best: 2048
```

### Artifacts Saved (in `./data/` folder)
1.  **Network Weights:** `dqn_local_*.pth`, `dqn_target_*.pth`. Auto-saves `_best.pth` on new record tile.
2.  **Training State:** `dqn_optimizer_*.pth`, `dqn_state_*.pkl`
3.  **Visualization:** `*_results.png` (6-panel graph: Score, Max Tile, Reward, Loss, Tile Distribution, Rolling Avg).

---

##  Project Structure

*   `game.py`: The 2048 environment logic, including optimized pure-numpy Afterstate calculations and `_CALC_CACHE` for O(1) row operations.
*   `agent_dqn.py`: Contains `DQNAgent`, `AfterstateValueNetwork`, `NoisyLinear` (with `decay_sigma`), `SumTree` (vectorized `get_batch`), `PrioritizedReplayBuffer`, and `NStepBuffer`.
*   `training_dqn.py`: Training loop, state encoding, PBRS reward shaping, rotational data augmentation, greedy evaluation, and Matplotlib plotting.
