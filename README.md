# 2048 Deep Reinforcement Learning Agent 

An advanced Deep Reinforcement Learning project that trains an AI agent to play the classic **2048 puzzle game**. 

This project evolved from a basic reward-prediction MLP into a **State-of-the-Art (SOTA) Dueling Double DQN architecture** equipped with a multi-scale Convolutional Neural Network and several cutting-edge RL enhancements.

---

##  Key Features & Architecture

To overcome the specific spatial challenges of the 2048 board and sparse long-term rewards, the agent implements several advanced techniques:

### 1. Dueling Double DQN + CNN (Core Architecture)
*   **Multi-scale CNN:** Instead of flattening the 4x4 board, the state is one-hot encoded into a `(18, 4, 4)` 3D tensor. The network uses multi-branch convolutions (`2x2`, `1x4`, `4x1`, `3x3`) to perfectly capture spatial relationships (e.g., adjacent tiles that can merge, or monotonic rows/columns).
*   **Double Q-Learning:** Prevents the agent from overestimating the value of states by decoupling action selection (Local Network) from action evaluation (Target Network).
*   **Dueling Streams:** Splits the network output into two streams: `Value V(s)` (how good is the board) and `Advantage A(s,a)` (how good is this move compared to others).

### 2. Advanced Training Enhancements
*   **Prioritized Experience Replay (PER):** Uses a `SumTree` data structure to prioritize training on experiences with high TD-error. The agent learns much faster by focusing on rare "Aha!" moments (like merging a 512 tile) rather than boring early-game moves. TD-errors are clipped to `[-1.0, 1.0]` to prevent extreme sampling bias.
*   **N-Step Returns ($N=3$):** Replaces the standard 1-step Bellman equation. Rewards are accumulated over 3 steps before bootstrapping, allowing the reward signal to propagate backward faster while reducing variance caused by the game's stochastic tile spawns.
*   **Pure RL Action Selection:** Strict Epsilon-Greedy implementation. If the agent selects an invalid action (e.g., swiping into a wall), the environment rejects the move and the agent receives an immediate penalty (`-0.1`). This forces the agent to learn the rules of the game naturally without environment hacks.
*   **Potential-Based Reward Shaping (PBRS):** Prevents the agent from "farming" passive income by remaining in a good state. Shaping is applied mathematically via $R_{shaped} = R_{raw} + \gamma \Phi(S') - \Phi(S)$. The potential $\Phi(S)$ includes:
    *   **Dynamic Corner Potential:** `+0.1 * log2(max_tile)` if the highest tile is securely in a corner. The gravitational pull scales exponentially as the tile grows!
    *   **Empty Cell Potential:** `+0.02` per empty cell to encourage space management.
    *   **Monotonicity Potential:** `+0.25` for sorting rows/columns (the classic "snake" strategy).

---

##  Installation

Requirements: Python 3.8+

```bash
# Clone the repository
git clone <repository_url>
cd 2048_RL-master

# Install required dependencies
pip install numpy torch matplotlib
```

---

##  Usage & Training

The main training script is `training_dqn.py`. It is highly configurable via command-line arguments. 

### 1. The Recommended Run (All Features ON)
By default, the script runs the optimal Dueling CNN architecture with Double DQN, PER, 3-step returns, and Potential-Based Reward Shaping enabled.

```bash
python3 training_dqn.py --episodes 50000
```

### 2. The "Hardcore" Run for 2048
If you want to maximize the chance of the AI seeing the 2048 tile, train it for a long duration with a slow epsilon decay:

```bash
python3 training_dqn.py --episodes 200000 --eps-decay 0.9997
```

### 3. Ablation Studies (Turning features off)
You can test the impact of the enhancements by turning them off to run a Vanilla DQN comparison:

```bash
# Pure Vanilla DQN with MLP (Flat Network)
python3 training_dqn.py --network-type mlp --no-double-dqn --no-per --n-step 1 --no-reward-shaping

# Dueling CNN but NO Prioritized Replay and 1-step
python3 training_dqn.py --no-per --n-step 1
```

---

##  Outputs & Monitoring

During training, the console will output progress every `100` episodes, showing the Average Score, Average Max Tile, Epsilon exploration rate, and the percentage of games reaching $\ge256$, $\ge512$, and $\ge1024$ tiles.

**Artifacts Saved (in `./data/` folder):**
1.  **Network Weights:** `dqn_local_*.pth`, `dqn_target_*.pth`
2.  **Training State:** `dqn_optimizer_*.pth`, `dqn_state_*.pkl` (Includes replay data)
3.  **Visualization:** `optimized_dqn_results.png` (A 6-panel graph showing Game Score, Max Tile progression, Loss curve, and Tile Distribution charts).

---

##  Project Structure

*   `game.py`: The 2048 environment logic and log2 raw reward calculation.
*   `agent_dqn.py`: Contains the Replay Buffers (`SumTree`, `PER`, `NStepBuffer`) and the PyTorch Network definitions (`DuelingCNNNetwork`, `QNetwork`, `DQNAgent`).
*   `training_dqn.py`: The primary training loop, epsilon-greedy logic, reward shaping, and Matplotlib plotting.
*   *(Legacy/Original)* `agent.py` & `training.ipynb`: The original reward-regression MLP implementation.
