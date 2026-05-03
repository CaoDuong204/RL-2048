"""
Training Script for Optimized DQN on 2048 Game
================================================
Supports two modes:
  1. Standard Q-learning (default, backward-compatible)
  2. Afterstate V-learning + NoisyNet (--afterstate --noisy-net)

Usage:
    # Default (original Dueling DDQN):
    python training_dqn.py --episodes 50000

    # Afterstate + NoisyNet (recommended for 2048 tile):
    python training_dqn.py --episodes 200000 --afterstate --noisy-net --lr-schedule

    # Vanilla DQN comparison:
    python training_dqn.py --network-type mlp --no-double-dqn --no-per --n-step 1
"""

import numpy as np
from game import Game, transform_state_cnn, transform_state_flat
from agent_dqn import DQNAgent, device
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import pickle
import argparse

# State Transformation functions have been moved to game.py to prevent circular imports.


# =============================================================================
# Reward Shaping (heuristic bonuses for 2048 strategy)
# =============================================================================
def get_potential(board):
    """
    Potential function for Reward Shaping.
    Higher potential means a better board state.
    Coefficients reduced by 50% to prevent agent from over-fitting to heuristics
    instead of learning true game value.
    """
    b = board.reshape(4, 4)
    max_val = b.max()
    potential = 0.0

    # 1. Corner potential (dynamic: +0.25 * log2(max_tile))
    corners = [b[0, 0], b[0, 3], b[3, 0], b[3, 3]]
    if max_val > 0 and max_val == max(corners):
        potential += 0.25 * np.log2(max_val)

    # 2. Empty cell potential (+0.04 per empty cell)
    empty_count = np.sum(b == 0)
    potential += empty_count * 0.04

    # 3. Monotonicity potential (+0.4 per sorted edge)
    log_b = np.log2(np.where(b > 0, b, 1))
    for row in log_b:  # Check ALL rows, not just edges
        diffs = np.diff(row)
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            potential += 0.4
    for col_idx in range(4):  # Check ALL columns
        col = log_b[:, col_idx]
        diffs = np.diff(col)
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            potential += 0.4

    # 4. Smoothness penalty (penalize large differences between adjacent tiles)
    for i in range(4):
        for j in range(3):
            if b[i, j] > 0 and b[i, j+1] > 0:
                potential -= 0.05 * abs(np.log2(b[i, j]) - np.log2(b[i, j+1]))
            if b[j, i] > 0 and b[j+1, i] > 0:
                potential -= 0.05 * abs(np.log2(b[j, i]) - np.log2(b[j+1, i]))

    # 5. Adjacent same-tile potential (snake strategy, halved)
    for i in range(4):
        for j in range(3):
            if b[i, j] > 0 and b[i, j] == b[i, j+1]:
                potential += 0.1 * np.log2(b[i, j])
            if b[j, i] > 0 and b[j, i] == b[j+1, i]:
                potential += 0.1 * np.log2(b[j, i])

    return potential


# =============================================================================
# Main Training Loop
# =============================================================================
def train(n_episodes=50000,
          eps_start=1.0, eps_end=0.05, eps_decay=0.9995,
          # Network
          network_type='dueling_cnn', n_filters=64,
          fc1=512, fc2=512, fc3=256,
          # Hyperparameters
          lr=1e-4, gamma=0.99, tau=1e-3,
          buffer_size=200000, batch_size=256, update_every=4,
          # Enhancements
          double_dqn=True, use_per=True, n_step=3,
          reward_shaping=True,
          # New features
          afterstate=False, noisy_net=False, sigma_init=0.5,
          lr_schedule=False,
          # Logging
          save_every=1000, print_every=100,
          save_name='optimized_dqn',
          # Early stopping
          patience=5, min_episodes=3000):

    # Memory optimization for afterstate's 5D tensors
    if afterstate and buffer_size > 100000:
        buffer_size = 100000

    # --- Environment ---
    env = Game(4, reward_mode='log2', negative_reward=-5, cell_move_penalty=0.1)

    # --- State transform ---
    if network_type == 'dueling_cnn' or afterstate:
        transform_fn = transform_state_cnn
        state_size = 18 * 4 * 4
    else:
        transform_fn = lambda s: transform_state_flat(s, 'one_hot')
        state_size = env.state_size * 18

    # --- Create Agent (Early initialization to get true n_step) ---
    agent = DQNAgent(
        state_size=state_size,
        action_size=env.action_size,
        seed=42,
        network_type=network_type,
        n_filters=n_filters,
        fc1_units=fc1, fc2_units=fc2, fc3_units=fc3,
        lr=lr, gamma=gamma, tau=tau,
        buffer_size=buffer_size, batch_size=batch_size,
        update_every=update_every,
        double_dqn=double_dqn,
        use_per=use_per,
        per_beta_frames=buffer_size, # beta reaches 1.0 after 1 buffer fill cycle
        n_step=n_step,
        afterstate=afterstate,
        noisy_net=noisy_net,
        sigma_init=sigma_init,
    )

    # --- Banner ---
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
    if agent.n_step > 1:
        features.append(f"{agent.n_step}-step")
    if reward_shaping:
        features.append("RewardShaping")
    mode_name = " + ".join(features)

    print("=" * 76)
    print(f"  {mode_name} — TRAINING FOR 2048")
    print("=" * 76)

    # --- LR Scheduler ---
    scheduler = None
    if lr_schedule:
        import torch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.optimizer, T_max=n_episodes, eta_min=1e-5
        )
        print(f"  LR Schedule    : CosineAnnealing → 1e-5")

    print(f"  Episodes        : {n_episodes:,}")
    if not noisy_net:
        print(f"  Epsilon         : {eps_start:.2f} → {eps_end:.4f} (decay: {eps_decay})")
    else:
        print(f"  Epsilon         : DISABLED (NoisyNet)")
    print(f"  Reward shaping  : {'ON' if reward_shaping else 'OFF'}")
    print(f"  Afterstate      : {'ON' if afterstate else 'OFF'}")
    print(f"  Device          : {device}")
    print("=" * 76)

    # --- Metrics ---
    scores = []
    max_tiles = []
    total_rewards = []
    steps_per_episode = []
    tile_distribution = {}
    best_score = 0
    best_max_tile = 0
    eps = eps_start

    # --- Early Stopping ---
    best_eval_metric = 0.0
    patience_counter = 0
    early_stop = False

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    for episode in range(1, n_episodes + 1):
        if early_stop:
            break
        t_start = time.time()
        env.reset(2, 0)
        state = transform_fn(env.current_state())
        total_reward = 0
        steps = 0

        # Safety reset for n-step buffer
        if agent.n_step_buffer is not None:
            agent.n_step_buffer.reset()
            
        if agent.noisy_net:
            agent.qnetwork_local.reset_noise()

        invalid_move_count = 0
        while not env.done:

            # =============================================================
            # ACTION SELECTION
            # =============================================================
            if afterstate:
                # --- Afterstate mode: try all actions, pick best V ---
                valid_actions = []
                afterstate_tensors = []
                for a in range(env.action_size):
                    astate_board, merge_reward, is_valid = env.get_afterstate(a)
                    if is_valid:
                        valid_actions.append((a, merge_reward, astate_board))
                        afterstate_tensors.append(
                            transform_fn(astate_board.flatten()))

                if not valid_actions:
                    break  # No valid moves = game over

                # Evaluate all afterstates
                values = agent.evaluate_batch(
                    np.array(afterstate_tensors))  # (n_valid, 1)

                # NoisyNet provides exploration via noise;
                # otherwise use epsilon-greedy over afterstate values
                if noisy_net or np.random.random() >= eps:
                    best_idx = int(np.argmax(values))
                else:
                    best_idx = int(np.random.randint(len(valid_actions)))

                action, merge_reward_val, chosen_astate = valid_actions[
                    best_idx]
                chosen_astate_encoded = afterstate_tensors[best_idx]

                # Record potential before move
                phi_s = get_potential(
                    env.game_board) if reward_shaping else 0.0

                # Execute action
                env.step(action)
                done = env.done

                # Precompute next afterstates to save CPU time during ReplayBuffer sample
                if afterstate:
                    next_state = np.zeros((4, 18, 4, 4), dtype=np.float32)
                    if not done:
                        for a in range(4):
                            astate, _, is_valid = env.get_afterstate(a)
                            if is_valid:
                                next_state[a] = transform_fn(astate.flatten())
                else:
                    next_state = transform_fn(env.current_state())

                # Reward = merge reward from afterstate computation
                raw_reward = merge_reward_val
                total_reward += raw_reward
                steps += 1

                # PBRS
                phi_next = 0.0 if done else (
                    get_potential(env.game_board) if reward_shaping else 0.0)
                # Death penalty: teach agent that dying is catastrophic
                if done:
                    raw_reward -= 10.0

                if reward_shaping:
                    reward = raw_reward + (gamma * phi_next) - phi_s
                else:
                    reward = raw_reward

                # Store (afterstate, dummy_action, reward, next_afterstates, done)
                agent.step(chosen_astate_encoded, 0, reward, next_state, done)

                # --- Data Augmentation: 3 rotations (90°, 180°, 270°) ---
                # Board 2048 has rotational symmetry: V(board) == V(rot(board))
                # Action permutation: rot k*90° CCW maps action a → action (a+k)%4
                # in the next_states_4 array, so we permute the action indices.
                # Uses store_only() to avoid triggering 4x gradient updates.
                _ROT_PERM = {1: [1, 2, 3, 0], 2: [2, 3, 0, 1], 3: [3, 0, 1, 2]}
                for k in [1, 2, 3]:
                    aug_state = np.rot90(chosen_astate_encoded, k=k, axes=(-2, -1)).copy()
                    perm = _ROT_PERM[k]
                    aug_next = np.zeros_like(next_state)
                    for ai in range(4):
                        aug_next[ai] = np.rot90(next_state[perm[ai]], k=k, axes=(-2, -1))
                    agent.store_only(aug_state, 0, reward, aug_next, done)

                if afterstate:
                    state = transform_fn(env.current_state())
                else:
                    state = next_state

            else:
                # --- Standard Q-learning mode (original logic) ---
                action_values = agent.act(state)

                if noisy_net or np.random.random() >= eps:
                    action = int(np.argmax(action_values[0]))
                else:
                    action = int(np.random.randint(env.action_size))

                phi_s = get_potential(
                    env.game_board) if reward_shaping else 0.0

                env.step(action)
                next_state = transform_fn(env.current_state())
                raw_reward = env.reward
                done = env.done

                if not env.moved:
                    reward = -0.1
                    invalid_move_count += 1
                    if invalid_move_count >= 25:
                        if agent.n_step_buffer is not None:
                            agent.n_step_buffer.reset()
                        done = True
                        reward = -5.0
                else:
                    invalid_move_count = 0
                    total_reward += raw_reward
                    steps += 1
                    phi_next = 0.0 if done else (
                        get_potential(
                            env.game_board) if reward_shaping else 0.0)
                    if reward_shaping:
                        reward = raw_reward + (gamma * phi_next) - phi_s
                    else:
                        reward = raw_reward

                agent.step(state, action, reward, next_state, done)
                state = next_state

            if done:
                break

        t_elapsed = time.time() - t_start

        # --- Record ---
        score = env.score
        max_tile = int(env.game_board.max())
        scores.append(score)
        max_tiles.append(max_tile)
        total_rewards.append(total_reward)
        steps_per_episode.append(steps)
        tile_distribution[max_tile] = tile_distribution.get(max_tile, 0) + 1

        if score > best_score:
            best_score = score
        if max_tile > best_max_tile:
            best_max_tile = max_tile
            agent.save(f'{save_name}_best')
            print(f"    ★ New best tile: {best_max_tile}!")

        # Epsilon decay (only when NOT using NoisyNet)
        if not noisy_net:
            eps = max(eps_end, eps * eps_decay)
        else:
            # NoisyNet sigma decay: gradually reduce exploration
            if hasattr(agent.qnetwork_local, 'decay_noise'):
                agent.qnetwork_local.decay_noise(factor=0.9999)
                agent.qnetwork_target.decay_noise(factor=0.9999)

        # LR scheduler step
        if scheduler is not None and agent.learn_step > 0:
            scheduler.step()

        # Hard target sync removed: using soft update (tau) in _soft_update

        # --- Print ---
        if episode % print_every == 0:
            n = min(print_every, len(scores))
            avg_score = np.mean(scores[-n:])
            avg_tile = np.mean(max_tiles[-n:])
            med_tile = int(np.median(max_tiles[-n:]))
            std_tile = np.std(max_tiles[-n:])
            avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
            recent = max_tiles[-n:]
            p256 = sum(1 for t in recent if t >= 256) / n * 100
            p512 = sum(1 for t in recent if t >= 512) / n * 100
            p1024 = sum(1 for t in recent if t >= 1024) / n * 100
            p2048 = sum(1 for t in recent if t >= 2048) / n * 100
            fail = sum(1 for t in recent if t < 256) / n * 100

            lr_now = agent.optimizer.param_groups[0]['lr']
            print(
                f"  Ep {episode:6,d} | "
                f"Score:{avg_score:7.0f} | "
                f"AvgTile:{avg_tile:5.0f} | Med:{med_tile:5d} | Std:{std_tile:5.0f} | "
                f"≥256:{p256:3.0f}% ≥512:{p512:3.0f}% ≥1K:{p1024:3.0f}% ≥2K:{p2048:3.0f}% | "
                f"Fail<256:{fail:3.0f}% | "
                f"Loss:{avg_loss:.4f} | LR:{lr_now:.2e} | "
                f"{t_elapsed:.2f}s"
            )

        if episode % save_every == 0:
            _save_all(agent, save_name, scores, max_tiles,
                      total_rewards, steps_per_episode, tile_distribution)
            print(f"    → Saved at ep {episode:,d}")

        # --- Greedy Evaluation (no noise) every 1000 episodes ---
        if episode % 1000 == 0:
            eval_tiles = []
            eval_scores_list = []
            was_training = agent.qnetwork_local.training
            agent.qnetwork_local.eval()  # Disables noise in NoisyLinear
            for _ in range(200):
                eval_env = Game(4, reward_mode='log2', negative_reward=-5, cell_move_penalty=0.1)
                eval_env.reset(2, 0)
                while not eval_env.done:
                    if afterstate:
                        va = []
                        at = []
                        for a_ev in range(eval_env.action_size):
                            as_b, _, is_v = eval_env.get_afterstate(a_ev)
                            if is_v:
                                va.append(a_ev)
                                at.append(transform_fn(as_b.flatten()))
                        if not va:
                            break
                        vals = agent.evaluate_batch(np.array(at))
                        best_a = va[int(np.argmax(vals))]
                    else:
                        av = agent.act(transform_fn(eval_env.current_state()))
                        best_a = int(np.argmax(av[0]))
                    eval_env.step(best_a)
                eval_tiles.append(int(eval_env.game_board.max()))
                eval_scores_list.append(eval_env.score)
            agent.qnetwork_local.train(was_training)
            e_avg = np.mean(eval_tiles)
            e_med = int(np.median(eval_tiles))
            e_std = np.std(eval_tiles)
            e512 = sum(1 for t in eval_tiles if t >= 512) / len(eval_tiles) * 100
            e1024 = sum(1 for t in eval_tiles if t >= 1024) / len(eval_tiles) * 100
            e2048 = sum(1 for t in eval_tiles if t >= 2048) / len(eval_tiles) * 100
            e_best = max(eval_tiles)
            avg_eval = np.mean(eval_scores_list)
            # Composite metric: 60% weight on ≥1024, 40% on ≥2048
            eval_metric = e1024 * 0.6 + e2048 * 0.4

            # Early stopping check
            improved = eval_metric > best_eval_metric + 1.0  # min_delta = 1%
            if improved:
                best_eval_metric = eval_metric
                patience_counter = 0
                agent.save(f'{save_name}_eval_best')
                status = "★ NEW BEST"
            else:
                patience_counter += 1
                status = f"no improve ({patience_counter}/{patience})"

            print(f"    ┌─ [EVAL - 200 games, greedy]")
            print(f"    │  Score:{avg_eval:7.0f} | "
                  f"AvgTile:{e_avg:5.0f} | Med:{e_med:5d} | Std:{e_std:5.0f}")
            print(f"    │  ≥512:{e512:3.0f}% | ≥1024:{e1024:3.0f}% | ≥2048:{e2048:3.0f}%")
            print(f"    │  Metric:{eval_metric:5.1f} | {status}")
            print(f"    └─ Best: {e_best}")

            if patience_counter >= patience and episode >= min_episodes:
                print(f"\n  ⛔ EARLY STOP at ep {episode:,d} — no improvement for "
                      f"{patience} consecutive evals ({patience * 1000} episodes)")
                print(f"     Best eval metric: {best_eval_metric:.1f}")
                early_stop = True

    # --- Final ---
    _save_all(agent, save_name, scores, max_tiles,
              total_rewards, steps_per_episode, tile_distribution)
    plot_results(scores, max_tiles, total_rewards, agent.losses,
                 tile_distribution, save_name)

    actual_episodes = len(scores)
    print("=" * 76)
    if early_stop:
        print(f"  TRAINING STOPPED EARLY at ep {actual_episodes:,d}/{n_episodes:,d}")
    else:
        print("  TRAINING COMPLETE")
    print(f"  Best Score: {best_score:,.0f}  |  Best Tile: {best_max_tile}")
    print(f"  Best Eval Metric: {best_eval_metric:.1f}")
    for tile in sorted(tile_distribution.keys()):
        count = tile_distribution[tile]
        pct = count / actual_episodes * 100 if actual_episodes > 0 else 0
        print(f"    {tile:6d}: {count:6d} ({pct:5.1f}%)")
    print("=" * 76)

    return agent, scores, max_tiles, total_rewards


# =============================================================================
# Helpers
# =============================================================================
def _save_all(agent, name, scores, max_tiles, rewards, steps, tile_dist):
    agent.save(name)
    os.makedirs('./data/', exist_ok=True)
    with open(f'./data/metrics_{name}.pkl', 'wb') as f:
        pickle.dump({
            'scores': scores, 'max_tiles': max_tiles,
            'total_rewards': rewards, 'steps_per_episode': steps,
            'losses': agent.losses.copy(), 'tile_distribution': tile_dist,
        }, f)


def plot_results(scores, max_tiles, total_rewards, losses,
                 tile_dist, save_name, window=200):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Optimized DQN Training — 2048', fontsize=14, fontweight='bold')

    def _plot(ax, data, title, ylabel, color, w=window):
        ax.plot(data, alpha=0.12, color=color)
        if len(data) >= w:
            ma = np.convolve(data, np.ones(w) / w, mode='valid')
            ax.plot(range(w - 1, len(data)), ma, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], scores, 'Score', 'Score', '#2196F3')
    _plot(axes[0, 1], max_tiles, 'Max Tile', 'Tile', '#4CAF50')
    _plot(axes[0, 2], total_rewards, 'Total Reward', 'Reward', '#FF5722')

    # Loss
    ax = axes[1, 0]
    if losses:
        ax.plot(losses, alpha=0.1, color='#9C27B0')
        wl = 2000
        if len(losses) >= wl:
            ma = np.convolve(losses, np.ones(wl) / wl, mode='valid')
            ax.plot(range(wl - 1, len(losses)), ma, color='#9C27B0', linewidth=2)
        ax.set_yscale('log')
    ax.set_title('Loss (log)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    # Tile distribution
    ax = axes[1, 1]
    if tile_dist:
        tiles = sorted(tile_dist.keys())
        total = sum(tile_dist.values())
        pcts = [tile_dist[t] / total * 100 for t in tiles]
        colors = ['#E0E0E0' if t < 256 else '#FFD54F' if t < 512
                  else '#FF9800' if t < 1024 else '#F44336' if t < 2048
                  else '#4CAF50' for t in tiles]
        bars = ax.bar([str(t) for t in tiles], pcts, color=colors,
                      edgecolor='#333', linewidth=0.5)
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_title('Tile Distribution')
    ax.set_xlabel('Tile')
    ax.set_ylabel('% Games')
    ax.grid(True, alpha=0.3, axis='y')

    # Rolling avg max tile
    ax = axes[1, 2]
    if max_tiles:
        rolling = [np.mean(max_tiles[max(0, i - window):i + 1])
                   for i in range(len(max_tiles))]
        ax.plot(rolling, color='#009688', linewidth=1.5)
    ax.set_title('Rolling Avg Max Tile')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Tile')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'./data/{save_name}_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Train Optimized DQN for 2048',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Training
    p.add_argument('--episodes', type=int, default=50000)
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay', type=float, default=0.9995)

    # Network
    p.add_argument('--network-type', default='dueling_cnn',
                   choices=['dueling_cnn', 'mlp'])
    p.add_argument('--n-filters', type=int, default=64)
    p.add_argument('--fc1', type=int, default=512)
    p.add_argument('--fc2', type=int, default=512)
    p.add_argument('--fc3', type=int, default=256)

    # Hyperparameters
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=1e-3)
    p.add_argument('--buffer-size', type=int, default=200000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--update-every', type=int, default=4)

    # Enhancements
    p.add_argument('--double-dqn', action='store_true', default=True)
    p.add_argument('--no-double-dqn', action='store_true')
    p.add_argument('--per', action='store_true', default=True,
                   help='Use Prioritized Experience Replay')
    p.add_argument('--no-per', action='store_true')
    p.add_argument('--n-step', type=int, default=3,
                   help='N-step returns (1=standard, 3=recommended)')
    p.add_argument('--reward-shaping', action='store_true', default=True)
    p.add_argument('--no-reward-shaping', action='store_true')

    # New features
    p.add_argument('--afterstate', action='store_true', default=False,
                   help='Use afterstate V-learning (recommended for 2048)')
    p.add_argument('--noisy-net', action='store_true', default=False,
                   help='Use NoisyNet for exploration (replaces epsilon)')
    p.add_argument('--sigma-init', type=float, default=0.5,
                   help='NoisyNet initial sigma')
    p.add_argument('--lr-schedule', action='store_true', default=False,
                   help='Use cosine annealing LR scheduler')

    # Logging
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--print-every', type=int, default=100)
    p.add_argument('--save-name', type=str, default='optimized_dqn')

    # Early stopping
    p.add_argument('--patience', type=int, default=5,
                   help='Stop if no EVAL improvement for N consecutive checks (each 1000 ep)')
    p.add_argument('--min-episodes', type=int, default=3000,
                   help='Minimum episodes before early stopping can trigger')

    args = p.parse_args()

    train(
        n_episodes=args.episodes,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        network_type=args.network_type,
        n_filters=args.n_filters,
        fc1=args.fc1, fc2=args.fc2, fc3=args.fc3,
        lr=args.lr, gamma=args.gamma, tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_every=args.update_every,
        double_dqn=args.double_dqn and not args.no_double_dqn,
        use_per=args.per and not args.no_per,
        n_step=args.n_step,
        reward_shaping=args.reward_shaping and not args.no_reward_shaping,
        afterstate=args.afterstate,
        noisy_net=args.noisy_net,
        sigma_init=args.sigma_init,
        lr_schedule=args.lr_schedule,
        save_every=args.save_every,
        print_every=args.print_every,
        save_name=args.save_name,
        patience=args.patience,
        min_episodes=args.min_episodes,
    )
