# SARSA vs Q-Learning — Step-by-step (CliffWalking-v1)

This document contains two runnable Python scripts (formatted for Jupyter/Colab) that compare **SARSA** and **Q-Learning** on OpenAI Gym's `CliffWalking-v1` environment. Each section explains the code and RL concepts used, step by step. Use this as a reference for GitHub or a portfolio README.

---

## Table of Contents

1. Overview
2. Prerequisites
3. Code: Visual comparison (short episodes)
4. Line-by-line explanation (visual script)
5. Code: Faster comparison (longer training + final graphs)
6. Line-by-line explanation (faster script)
7. How to run (Colab / local)
8. Extensions & improvements
9. Appendix: Key RL concepts

---

## 1. Overview

- **Goal:** Compare on-policy SARSA and off-policy Q-Learning on the CliffWalking environment and visualize their behavior side-by-side.
- **Environment:** `CliffWalking-v1` — gridworld where stepping on cliff gives a large negative reward; goal is top-right corner from bottom-left.
- **Why compare?** SARSA learns values for the policy actually being followed (epsilon-greedy behavior) while Q-Learning learns the optimal action-values (greedy w.r.t. Q). In risky environments like the cliff, behaviour can differ substantially.

---

## 2. Prerequisites

Install required packages (Colab already has most):

```bash
pip install gymnasium[box2d] matplotlib
```

> Use a Python 3.8+ environment. If using Colab, pick a Python 3 runtime and install `gymnasium` as needed.

---

## 3. Code — Visual comparison (short episodes)

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
import time

# ---------------- CREATE TWO ENVIRONMENTS ----------------
env_sarsa = gym.make('CliffWalking-v1', render_mode="rgb_array")
env_q = gym.make('CliffWalking-v1', render_mode="rgb_array")

# ---------------- CONSTANTS ----------------
EPISODES = 50
EPSILON_SARSA = 0.01
ALPHA_SARSA = 0.5
EPSILON_Q = 0.1
ALPHA_Q = 0.85
GAMMA = 0.99

# ---------------- Q-TABLES ----------------
Q_sarsa = np.random.rand(env_sarsa.observation_space.n, env_sarsa.action_space.n)
Q_sarsa[env_sarsa.observation_space.n-1] = np.zeros(env_sarsa.action_space.n)  # terminal state

Q_q = np.zeros((env_q.observation_space.n, env_q.action_space.n))

# ---------------- EPSILON-GREEDY FUNCTION ----------------
def epsilon_greedy(Q, state, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        return np.argmax(Q[state])

# ---------------- MAIN LOOP ----------------
for episode in range(EPISODES):
    obs_sarsa, _ = env_sarsa.reset()
    obs_q, _ = env_q.reset()

    done_sarsa = False
    done_q = False

    step_sarsa = 0
    step_q = 0
    rewards_sarsa = 0
    rewards_q = 0

    # Initialize SARSA first action
    action_sarsa = epsilon_greedy(Q_sarsa, obs_sarsa, EPSILON_SARSA, env_sarsa.action_space)

    while not (done_sarsa and done_q):
        # ---------------- SARSA STEP ----------------
        if not done_sarsa:
            next_obs_sarsa, reward_sarsa, terminated_sarsa, truncated_sarsa, _ = env_sarsa.step(action_sarsa)
            done_sarsa = terminated_sarsa or truncated_sarsa
            next_action_sarsa = epsilon_greedy(Q_sarsa, next_obs_sarsa, EPSILON_SARSA, env_sarsa.action_space)
            Q_sarsa[obs_sarsa][action_sarsa] += ALPHA_SARSA * (
                reward_sarsa + GAMMA * Q_sarsa[next_obs_sarsa][next_action_sarsa] - Q_sarsa[obs_sarsa][action_sarsa]
            )
            obs_sarsa, action_sarsa = next_obs_sarsa, next_action_sarsa
            step_sarsa += 1
            rewards_sarsa += reward_sarsa

        # ---------------- Q-LEARNING STEP ----------------
        if not done_q:
            action_q = epsilon_greedy(Q_q, obs_q, EPSILON_Q, env_q.action_space)
            next_obs_q, reward_q, terminated_q, truncated_q, _ = env_q.step(action_q)
            done_q = terminated_q or truncated_q
            Q_q[obs_q][action_q] += ALPHA_Q * (
                reward_q + GAMMA * np.max(Q_q[next_obs_q]) - Q_q[obs_q][action_q]
            )
            obs_q = next_obs_q
            step_q += 1
            rewards_q += reward_q

        # ---------------- RENDER BOTH WITH INFO ----------------
        frame_sarsa = env_sarsa.render()
        frame_q = env_q.render()

        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(frame_sarsa)
        axes[0].set_title(f"SARSA\nEpisode: {episode} | Step: {step_sarsa} | Rewards: {rewards_sarsa}")
        axes[0].axis('off')

        axes[1].imshow(frame_q)
        axes[1].set_title(f"Q-Learning\nEpisode: {episode} | Step: {step_q} | Rewards: {rewards_q}")
        axes[1].axis('off')

        plt.show()
        time.sleep(1)  # 1-second delay per step
```

---

## 4. Explanation — Visual comparison script (line-by-line)

**Top imports**
- `gymnasium` for the environment. `render_mode='rgb_array'` returns frames as images that matplotlib can show.
- `numpy` numerical arrays, `matplotlib` to visualize frames.
- `clear_output` to update the visual frames in Jupyter/Colab.

**Two environments**
- We create two separate `CliffWalking-v1` environments (`env_sarsa` and `env_q`) so both agents run on the same problem independently and can be rendered side-by-side.

**Constants**
- `EPISODES` — number of episodes to run.
- `EPSILON_SARSA`, `EPSILON_Q` — exploration rates used for epsilon-greedy action selection.
- `ALPHA_SARSA`, `ALPHA_Q` — learning rates.
- `GAMMA` — discount factor for future rewards.

**Q-tables**
- `Q_sarsa` initialized randomly — helps create initial exploration variety.
- `Q_q` initialized to zeros.
- Terminal state (often the goal index) is set to zero vector for SARSA table (optional but common to anchor terminal values).

**Epsilon-greedy function**
- With probability `epsilon`, choose a random action; otherwise choose the greedy action `argmax(Q[state])`.

**Main loop**
- Each episode resets both environments and variables used to track steps and rewards.
- SARSA: On-policy update uses the next action (`next_action_sarsa`) chosen by the current policy (epsilon-greedy). Update rule:
  ```
  Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
  ```
- Q-Learning: Off-policy update uses the maximum over next state's actions `max_a' Q(s', a')`:
  ```
  Q(s,a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s,a))
  ```
- Rendering loop: We render both environments and display them side by side with episode/step/reward stats.

**Behavioral difference to look for**
- In CliffWalking (risky environment), SARSA often learns safer paths because it accounts for the epsilon-greedy policy (it knows it can take random actions). Q-Learning tends to be more optimistic and may learn risky, shorter paths that assume greedy execution.

---

## 5. Code — Faster comparison (longer training + final graphs)

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random

# ---------------- ENV ----------------
env_sarsa = gym.make('CliffWalking-v1', render_mode="rgb_array")
env_q = gym.make('CliffWalking-v1', render_mode="rgb_array")

# ---------------- CONSTANTS ----------------
TOTAL_EPISODES = 200
EPSILON_SARSA = 0.01
ALPHA_SARSA = 0.5
EPSILON_Q = 0.1
ALPHA_Q = 0.85
GAMMA = 0.99

# ---------------- Q-TABLES ----------------
Q_sarsa = np.random.rand(env_sarsa.observation_space.n, env_sarsa.action_space.n)
Q_sarsa[env_sarsa.observation_space.n-1] = np.zeros(env_sarsa.action_space.n)
Q_q = np.zeros((env_q.observation_space.n, env_q.action_space.n))

# ---------------- EPSILON-GREEDY FUNCTION ----------------
def epsilon_greedy(Q, state, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        return np.argmax(Q[state])

# ---------------- TRACK PERFORMANCE ----------------
rewards_sarsa_list = []
steps_sarsa_list = []
rewards_q_list = []
steps_q_list = []

# ---------------- RUN ALL EPISODES ----------------
for episode in range(1, TOTAL_EPISODES + 1):
    obs_sarsa, _ = env_sarsa.reset()
    obs_q, _ = env_q.reset()
    done_sarsa = False
    done_q = False

    step_sarsa = 0
    step_q = 0
    rewards_sarsa = 0
    rewards_q = 0

    action_sarsa = epsilon_greedy(Q_sarsa, obs_sarsa, EPSILON_SARSA, env_sarsa.action_space)

    while not (done_sarsa and done_q):
        # SARSA STEP
        if not done_sarsa:
            next_obs_sarsa, reward_sarsa, terminated_sarsa, truncated_sarsa, _ = env_sarsa.step(action_sarsa)
            done_sarsa = terminated_sarsa or truncated_sarsa
            next_action_sarsa = epsilon_greedy(Q_sarsa, next_obs_sarsa, EPSILON_SARSA, env_sarsa.action_space)
            Q_sarsa[obs_sarsa][action_sarsa] += ALPHA_SARSA * (
                reward_sarsa + GAMMA * Q_sarsa[next_obs_sarsa][next_action_sarsa] - Q_sarsa[obs_sarsa][action_sarsa]
            )
            obs_sarsa, action_sarsa = next_obs_sarsa, next_action_sarsa
            rewards_sarsa += reward_sarsa
            step_sarsa += 1

        # Q-LEARNING STEP
        if not done_q:
            action_q = epsilon_greedy(Q_q, obs_q, EPSILON_Q, env_q.action_space)
            next_obs_q, reward_q, terminated_q, truncated_q, _ = env_q.step(action_q)
            done_q = terminated_q or truncated_q
            Q_q[obs_q][action_q] += ALPHA_Q * (
                reward_q + GAMMA * np.max(Q_q[next_obs_q]) - Q_q[obs_q][action_q]
            )
            obs_q = next_obs_q
            rewards_q += reward_q
            step_q += 1

        # ---------------- RENDER ----------------
        frame_sarsa = env_sarsa.render()
        frame_q = env_q.render()
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(frame_sarsa)
        axes[0].set_title(f"SARSA\nEpisode: {episode} | Step: {step_sarsa} | Rewards: {rewards_sarsa}")
        axes[0].axis('off')

        axes[1].imshow(frame_q)
        axes[1].set_title(f"Q-Learning\nEpisode: {episode} | Step: {step_q} | Rewards: {rewards_q}")
        axes[1].axis('off')

        plt.show()

    # Save performance stats
    rewards_sarsa_list.append(rewards_sarsa)
    steps_sarsa_list.append(step_sarsa)
    rewards_q_list.append(rewards_q)
    steps_q_list.append(step_q)

# ---------------- FINAL PERFORMANCE GRAPH ----------------
episodes = np.arange(1, TOTAL_EPISODES + 1)
plt.figure(figsize=(14, 6))

# Reward comparison
plt.subplot(1, 2, 1)
plt.plot(episodes, rewards_sarsa_list, label='SARSA')
plt.plot(episodes, rewards_q_list, label='Q-Learning')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SARSA vs Q-Learning: Total Reward per Episode")
plt.legend()
plt.grid(True)

# Step comparison
plt.subplot(1, 2, 2)
plt.plot(episodes, steps_sarsa_list, label='SARSA')
plt.plot(episodes, steps_q_list, label='Q-Learning')
plt.xlabel("Episode")
plt.ylabel("Steps Taken")
plt.title("SARSA vs Q-Learning: Steps per Episode")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 6. Explanation — Faster script (line-by-line)

The faster script shares the same components as the visual version, with these differences:

- **TOTAL_EPISODES** is larger (200) so you can see learning trends across many episodes.
- Performance metrics (`rewards_*_list`, `steps_*_list`) are stored each episode and plotted at the end to compare learning curves.
- No artificial `time.sleep` delays are used — the script focuses on training speed and collects stats.

**Important details:**
- The SARSA update uses `Q(s', a')` where `a'` is chosen by the current policy (on-policy).
- Q-Learning update uses `max_a' Q(s',a')` independent of the current action choice (off-policy).
- Because CliffWalking is risky (stepping off cliff is catastrophic), SARSA may prefer safer routes under epsilon exploration.

---

## 7. How to run (Colab / local)

**Colab**
1. Open Google Colab and create a new notebook.
2. Install gymnasium if missing: `!pip install gymnasium`.
3. Copy code cells into the notebook. Use small `TOTAL_EPISODES` first to test.
4. If rendering shows blank images, ensure `render_mode='rgb_array'` and display with `plt.imshow`.

**Local**
1. Create a virtualenv and `pip install gymnasium matplotlib`.
2. Run in Jupyter or a Python script. If running as a script, remove `clear_output` and use saving frames instead.

---

## 8. Extensions & improvements (ideas for portfolio)

- **Decay epsilon** with episodes (e.g., `epsilon = max(0.01, 1 - episode/1000)`) to encourage exploration early and exploitation later.
- **Track moving-average rewards** instead of raw episode rewards to smooth plots.
- **Use optimistic initial values** to encourage exploration.
- **Compare on other environments** (e.g., `FrozenLake-v1` or custom gridworlds) to show generality.
- **Evaluate deterministic policy** after training (set `epsilon=0`) and compare final paths.
- **Add seed control** to make experiments reproducible: `env.reset(seed=...)`, `np.random.seed(...)`, `random.seed(...)`.
- **Log & save Q-tables** and visualize greedy policy arrows on the grid.

---

## 9. Appendix: Key RL concepts (brief)

- **SARSA (State-Action-Reward-State-Action):** on-policy TD control algorithm. Update uses the action actually taken in the next state.
- **Q-Learning:** off-policy TD control algorithm. Update uses the best possible next action value (max), regardless of policy.
- **Epsilon-greedy:** exploration strategy that picks a random action with probability epsilon.
- **Learning rate (alpha):** how much new information overrides old value.
- **Discount factor (gamma):** preference for immediate vs delayed rewards.

---

### If you want this as a single downloadable `.md` file for GitHub
Tell me and I will export this document and provide a download link (or directly create a file in your workspace).

