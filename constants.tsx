import { PythonFile } from './types';

export const PYTHON_CODEBASE: PythonFile[] = [
  {
    path: 'README.md',
    name: 'README.md',
    language: 'markdown',
    /* Escape backticks within the template literal to prevent syntax errors and incorrect parsing */
    content: `# 🚦 traffic-rl: Deep Reinforcement Learning for Traffic Flow

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

Autonomous Traffic Signal Control (ATSC) using Deep Q-Networks. This project implements a sophisticated 4-way intersection simulation where an agent learns to optimize signal phases based on real-time vehicle kinematics.

## ⚙️ Setup Guidelines

### 1. Prerequisites
- Python 3.8 or 3.9
- \`pip\` (Python package manager)

### 2. Environment Setup
It is recommended to use a virtual environment:
\`\`\`bash
# Create venv
python -m venv venv

# Activate venv
# Linux/macOS:
source venv/bin/activate
# Windows:
.\\\\venv\\\\Scripts\\\\activate
\`\`\`

### 3. Installation
Install core requirements for training and visualization:
\`\`\`bash
pip install torch numpy matplotlib pytest
# OR
pip install -r requirements.txt
\`\`\`

## 🚀 Running the Project

### Training
Start the DQN training loop. Use \`--render\` to see the agent's decisions via Matplotlib:
\`\`\`bash
python train.py --episodes 100 --render
\`\`\`

### Evaluation
Benchmark a saved model (\`models/dqn_checkpoint.pth\`):
\`\`\`bash
python evaluate.py --checkpoint models/dqn_checkpoint.pth --render
\`\`\`

### Testing
Verify the environment physics and observation spaces:
\`\`\`bash
pytest tests/test_env.py
\`\`\`

## 🧠 Architecture Overview
- **Kinetic State Space**: 23 features including lane-wise momentum ($m \\cdot v$), density, and priority flags.
- **Reward Shaping**:
  - Positive: Throughput weight (mass cleared).
  - Negative: Average wait time penalty.
  - Negative: Massive penalty for collision ($r = -200$).
- **Action Masking**: Prevents rapid toggling by enforcing a minimum phase time (e.g., 10 steps).

## 📂 Repository Layout
- \`envs/\`: TrafficEnv implementation (gym-like).
- \`agents/\`: DQN Network and Agent logic.
- \`utils/\`: Visualization engine and metrics plotting.
- \`models/\`: Saved model weights.
- \`tests/\`: Automated test suite.`
  },
  {
    path: 'requirements.txt',
    name: 'requirements.txt',
    language: 'text',
    content: `numpy\ntorch\nmatplotlib\npytest`
  },
  {
    path: 'envs/traffic_env.py',
    name: 'traffic_env.py',
    language: 'python',
    content: `import numpy as np\n\nclass TrafficEnv:\n    VEHICLE_TYPES = {\n        "car":   {"weight": 1.0, "v_max": 1.0, "accel": 0.25, "clear_t": 2},\n        "truck": {"weight": 4.5, "v_max": 0.5, "accel": 0.06, "clear_t": 6},\n        "bike":  {"weight": 0.3, "v_max": 1.5, "accel": 0.50, "clear_t": 1}\n    }\n    \n    def __init__(self, arrival_rates=None):\n        self.n_lanes = 4\n        self.arrival_rates = arrival_rates if arrival_rates else [0.22, 0.22, 0.12, 0.12]\n        self.min_phase_duration = 10\n        self.reset_all()\n\n    def reset_all(self):\n        self.queues = [[] for _ in range(self.n_lanes)]\n        self.phase = 0\n        self.steps_in_phase = 0\n        self.steps = 0\n        self.ambulance_lane = -1\n        self.intersection_occupancy = 0 \n        self.occupancy_phase = -1 \n        self.collision = False\n\n    def reset(self):\n        self.reset_all()\n        return self._get_obs(), {"action_mask": self.get_action_mask()}\n\n    def _get_obs(self):\n        lane_counts = np.array([len(q) for q in self.queues], dtype=np.float32)\n        lane_weights = np.array([sum(v['weight'] for v in q) for q in self.queues], dtype=np.float32)\n        avg_speeds = np.array([np.mean([v['v'] for v in q]) if q else 1.0 for q in self.queues], dtype=np.float32)\n        lane_momentum = np.array([sum(v['weight'] * v['v'] for v in q) if q else 0.0 for q in self.queues], dtype=np.float32)\n        lane_urgency = np.array([(q[0]['weight'] * q[0]['wait_steps'] / 50.0) if q else 0.0 for q in self.queues], dtype=np.float32)\n        amb_flag = 1.0 if self.ambulance_lane != -1 else 0.0\n        return np.concatenate([lane_counts, lane_weights, avg_speeds, lane_momentum, lane_urgency, [float(self.phase)], [amb_flag], [float(self.intersection_occupancy / 10.0)]])\n\n    def get_action_mask(self):\n        mask = np.ones(2, dtype=np.float32)\n        if self.steps_in_phase < self.min_phase_duration: mask[1] = 0.0\n        return mask\n\n    def step(self, action):\n        mask = self.get_action_mask()\n        if mask[action] == 0: action = 0\n\n        if action == 1:\n            self.phase = 1 - self.phase\n            self.steps_in_phase = 0\n        else:\n            self.steps_in_phase += 1\n\n        for i, rate in enumerate(self.arrival_rates):\n            if np.random.random() < rate:\n                vtype = np.random.choice(list(self.VEHICLE_TYPES.keys()), p=[0.7, 0.2, 0.1])\n                cfg = self.VEHICLE_TYPES[vtype]\n                self.queues[i].append({\n                    "weight": cfg["weight"], "v": 0.0, "v_max": cfg["v_max"], \n                    "a": cfg["accel"], "clear_t": cfg["clear_t"], "wait_steps": 0, "type": vtype\n                })\n\n        green_lanes = [0, 1] if self.phase == 0 else [2, 3]\n        total_reward = 0.0\n        if self.intersection_occupancy > 0: self.intersection_occupancy -= 1\n        \n        for i in range(self.n_lanes):\n            if i in green_lanes:\n                for j, v in enumerate(self.queues[i]):\n                    if j < 4: v['v'] = min(v['v_max'], v['v'] + v['a'])\n                    if v['v'] < 0.1: v['wait_steps'] += 1\n                if self.queues[i] and self.queues[i][0]['v'] >= (self.queues[i][0]['v_max'] * 0.6):\n                    if self.intersection_occupancy > 0 and self.occupancy_phase != self.phase: self.collision = True\n                    departed = self.queues[i].pop(0)\n                    total_reward += departed['weight'] * 5.0\n                    self.intersection_occupancy = departed['clear_t']\n                    self.occupancy_phase = self.phase\n            else:\n                for v in self.queues[i]:\n                    v['v'] = max(0.0, v['v'] - 0.8)\n                    v['wait_steps'] += 1\n        \n        if np.random.random() < 0.015 and self.ambulance_lane == -1:\n            self.ambulance_lane = np.random.randint(0, 4)\n            self.queues[self.ambulance_lane].insert(0, {"weight": 10.0, "v": 0.5, "v_max": 2.0, "a": 0.5, "clear_t": 2, "wait_steps": 0, "type": "ambulance"})\n        elif self.ambulance_lane != -1:\n            if not self.queues[self.ambulance_lane] or self.queues[self.ambulance_lane][0].get('type') != 'ambulance':\n                self.ambulance_lane = -1\n\n        self.steps += 1\n        reward = total_reward - (200.0 if self.collision else 0.0)\n        done = (self.steps >= 150) or self.collision\n        \n        info = {\n            "action_mask": self.get_action_mask(),\n            "collision": self.collision\n        }\n        return self._get_obs(), float(reward), done, info`
  },
  {
    path: 'agents/dqn_agent.py',
    name: 'dqn_agent.py',
    language: 'python',
    content: `import torch\nimport torch.nn as nn\nimport random\nimport numpy as np\n\nclass DQN(nn.Module):\n    def __init__(self, in_dim, out_dim):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(in_dim, 256), nn.ReLU(),\n            nn.Linear(256, 128), nn.ReLU(),\n            nn.Linear(128, out_dim)\n        )\n    def forward(self, x): return self.net(x)\n\nclass DQNAgent:\n    def __init__(self, state_dim, action_dim):\n        self.model = DQN(state_dim, action_dim)\n        self.epsilon = 1.0\n        self.epsilon_min = 0.05\n        self.epsilon_decay = 0.99\n\n    def act(self, state, action_mask=None):\n        if np.random.rand() <= self.epsilon:\n            valid_actions = np.where(action_mask == 1)[0] if action_mask is not None else [0, 1]\n            return random.choice(valid_actions)\n        \n        state_t = torch.FloatTensor(state).unsqueeze(0)\n        with torch.no_grad():\n            q_values = self.model(state_t).numpy()[0]\n        \n        if action_mask is not None:\n            q_values[action_mask == 0] = -1e10\n            \n        return np.argmax(q_values)\n\n    def train_step(self):\n        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)\n\n    def save(self, path): torch.save(self.model.state_dict(), path)\n    def load(self, path): self.model.load_state_dict(torch.load(path))`
  },
  {
    path: 'train.py',
    name: 'train.py',
    language: 'python',
    content: `import argparse\nimport os\nfrom envs.traffic_env import TrafficEnv\nfrom agents.dqn_agent import DQNAgent\nfrom utils.visualize import render_env\n\ndef main(episodes, render):\n    os.makedirs('models', exist_ok=True)\n    env = TrafficEnv()\n    agent = DQNAgent(23, 2)\n    \n    for e in range(episodes):\n        state, info = env.reset()\n        mask = info['action_mask']\n        total_reward = 0\n        while True:\n            action = agent.act(state, action_mask=mask)\n            next_state, reward, done, info = env.step(action)\n            state = next_state\n            mask = info['action_mask']\n            total_reward += reward\n            \n            if render:\n                render_env(env, episode=e+1, step=env.steps, reward=total_reward)\n                \n            if done: break\n        \n        agent.train_step()\n        if (e + 1) % 10 == 0:\n            print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.2f}")\n    \n    agent.save('models/dqn_checkpoint.pth')\n    print("Training complete. Model saved.")\n\nif __name__ == '__main__':\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--episodes', type=int, default=50)\n    parser.add_argument('--render', action='store_true', help='Render simulation')\n    args = parser.parse_args()\n    main(args.episodes, args.render)`
  },
  {
    path: 'evaluate.py',
    name: 'evaluate.py',
    language: 'python',
    content: `import argparse\nfrom envs.traffic_env import TrafficEnv\nfrom agents.dqn_agent import DQNAgent\nfrom utils.visualize import render_env\n\ndef evaluate(checkpoint, render):\n    env = TrafficEnv()\n    agent = DQNAgent(23, 2)\n    agent.load(checkpoint)\n    agent.epsilon = 0.0\n\n    for i in range(5):\n        state, info = env.reset()\n        mask = info['action_mask']\n        total_reward = 0\n        while True:\n            action = agent.act(state, action_mask=mask)\n            state, reward, done, info = env.step(action)\n            mask = info['action_mask']\n            total_reward += reward\n            \n            if render:\n                render_env(env, episode=i+1, step=env.steps, reward=total_reward)\n                \n            if done: break\n        print(f"Eval Episode {i+1} | Reward: {total_reward:.2f}")\n\nif __name__ == '__main__':\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--checkpoint', type=str, required=True)\n    parser.add_argument('--render', action='store_true', help='Render simulation')\n    args = parser.parse_args()\n    evaluate(args.checkpoint, args.render)`
  },
  {
    path: 'utils/visualize.py',
    name: 'visualize.py',
    language: 'python',
    content: `import matplotlib.pyplot as plt\nimport matplotlib.patches as patches\nimport numpy as np\n\n_fig = None\n_ax = None\n\ndef render_env(env, episode=0, step=0, reward=0):\n    global _fig, _ax\n    if _fig is None:\n        plt.ion()\n        _fig, _ax = plt.subplots(figsize=(6, 6))\n        _ax.set_facecolor('#1e293b')\n\n    _ax.clear()\n    _ax.set_xlim(0, 500)\n    _ax.set_ylim(0, 500)\n    _ax.set_aspect('equal')\n    _ax.axis('off')\n\n    # Roads\n    _ax.add_patch(patches.Rectangle((0, 195), 500, 110, color='#334155'))\n    _ax.add_patch(patches.Rectangle((195, 0), 110, 500, color='#334155'))\n    \n    # Lanes markers\n    _ax.plot([0, 500], [250, 250], color='#475569', linestyle='--', linewidth=1)\n    _ax.plot([250, 250], [0, 500], color='#475569', linestyle='--', linewidth=1)\n\n    # Signals\n    ns_color = '#00ff9d' if env.phase == 0 else '#ff4d4d'\n    ew_color = '#00ff9d' if env.phase == 1 else '#ff4d4d'\n    \n    # NS Signals\n    _ax.add_patch(patches.Circle((225, 315), 10, color=ns_color))\n    _ax.add_patch(patches.Circle((275, 185), 10, color=ns_color))\n    # EW Signals\n    _ax.add_patch(patches.Circle((315, 275), 10, color=ew_color))\n    _ax.add_patch(patches.Circle((185, 225), 10, color=ew_color))\n\n    # Vehicles\n    for i, queue in enumerate(env.queues):\n        for j, v in enumerate(queue[:15]):\n            color = '#3b82f6' if v.get('type') == 'car' else '#f59e0b' if v.get('type') == 'truck' else '#ff0000' if v.get('type') == 'ambulance' else '#10b981'\n            if i == 0: # North\n                x, y = 225, 300 + j*18\n            elif i == 1: # South\n                x, y = 275, 200 - j*18\n            elif i == 2: # East\n                x, y = 300 + j*18, 275\n            else: # West\n                x, y = 200 - j*18, 225\n            _ax.add_patch(patches.Rectangle((x-6, y-6), 12, 12, color=color, alpha=0.8))\n\n    _ax.text(10, 480, f"EPISODE: {episode}", color='white', fontweight='bold', fontfamily='monospace')\n    _ax.text(10, 460, f"STEP: {step}", color='white', fontweight='bold', fontfamily='monospace')\n    _ax.text(10, 440, f"REWARD: {reward:.1f}", color='white', fontweight='bold', fontfamily='monospace')\n    \n    if env.collision:\n        _ax.text(250, 250, "COLLISION!", color='red', fontsize=20, fontweight='black', ha='center', va='center')\n\n    plt.draw()\n    plt.pause(0.01)`
  },
  {
    path: 'tests/test_env.py',
    name: 'test_env.py',
    language: 'python',
    content: `import pytest\nimport numpy as np\nfrom envs.traffic_env import TrafficEnv\n\ndef test_env_reset():\n    env = TrafficEnv()\n    obs, info = env.reset()\n    assert obs.shape == (23,)\n    assert "action_mask" in info\n    assert env.steps == 0\n\ndef test_env_step():\n    env = TrafficEnv()\n    obs, info = env.reset()\n    mask = info["action_mask"]\n    action = 1 if mask[1] == 1.0 else 0\n    next_obs, reward, done, next_info = env.step(action)\n    assert next_obs.shape == (23,)\n    assert "action_mask" in next_info\n    assert isinstance(reward, float)\n    assert isinstance(done, bool)`
  },
  {
    path: 'utils/plot.py',
    name: 'plot.py',
    language: 'python',
    content: `import matplotlib.pyplot as plt\n\ndef plot_learning_curve(data, title="DQN Training Progress"):\n    plt.figure(figsize=(10, 5))\n    plt.plot(data)\n    plt.title(title)\n    plt.xlabel("Episode")\n    plt.ylabel("Reward")\n    plt.grid(True)\n    plt.show()`
  }
];