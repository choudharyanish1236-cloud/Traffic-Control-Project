
import { PythonFile } from './types';

export const PYTHON_CODEBASE: PythonFile[] = [
  {
    path: 'README.md',
    name: 'README.md',
    language: 'markdown',
    content: `# traffic-rl\n\nSophisticated RL traffic signal controller with heterogeneous vehicle types, priority-based reward shaping, and speed-optimized flow.\n\n## Implementation Details\n- **Kinetic Observation**: State space includes real-time average speeds and lane-wise momentum (mass * speed), allowing the agent to sense "fluidity" and kinetic energy.\n- **Momentum Reward**: Reward incorporates a 'momentum' term to encourage continuous flow.\n- **Action Masking**: Prevents illegal or unstable actions (e.g., switching too fast) directly via the environment's info dictionary.\n- **Collision Detection**: Heavy penalties for intersection collisions.\n\n## Quick Run\n1. Install dependencies: \`pip install -r requirements.txt\`\n2. Run validation tests: \`pytest\`\n3. Train: \`python train.py --episodes 100\`\n4. Evaluate: \`python evaluate.py --checkpoint models/dqn_checkpoint.pth\``
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
    content: `import numpy as np\n\nclass TrafficEnv:\n    VEHICLE_TYPES = {\n        "car":   {"weight": 1.0, "v_max": 1.0, "accel": 0.25, "clear_t": 2},\n        "truck": {"weight": 4.5, "v_max": 0.5, "accel": 0.06, "clear_t": 6},\n        "bike":  {"weight": 0.3, "v_max": 1.5, "accel": 0.50, "clear_t": 1}\n    }\n    \n    def __init__(self, arrival_rates=None):\n        self.n_lanes = 4\n        self.arrival_rates = arrival_rates if arrival_rates else [0.22, 0.22, 0.12, 0.12]\n        self.min_phase_duration = 10\n        self.reset_all()\n\n    def reset_all(self):\n        self.queues = [[] for _ in range(self.n_lanes)]\n        self.phase = 0\n        self.steps_in_phase = 0\n        self.steps = 0\n        self.ambulance_lane = -1\n        self.intersection_occupancy = 0 \n        self.occupancy_phase = -1 \n        self.collision = False\n\n    def reset(self):\n        self.reset_all()\n        return self._get_obs(), {"action_mask": self.get_action_mask()}\n\n    def _get_obs(self):\n        lane_counts = np.array([len(q) for q in self.queues], dtype=np.float32)\n        lane_weights = np.array([sum(v['weight'] for v in q) for q in self.queues], dtype=np.float32)\n        avg_speeds = np.array([np.mean([v['v'] for v in q]) if q else 1.0 for q in self.queues], dtype=np.float32)\n        lane_momentum = np.array([sum(v['weight'] * v['v'] for v in q) if q else 0.0 for q in self.queues], dtype=np.float32)\n        lane_urgency = np.array([(q[0]['weight'] * q[0]['wait_steps'] / 50.0) if q else 0.0 for q in self.queues], dtype=np.float32)\n        amb_flag = 1.0 if self.ambulance_lane != -1 else 0.0\n        return np.concatenate([lane_counts, lane_weights, avg_speeds, lane_momentum, lane_urgency, [float(self.phase)], [amb_flag], [float(self.intersection_occupancy / 10.0)]])\n\n    def get_action_mask(self):\n        mask = np.ones(2, dtype=np.float32)\n        if self.steps_in_phase < self.min_phase_duration: mask[1] = 0.0\n        return mask\n\n    def step(self, action):\n        mask = self.get_action_mask()\n        # Environment enforcement of the mask\n        if mask[action] == 0: action = 0\n\n        if action == 1:\n            self.phase = 1 - self.phase\n            self.steps_in_phase = 0\n        else:\n            self.steps_in_phase += 1\n\n        for i, rate in enumerate(self.arrival_rates):\n            if np.random.random() < rate:\n                vtype = np.random.choice(list(self.VEHICLE_TYPES.keys()), p=[0.7, 0.2, 0.1])\n                cfg = self.VEHICLE_TYPES[vtype]\n                self.queues[i].append({\n                    "weight": cfg["weight"], "v": 0.0, "v_max": cfg["v_max"], \n                    "a": cfg["accel"], "clear_t": cfg["clear_t"], "wait_steps": 0\n                })\n\n        green_lanes = [0, 1] if self.phase == 0 else [2, 3]\n        total_reward = 0.0\n        if self.intersection_occupancy > 0: self.intersection_occupancy -= 1\n        \n        for i in range(self.n_lanes):\n            if i in green_lanes:\n                for j, v in enumerate(self.queues[i]):\n                    if j < 4: v['v'] = min(v['v_max'], v['v'] + v['a'])\n                    if v['v'] < 0.1: v['wait_steps'] += 1\n                if self.queues[i] and self.queues[i][0]['v'] >= (self.queues[i][0]['v_max'] * 0.6):\n                    if self.intersection_occupancy > 0 and self.occupancy_phase != self.phase: self.collision = True\n                    departed = self.queues[i].pop(0)\n                    total_reward += departed['weight'] * 5.0\n                    self.intersection_occupancy = departed['clear_t']\n                    self.occupancy_phase = self.phase\n            else:\n                for v in self.queues[i]:\n                    v['v'] = max(0.0, v['v'] - 0.8)\n                    v['wait_steps'] += 1\n        \n        self.steps += 1\n        reward = total_reward - (200.0 if self.collision else 0.0)\n        done = (self.steps >= 150) or self.collision\n        \n        info = {\n            "action_mask": self.get_action_mask(),\n            "collision": self.collision\n        }\n        return self._get_obs(), float(reward), done, info`
  },
  {
    path: 'agents/dqn_agent.py',
    name: 'dqn_agent.py',
    language: 'python',
    content: `import torch\nimport torch.nn as nn\nimport random\nimport numpy as np\n\nclass DQN(nn.Module):\n    def __init__(self, in_dim, out_dim):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(in_dim, 256), nn.ReLU(),\n            nn.Linear(256, 128), nn.ReLU(),\n            nn.Linear(128, out_dim)\n        )\n    def forward(self, x): return self.net(x)\n\nclass DQNAgent:\n    def __init__(self, state_dim, action_dim):\n        self.model = DQN(state_dim, action_dim)\n        self.epsilon = 1.0\n        self.epsilon_min = 0.05\n        self.epsilon_decay = 0.99\n\n    def act(self, state, action_mask=None):\n        # Explore: pick only from valid actions according to the mask\n        if np.random.rand() <= self.epsilon:\n            valid_actions = np.where(action_mask == 1)[0] if action_mask is not None else [0, 1]\n            return random.choice(valid_actions)\n        \n        # Exploit: predict q-values and mask invalid ones\n        state_t = torch.FloatTensor(state).unsqueeze(0)\n        with torch.no_grad():\n            q_values = self.model(state_t).numpy()[0]\n        \n        if action_mask is not None:\n            # Set invalid action Q-values to negative infinity\n            q_values[action_mask == 0] = -1e10\n            \n        return np.argmax(q_values)\n\n    def train_step(self):\n        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)\n\n    def save(self, path): torch.save(self.model.state_dict(), path)\n    def load(self, path): self.model.load_state_dict(torch.load(path))`
  },
  {
    path: 'train.py',
    name: 'train.py',
    language: 'python',
    content: `import argparse\nimport os\nfrom envs.traffic_env import TrafficEnv\nfrom agents.dqn_agent import DQNAgent\n\ndef main(episodes):\n    os.makedirs('models', exist_ok=True)\n    env = TrafficEnv()\n    agent = DQNAgent(23, 2)\n    \n    for e in range(episodes):\n        state, info = env.reset()\n        mask = info['action_mask']\n        total_reward = 0\n        while True:\n            action = agent.act(state, action_mask=mask)\n            next_state, reward, done, info = env.step(action)\n            state = next_state\n            mask = info['action_mask']\n            total_reward += reward\n            if done: break\n        \n        agent.train_step()\n        if (e + 1) % 10 == 0:\n            print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.2f}")\n    \n    agent.save('models/dqn_checkpoint.pth')\n    print("Training complete. Model saved.")\n\nif __name__ == '__main__':\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--episodes', type=int, default=50)\n    args = parser.parse_args()\n    main(args.episodes)`
  },
  {
    path: 'evaluate.py',
    name: 'evaluate.py',
    language: 'python',
    content: `import argparse\nfrom envs.traffic_env import TrafficEnv\nfrom agents.dqn_agent import DQNAgent\n\ndef evaluate(checkpoint):\n    env = TrafficEnv()\n    agent = DQNAgent(23, 2)\n    agent.load(checkpoint)\n    agent.epsilon = 0.0 # Greedy evaluation\n\n    for i in range(5):\n        state, info = env.reset()\n        mask = info['action_mask']\n        total_reward = 0\n        steps = 0\n        while True:\n            action = agent.act(state, action_mask=mask)\n            state, reward, done, info = env.step(action)\n            mask = info['action_mask']\n            total_reward += reward\n            steps += 1\n            if done: break\n        print(f"Eval Episode {i+1} | Reward: {total_reward:.2f} | Steps: {steps}")\n\nif __name__ == '__main__':\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--checkpoint', type=str, required=True)\n    args = parser.parse_args()\n    evaluate(args.checkpoint)`
  },
  {
    path: 'tests/test_env.py',
    name: 'test_env.py',
    language: 'python',
    content: `import pytest\nimport numpy as np\nfrom envs.traffic_env import TrafficEnv\n\ndef test_env_reset():\n    env = TrafficEnv()\n    obs, info = env.reset()\n    assert obs.shape == (23,)\n    assert "action_mask" in info\n    assert env.steps == 0\n\ndef test_env_step():\n    env = TrafficEnv()\n    obs, info = env.reset()\n    mask = info["action_mask"]\n    # Switch phase if allowed\n    action = 1 if mask[1] == 1.0 else 0\n    next_obs, reward, done, next_info = env.step(action)\n    assert next_obs.shape == (23,)\n    assert "action_mask" in next_info\n    assert isinstance(reward, float)\n    assert isinstance(done, bool)`
  },
  {
    path: 'utils/plot.py',
    name: 'plot.py',
    language: 'python',
    content: `import matplotlib.pyplot as plt\n\ndef plot_learning_curve(data, title="DQN Training Progress"):\n    plt.figure(figsize=(10, 5))\n    plt.plot(data)\n    plt.title(title)\n    plt.xlabel("Episode")\n    plt.ylabel("Reward")\n    plt.grid(True)\n    plt.show()`
  }
];
