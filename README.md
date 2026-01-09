# 🚦 Traffic-RL Explorer: Interactive AI Dashboard

This project is a high-fidelity web application that visualizes a Reinforcement Learning (RL) based traffic signal controller. It serves as both a live simulation environment and a documentation hub for the underlying Python codebase.

## 🚀 Quick Start

### Prerequisites
- **Node.js**: v18.0 or higher
- **npm**: v9.0 or higher

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd traffic-rl-explorer
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure Environment Variables**:
   Create a `.env.local` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the development server**:
   ```bash
   npm run dev
   ```

## 🏗️ Project Structure

- `App.tsx`: Main application shell and routing logic.
- `components/SimulationDashboard.tsx`: High-performance SVG-based traffic simulation engine.
- `components/CodeViewer.tsx`: Interactive source code browser with syntax highlighting.
- `constants.tsx`: Contains the full "traffic-rl" Python project codebase represented as strings for browser-side viewing.
- `types.ts`: Centralized TypeScript definitions for simulation states and metrics.

## 🧠 The RL Project (Python)

The core logic of the traffic controller is implemented in Python (visible in the "Codebase" tab). It utilizes:
- **PyTorch**: For the Deep Q-Network (DQN) implementation.
- **NumPy**: For physical simulation of vehicle dynamics.
- **Matplotlib**: For standalone desktop visualization.

## 🛠️ Features
- **Live RL Training Simulation**: Real-time visualization of agent learning.
- **Physics Observables**: Dynamic tracking of throughput, wait times, and lane mass.
- **Collision Feedback**: Animated and acoustic alerts for intersection failures.
- **Emergency Overrides**: Simulation of priority vehicles (Ambulances) and their impact on flow.

## 📜 License
MIT License - See the codebase for full details.