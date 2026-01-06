
export interface LaneState {
  cars: number;
  waitTime: number;
  arrivals: number;
}

export interface SimulationState {
  lanes: LaneState[];
  phase: number; // 0: N-S Green, 1: E-W Green
  step: number;
  totalReward: number;
  avgWait: number;
  ambulanceLane: number | null;
}

export interface SimulationMetrics {
  step: number;
  avgWait: number;
  throughput: number;
  collision: boolean;
  totalMass: number;
}

export enum TrafficPhase {
  NORTH_SOUTH = 0,
  EAST_WEST = 1
}

export interface PythonFile {
  name: string;
  path: string;
  content: string;
  language: 'python' | 'markdown' | 'text';
}
