
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { TrafficPhase, SimulationState, LaneState, SimulationMetrics } from '../types';
import { AlertTriangle, Clock, Car, Settings2, Info, Lock, Unlock, Truck, Bike, FastForward, Zap, Skull, RotateCcw } from 'lucide-react';

interface Vehicle {
  type: 'car' | 'truck' | 'bike';
  weight: number;
  speed: number;
  maxSpeed: number;
  accel: number;
}

interface EnhancedSimulationState extends SimulationState {
  laneVehicles: Vehicle[][];
  minPhaseSteps: number;
  currentPhaseSteps: number;
  intersectionOccupancy: number;
  occupancyPhase: number;
  hasCollision: boolean;
  totalEpisodeReward: number;
}

interface SimulationDashboardProps {
  onStep?: (metrics: SimulationMetrics) => void;
  onEpisodeComplete?: (stats: { avgWait: number, totalReward: number }) => void;
  isTrainingMode?: boolean;
}

const MAX_EPISODE_STEPS = 150;

const playCollisionSound = () => {
  try {
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = 'sawtooth';
    oscillator.frequency.setValueAtTime(150, audioCtx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(40, audioCtx.currentTime + 0.6);

    gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.6);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + 0.6);
  } catch (e) {
    console.warn("Audio Context could not be initialized", e);
  }
};

const SimulationDashboard: React.FC<SimulationDashboardProps> = ({ 
  onStep, 
  onEpisodeComplete,
  isTrainingMode = false
}) => {
  const [simState, setSimState] = useState<EnhancedSimulationState>({
    lanes: Array(4).fill({ cars: 0, waitTime: 0, arrivals: 0 }),
    laneVehicles: [[], [], [], []],
    phase: TrafficPhase.NORTH_SOUTH,
    step: 0,
    totalReward: 0,
    totalEpisodeReward: 0,
    avgWait: 0,
    ambulanceLane: null,
    minPhaseSteps: 12,
    currentPhaseSteps: 0,
    intersectionOccupancy: 0,
    occupancyPhase: -1,
    hasCollision: false
  });
  
  const [isRunning, setIsRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [policy, setPolicy] = useState<'DQN' | 'Manual' | 'Heuristic'>('DQN');
  const [arrivalIntensity, setArrivalIntensity] = useState(0.22);

  const isSwitchLocked = simState.currentPhaseSteps < simState.minPhaseSteps;

  const handleReset = useCallback(() => {
    setSimState({
      lanes: Array(4).fill({ cars: 0, waitTime: 0, arrivals: 0 }),
      laneVehicles: [[], [], [], []],
      phase: TrafficPhase.NORTH_SOUTH,
      step: 0,
      totalReward: 0,
      totalEpisodeReward: 0,
      avgWait: 0,
      ambulanceLane: null,
      minPhaseSteps: 12,
      currentPhaseSteps: 0,
      intersectionOccupancy: 0,
      occupancyPhase: -1,
      hasCollision: false
    });
  }, []);

  const updateSim = useCallback(() => {
    if (simState.hasCollision) return;

    setSimState(prev => {
      // Check for episode completion
      const isDone = prev.step >= MAX_EPISODE_STEPS;
      
      if (isDone || prev.hasCollision) {
        if (onEpisodeComplete) {
          onEpisodeComplete({
            avgWait: prev.avgWait,
            totalReward: prev.totalEpisodeReward
          });
        }
        
        if (isTrainingMode) {
          // Auto reset for training
          return {
            lanes: Array(4).fill({ cars: 0, waitTime: 0, arrivals: 0 }),
            laneVehicles: [[], [], [], []],
            phase: TrafficPhase.NORTH_SOUTH,
            step: 0,
            totalReward: 0,
            totalEpisodeReward: 0,
            avgWait: 0,
            ambulanceLane: null,
            minPhaseSteps: 12,
            currentPhaseSteps: 0,
            intersectionOccupancy: 0,
            occupancyPhase: -1,
            hasCollision: false
          };
        }
        return prev;
      }

      // 1. Stochastic Arrivals
      const newLaneVehicles = prev.laneVehicles.map((q, idx) => {
        const nextQ = [...q];
        if (Math.random() < arrivalIntensity) {
          const rand = Math.random();
          let type: 'car' | 'truck' | 'bike' = 'car';
          let weight = 1.0, maxSpeed = 1.0, accel = 0.25;

          if (rand < 0.15) {
            type = 'truck'; weight = 3.5; maxSpeed = 0.6; accel = 0.08;
          } else if (rand > 0.85) {
            type = 'bike'; weight = 0.4; maxSpeed = 1.4; accel = 0.45;
          }

          nextQ.push({ type, weight, speed: 0, maxSpeed, accel });
        }
        return nextQ;
      });
      
      // 2. Policy Decisions
      let nextPhase = prev.phase;
      let nextPhaseSteps = prev.currentPhaseSteps + 1;
      
      if (!isSwitchLocked && policy === 'DQN') {
        const nsWeight = newLaneVehicles[0].reduce((a, b) => a + b.weight, 0) + newLaneVehicles[1].reduce((a, b) => a + b.weight, 0);
        const ewWeight = newLaneVehicles[2].reduce((a, b) => a + b.weight, 0) + newLaneVehicles[3].reduce((a, b) => a + b.weight, 0);
        
        // Simulating the agent choosing to switch
        if (prev.phase === TrafficPhase.NORTH_SOUTH && ewWeight > nsWeight + 8) {
          nextPhase = TrafficPhase.EAST_WEST;
          nextPhaseSteps = 0;
        } else if (prev.phase === TrafficPhase.EAST_WEST && nsWeight > ewWeight + 8) {
          nextPhase = TrafficPhase.NORTH_SOUTH;
          nextPhaseSteps = 0;
        }
      }

      // 3. Emergency Override
      let ambulanceLane = prev.ambulanceLane;
      if (!ambulanceLane && Math.random() < 0.015) ambulanceLane = Math.floor(Math.random() * 4);
      
      if (ambulanceLane !== null) {
        const target = ambulanceLane < 2 ? TrafficPhase.NORTH_SOUTH : TrafficPhase.EAST_WEST;
        if (nextPhase !== target) {
          nextPhase = target;
          nextPhaseSteps = 0;
        }
        if (newLaneVehicles[ambulanceLane].length === 0) ambulanceLane = null;
      }

      // 4. Physics Engine & Collision Detection
      const greenLanes = nextPhase === TrafficPhase.NORTH_SOUTH ? [0, 1] : [2, 3];
      let hasCollision = false;
      let nextOccupancy = Math.max(0, prev.intersectionOccupancy - 1);
      let nextOccupancyPhase = nextOccupancy === 0 ? -1 : prev.occupancyPhase;
      let stepThroughput = 0;
      let stepReward = 0;

      const finalVehicles = newLaneVehicles.map((q, idx) => {
        let nextQ = q.map((v, i) => {
          let s = v.speed;
          if (greenLanes.includes(idx)) {
            if (i < 4) s = Math.min(v.maxSpeed, s + v.accel);
          } else {
            s = Math.max(0, s - 0.5);
          }
          return { ...v, speed: s };
        });

        // Departure threshold
        if (nextQ.length > 0 && greenLanes.includes(idx) && nextQ[0].speed >= 0.5) {
          // Collision check
          if (nextOccupancy > 0 && nextOccupancyPhase !== nextPhase) {
            hasCollision = true;
            stepReward -= 200; // Large penalty
          }
          const departed = nextQ.shift();
          if (departed) {
            stepThroughput += departed.weight;
            stepReward += departed.weight * 5.0; // Positive reinforcement for throughput
            nextOccupancy = departed.type === 'truck' ? 6 : 3;
            nextOccupancyPhase = nextPhase;
          }
        }

        return nextQ;
      });

      const totalStationaryWeight = finalVehicles.flat().reduce((acc, v) => acc + (v.speed < 0.1 ? v.weight : 0), 0);
      const totalVehicles = finalVehicles.flat().length;
      const currentAvgWait = totalVehicles > 0 ? totalStationaryWeight / totalVehicles : 0;
      
      // Wait penalty reinforcement
      stepReward -= currentAvgWait * 0.5;

      // Report metrics back to parent
      if (onStep) {
        onStep({
          step: prev.step + 1,
          avgWait: currentAvgWait,
          throughput: stepThroughput,
          collision: hasCollision,
          totalMass: totalStationaryWeight
        });
      }

      return {
        ...prev,
        laneVehicles: finalVehicles,
        lanes: finalVehicles.map(v => ({ cars: v.length, waitTime: v.reduce((a,b)=>a + (b.speed < 0.1 ? 1 : 0), 0), arrivals: 0 })),
        phase: nextPhase,
        currentPhaseSteps: nextPhaseSteps,
        step: prev.step + 1,
        avgWait: currentAvgWait,
        ambulanceLane,
        intersectionOccupancy: nextOccupancy,
        occupancyPhase: nextOccupancyPhase,
        hasCollision,
        totalEpisodeReward: prev.totalEpisodeReward + stepReward
      };
    });
  }, [policy, arrivalIntensity, isSwitchLocked, simState.hasCollision, simState.step, onStep, onEpisodeComplete, isTrainingMode]);

  useEffect(() => {
    let interval: any;
    // When training, we force 5X speed to converge faster visually
    const effectiveSpeed = isTrainingMode ? 5 : speed;
    if (isRunning && (!simState.hasCollision || isTrainingMode)) {
      interval = setInterval(updateSim, 1000 / (effectiveSpeed * 10)); 
    }
    return () => clearInterval(interval);
  }, [isRunning, speed, updateSim, simState.hasCollision, isTrainingMode]);

  // Audio feedback for collision
  useEffect(() => {
    if (simState.hasCollision && !isTrainingMode) {
      playCollisionSound();
    }
  }, [simState.hasCollision, isTrainingMode]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
      <div className={`lg:col-span-2 bg-slate-950 rounded-xl border border-slate-800 shadow-2xl p-6 flex flex-col items-center justify-center relative overflow-hidden ${simState.hasCollision ? 'animate-shake border-red-900/50 shadow-red-900/20' : ''}`}>
        {/* Collision Overlay */}
        {simState.hasCollision && !isTrainingMode && (
          <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-red-950/80 backdrop-blur-lg animate-in fade-in duration-500">
             <Skull size={80} className="text-red-500 mb-4 animate-bounce" />
             <h3 className="text-4xl font-black text-white uppercase tracking-tighter mb-2">System Failure</h3>
             <p className="text-red-200 text-lg font-bold mb-8 uppercase tracking-widest">Intersection Collision Detected</p>
             <div className="flex gap-4">
               <button 
                  onClick={handleReset}
                  className="px-8 py-4 bg-red-600 hover:bg-red-500 text-white font-black rounded-full shadow-2xl shadow-red-500/50 transition-all hover:scale-105 active:scale-95 uppercase tracking-widest flex items-center gap-2"
                >
                  <RotateCcw size={18} />
                  Retry Episode
                </button>
                <button 
                  onClick={handleReset}
                  className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-slate-100 font-black rounded-full shadow-2xl transition-all hover:scale-105 active:scale-95 uppercase tracking-widest"
                >
                  Reset Environment
                </button>
             </div>
          </div>
        )}

        {/* Real-time Status */}
        <div className="absolute top-4 left-4 flex flex-wrap gap-2 z-10">
          <div className="bg-slate-900/90 backdrop-blur-md px-4 py-2 rounded-full flex items-center gap-2 border border-slate-700 shadow-xl text-slate-100">
            <Zap size={14} className="text-amber-400 fill-amber-400" />
            <span className="text-xs font-black tracking-widest font-mono">SIM_T.{simState.step} / {MAX_EPISODE_STEPS}</span>
          </div>
          <div className={`px-4 py-2 rounded-full flex items-center gap-2 border shadow-xl transition-all duration-300 ${isSwitchLocked ? 'bg-amber-950/60 border-amber-800 text-amber-200' : 'bg-emerald-950/60 border-emerald-800 text-emerald-200'}`}>
            {isSwitchLocked ? <Lock size={14} className="animate-pulse" /> : <Unlock size={14} />}
            <span className="text-xs font-bold uppercase tracking-widest">{isSwitchLocked ? `LOCKED: ${simState.minPhaseSteps - simState.currentPhaseSteps}` : 'PHASE_RDY'}</span>
          </div>
          {isTrainingMode && (
            <div className="bg-blue-900/90 backdrop-blur-md px-4 py-2 rounded-full flex items-center gap-2 border border-blue-700 shadow-xl text-blue-100">
              <FastForward size={14} className="animate-pulse" />
              <span className="text-xs font-black tracking-widest font-mono uppercase">TRAINING_OPTIMIZATION</span>
            </div>
          )}
        </div>

        {/* 4-Way Intersection SVG */}
        <div className="relative w-full max-w-[620px] aspect-square">
          <svg viewBox="0 0 500 500" className={`w-full h-full ${simState.hasCollision ? 'opacity-50' : ''}`}>
            <rect x="0" y="195" width="500" height="110" fill="#1e293b" />
            <rect x="195" y="0" width="110" height="500" fill="#1e293b" />
            <line x1="0" y1="250" x2="500" y2="250" stroke="#475569" strokeWidth="2" strokeDasharray="20 15" />
            <line x1="250" y1="0" x2="250" y2="500" stroke="#475569" strokeWidth="2" strokeDasharray="20 15" />
            <rect x="195" y="195" width="110" height="110" fill="#334155" />

            <LaneGroup 
              x={225} y={110} rotate={0} 
              vehicles={simState.laneVehicles[0]} 
              isGreen={simState.phase === TrafficPhase.NORTH_SOUTH}
              isAmbulance={simState.ambulanceLane === 0}
            />
            <LaneGroup 
              x={275} y={390} rotate={180} 
              vehicles={simState.laneVehicles[1]} 
              isGreen={simState.phase === TrafficPhase.NORTH_SOUTH}
              isAmbulance={simState.ambulanceLane === 1}
            />
            <LaneGroup 
              x={390} y={225} rotate={90} 
              vehicles={simState.laneVehicles[2]} 
              isGreen={simState.phase === TrafficPhase.EAST_WEST}
              isAmbulance={simState.ambulanceLane === 2}
            />
            <LaneGroup 
              x={110} y={275} rotate={270} 
              vehicles={simState.laneVehicles[3]} 
              isGreen={simState.phase === TrafficPhase.EAST_WEST}
              isAmbulance={simState.ambulanceLane === 3}
            />
          </svg>
        </div>

        {simState.ambulanceLane !== null && !simState.hasCollision && (
          <div className="absolute bottom-6 bg-red-900/80 backdrop-blur-md text-red-100 px-6 py-3 rounded-full border border-red-700 flex items-center gap-3 animate-pulse shadow-2xl z-20">
            <AlertTriangle className="text-red-400" />
            <span className="font-black text-xs uppercase tracking-[0.2em]">Priority Emergency Override Active</span>
          </div>
        )}
      </div>

      <div className="flex flex-col gap-6">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
          <div className="flex items-center gap-2 mb-6">
            <Settings2 size={20} className="text-slate-400" />
            <h3 className="font-bold text-slate-800 uppercase text-xs tracking-[0.1em]">Simulation Engine</h3>
          </div>
          
          <div className="space-y-6">
            <div>
              <div className="flex justify-between mb-2 items-center">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-tighter">Spawn Probability</label>
                <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{(arrivalIntensity * 100).toFixed(0)}%</span>
              </div>
              <input 
                type="range" min="0.05" max="0.6" step="0.01" 
                value={arrivalIntensity} onChange={e => setArrivalIntensity(Number(e.target.value))}
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>

            <div>
              <label className="text-[10px] font-black text-slate-500 uppercase block mb-3 tracking-tighter">Controller Agent</label>
              <div className="grid grid-cols-3 gap-1 bg-slate-100 p-1 rounded-lg shadow-inner">
                {(['DQN', 'Manual', 'Heuristic'] as const).map(p => (
                  <button
                    key={p}
                    onClick={() => setPolicy(p)}
                    className={`py-2 text-[10px] font-black rounded-md transition-all ${policy === p ? 'bg-white text-blue-600 shadow-sm border border-blue-100' : 'text-slate-400 hover:text-slate-600'}`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between pt-6 border-t border-slate-100">
               <span className="text-[10px] font-black text-slate-500 uppercase tracking-tighter">Temporal Warp</span>
               <div className="flex gap-2">
                  {[1, 2, 5].map(v => (
                    <button 
                      key={v}
                      onClick={() => setSpeed(v)}
                      className={`w-10 h-10 rounded-lg flex items-center justify-center text-[10px] font-black transition-all ${speed === v ? 'bg-slate-900 text-white scale-105 shadow-md' : 'bg-slate-50 text-slate-400 border border-slate-100 hover:bg-slate-100'}`}
                    >
                      {v}X
                    </button>
                  ))}
               </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 flex-1 flex flex-col">
          <div className="flex items-center gap-2 mb-6">
            <FastForward size={20} className="text-slate-400" />
            <h3 className="font-bold text-slate-800 uppercase text-xs tracking-[0.1em]">Physics Observables</h3>
          </div>
          
          <div className="flex-1 space-y-6">
             <div className="p-5 bg-blue-50/50 rounded-2xl border border-blue-100 shadow-sm">
                <span className="text-[10px] font-black text-blue-400 uppercase block mb-2 tracking-tighter">Traffic Density Metric</span>
                <div className="flex items-baseline gap-2">
                   <span className="text-4xl font-black text-slate-900 tracking-tighter">
                     {simState.laneVehicles.flat().length}
                   </span>
                   <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Active Units</span>
                </div>
             </div>

             <div className="space-y-4">
                {simState.laneVehicles.map((v, i) => (
                  <div key={i} className="space-y-2">
                    <div className="flex justify-between items-center text-[9px] font-black tracking-widest">
                       <span className="text-slate-400">LN_{i}</span>
                       <span className={v.length > 6 ? 'text-red-500' : 'text-blue-500'}>
                        {v.reduce((a, b) => a + b.weight, 0).toFixed(1)} <span className="text-slate-300">MASS</span>
                       </span>
                    </div>
                    <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden flex gap-[1px]">
                       {v.map((veh, idx) => (
                         <div 
                          key={idx} 
                          className={`h-full transition-all duration-300 ${veh.type === 'car' ? 'bg-blue-500' : veh.type === 'truck' ? 'bg-amber-500' : 'bg-emerald-500'}`}
                          style={{ 
                            width: `${Math.max(5, (veh.weight / 10) * 100)}%`, 
                            opacity: veh.speed > 0.1 ? 1 : 0.4,
                            flexShrink: 0
                          }}
                         />
                       ))}
                    </div>
                  </div>
                ))}
             </div>
          </div>
          
          <div className="mt-8 pt-6 border-t border-slate-100 flex items-center justify-between text-[10px] font-bold text-slate-400">
             <div className="flex items-center gap-2">
               <div className="w-2 h-2 rounded-full bg-blue-500"></div>
               <span>CAR</span>
             </div>
             <div className="flex items-center gap-2">
               <div className="w-2 h-2 rounded-full bg-amber-500"></div>
               <span>TRUCK</span>
             </div>
             <div className="flex items-center gap-2">
               <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
               <span>BIKE</span>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const LaneGroup: React.FC<{ x: number, y: number, rotate: number, vehicles: Vehicle[], isGreen: boolean, isAmbulance: boolean }> = ({ x, y, rotate, vehicles, isGreen, isAmbulance }) => (
  <g transform={`translate(${x},${y}) rotate(${rotate})`}>
    <rect x="-14" y="-60" width="28" height="50" rx="6" fill="#0f172a" stroke="#334155" strokeWidth="1" />
    <circle cx="0" cy="-45" r="8" fill={isGreen ? "#00ff9d" : "#1e293b"} filter={isGreen ? "drop-shadow(0 0 8px #00ff9d)" : ""} />
    <circle cx="0" cy="-25" r="8" fill={!isGreen ? "#ff4d4d" : "#1e293b"} filter={!isGreen ? "drop-shadow(0 0 8px #ff4d4d)" : ""} />

    {vehicles.slice(0, 16).map((v, i) => {
      const offset = i === 0 ? -v.speed * 45 : 0;
      const width = v.type === 'truck' ? 24 : v.type === 'bike' ? 10 : 18;
      const height = v.type === 'truck' ? 16 : v.type === 'bike' ? 8 : 12;
      
      return (
        <rect 
          key={i} 
          x={-width/2} 
          y={20 + i * 16 + offset} 
          width={width} 
          height={height} 
          rx={v.type === 'bike' ? 4 : 2} 
          fill={isAmbulance && i === 0 ? "#ff0000" : v.type === 'car' ? "#3b82f6" : v.type === 'truck' ? "#f59e0b" : "#10b981"}
          className={isAmbulance && i === 0 ? 'animate-pulse' : ''}
          style={{ 
            opacity: v.speed > 0.1 ? 1 : 0.6,
            transition: 'y 0.1s linear, fill 0.3s ease'
          }}
          stroke={isAmbulance && i === 0 ? "#ffffff" : "none"}
          strokeWidth="1"
        />
      );
    })}
  </g>
);

export default SimulationDashboard;
