
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { TrafficPhase, SimulationState, LaneState } from '../types';
import { AlertTriangle, Clock, Car, Settings2, Info, Lock, Unlock, Truck, Bike, FastForward } from 'lucide-react';

interface EnhancedSimulationState extends SimulationState {
  laneVehicles: { type: 'car' | 'truck' | 'bike', wait: number, speed: number, maxSpeed: number }[][];
  minPhaseSteps: number;
  currentPhaseSteps: number;
}

const SimulationDashboard: React.FC = () => {
  const [simState, setSimState] = useState<EnhancedSimulationState>({
    lanes: Array(4).fill({ cars: 0, waitTime: 0, arrivals: 0 }),
    laneVehicles: [[], [], [], []],
    phase: TrafficPhase.NORTH_SOUTH,
    step: 0,
    totalReward: 0,
    avgWait: 0,
    ambulanceLane: null,
    minPhaseSteps: 10,
    currentPhaseSteps: 0
  });
  
  const [isRunning, setIsRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [policy, setPolicy] = useState<'DQN' | 'Manual' | 'Heuristic'>('DQN');
  const [arrivalIntensity, setArrivalIntensity] = useState(0.2);

  const isSwitchLocked = simState.currentPhaseSteps < simState.minPhaseSteps;

  const updateSim = useCallback(() => {
    setSimState(prev => {
      // 1. New Arrivals with speeds
      const newLaneVehicles = prev.laneVehicles.map((q, idx) => {
        const nextQ = [...q];
        if (Math.random() < arrivalIntensity) {
          const types: ('car' | 'truck' | 'bike')[] = ['car', 'truck', 'bike'];
          const type = types[Math.floor(Math.random() * types.length)];
          const maxSpeed = type === 'bike' ? 1.5 : type === 'truck' ? 0.6 : 1.0;
          nextQ.push({ type, wait: 0, speed: 0, maxSpeed });
        }
        return nextQ;
      });
      
      // 2. Policy Logic
      let nextPhase = prev.phase;
      let nextPhaseSteps = prev.currentPhaseSteps + 1;
      
      if (!isSwitchLocked && policy === 'DQN') {
        const nsWeight = newLaneVehicles[0].length + newLaneVehicles[1].length;
        const ewWeight = newLaneVehicles[2].length + newLaneVehicles[3].length;
        
        if (prev.phase === TrafficPhase.NORTH_SOUTH && ewWeight > nsWeight + 6) {
          nextPhase = TrafficPhase.EAST_WEST;
          nextPhaseSteps = 0;
        } else if (prev.phase === TrafficPhase.EAST_WEST && nsWeight > ewWeight + 6) {
          nextPhase = TrafficPhase.NORTH_SOUTH;
          nextPhaseSteps = 0;
        }
      }

      // 3. Ambulance Override
      let ambulanceLane = prev.ambulanceLane;
      if (!ambulanceLane && Math.random() < 0.01) ambulanceLane = Math.floor(Math.random() * 4);
      
      if (ambulanceLane !== null) {
        const target = ambulanceLane < 2 ? TrafficPhase.NORTH_SOUTH : TrafficPhase.EAST_WEST;
        if (nextPhase !== target) {
          nextPhase = target;
          nextPhaseSteps = 0;
        }
        if (newLaneVehicles[ambulanceLane].length === 0) ambulanceLane = null;
      }

      // 4. Update Speeds and Departures
      const greenLanes = nextPhase === TrafficPhase.NORTH_SOUTH ? [0, 1] : [2, 3];
      const finalVehicles = newLaneVehicles.map((q, idx) => {
        let nextQ = q.map(v => {
          let s = v.speed;
          if (greenLanes.includes(idx)) {
            // Accelerate leading vehicles
            s = Math.min(v.maxSpeed, s + 0.15);
          } else {
            // Decelerate
            s = Math.max(0, s - 0.4);
          }
          return { ...v, speed: s, wait: s < 0.1 ? v.wait + 1 : v.wait };
        });

        // Departure logic: first vehicle leaves if it has speed
        if (nextQ.length > 0 && greenLanes.includes(idx) && nextQ[0].speed > 0.4) {
          nextQ.shift();
        }

        return nextQ;
      });

      const totalWait = finalVehicles.flat().reduce((acc, v) => acc + v.wait, 0);
      const totalCars = finalVehicles.flat().length;

      return {
        ...prev,
        laneVehicles: finalVehicles,
        lanes: finalVehicles.map(v => ({ cars: v.length, waitTime: v.reduce((a,b)=>a+b.wait,0), arrivals: 0 })),
        phase: nextPhase,
        currentPhaseSteps: nextPhaseSteps,
        step: prev.step + 1,
        avgWait: totalCars > 0 ? totalWait / totalCars : 0,
        ambulanceLane
      };
    });
  }, [policy, arrivalIntensity, isSwitchLocked]);

  useEffect(() => {
    let interval: any;
    if (isRunning) {
      interval = setInterval(updateSim, 1000 / (speed * 2)); // Double actual speed for smoother UI
    }
    return () => clearInterval(interval);
  }, [isRunning, speed, updateSim]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
      <div className="lg:col-span-2 bg-slate-900 rounded-xl border border-slate-800 shadow-2xl p-6 flex flex-col items-center justify-center relative overflow-hidden">
        {/* Status Overlay */}
        <div className="absolute top-4 left-4 flex flex-wrap gap-2 z-10">
          <div className="bg-slate-800/80 backdrop-blur px-3 py-1.5 rounded-full flex items-center gap-2 border border-slate-700 shadow-sm text-slate-200">
            <Clock size={14} className="text-blue-400" />
            <span className="text-xs font-bold font-mono">STEP_{simState.step}</span>
          </div>
          <div className={`px-3 py-1.5 rounded-full flex items-center gap-2 border shadow-sm transition-colors ${isSwitchLocked ? 'bg-amber-900/40 border-amber-700 text-amber-300' : 'bg-emerald-900/40 border-emerald-700 text-emerald-300'}`}>
            {isSwitchLocked ? <Lock size={14} /> : <Unlock size={14} />}
            <span className="text-xs font-bold uppercase tracking-tight">{isSwitchLocked ? `LOCKED (${simState.minPhaseSteps - simState.currentPhaseSteps})` : 'READY'}</span>
          </div>
        </div>

        {/* Intersection Visualization */}
        <div className="relative w-full max-w-[600px] aspect-square">
          <svg viewBox="0 0 500 500" className="w-full h-full drop-shadow-2xl">
            {/* Roads */}
            <rect x="0" y="190" width="500" height="120" fill="#2d3436" />
            <rect x="190" y="0" width="120" height="500" fill="#2d3436" />
            
            <line x1="0" y1="250" x2="500" y2="250" stroke="#636e72" strokeWidth="2" strokeDasharray="15 10" />
            <line x1="250" y1="0" x2="250" y2="500" stroke="#636e72" strokeWidth="2" strokeDasharray="15 10" />

            <LaneGroup 
              x={220} y={100} rotate={0} 
              vehicles={simState.laneVehicles[0]} 
              isGreen={simState.phase === TrafficPhase.NORTH_SOUTH}
              isAmbulance={simState.ambulanceLane === 0}
            />
            <LaneGroup 
              x={280} y={400} rotate={180} 
              vehicles={simState.laneVehicles[1]} 
              isGreen={simState.phase === TrafficPhase.NORTH_SOUTH}
              isAmbulance={simState.ambulanceLane === 1}
            />
            <LaneGroup 
              x={400} y={220} rotate={90} 
              vehicles={simState.laneVehicles[2]} 
              isGreen={simState.phase === TrafficPhase.EAST_WEST}
              isAmbulance={simState.ambulanceLane === 2}
            />
            <LaneGroup 
              x={100} y={280} rotate={270} 
              vehicles={simState.laneVehicles[3]} 
              isGreen={simState.phase === TrafficPhase.EAST_WEST}
              isAmbulance={simState.ambulanceLane === 3}
            />
          </svg>
        </div>
      </div>

      <div className="flex flex-col gap-6">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
          <div className="flex items-center gap-2 mb-4">
            <Settings2 size={20} className="text-slate-500" />
            <h3 className="font-semibold text-slate-800 uppercase text-xs tracking-wider">Simulation Parameters</h3>
          </div>
          
          <div className="space-y-5">
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-[10px] font-bold text-slate-500 uppercase">Traffic Intensity</label>
                <span className="text-[10px] font-mono text-blue-600">{(arrivalIntensity * 100).toFixed(0)}%</span>
              </div>
              <input 
                type="range" min="0.05" max="0.5" step="0.05" 
                value={arrivalIntensity} onChange={e => setArrivalIntensity(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>

            <div>
              <label className="text-[10px] font-bold text-slate-500 uppercase block mb-2">Policy</label>
              <div className="grid grid-cols-3 gap-1 bg-slate-100 p-1 rounded-lg">
                {(['DQN', 'Manual', 'Heuristic'] as const).map(p => (
                  <button
                    key={p}
                    onClick={() => setPolicy(p)}
                    className={`py-1.5 text-[10px] font-bold rounded-md transition-all ${policy === p ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-slate-50">
               <span className="text-[10px] font-bold text-slate-500 uppercase">Speed Control</span>
               <div className="flex gap-2">
                  {[1, 2, 5].map(v => (
                    <button 
                      key={v}
                      onClick={() => setSpeed(v)}
                      className={`w-8 h-8 rounded flex items-center justify-center text-[10px] font-bold ${speed === v ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-500'}`}
                    >
                      {v}x
                    </button>
                  ))}
               </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 flex-1">
          <div className="flex items-center gap-2 mb-4">
            <FastForward size={20} className="text-slate-500" />
            <h3 className="font-semibold text-slate-800 uppercase text-xs tracking-wider">Dynamic Metrics</h3>
          </div>
          
          <div className="space-y-4">
             <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-[10px] font-bold text-slate-400 uppercase block mb-1">Intersection Avg Speed</span>
                <div className="flex items-end gap-2">
                   <span className="text-2xl font-black text-slate-900">
                     {(simState.laneVehicles.flat().reduce((a,b)=>a+b.speed, 0) / (simState.laneVehicles.flat().length || 1) * 60).toFixed(1)}
                   </span>
                   <span className="text-[10px] font-bold text-slate-400 mb-1">km/h (sim)</span>
                </div>
             </div>

             <div className="space-y-2">
                {simState.laneVehicles.map((v, i) => (
                  <div key={i} className="group">
                    <div className="flex justify-between text-[10px] font-bold mb-1">
                       <span className="text-slate-400">LANE {i}</span>
                       <span className={v.length > 5 ? 'text-red-500' : 'text-slate-400'}>{v.length} QUEUED</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden flex">
                       {v.map((veh, idx) => (
                         <div 
                          key={idx} 
                          className={`h-full transition-all duration-500 ${veh.type === 'car' ? 'bg-blue-400' : veh.type === 'truck' ? 'bg-amber-400' : 'bg-emerald-400'}`}
                          style={{ width: `${100/Math.max(10, v.length)}%`, opacity: veh.speed > 0.1 ? 1 : 0.4 }}
                         />
                       ))}
                    </div>
                  </div>
                ))}
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const LaneGroup: React.FC<{ x: number, y: number, rotate: number, vehicles: any[], isGreen: boolean, isAmbulance: boolean }> = ({ x, y, rotate, vehicles, isGreen, isAmbulance }) => (
  <g transform={`translate(${x},${y}) rotate(${rotate})`}>
    {/* Traffic Light */}
    <rect x="-12" y="-55" width="24" height="45" rx="4" fill="#1e293b" />
    <circle cx="0" cy="-42" r="7" fill={isGreen ? "#10b981" : "#334155"} className={isGreen ? "shadow-emerald-500/50" : ""} />
    <circle cx="0" cy="-22" r="7" fill={!isGreen ? "#ef4444" : "#334155"} />

    {/* Vehicle Rendering with offset for speed */}
    {vehicles.slice(0, 15).map((v, i) => {
      // Leading vehicle moves forward slightly based on speed
      const offset = i === 0 ? -v.speed * 40 : 0;
      return (
        <rect 
          key={i} 
          x={v.type === 'truck' ? -10 : -8} 
          y={15 + i * 14 + offset} 
          width={v.type === 'truck' ? 20 : 16} 
          height={v.type === 'truck' ? 12 : 10} 
          rx="2" 
          fill={isAmbulance && i === 0 ? "#ef4444" : v.type === 'car' ? "#3b82f6" : v.type === 'truck' ? "#f59e0b" : "#10b981"}
          className={isAmbulance && i === 0 ? 'animate-pulse' : ''}
          style={{ opacity: v.speed > 0.1 ? 1 : 0.7 }}
        />
      );
    })}
  </g>
);

export default SimulationDashboard;
