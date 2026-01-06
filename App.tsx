
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Activity, 
  Code, 
  Play, 
  Settings, 
  ChevronRight, 
  FileCode, 
  Terminal, 
  AlertCircle,
  BarChart3,
  Cpu,
  Trophy,
  Zap,
  ShieldAlert
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from 'recharts';
import { PYTHON_CODEBASE } from './constants';
import { SimulationState, TrafficPhase, LaneState, SimulationMetrics } from './types';

// Components
import CodeViewer from './components/CodeViewer';
import SimulationDashboard from './components/SimulationDashboard';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'sim' | 'code' | 'stats'>('sim');
  const [selectedFile, setSelectedFile] = useState(PYTHON_CODEBASE[0]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState<{episode: number, wait: number}[]>([]);
  
  // Real-time metrics from the simulation
  const [realTimeMetrics, setRealTimeMetrics] = useState<SimulationMetrics[]>([]);
  const [cumulativeStats, setCumulativeStats] = useState({
    totalThroughput: 0,
    collisions: 0,
    peakWait: 0
  });

  const handleSimStep = useCallback((metrics: SimulationMetrics) => {
    setRealTimeMetrics(prev => {
      const next = [...prev, metrics];
      // Keep only last 100 steps for charts
      return next.length > 100 ? next.slice(1) : next;
    });

    setCumulativeStats(prev => ({
      totalThroughput: prev.totalThroughput + metrics.throughput,
      collisions: prev.collisions + (metrics.collision ? 1 : 0),
      peakWait: Math.max(prev.peakWait, metrics.avgWait)
    }));
  }, []);

  // Mock training simulation
  const startTraining = () => {
    setIsTraining(true);
    setTrainingData([]);
    let episode = 0;
    const interval = setInterval(() => {
      setTrainingData(prev => [
        ...prev, 
        { 
          episode: episode++, 
          wait: 50 * Math.exp(-episode / 30) + Math.random() * 5 + 10 
        }
      ]);
      if (episode > 100) {
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 50);
  };

  const currentWait = realTimeMetrics.length > 0 ? realTimeMetrics[realTimeMetrics.length - 1].avgWait : 0;
  const currentThroughput = realTimeMetrics.length > 0 ? realTimeMetrics[realTimeMetrics.length - 1].throughput : 0;

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Sidebar */}
      <div className="w-16 md:w-64 bg-slate-900 text-slate-400 flex flex-col border-r border-slate-800">
        <div className="p-4 flex items-center gap-3 text-white border-b border-slate-800">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Cpu size={20} />
          </div>
          <span className="hidden md:block font-bold text-lg tracking-tight">Traffic-RL</span>
        </div>
        
        <nav className="flex-1 p-2 space-y-1">
          <button 
            onClick={() => setActiveTab('sim')}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors ${activeTab === 'sim' ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 hover:text-slate-200'}`}
          >
            <Activity size={20} />
            <span className="hidden md:block font-medium text-sm">Live Simulation</span>
          </button>
          
          <button 
            onClick={() => setActiveTab('code')}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors ${activeTab === 'code' ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 hover:text-slate-200'}`}
          >
            <Code size={20} />
            <span className="hidden md:block font-medium text-sm">Python Codebase</span>
          </button>
          
          <button 
            onClick={() => setActiveTab('stats')}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors ${activeTab === 'stats' ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 hover:text-slate-200'}`}
          >
            <BarChart3 size={20} />
            <span className="hidden md:block font-medium text-sm">Training Analytics</span>
          </button>
        </nav>
        
        <div className="p-4 border-t border-slate-800">
          <div className="hidden md:block">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span className="text-xs font-medium uppercase tracking-wider text-slate-200">Simulation Stream: ON</span>
            </div>
            <p className="text-[10px] text-slate-500 font-mono">WS://LOCAL_ENV:8080</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6">
          <div className="flex flex-col">
            <h2 className="text-lg font-bold text-slate-800 leading-tight">
              {activeTab === 'sim' && 'Environment Visualization'}
              {activeTab === 'code' && 'Project Repository: traffic-rl'}
              {activeTab === 'stats' && 'RL Performance Metrics'}
            </h2>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              {activeTab === 'stats' ? 'Real-time Telemetry Active' : 'Neural Network Controller v1.0.4'}
            </p>
          </div>
          <div className="flex items-center gap-3">
             <button 
              onClick={startTraining}
              disabled={isTraining}
              className={`flex items-center gap-2 px-4 py-2 rounded-md font-bold text-xs uppercase tracking-widest transition-all ${isTraining ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200' : 'bg-slate-900 text-white hover:bg-slate-800 active:scale-95 shadow-lg shadow-slate-200'}`}
            >
              <Zap size={14} className={isTraining ? '' : 'fill-amber-400 text-amber-400'} />
              {isTraining ? 'Training...' : 'Recalibrate Agent'}
            </button>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 overflow-auto p-6 bg-slate-50/50">
          {activeTab === 'sim' && <SimulationDashboard onStep={handleSimStep} />}
          
          {activeTab === 'code' && (
            <div className="h-full flex gap-6">
              <div className="w-64 flex flex-col gap-2">
                <h3 className="text-[10px] font-black uppercase text-slate-400 px-2 tracking-widest mb-2">File Explorer</h3>
                {PYTHON_CODEBASE.map(file => (
                  <button
                    key={file.path}
                    onClick={() => setSelectedFile(file)}
                    className={`flex items-center gap-2 px-3 py-2.5 rounded-lg text-xs transition-all ${selectedFile.path === file.path ? 'bg-white border border-slate-200 text-blue-600 font-black shadow-sm' : 'text-slate-500 hover:bg-slate-100'}`}
                  >
                    <FileCode size={14} className={selectedFile.path === file.path ? 'text-blue-500' : 'text-slate-400'} />
                    {file.name}
                  </button>
                ))}
              </div>
              <div className="flex-1 min-w-0">
                <CodeViewer file={selectedFile} />
              </div>
            </div>
          )}

          {activeTab === 'stats' && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <StatCard 
                  title="Real-time Avg Wait" 
                  value={`${currentWait.toFixed(1)}s`} 
                  sub="Current Step"
                  icon={<Activity size={18} className="text-blue-500" />}
                />
                <StatCard 
                  title="Total Throughput" 
                  value={`${cumulativeStats.totalThroughput.toFixed(0)}`} 
                  sub="Accumulated Mass"
                  icon={<Trophy size={18} className="text-amber-500" />}
                />
                <StatCard 
                  title="Catastrophic Failures" 
                  value={`${cumulativeStats.collisions}`} 
                  sub="Collision Count"
                  icon={<ShieldAlert size={18} className="text-red-500" />}
                />
                <StatCard 
                  title="Flow Rate" 
                  value={`${currentThroughput.toFixed(1)}`} 
                  sub="Current Throughput"
                  icon={<Zap size={18} className="text-emerald-500" />}
                />
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-sm font-black uppercase tracking-widest text-slate-800">Live Wait Time Dynamics</h3>
                    <div className="px-2 py-1 bg-blue-50 text-[10px] font-bold text-blue-600 rounded">UPDATES 30Hz</div>
                  </div>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={realTimeMetrics}>
                        <defs>
                          <linearGradient id="colorWaitLive" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="step" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#fff', borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="avgWait" 
                          stroke="#3b82f6" 
                          strokeWidth={3}
                          fillOpacity={1} 
                          fill="url(#colorWaitLive)" 
                          isAnimationActive={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-sm font-black uppercase tracking-widest text-slate-800">Throughput Mass Flow</h3>
                    <div className="px-2 py-1 bg-emerald-50 text-[10px] font-bold text-emerald-600 rounded">DYNAMIC FLOW</div>
                  </div>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={realTimeMetrics.slice(-20)}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="step" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#fff', borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                        />
                        <Bar 
                          dataKey="throughput" 
                          fill="#10b981" 
                          radius={[4, 4, 0, 0]}
                          isAnimationActive={false}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 shadow-2xl relative overflow-hidden">
                <div className="relative z-10 flex flex-col md:flex-row items-center gap-8">
                  <div className="flex-1">
                    <h3 className="text-white text-xl font-black mb-2 uppercase tracking-tighter">Optimization Training Progress</h3>
                    <p className="text-slate-400 text-sm mb-6">Convergence behavior of the Deep Q-Network over historical training episodes. Lower wait time indicates successful policy learning.</p>
                    <div className="flex gap-4">
                      <div className="bg-white/5 px-4 py-3 rounded-xl border border-white/10">
                        <span className="text-slate-500 text-[10px] font-black uppercase block mb-1">Total Episodes</span>
                        <span className="text-white font-mono font-bold">{trainingData.length}</span>
                      </div>
                      <div className="bg-white/5 px-4 py-3 rounded-xl border border-white/10">
                        <span className="text-slate-500 text-[10px] font-black uppercase block mb-1">Convergence</span>
                        <span className="text-emerald-400 font-mono font-bold">ALPHA_READY</span>
                      </div>
                    </div>
                  </div>
                  <div className="w-full md:w-[400px] h-[150px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingData}>
                        <Line type="monotone" dataKey="wait" stroke="#3b82f6" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/2 w-64 h-64 bg-blue-600/10 blur-[100px] rounded-full"></div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

const StatCard: React.FC<{ title: string; value: string; sub: string; icon: React.ReactNode }> = ({ title, value, sub, icon }) => (
  <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
    <div className="flex justify-between items-start mb-4">
      <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none">{title}</span>
      <div className="p-1.5 bg-slate-50 rounded-lg">{icon}</div>
    </div>
    <div className="flex flex-col">
      <span className="text-2xl font-black text-slate-900 tracking-tighter leading-none mb-1">{value}</span>
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">{sub}</span>
    </div>
  </div>
);

export default App;
