
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
  Cpu
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
  Area
} from 'recharts';
import { PYTHON_CODEBASE } from './constants';
import { SimulationState, TrafficPhase, LaneState } from './types';

// Components
import CodeViewer from './components/CodeViewer';
import SimulationDashboard from './components/SimulationDashboard';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'sim' | 'code' | 'stats'>('sim');
  const [selectedFile, setSelectedFile] = useState(PYTHON_CODEBASE[0]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState<{episode: number, wait: number}[]>([]);
  
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
    }, 100);
  };

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
              <span className="text-xs font-medium uppercase tracking-wider">RL Agent Ready</span>
            </div>
            <p className="text-[10px] text-slate-500">v1.0.4-alpha</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6">
          <h2 className="text-xl font-semibold text-slate-800">
            {activeTab === 'sim' && 'Environment Visualization'}
            {activeTab === 'code' && 'Project Repository: traffic-rl'}
            {activeTab === 'stats' && 'RL Performance Metrics'}
          </h2>
          <div className="flex items-center gap-3">
             <button 
              onClick={startTraining}
              disabled={isTraining}
              className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium text-sm transition-all ${isTraining ? 'bg-slate-100 text-slate-400 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95'}`}
            >
              <Play size={16} fill="currentColor" />
              {isTraining ? 'Training Agent...' : 'Train DQN'}
            </button>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 overflow-auto p-6">
          {activeTab === 'sim' && <SimulationDashboard />}
          
          {activeTab === 'code' && (
            <div className="h-full flex gap-6">
              <div className="w-64 flex flex-col gap-2">
                <h3 className="text-xs font-bold uppercase text-slate-500 px-2">Files</h3>
                {PYTHON_CODEBASE.map(file => (
                  <button
                    key={file.path}
                    onClick={() => setSelectedFile(file)}
                    className={`flex items-center gap-2 p-2 rounded-md text-sm transition-all ${selectedFile.path === file.path ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-slate-600 hover:bg-slate-100'}`}
                  >
                    <FileCode size={16} className={selectedFile.path === file.path ? 'text-blue-600' : 'text-slate-400'} />
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
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard 
                  title="Average Reward" 
                  value={isTraining ? "-42.5" : "-12.1"} 
                  change="-14%" 
                  trend="up" 
                />
                <StatCard 
                  title="Success Rate" 
                  value="98.2%" 
                  change="+2.4%" 
                  trend="up" 
                />
                <StatCard 
                  title="Collision Buffer" 
                  value="1.2s" 
                  change="0.0s" 
                  trend="neutral" 
                />
              </div>
              
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h3 className="text-lg font-semibold mb-4 text-slate-800">Training Progress (Avg Wait Time)</h3>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trainingData}>
                      <defs>
                        <linearGradient id="colorWait" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis dataKey="episode" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0' }}
                        cursor={{ stroke: '#3b82f6', strokeWidth: 2 }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="wait" 
                        stroke="#3b82f6" 
                        strokeWidth={3}
                        fillOpacity={1} 
                        fill="url(#colorWait)" 
                        animationDuration={500}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

const StatCard: React.FC<{ title: string; value: string; change: string; trend: 'up' | 'down' | 'neutral' }> = ({ title, value, change, trend }) => (
  <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
    <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
    <div className="flex items-end gap-2">
      <span className="text-2xl font-bold text-slate-900">{value}</span>
      <span className={`text-xs font-semibold mb-1 ${trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-slate-400'}`}>
        {change}
      </span>
    </div>
  </div>
);

export default App;
