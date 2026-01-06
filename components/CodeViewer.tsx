
import React from 'react';
import { PythonFile } from '../types';
import { Copy, Terminal } from 'lucide-react';

interface CodeViewerProps {
  file: PythonFile;
}

const CodeViewer: React.FC<CodeViewerProps> = ({ file }) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(file.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full bg-[#1e293b] rounded-xl overflow-hidden shadow-xl border border-slate-800">
      <div className="flex items-center justify-between px-4 py-3 bg-[#0f172a] border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5 mr-3">
            <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
            <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
            <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
          </div>
          <span className="text-xs font-mono text-slate-400">{file.path}</span>
        </div>
        <button 
          onClick={handleCopy}
          className="text-slate-400 hover:text-white transition-colors p-1.5 rounded-md hover:bg-slate-800"
          title="Copy code"
        >
          {copied ? <span className="text-[10px] uppercase font-bold text-emerald-500">Copied!</span> : <Copy size={16} />}
        </button>
      </div>
      
      <div className="flex-1 overflow-auto p-6 code-font">
        <pre className="text-sm leading-relaxed text-slate-300 whitespace-pre">
          <code>{file.content}</code>
        </pre>
      </div>
      
      <div className="p-3 bg-[#0f172a] border-t border-slate-800 flex items-center justify-between">
        <div className="flex gap-4 text-[10px] text-slate-500 font-mono uppercase">
          <span>{file.language}</span>
          <span>UTF-8</span>
          <span>{file.content.split('\n').length} lines</span>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-blue-400 font-mono italic">
          <Terminal size={12} />
          <span>Ready for execution</span>
        </div>
      </div>
    </div>
  );
};

export default CodeViewer;
