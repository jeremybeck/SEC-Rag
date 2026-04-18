import { useState } from 'react';
import { useQuery } from './hooks/useQuery';
import { ChatInput } from './components/ChatInput';
import { QueryPlanPanel } from './components/QueryPlanPanel';
import { NodesSidebar } from './components/NodesSidebar';
import { StreamingAnswer } from './components/StreamingAnswer';
import { SourceChips } from './components/SourceChips';

export default function App() {
  const { plan, nodes, answer, sources, isStreaming, error, submitQuery } = useQuery();
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(new Set());
  const [scrollTargetId, setScrollTargetId] = useState<string | null>(null);

  function activateNode(nodeId: string) {
    setExpandedNodeIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
        return next;
      }
      next.add(nodeId);
      setScrollTargetId(nodeId);
      return next;
    });
  }

  function handleSubmit(query: string) {
    setExpandedNodeIds(new Set());
    setScrollTargetId(null);
    submitQuery(query);
  }

  return (
    <div className="h-screen flex flex-col bg-white text-slate-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-200 bg-white flex items-center gap-3 shrink-0">
        <div className="w-2 h-2 rounded-full bg-indigo-500" />
        <h1 className="text-sm font-semibold text-slate-700 tracking-wide">SEC Filings RAG</h1>
      </div>

      {/* Query Plan banner */}
      <div className="shrink-0">
        <QueryPlanPanel plan={plan} />
      </div>

      {/* Main body: answer (left) + nodes sidebar (right) */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: answer + sources */}
        <div className="flex flex-col flex-1 overflow-hidden">
          {error ? (
            <div className="flex-1 flex items-center justify-center px-6">
              <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-red-700 text-sm max-w-lg">
                <strong>Error:</strong> {error}
              </div>
            </div>
          ) : (
            <StreamingAnswer answer={answer} isStreaming={isStreaming} />
          )}

          <div className="shrink-0">
            <SourceChips sources={sources} onActivate={activateNode} expandedNodeIds={expandedNodeIds} />
          </div>
        </div>

        {/* Right: nodes sidebar */}
        <NodesSidebar
          nodes={nodes}
          expandedNodeIds={expandedNodeIds}
          onToggle={(id) =>
            setExpandedNodeIds((prev) => {
              const next = new Set(prev);
              next.has(id) ? next.delete(id) : next.add(id);
              return next;
            })
          }
          scrollTargetId={scrollTargetId}
          onScrolled={() => setScrollTargetId(null)}
        />
      </div>

      {/* Chat input */}
      <div className="shrink-0">
        <ChatInput onSubmit={handleSubmit} disabled={isStreaming} />
      </div>
    </div>
  );
}
