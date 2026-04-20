import { useEffect, useRef, useMemo } from 'react';
import type { NodeData } from '../types';

interface Props {
  nodes: NodeData[];
  expandedNodeIds: Set<string>;
  onToggle: (id: string) => void;
  scrollTargetId: string | null;
  onScrolled: () => void;
}

function scoreColor(score: number | null): string {
  if (score === null) return 'bg-slate-100 text-slate-500';
  if (score >= 0.85) return 'bg-emerald-100 text-emerald-700';
  if (score >= 0.70) return 'bg-amber-100 text-amber-700';
  return 'bg-red-100 text-red-600';
}

interface NodeCardProps {
  node: NodeData;
  expanded: boolean;
  onToggle: () => void;
  nodeRef: (el: HTMLDivElement | null) => void;
}

function NodeCard({ node, expanded, onToggle, nodeRef }: NodeCardProps) {
  return (
    <div
      ref={nodeRef}
      className="bg-white rounded-lg p-3 space-y-1.5 border border-slate-200 shadow-sm scroll-mt-8"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs text-slate-700 font-medium leading-tight flex-1">
          {node.section_label}
        </p>
        {node.score !== null && (
          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded shrink-0 ${scoreColor(node.score)}`}>
            {node.score.toFixed(3)}
          </span>
        )}
      </div>

      {expanded && (
        <p className="text-[11px] text-slate-500 leading-relaxed whitespace-pre-wrap">
          {node.text_preview}
        </p>
      )}

      <button
        onClick={onToggle}
        className="text-[10px] text-indigo-500 hover:text-indigo-700 transition-colors"
      >
        {expanded ? 'Hide preview' : 'Show preview'}
      </button>
    </div>
  );
}

export function NodesSidebar({ nodes, expandedNodeIds, onToggle, scrollTargetId, onScrolled }: Props) {
  const nodeRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Scroll to the target node when activated from a source chip
  useEffect(() => {
    if (!scrollTargetId) return;
    const el = nodeRefs.current.get(scrollTargetId);
    if (el && scrollContainerRef.current) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    onScrolled();
  }, [scrollTargetId, onScrolled]);

  const groups = useMemo(() => {
    const map = new Map<string, NodeData[]>();
    for (const n of nodes) {
      const key = `${n.ticker} ${n.filing_type} FY${n.fiscal_year}${n.fiscal_quarter ? ` Q${n.fiscal_quarter}` : ''}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(n);
    }
    return Array.from(map.entries());
  }, [nodes]);

  return (
    <div className="h-full flex flex-col border-l border-slate-200 bg-slate-50 w-72 shrink-0">
      <div className="px-3 py-2.5 border-b border-slate-200">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-indigo-600">
          Relevant Docs
          {nodes.length > 0 && (
            <span className="ml-2 text-slate-400 normal-case font-normal">
              ({nodes.length})
            </span>
          )}
        </h2>
      </div>

      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {nodes.length === 0 && (
          <p className="text-xs text-slate-400 text-center mt-8">
            Nodes will appear here after retrieval.
          </p>
        )}

        {groups.map(([groupKey, groupNodes]) => (
          <div key={groupKey} className="space-y-2">
            <div className="sticky top-0 bg-slate-50/90 py-1">
              <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
                {groupKey}
              </p>
            </div>
            {groupNodes.map((n) => (
              <NodeCard
                key={n.node_id}
                node={n}
                expanded={expandedNodeIds.has(n.node_id)}
                onToggle={() => onToggle(n.node_id)}
                nodeRef={(el) => {
                  if (el) nodeRefs.current.set(n.node_id, el);
                  else nodeRefs.current.delete(n.node_id);
                }}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
