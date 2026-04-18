import { useState } from 'react';
import type { PlanData } from '../types';

interface Props {
  plan: PlanData | null;
}

const TYPE_COLORS: Record<string, string> = {
  '10-K': 'bg-violet-100 text-violet-700 border border-violet-200',
  '10-Q': 'bg-blue-100 text-blue-700 border border-blue-200',
};

export function QueryPlanPanel({ plan }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  if (!plan) return null;

  return (
    <div className="border-b border-slate-200 bg-slate-50 px-4 py-3 transition-all">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-600">
          Query Plan
        </span>
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="text-xs text-slate-400 hover:text-slate-600 transition-colors"
        >
          {collapsed ? 'Show' : 'Hide'}
        </button>
      </div>

      {!collapsed && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-1 items-center">
            <span className="text-xs text-slate-500 mr-1">Tickers:</span>
            {plan.tickers.map((t) => (
              <span
                key={t}
                className="px-2 py-0.5 rounded bg-slate-200 text-slate-700 text-xs font-mono font-semibold"
              >
                {t}
              </span>
            ))}
          </div>

          <div className="flex flex-wrap gap-1 items-center">
            <span className="text-xs text-slate-500 mr-1">Years:</span>
            {plan.years.map((y) => (
              <span
                key={y}
                className="px-2 py-0.5 rounded bg-slate-200 text-slate-700 text-xs"
              >
                {y}
              </span>
            ))}
            <span className="text-slate-300 mx-1">·</span>
            <span className="text-xs text-slate-500 mr-1">Types:</span>
            {plan.filing_types.map((ft) => (
              <span
                key={ft}
                className={`px-2 py-0.5 rounded text-xs font-medium ${TYPE_COLORS[ft] ?? 'bg-slate-200 text-slate-700'}`}
              >
                {ft}
              </span>
            ))}
          </div>

          {plan.reasoning && (
            <p className="text-xs text-slate-500 italic leading-relaxed">{plan.reasoning}</p>
          )}
        </div>
      )}
    </div>
  );
}
