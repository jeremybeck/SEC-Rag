import type { SourceData } from '../types';

interface Props {
  sources: SourceData[];
  onActivate: (nodeId: string) => void;
  expandedNodeIds: Set<string>;
}

const CHIP_COLORS = [
  { inactive: 'bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100',   active: 'bg-indigo-600 text-white border-indigo-600' },
  { inactive: 'bg-violet-50 text-violet-700 border-violet-200 hover:bg-violet-100',   active: 'bg-violet-600 text-white border-violet-600' },
  { inactive: 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100',           active: 'bg-blue-600 text-white border-blue-600' },
  { inactive: 'bg-teal-50 text-teal-700 border-teal-200 hover:bg-teal-100',           active: 'bg-teal-600 text-white border-teal-600' },
  { inactive: 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100', active: 'bg-emerald-600 text-white border-emerald-600' },
  { inactive: 'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100',       active: 'bg-amber-500 text-white border-amber-500' },
  { inactive: 'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100',           active: 'bg-rose-600 text-white border-rose-600' },
  { inactive: 'bg-sky-50 text-sky-700 border-sky-200 hover:bg-sky-100',               active: 'bg-sky-600 text-white border-sky-600' },
];

function tickerColor(ticker: string) {
  let hash = 0;
  for (let i = 0; i < ticker.length; i++) hash += ticker.charCodeAt(i);
  return CHIP_COLORS[hash % CHIP_COLORS.length];
}

export function SourceChips({ sources, onActivate, expandedNodeIds }: Props) {
  if (sources.length === 0) return null;

  return (
    <div className="px-6 py-3 border-t border-slate-200 bg-slate-50">
      <p className="text-[10px] uppercase tracking-wider text-slate-400 mb-2">
        Sources — click to show/hide preview
      </p>
      <div className="flex flex-wrap gap-2">
        {sources.map((s) => {
          const active = expandedNodeIds.has(s.node_id);
          const colors = tickerColor(s.ticker);
          return (
            <div key={s.node_id} className="relative group">
              <button
                onClick={() => onActivate(s.node_id)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border cursor-pointer transition-colors ${active ? colors.active : colors.inactive}`}
              >
                {s.citation_indices.map((idx) => (
                  <span key={idx} className={`font-mono text-[10px] ${active ? 'opacity-90' : 'opacity-50'}`}>[{idx}]</span>
                ))}
                <span className="font-semibold">{s.ticker}</span>
                <span className={active ? 'opacity-90' : 'opacity-70'}>{s.filing_type}</span>
                <span className={active ? 'opacity-90' : 'opacity-70'}>FY{s.fiscal_year}</span>
                <span className={active ? 'opacity-60' : 'opacity-40'}>·</span>
                <span className={`max-w-[120px] truncate ${active ? 'opacity-90' : 'opacity-70'}`}>{s.section_label}</span>
              </button>

              {s.quote && (
                <div className="absolute bottom-full left-0 mb-2 w-80 bg-slate-800 text-slate-100 text-xs leading-relaxed rounded-lg p-3 shadow-xl z-50 pointer-events-none opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity duration-150">
                  <span className="opacity-50">...</span>{s.quote}<span className="opacity-50">...</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
