import { useState } from 'react';
import type { DataQualityAssessment } from '../types';

interface Props {
  quality: DataQualityAssessment;
}

const RATING_STYLES: Record<DataQualityAssessment['rating'], string> = {
  HIGH:   'bg-emerald-100 text-emerald-700 border border-emerald-200',
  MEDIUM: 'bg-amber-100   text-amber-700   border border-amber-200',
  LOW:    'bg-red-100     text-red-700     border border-red-200',
};

export function DataQualityBadge({ quality }: Props) {
  const [expanded, setExpanded] = useState(false);
  const hasGaps = quality.missing_coverage.length > 0;

  return (
    <div className="px-6 py-2.5 border-t border-slate-100 bg-white shrink-0">
      <div className="flex items-start gap-2.5">
        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full shrink-0 mt-0.5 ${RATING_STYLES[quality.rating]}`}>
          {quality.rating}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] text-slate-600 leading-snug">{quality.summary}</p>
          {hasGaps && (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="mt-1 text-[10px] text-slate-400 hover:text-slate-600 flex items-center gap-1 transition-colors"
            >
              <span>{expanded ? '▴' : '▾'}</span>
              <span>{quality.missing_coverage.length} gap{quality.missing_coverage.length !== 1 ? 's' : ''}</span>
            </button>
          )}
          {expanded && hasGaps && (
            <ul className="mt-1.5 space-y-0.5">
              {quality.missing_coverage.map((gap, i) => (
                <li key={i} className="text-[10px] text-slate-500 flex items-start gap-1.5">
                  <span className="text-slate-300 shrink-0 mt-px">•</span>
                  <span>{gap}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
