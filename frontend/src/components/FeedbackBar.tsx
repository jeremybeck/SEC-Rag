import { useEffect, useState } from 'react';
import type { DataQualityAssessment, SourceData } from '../types';

interface FeedbackBarProps {
  query:       string;
  answer:      string;
  sources:     SourceData[];
  dataQuality: DataQualityAssessment | null;
  isStreaming: boolean;
}

export function FeedbackBar({ query, answer, sources, dataQuality, isStreaming }: FeedbackBarProps) {
  const [rating, setRating]           = useState<'up' | 'down' | null>(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [expanded, setExpanded]       = useState(false);
  const [submitted, setSubmitted]     = useState(false);
  const [submitting, setSubmitting]   = useState(false);

  // Reset when a new query starts
  useEffect(() => {
    setRating(null);
    setFeedbackText('');
    setExpanded(false);
    setSubmitted(false);
  }, [query]);

  if (isStreaming || !answer) return null;

  function handleRating(value: 'up' | 'down') {
    if (submitted) return;
    setRating(prev => prev === value ? null : value);
    setExpanded(true);
  }

  async function handleSubmit() {
    if (submitting || submitted) return;
    setSubmitting(true);
    try {
      await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_text:    query,
          answer_text:   answer,
          cited_sources: sources,
          data_quality:  dataQuality?.rating ?? null,
          rating:        rating,
          feedback_text: feedbackText.trim() || null,
        }),
      });
      setSubmitted(true);
    } finally {
      setSubmitting(false);
    }
  }

  if (submitted) {
    return (
      <div className="px-6 py-3 shrink-0">
        <p className="text-xs text-slate-400">Thanks for your feedback.</p>
      </div>
    );
  }

  const canSubmit = (rating !== null || feedbackText.trim().length > 0) && !submitting;

  return (
    <div className="px-6 py-3 shrink-0 border-t border-slate-100">
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-400">Was this helpful?</span>

        {/* Thumbs up */}
        <button
          onClick={() => handleRating('up')}
          title="Thumbs up"
          className={`rounded p-1 transition-colors ${
            rating === 'up'
              ? 'text-emerald-600 bg-emerald-50'
              : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path d="M1 8.25a1.25 1.25 0 1 1 2.5 0v7.5a1.25 1.25 0 1 1-2.5 0v-7.5ZM11 3V1.7c0-.268.14-.526.395-.607A2 2 0 0 1 14 3c0 .995-.182 1.948-.514 2.826-.204.54.166 1.174.744 1.174h2.52c1.243 0 2.261 1.01 2.146 2.247a23.864 23.864 0 0 1-1.341 5.974C17.153 16.323 16.072 17 14.9 17H8.731c-.637 0-1.249-.246-1.7-.686L4.5 13.81V8.25c2.664 0 4.534-1.468 5.725-2.938C10.9 4.44 11 3.68 11 3Z" />
          </svg>
        </button>

        {/* Thumbs down */}
        <button
          onClick={() => handleRating('down')}
          title="Thumbs down"
          className={`rounded p-1 transition-colors ${
            rating === 'down'
              ? 'text-red-500 bg-red-50'
              : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path d="M18.905 12.75a1.25 1.25 0 1 1-2.5 0v-7.5a1.25 1.25 0 1 1 2.5 0v7.5ZM8.905 17v1.3c0 .268-.14.526-.395.607A2 2 0 0 1 5.905 17c0-.995.182-1.948.514-2.826.204-.54-.166-1.174-.744-1.174h-2.52c-1.243 0-2.261-1.01-2.146-2.247a23.862 23.862 0 0 1 1.341-5.974C2.752 3.677 3.833 3 5.005 3h6.169c.637 0 1.249.246 1.7.686l2.531 2.524v5.56c-2.664 0-4.534 1.468-5.725 2.938-.666.822-.775 1.581-.775 2.292Z" />
          </svg>
        </button>

        {/* Add comment toggle */}
        {!expanded && (
          <button
            onClick={() => setExpanded(true)}
            className="text-xs text-slate-400 hover:text-slate-600 underline underline-offset-2"
          >
            Add comment
          </button>
        )}
      </div>

      {expanded && (
        <div className="mt-2 flex flex-col gap-2">
          <textarea
            value={feedbackText}
            onChange={e => setFeedbackText(e.target.value)}
            placeholder="Optional: tell us what was wrong or what could be better…"
            rows={2}
            className="w-full text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-3 py-2 resize-none focus:outline-none focus:ring-1 focus:ring-indigo-400 placeholder:text-slate-400"
          />
          <div className="flex items-center gap-2">
            <button
              onClick={handleSubmit}
              disabled={!canSubmit}
              className="text-xs px-3 py-1.5 rounded bg-indigo-600 text-white font-medium disabled:opacity-40 hover:bg-indigo-700 transition-colors"
            >
              {submitting ? 'Submitting…' : 'Submit feedback'}
            </button>
            <button
              onClick={() => setExpanded(false)}
              className="text-xs text-slate-400 hover:text-slate-600"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
