import { useState, type KeyboardEvent } from 'react';

interface Props {
  onSubmit: (query: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSubmit, disabled }: Props) {
  const [value, setValue] = useState('');

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    setValue('');
    onSubmit(trimmed);
  }

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-3">
      <div className="flex gap-3 items-end max-w-full">
        <textarea
          className="flex-1 resize-none rounded-lg bg-white border border-slate-300 text-slate-800
                     placeholder-slate-400 px-4 py-3 text-sm focus:outline-none focus:border-indigo-400
                     focus:ring-2 focus:ring-indigo-100 transition-colors min-h-[44px] max-h-40
                     disabled:opacity-50 disabled:bg-slate-50"
          rows={1}
          placeholder="Ask about SEC filings… (Enter to send, Shift+Enter for newline)"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="px-4 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50
                     disabled:cursor-not-allowed text-white text-sm font-medium transition-colors
                     whitespace-nowrap"
        >
          {disabled ? 'Thinking…' : 'Send'}
        </button>
      </div>
    </div>
  );
}
