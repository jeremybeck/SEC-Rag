import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

interface Props {
  answer: string;
  isStreaming: boolean;
}

export function StreamingAnswer({ answer, isStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [answer]);

  if (!answer && !isStreaming) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-slate-400 text-sm">Ask a question to get started.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-5">
      <div className="prose text-slate-700 text-sm max-w-none">
        <ReactMarkdown>{answer}</ReactMarkdown>
        {isStreaming && (
          <span className="inline-block w-1.5 h-4 bg-indigo-400 ml-0.5 animate-pulse align-middle" />
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}
