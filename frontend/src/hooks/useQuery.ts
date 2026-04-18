import { useState, useRef } from 'react';
import type { NodeData, SourceData, SSEEvent } from '../types';

const API_URL = 'http://localhost:8000/query';

export function useQuery() {
  const [nodes, setNodes] = useState<NodeData[]>([]);
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<SourceData[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Accumulate tokens without stale closure issues
  const answerRef = useRef('');

  async function submitQuery(query: string) {
    // Reset state
    setNodes([]);
    setAnswer('');
    setSources([]);
    setError(null);
    answerRef.current = '';
    setIsStreaming(true);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // sse-starlette uses \r\n\r\n between frames; also handle \n\n
        const frames = buffer.split(/\r\n\r\n|\n\n/);
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          const dataLine = frame
            .split(/\r\n|\n/)
            .find((line) => line.startsWith('data: '));
          if (!dataLine) continue;

          const jsonStr = dataLine.slice('data: '.length).trimEnd();
          let event: SSEEvent;
          try {
            event = JSON.parse(jsonStr);
          } catch {
            continue;
          }

          switch (event.type) {
            case 'nodes':
              setNodes(event.data);
              break;
            case 'token':
              answerRef.current += event.data;
              setAnswer(answerRef.current);
              break;
            case 'sources':
              setSources(event.data);
              break;
            case 'error':
              setError(event.data);
              break;
            case 'done':
              break;
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsStreaming(false);
    }
  }

  return { nodes, answer, sources, isStreaming, error, submitQuery };
}
