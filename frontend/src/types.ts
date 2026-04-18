export interface PlanData {
  tickers: string[];
  years: number[];
  filing_types: string[];
  reasoning: string;
}

export interface NodeData {
  node_id: string;
  ticker: string;
  filing_type: string;
  fiscal_year: number;
  fiscal_quarter: number | null;
  section_label: string;
  score: number | null;
  text_preview: string;
}

export interface SourceData {
  ticker: string;
  filing_type: string;
  fiscal_year: number;
  section_label: string;
  node_id: string;
}

export type SSEEvent =
  | { type: 'plan'; data: PlanData }
  | { type: 'nodes'; data: NodeData[] }
  | { type: 'token'; data: string }
  | { type: 'sources'; data: SourceData[] }
  | { type: 'done' }
  | { type: 'error'; data: string };
