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
  citation_index: number;
  quote: string;
}

export interface DataQualityAssessment {
  rating: 'HIGH' | 'MEDIUM' | 'LOW';
  summary: string;
  missing_coverage: string[];
}

export interface FilterInfo {
  tickers:    string[];
  years:      number[];
  industries: string[];
}

export type SSEEvent =
  | { type: 'nodes';   data: NodeData[] }
  | { type: 'filters'; data: FilterInfo }
  | { type: 'token';   data: string }
  | { type: 'answer';  data: string }
  | { type: 'quality'; data: DataQualityAssessment }
  | { type: 'sources'; data: SourceData[] }
  | { type: 'done' }
  | { type: 'error';   data: string };
