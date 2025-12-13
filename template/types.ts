// types.ts
export interface PredictionResult {
  status: string;
  predicted_label: string;
  confidence: number;
  threat_level: string;
  encoded_value?: number;
}

export interface AlertLog {
  timestamp: string;
  type: string;
  confidence: number;
  level: string;
  frequency?: number;  // 新增
}

export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc?: number;
  FPR?: number;
}