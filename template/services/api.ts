import { PerformanceMetrics, PredictionResult, TrafficData } from '../types';

const API_BASE_URL = 'http://127.0.0.1:5050';
const TIMEOUT_MS = 8000;
const RETRAIN_TIMEOUT_MS = 120000;

async function fetchWithTimeout(resource: string, options: RequestInit = {}, timeoutMs = TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const url = resource.startsWith('http') ? resource : `${API_BASE_URL}${resource}`;
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return res;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

export type AttackStreamSample = { features: number[]; label?: string; at_ms: number; ts?: number };
export type AttackStreamResponse = {
  status: string;
  mode: 'Low' | 'Medium' | 'High' | string;
  time_window_seconds: number;
  attack_frequency: number;
  frequency_level: 'Low' | 'Medium' | 'High' | string;
  thresholds: { low_max: number; medium_max: number };
  stream: AttackStreamSample[];
};

export const api = {
  async getPerformance(): Promise<PerformanceMetrics> {
    const res = await fetchWithTimeout('/api/performance', { method: 'GET' });
    if (!res.ok) throw new Error('Server offline');
    return res.json();
  },

  async getTrafficData(type: 'normal' | 'attack' | 'random'): Promise<TrafficData | AttackStreamResponse> {
    // normal/random 你原本怎么取就怎么取；这里保持 /api/random
    const url = type === 'attack' ? '/api/stream' : '/api/random';
    const res = await fetchWithTimeout(url, { method: 'GET' });
    if (!res.ok) throw new Error('Failed to get traffic data');
    return res.json();
  },

  async predict(features: number[]): Promise<PredictionResult> {
    const res = await fetchWithTimeout('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    if (!res.ok) throw new Error('Prediction API failed');
    return res.json();
  },

  async retrain(file: File): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetchWithTimeout('/api/upload-and-retrain', { method: 'POST', body: formData }, RETRAIN_TIMEOUT_MS);
    if (!res.ok) throw new Error('Retraining failed');
    return res.json();
  }
};
