import { PerformanceMetrics, PredictionResult, TrafficData } from '../types';

const API_BASE_URL = 'http://127.0.0.1:5050';
const TIMEOUT_MS = 8000;

async function fetchWithTimeout(resource: string, options: RequestInit = {}): Promise<Response> {
  const { signal, ...rest } = options;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT_MS);

  // If a signal was passed, we need to respect it too, but for simplicity here
  // we primarily rely on our timeout controller.
  
  try {
    const url = resource.startsWith('http') ? resource : `${API_BASE_URL}${resource}`;
    const response = await fetch(url, {
      ...rest,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

export const api = {
  async getPerformance(): Promise<PerformanceMetrics> {
    const res = await fetchWithTimeout('/api/performance', { method: 'GET' });
    if (!res.ok) throw new Error("Server offline");
    const data = await res.json();
    // Convert string values to numbers if needed (backend may return strings)
    return {
      accuracy: typeof data.accuracy === 'string' ? parseFloat(data.accuracy) : data.accuracy,
      precision: typeof data.precision === 'string' ? parseFloat(data.precision) : data.precision,
      recall: typeof data.recall === 'string' ? parseFloat(data.recall) : data.recall,
      FPR: typeof data.FPR === 'string' ? parseFloat(data.FPR) : data.FPR,
      auc: typeof data.auc === 'string' ? parseFloat(data.auc) : data.auc
    };
  },

  async getTrafficData(type: 'normal' | 'attack' | 'random'): Promise<TrafficData> {
    let url = '/api/random';
    if (type === 'normal') {
      // For normal traffic, use random data (backend doesn't have dedicated normal endpoint)
      url = '/api/random';
    }
    if (type === 'attack') {
      // Use the /api/stream endpoint for real attack samples
      url = '/api/stream';
    }
    
    const res = await fetchWithTimeout(url);
    if (!res.ok) throw new Error("Failed to generate traffic data");
    return res.json();
  },

  async predict(features: number[]): Promise<PredictionResult> {
    const res = await fetchWithTimeout('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    if (!res.ok) throw new Error("Prediction API failed");
    return res.json();
  },

  async retrain(file: File): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append('files', file);

    // Retraining needs a longer timeout
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), 30000);

    try {
      const res = await fetch(`${API_BASE_URL}/api/upload-and-retrain`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      clearTimeout(id);
      if (!res.ok) throw new Error("Retraining failed");
      return res.json();
    } catch (error) {
      clearTimeout(id);
      throw error;
    }
  }
};
