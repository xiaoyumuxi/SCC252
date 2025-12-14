<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { api } from './services/api';
import { PerformanceMetrics, PredictionResult, LogEntry } from './types';
import StatCard from './components/StatCard.vue';
import RadarChart from './components/RadarChart.vue';

// --- State ---
const performanceData = ref<PerformanceMetrics | null>(null);
const isLoadingPerf = ref(true);
const isAnalyzing = ref(false);
const isRetraining = ref(false);
const predictionResult = ref<PredictionResult | null>(null);
const logs = ref<LogEntry[]>([
  { id: 1, timestamp: new Date().toLocaleTimeString(), message: "[SYSTEM] IDS initialized successfully...", type: "success" },
  { id: 2, timestamp: new Date().toLocaleTimeString(), message: "[SYSTEM] Connected to RF Model (78 features)...", type: "info" }
]);
const fileInput = ref<HTMLInputElement | null>(null);

// --- Actions ---

// Load Performance Metrics
const loadPerformance = async () => {
  isLoadingPerf.value = true;
  try {
    const data = await api.getPerformance();
    performanceData.value = data;
  } catch (error) {
    console.warn("Load failed", error);
    performanceData.value = null; // Triggers 'Offline' in UI
  } finally {
    isLoadingPerf.value = false;
  }
};

// Simulate Traffic
const simulateTraffic = async (type: 'normal' | 'attack' | 'random') => {
  isAnalyzing.value = true;
  try {
    // 1. Get Fake Data
    const data = await api.getTrafficData(type);
    
    // 2. Predict
    const result = await api.predict(data.features);
    predictionResult.value = result;

    // 3. Log
    addLog(result);
  } catch (error: any) {
    console.error(error);
    const msg = error.name === 'AbortError' 
      ? "⚠️ Request Timed Out! Server took too long." 
      : `❌ Error: ${error.message}`;
    alert(msg);
  } finally {
    isAnalyzing.value = false;
  }
};

// Handle Log Adding
const addLog = (res: PredictionResult) => {
  const newLog: LogEntry = {
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString(),
    message: `DETECTED: `,
    label: res.predicted_label,
    confidence: `(Conf: ${(res.confidence * 100).toFixed(1)}%)`,
    type: res.threat_level === 'None' ? 'success' : 'danger'
  };
  logs.value.unshift(newLog); // Add to top
};

// Upload & Retrain
const handleRetrain = async () => {
  const file = fileInput.value?.files?.[0];
  if (!file) {
    alert("Please select a CSV file first!");
    return;
  }

  isRetraining.value = true;
  try {
    const res = await api.retrain(file);
    alert("✅ " + res.message);
    await loadPerformance(); // Refresh stats
    if (fileInput.value) fileInput.value.value = ''; // Clear input
  } catch (error: any) {
    const msg = error.name === 'AbortError' 
      ? "⚠️ Training Timeout! Dataset might be too large." 
      : `❌ Training Failed: ${error.message}`;
    alert(msg);
  } finally {
    isRetraining.value = false;
  }
};

onMounted(() => {
  loadPerformance();
});
</script>

<template>
  <div class="w-full min-h-screen p-6 flex flex-col">
    <!-- Header -->
    <header class="flex flex-col md:flex-row justify-between items-center mb-6 pb-4 border-b border-gray-300">
      <div class="mb-4 md:mb-0 text-center md:text-left">
        <h2 class="text-3xl font-bold text-primary flex items-center gap-2">
          <i class="fas fa-shield-virus"></i> DDoS Defense Shield
        </h2>
        <small class="text-gray-500 font-medium">Real-time Network Traffic Analysis & Threat Intelligence</small>
      </div>
      <span class="bg-green-100 text-green-700 border border-green-200 px-3 py-1.5 rounded text-sm font-semibold flex items-center">
        System Active <i class="fas fa-satellite-dish fa-spin ml-2"></i>
      </span>
    </header>

    <!-- Top Stats Row -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <StatCard 
        title="Accuracy" 
        :value="performanceData?.accuracy" 
        color-class="text-green-600" 
        :is-loading="isLoadingPerf" 
      />
      <StatCard 
        title="Precision" 
        :value="performanceData?.precision" 
        color-class="text-blue-600" 
        :is-loading="isLoadingPerf" 
      />
      <StatCard 
        title="Recall" 
        :value="performanceData?.recall" 
        color-class="text-orange-500" 
        :is-loading="isLoadingPerf" 
      />
      <StatCard 
        title="FPR" 
        :value="performanceData?.FPR" 
        color-class="text-red-600" 
        :is-loading="isLoadingPerf" 
      />
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
      <!-- Left: Simulator Controls -->
      <div class="lg:col-span-3">
        <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5 h-full">
          <h5 class="text-gray-800 border-b border-gray-200 pb-2 mb-4 font-bold flex items-center gap-2">
            <i class="fas fa-network-wired text-primary"></i> Traffic Simulator
          </h5>
          <p class="text-gray-500 text-sm mb-4">Inject simulated packets to test IDS response.</p>

          <div class="space-y-3">
            <button 
              @click="simulateTraffic('normal')"
              :disabled="isAnalyzing"
              class="w-full py-2.5 px-4 bg-white border border-primary text-primary font-semibold uppercase tracking-wide hover:bg-primary hover:text-white transition-colors duration-300 rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <i class="fas fa-check-circle"></i> Normal Traffic
            </button>
            <button 
              @click="simulateTraffic('random')" 
              :disabled="isAnalyzing"
              class="w-full py-2.5 px-4 bg-white border border-orange-400 text-orange-500 font-semibold uppercase tracking-wide hover:bg-orange-400 hover:text-white transition-colors duration-300 rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <i class="fas fa-search-location"></i> PortScan Test
            </button>
            <button 
              @click="simulateTraffic('attack')"
              :disabled="isAnalyzing"
              class="w-full py-2.5 px-4 bg-white border border-red-500 text-red-500 font-semibold uppercase tracking-wide hover:bg-red-500 hover:text-white transition-colors duration-300 rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <i class="fas fa-biohazard"></i> DDoS Attack
            </button>
          </div>

          <div v-if="isAnalyzing" class="mt-4 text-center text-primary font-bold animate-pulse text-sm">
            <i class="fas fa-spinner fa-spin mr-1"></i> Analyzing Features...
          </div>
        </div>
      </div>

      <!-- Center: Results & Logs -->
      <div class="lg:col-span-6 flex flex-col gap-4 min-h-0">
        <!-- Result Card -->
        <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5">
          <div class="flex justify-between items-center">
            <div>
              <h6 class="text-gray-400 text-xs uppercase font-bold">Last Detection Result:</h6>
              <h3 
                class="text-3xl font-bold mt-1"
                :class="{
                  'text-gray-800': !predictionResult,
                  'text-green-600': predictionResult?.threat_level === 'None' || predictionResult?.threat_level === 'Low',
                  'text-orange-500': predictionResult?.threat_level === 'Medium',
                  'text-red-600': predictionResult?.threat_level === 'High',
                }"
              >
                {{ predictionResult?.predicted_label || 'Waiting for input...' }}
              </h3>
            </div>
            <span 
              class="px-4 py-2 rounded-full text-sm font-semibold shadow-sm"
              :class="{
                'bg-gray-200 text-gray-700': !predictionResult,
                'bg-green-100 text-green-800': predictionResult?.threat_level === 'None' || predictionResult?.threat_level === 'Low',
                'bg-orange-100 text-orange-800': predictionResult?.threat_level === 'Medium',
                'bg-red-100 text-red-800': predictionResult?.threat_level === 'High',
              }"
            >
              Status: {{ predictionResult?.threat_level || 'Idle' }}
            </span>
          </div>
        </div>

        <!-- Log Console -->
        <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5 flex-1 flex flex-col min-h-0">
          <h5 class="text-gray-800 border-b border-gray-200 pb-2 mb-2 font-bold flex items-center gap-2">
            <i class="fas fa-terminal text-primary"></i> Live Threat Logs
          </h5>
          <div class="bg-gray-50 border border-gray-200 rounded p-4 flex-1 overflow-y-auto font-mono text-sm">
            <div v-for="log in logs" :key="log.id" class="mb-1" :class="log.type === 'success' ? 'text-green-600' : (log.type === 'info' ? 'text-gray-500' : 'text-red-600')">
              <span class="opacity-70">[{{ log.timestamp }}]</span> 
              <span v-html="log.message"></span>
              <b v-if="log.label">{{ log.label }}</b>
              <span v-if="log.confidence" class="ml-1 opacity-80">{{ log.confidence }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Retrain & Chart -->
      <div class="lg:col-span-3 flex flex-col gap-4">
        <!-- Retrain Box -->
        <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5">
          <h5 class="text-gray-800 border-b border-gray-200 pb-2 mb-3 font-bold flex items-center gap-2">
            <i class="fas fa-sync text-primary"></i> Model Retraining
          </h5>
          <div class="mb-3">
            <label class="block text-gray-500 text-xs font-bold uppercase mb-1">Upload CSV Dataset:</label>
            <input 
              type="file" 
              ref="fileInput"
              class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-gray-700 file:text-white hover:file:bg-gray-800 cursor-pointer border border-gray-300 rounded"
            >
          </div>
          <button 
            @click="handleRetrain"
            :disabled="isRetraining"
            class="w-full py-2 px-4 border border-primary text-primary hover:bg-primary hover:text-white rounded transition-colors duration-300 text-sm font-semibold flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span v-if="isRetraining" class="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></span>
            <span v-else><i class="fas fa-upload"></i></span>
            {{ isRetraining ? 'Training...' : 'Upload & Retrain' }}
          </button>
        </div>

        <!-- Radar Chart -->
        <div class="flex-1 min-h-0">
          <RadarChart :data="performanceData" />
        </div>
      </div>
    </div>
  </div>
</template>
