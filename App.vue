<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { api } from './services/api';
import { PerformanceMetrics, PredictionResult, LogEntry } from './types';
import StatCard from './components/StatCard.vue';
import RadarChart from './components/RadarChart.vue';

type PredictionResultWithTimestamp = PredictionResult & { timestamp?: string };
type RealtimeLevel = 'Low' | 'Medium' | 'High';

// --- State ---
const performanceData = ref<PerformanceMetrics | null>(null);
const isLoadingPerf = ref(true);
const isAnalyzing = ref(false);
const isRetraining = ref(false);
const predictionResult = ref<PredictionResultWithTimestamp | null>(null);
const logs = ref<LogEntry[]>([
  { id: 1, timestamp: new Date().toLocaleTimeString(), message: "[SYSTEM] IDS initialized successfully...", type: "success" },
  { id: 2, timestamp: new Date().toLocaleTimeString(), message: "[SYSTEM] Connected to RF Model (78 features)...", type: "info" }
]);
const fileInput = ref<HTMLInputElement | null>(null);

// --- Realtime intensity (10s sliding window, ✅ BENIGN 不计入) ---
const isAttackRunning = ref(false);
const windowMs = ref(10000);
const realtimeCount = ref(0); // ✅ 只统计非BENIGN次数
const realtimeLevel = ref<RealtimeLevel>('Low');
const streamRunId = ref(0);

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

const calcLevel = (count: number, thresholds?: { low_max: number; medium_max: number }): RealtimeLevel => {
  const lowMax = thresholds?.low_max ?? 5;
  const medMax = thresholds?.medium_max ?? 10;
  if (count <= lowMax) return 'Low';
  if (count <= medMax) return 'Medium';
  return 'High';
};

// ✅ 展示用 threat：攻击运行时用 realtimeLevel，否则用模型 threat_level
const displayThreat = computed(() => {
  if (isAttackRunning.value) return realtimeLevel.value;
  return (predictionResult.value?.threat_level ?? 'Idle') as any;
});

// --- Actions ---

// Load Performance Metrics
const loadPerformance = async () => {
  isLoadingPerf.value = true;
  try {
    const data = await api.getPerformance();
    performanceData.value = data;
  } catch (error) {
    console.warn("Load failed", error);
    performanceData.value = null;
  } finally {
    isLoadingPerf.value = false;
  }
};

// Handle Log Adding (model result)
const addLog = (res: PredictionResult) => {
  const newLog: LogEntry = {
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString(),
    message: `DETECTED: `,
    label: res.predicted_label,
    confidence: `(Conf: ${(res.confidence * 100).toFixed(1)}%)`,
    type: res.threat_level === 'None' ? 'success' : 'danger',
    threat_level: res.threat_level,
    probabilities: res.probabilities
  };
  logs.value.unshift(newLog);
};

// Simulate Traffic
const simulateTraffic = async (type: 'normal' | 'attack' | 'random') => {
  isAnalyzing.value = true;
  const myRun = ++streamRunId.value;

  try {
    const data: any = await api.getTrafficData(type);
    const stream = Array.isArray(data?.stream) ? data.stream : null;

    // --- ATTACK: play a 10s plan, update intensity in real time (✅ BENIGN 不计入) ---
    if (type === 'attack' && stream && stream.length > 0) {
      isAttackRunning.value = true;

      const thresholds = data.thresholds ?? { low_max: 5, medium_max: 10 };
      const winSec = Number(data.time_window_seconds ?? 10);
      windowMs.value = Math.max(1000, winSec * 1000);

      // ✅ 只记录“非BENIGN”的事件时间戳
      const nonBenignTimes: number[] = [];
      realtimeCount.value = 0;
      realtimeLevel.value = 'Low';

      logs.value.unshift({
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        message: `[SIM] Attack plan started (mode=${data.mode ?? 'N/A'}, total=${data.attack_frequency ?? stream.length}, window=${winSec}s). NOTE: BENIGN will NOT be counted.`,
        type: 'info'
      });

      // ✅ 定时器：处理衰减（只对 nonBenignTimes）
      const refreshInterval = 200;
      const intensityTimer = setInterval(() => {
        const now = Date.now();
        while (nonBenignTimes.length && now - nonBenignTimes[0] > windowMs.value) nonBenignTimes.shift();

        const newCount = nonBenignTimes.length;
        const newLevel = calcLevel(newCount, thresholds);

        if (streamRunId.value === myRun) {
          realtimeCount.value = newCount;
          realtimeLevel.value = newLevel;
        }
      }, refreshInterval);

      let lastLevel: RealtimeLevel = realtimeLevel.value;

      const start = Date.now();

      // ✅ 按 at_ms 定点触发
      for (const sample of stream) {
        if (streamRunId.value !== myRun) break;

        const target = start + Number(sample.at_ms ?? 0);
        const wait = Math.max(0, target - Date.now());
        await sleep(wait);

        // 预测（保持原流程）
        const result = await api.predict(sample.features);

        predictionResult.value = {
          ...result,
          timestamp: new Date(sample.ts ?? Date.now()).toLocaleTimeString()
        };
        addLog(result);

        // ✅ BENIGN 不计入“10秒内攻击次数”
        const pred = String(result.predicted_label ?? '').trim().toUpperCase();
        const isBenign = (pred === 'BENIGN');

        if (!isBenign) {
          const now = Date.now();
          nonBenignTimes.push(now);

          // 立刻刷新一次（不等定时器）
          while (nonBenignTimes.length && now - nonBenignTimes[0] > windowMs.value) nonBenignTimes.shift();
          realtimeCount.value = nonBenignTimes.length;

          const newLevel = calcLevel(realtimeCount.value, thresholds);
          realtimeLevel.value = newLevel;

          if (newLevel !== lastLevel) {
            lastLevel = newLevel;
            logs.value.unshift({
              id: Date.now() + Math.random(),
              timestamp: new Date().toLocaleTimeString(),
              message: `[ALERT] Traffic intensity escalated to <b>${newLevel}</b> (10s non-BENIGN count=${realtimeCount.value})`,
              type: newLevel === 'High' ? 'danger' : 'info'
            });
          }
        }
      }

      logs.value.unshift({
        id: Date.now() + 1,
        timestamp: new Date().toLocaleTimeString(),
        message: `[SIM] Attack plan finished. Waiting for 10s window to decay...`,
        type: 'info'
      });

      // ✅ 等到 10 秒窗口自然衰减到 0（只看非BENIGN）
      const decayStart = Date.now();
      while (streamRunId.value === myRun) {
        const now = Date.now();
        while (nonBenignTimes.length && now - nonBenignTimes[0] > windowMs.value) nonBenignTimes.shift();
        if (nonBenignTimes.length === 0) break;

        if (now - decayStart > windowMs.value + 1000) break;
        await sleep(refreshInterval);
      }

      clearInterval(intensityTimer);
      isAttackRunning.value = false;

      logs.value.unshift({
        id: Date.now() + 2,
        timestamp: new Date().toLocaleTimeString(),
        message: `[SIM] Intensity window cleared (10s non-BENIGN count=${realtimeCount.value}, level=${realtimeLevel.value}).`,
        type: 'info'
      });
    } else {
      // --- NORMAL / RANDOM: single shot as before ---
      const result = await api.predict(data.features);
      predictionResult.value = result;
      addLog(result);

      isAttackRunning.value = false;
      realtimeCount.value = 0;
      realtimeLevel.value = 'Low';
    }
  } catch (error: any) {
    console.error(error);
    const msg = error.name === 'AbortError'
      ? "⚠️ Request Timed Out! Server took too long."
      : `❌ Error: ${error.message}`;
    alert(msg);

    isAttackRunning.value = false;
    realtimeCount.value = 0;
    realtimeLevel.value = 'Low';
  } finally {
    if (streamRunId.value === myRun) isAnalyzing.value = false;
  }
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
    await loadPerformance();
    if (fileInput.value) fileInput.value.value = '';
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

      <div class="lg:col-span-6 flex flex-col gap-4 min-h-0">
        <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5">
          <div class="flex justify-between items-start">
            <div class="flex-1">
              <h6 class="text-gray-400 text-xs uppercase font-bold">Last Detection Result:</h6>
              <h3
                class="text-3xl font-bold mt-1"
                :class="{
                  'text-gray-800': !predictionResult && !isAttackRunning,
                  'text-green-600': displayThreat === 'None' || displayThreat === 'Low',
                  'text-orange-500': displayThreat === 'Medium',
                  'text-red-600': displayThreat === 'High',
                }"
              >
                {{ predictionResult?.predicted_label || 'Waiting for input...' }}
              </h3>

              <div v-if="predictionResult" class="mt-2 text-sm text-gray-600 space-y-1">
                <div>
                  <span class="font-semibold">Confidence:</span>
                  <span class="font-mono">{{ (predictionResult.confidence * 100).toFixed(2) }}%</span>
                </div>
                <div v-if="predictionResult.timestamp">
                  <span class="font-semibold">Detected at:</span>
                  <span class="font-mono">{{ predictionResult.timestamp }}</span>
                </div>
              </div>
            </div>

            <span
              class="px-4 py-2 rounded-full text-sm font-semibold shadow-sm whitespace-nowrap"
              :class="{
                'bg-gray-200 text-gray-700': !predictionResult && !isAttackRunning,
                'bg-green-100 text-green-800': displayThreat === 'None' || displayThreat === 'Low',
                'bg-orange-100 text-orange-800': displayThreat === 'Medium',
                'bg-red-100 text-red-800': displayThreat === 'High',
              }"
            >
              Status: {{ displayThreat }}
              <span v-if="isAttackRunning" class="ml-2 text-xs opacity-70">(10s non-BENIGN: {{ realtimeCount }})</span>
            </span>
          </div>

          <div v-if="predictionResult?.probabilities && Object.keys(predictionResult.probabilities).length > 1" class="mt-4 pt-4 border-t border-gray-200">
            <h6 class="text-gray-500 text-xs uppercase font-bold mb-2">Probability Distribution:</h6>
            <div class="space-y-2">
              <div v-for="(prob, label) in predictionResult.probabilities" :key="label" class="flex items-center gap-3">
                <span class="text-xs font-medium w-36 truncate" :title="label">{{ label }}</span>
                <div class="flex-1 bg-gray-200 rounded-full h-3">
                  <div
                    class="h-3 rounded-full transition-all"
                    :class="{
                      'bg-green-500': predictionResult.predicted_label === label && predictionResult.threat_level === 'None',
                      'bg-orange-500': predictionResult.predicted_label === label && predictionResult.threat_level === 'Medium',
                      'bg-red-500': predictionResult.predicted_label === label && predictionResult.threat_level === 'High',
                      'bg-blue-400': predictionResult.predicted_label !== label
                    }"
                    :style="{ width: `${prob * 100}%` }"
                  ></div>
                </div>
                <span class="text-xs font-mono w-14 text-right">{{ (prob * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>

        <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5 flex-1 flex flex-col min-h-0">
          <h5 class="text-gray-800 border-b border-gray-200 pb-2 mb-2 font-bold flex items-center gap-2">
            <i class="fas fa-terminal text-primary"></i> Live Threat Logs
          </h5>
          <div class="bg-gray-50 border border-gray-200 rounded p-4 flex-1 overflow-y-auto font-mono text-sm">
            <div v-for="log in logs" :key="log.id" class="mb-3 pb-2 border-b border-gray-200 last:border-b-0" :class="log.type === 'success' ? 'text-green-600' : (log.type === 'info' ? 'text-gray-500' : 'text-red-600')">
              <div class="mb-1">
                <span class="opacity-70">[{{ log.timestamp }}]</span>
                <span v-html="log.message"></span>
                <b v-if="log.label">{{ log.label }}</b>
                <span v-if="log.confidence" class="ml-1 opacity-80">{{ log.confidence }}</span>
                <span v-if="log.threat_level" class="ml-2 px-2 py-0.5 rounded text-xs font-semibold"
                  :class="{
                    'bg-green-100 text-green-800': log.threat_level === 'None' || log.threat_level === 'Low',
                    'bg-orange-100 text-orange-800': log.threat_level === 'Medium',
                    'bg-red-100 text-red-800': log.threat_level === 'High'
                  }">
                  {{ log.threat_level }}
                </span>
              </div>

              <div v-if="log.probabilities && Object.keys(log.probabilities).length > 1" class="ml-6 text-xs opacity-70 space-y-0.5">
                <div v-for="(prob, label) in log.probabilities" :key="label" class="flex items-center gap-2">
                  <span class="w-32 truncate">{{ label }}:</span>
                  <div class="flex-1 bg-gray-200 rounded-full h-2 max-w-[150px]">
                    <div class="bg-current h-2 rounded-full transition-all" :style="{ width: `${prob * 100}%` }"></div>
                  </div>
                  <span class="w-12 text-right">{{ (prob * 100).toFixed(1) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="lg:col-span-3 flex flex-col gap-4 h-full">
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

        <div class="flex-1 min-h-[300px]">
          <RadarChart :data="performanceData" />
        </div>
      </div>
    </div>
  </div>
</template>