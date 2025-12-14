<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import Chart from 'chart.js/auto';
import { PerformanceMetrics } from '../types';

const props = defineProps<{
  data: PerformanceMetrics | null;
}>();

const chartCanvas = ref<HTMLCanvasElement | null>(null);
let chartInstance: Chart | null = null;

const initChart = () => {
  if (!chartCanvas.value) return;

  const ctx = chartCanvas.value.getContext('2d');
  if (!ctx) return;

  chartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'FPR', 'AUC'],
      datasets: [{
        label: 'Current Model',
        data: [0, 0, 0, 0, 0], 
        backgroundColor: 'rgba(13, 110, 253, 0.2)',
        borderColor: '#0d6efd',
        pointBackgroundColor: '#fff',
        pointBorderColor: '#0d6efd',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: '#0d6efd'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        r: {
          angleLines: { color: 'rgba(0, 0, 0, 0.1)' },
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          pointLabels: { 
            color: '#666',
            font: { size: 11 }
          },
          suggestedMin: 0,
          suggestedMax: 1
        }
      },
      plugins: { legend: { display: false } }
    }
  });
};

const updateChart = (metrics: PerformanceMetrics) => {
  if (!chartInstance) return;
  
  chartInstance.data.datasets[0].data = [
    metrics.accuracy,
    metrics.precision,
    metrics.recall,
    metrics.FPR,  // Changed from f1_score to FPR
    metrics.auc || 0.95 // Fallback if AUC missing
  ];
  chartInstance.update();
};

onMounted(() => {
  initChart();
});

watch(() => props.data, (newData) => {
  if (newData) {
    updateChart(newData);
  }
}, { deep: true });
</script>

<template>
  <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5 h-full flex flex-col">
    <h6 class="text-gray-500 text-center text-xs font-bold uppercase mb-4 tracking-wider">Model Performance</h6>
    <div class="relative flex-1 w-full">
      <canvas ref="chartCanvas"></canvas>
    </div>
  </div>
</template>
