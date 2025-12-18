<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  title: string;
  value: number | undefined;
  colorClass: string;
  isLoading?: boolean;
}>();

const displayValue = computed(() => {
  if (props.isLoading) return 'Loading...';
  if (props.value === undefined) return 'Offline';
  return (props.value * 100).toFixed(4) + '%';
});
</script>

<template>
  <div class="bg-white border border-gray-200 shadow-sm rounded-lg p-5 hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300 h-full flex flex-col items-center justify-center text-center">
    <h6 class="text-gray-500 uppercase text-xs font-bold tracking-wider mb-2">{{ title }}</h6>
    <div :class="['text-3xl font-bold', colorClass]">
      {{ displayValue }}
    </div>
  </div>
</template>
