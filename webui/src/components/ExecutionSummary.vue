<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api/client'
import type { ExecutionRun, Outcome } from '../execution'
import { executionTime } from '../execution'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ name: string }>()
const router = useRouter()
const run = ref<ExecutionRun | null>(null)
const error = ref('')
let generation = 0, loading = false
const attention = computed(() => run.value ? (['failed', 'partial', 'blocked', 'unknown'] as Outcome[]).reduce((sum, key) => sum + (run.value!.counts[key] || 0), 0) : 0)
async function refresh() {
  if (!props.name || loading) return
  const name = props.name, version = generation
  loading = true
  try {
    const data = await api.get(`/api/${encodeURIComponent(name)}/executions?limit=1`)
    if (version === generation) { run.value = data.runs[0] || null; error.value = '' }
  } catch (exception: any) { if (version === generation) error.value = exception.message }
  finally { loading = false }
}
watch(() => props.name, () => { generation++; run.value = null; void refresh() }, { immediate: true })
const timer = window.setInterval(refresh, 5000)
onBeforeUnmount(() => { generation++; window.clearInterval(timer) })
function open() { router.push({ path: `/i/${props.name}/executions`, query: run.value ? { run: run.value.id } : {} }) }
</script>

<template>
  <article class="card execution-summary">
    <div class="execution-summary-head"><b>最近执行</b><button class="btn sm" @click="open">执行记录 <AppIcon name="arrow-right" :size="13" /></button></div>
    <template v-if="run">
      <p>{{ executionTime(run.started_at) }} · {{ run.retrying ? '正在补做' : run.state === 'running' ? '执行中' : run.state === 'interrupted' ? '已中断' : '已结束' }}</p>
      <div class="execution-summary-counts"><span class="ok">已完成 {{ run.counts.success }}</span><span :class="{ attention: attention > 0 }">需要关注 {{ attention }}</span><span>正常跳过 {{ run.counts.skipped }}</span></div>
      <small>{{ run.retrying ? '重试结果会更新到原简报，历史记录保留。' : attention ? '查看未完成原因，选择需要补做的任务。' : '查看各项结果与尝试历史。' }}</small>
    </template>
    <p v-else>{{ error || '尚无执行记录，下次运行后会自动生成简报。' }}</p>
  </article>
</template>

<style scoped>
.execution-summary { padding: 18px; margin-bottom: 16px; }
.execution-summary-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.execution-summary p { color: var(--text-2); font-size: 13px; margin: 12px 0; line-height: 1.6; }
.execution-summary-counts { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; margin-bottom: 10px; color: var(--text-2); }
.execution-summary .ok { color: var(--green); }.execution-summary .attention { color: var(--yellow); }
.execution-summary small { color: var(--text-2); font-size: 12px; }
</style>
