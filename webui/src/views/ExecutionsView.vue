<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import AppIcon from '../components/AppIcon.vue'
import AppSelect from '../components/AppSelect.vue'
import { api } from '../api/client'
import { attentionStatuses, executionDuration, executionTime, outcomeLabels } from '../execution'
import type { ExecutionAttempt, ExecutionRun, ExecutionTask, Outcome } from '../execution'
import { useWorkspaceStore } from '../stores/workspace'
import { useToastStore } from '../stores/toast'

const { selectedName } = storeToRefs(useWorkspaceStore())
const route = useRoute(), router = useRouter(), toast = useToastStore()
const runs = ref<ExecutionRun[]>([]), run = ref<ExecutionRun | null>(null)
const selected = ref<string[]>([]), focused = ref<string | null>(null)
const loading = ref(false), busy = ref(false), error = ref(''), filter = ref('attention')
const openGroups = ref<Record<string, boolean>>({ attention: true })
const moreAvailable = ref(false), loadingOlder = ref(false), drawer = ref<HTMLElement | null>(null)
let generation = 0, initialized = '', returnFocus: HTMLElement | null = null
const detailTask = computed(() => run.value?.tasks.find(task => task.command === focused.value))
const groups = computed(() => [
  { key: 'attention', label: '需要关注', items: run.value?.tasks.filter(task => attentionStatuses.includes(task.status)) || [] },
  { key: 'success', label: '已完成', items: run.value?.tasks.filter(task => task.status === 'success') || [] },
  { key: 'skipped', label: '正常跳过', items: run.value?.tasks.filter(task => task.status === 'skipped') || [] },
].filter(group => filter.value === 'all' || group.key === 'attention'))
const failureTasks = computed(() => run.value?.tasks.filter(task => ['failed', 'partial'].includes(task.status) && task.retryable && task.can_retry && !task.retry_state) || [])
const retryDisabled = computed(() => busy.value || !run.value || run.value.state === 'running' || !selected.value.length)
const runOptions = computed(() => runs.value.map(item => ({ value: item.id, label: `${executionTime(item.started_at)} · ${item.state === 'running' ? '执行中' : item.state === 'interrupted' ? '已中断' : '已结束'}${item.source === 'retry' ? ' · 跨日补做' : ''}` })))
const chips: Outcome[] = ['success', 'failed', 'partial', 'blocked', 'unknown', 'skipped', 'pending', 'running']

async function refresh(force = false) {
  if (!selectedName.value || (loading.value && !force)) return
  const version = ++generation, name = selectedName.value
  loading.value = true
  try {
    const list = await api.get(`/api/${encodeURIComponent(name)}/executions?limit=30`)
    if (version !== generation) return
    if (!runs.value.length) moreAvailable.value = list.runs.length === 30
    const latestIds = new Set(list.runs.map((item: ExecutionRun) => item.id))
    runs.value = [...list.runs, ...runs.value.filter(item => !latestIds.has(item.id))]
    const id = String(route.query.run || runs.value[0]?.id || '')
    if (!id) { run.value = null; error.value = ''; return }
    const data: ExecutionRun = await api.get(`/api/${encodeURIComponent(name)}/executions/${encodeURIComponent(id)}`)
    if (version !== generation) return
    const justFinished = run.value?.id === id && ((run.value.state === 'running' && data.state !== 'running') || (run.value.retrying && !data.retrying))
    run.value = data
    if (initialized !== id || justFinished) { selected.value = failureTasks.value.map(task => task.command); initialized = id }
    else selected.value = selected.value.filter(command => data.tasks.some(task => task.command === command && task.can_retry && !task.retry_state && !['success', 'skipped', 'running'].includes(task.status)))
    error.value = ''
  } catch (exception: any) { if (version === generation) error.value = exception.message }
  finally { if (version === generation) loading.value = false }
}
function chooseRun(id: string) { router.replace({ query: { run: id } }) }
async function loadEarlier() {
  if (loadingOlder.value) return
  const name = selectedName.value
  loadingOlder.value = true
  try {
    const data = await api.get(`/api/${encodeURIComponent(name)}/executions?limit=30&offset=${runs.value.length}`)
    if (name !== selectedName.value) return
    const ids = new Set(runs.value.map(item => item.id))
    runs.value.push(...data.runs.filter((item: ExecutionRun) => !ids.has(item.id)))
    moreAvailable.value = data.runs.length === 30
  } catch (exception: any) { toast.error = exception.message }
  finally { loadingOlder.value = false }
}
function toggleTask(task: ExecutionTask) { selected.value = selected.value.includes(task.command) ? selected.value.filter(command => command !== task.command) : [...selected.value, task.command] }
async function retryTasks(commands = selected.value) {
  if (!run.value || busy.value || !commands.length) return
  busy.value = true
  try {
    const data = await api.post(`/api/${encodeURIComponent(selectedName.value)}/executions/${run.value.id}/retry`, {
      tasks: commands, revision: run.value.revision,
      request_id: Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join(''),
      new_day: run.value.stale,
    })
    toast.notify('已加入重试队列，将在当前任务结束后执行', 'ok')
    selected.value = []
    if (data.run_id !== run.value.id) await router.replace({ query: { run: data.run_id } })
    await refresh(true)
  } catch (exception: any) { toast.error = exception.message; await refresh(true) }
  finally { busy.value = false }
}
async function openDetail(task: ExecutionTask) { returnFocus = document.activeElement as HTMLElement; focused.value = task.command; await nextTick(); drawer.value?.focus() }
function closeDetail() { focused.value = null; returnFocus?.focus() }
function dialogKey(event: KeyboardEvent) {
  if (event.key === 'Escape') { closeDetail(); return }
  if (event.key !== 'Tab') return
  const nodes = drawer.value?.querySelectorAll<HTMLElement>('button,a[href],summary,[tabindex="0"]')
  if (!nodes?.length) return
  const first = nodes[0], last = nodes[nodes.length - 1]
  if (event.shiftKey && (document.activeElement === first || document.activeElement === drawer.value)) { event.preventDefault(); last.focus() }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
}
function screenshot(attempt: ExecutionAttempt) { return `/api/${encodeURIComponent(selectedName.value)}/executions/attempt/${attempt.id}/screenshot` }
function configure(task: ExecutionTask) { router.push(`/i/${selectedName.value}/task/${task.command}`) }
watch(() => [selectedName.value, route.query.run], (value, previous) => {
  if (value[0] !== previous?.[0]) { runs.value = []; run.value = null; initialized = '' }
  focused.value = null
  void refresh(true)
}, { immediate: true })
const timer = window.setInterval(() => { if (!document.hidden) void refresh() }, 4000)
onBeforeUnmount(() => { generation++; window.clearInterval(timer) })
</script>

<template>
  <section class="view reports-view">
    <div class="reports-heading"><div><h2>执行记录</h2><p>先看遗漏，再补做。重试成功后更新本份简报，尝试历史始终保留。</p></div><button class="btn sm" :disabled="loading" @click="refresh()"><AppIcon name="refresh" :size="14" /> 刷新</button></div>
    <div class="report-picker"><AppSelect :model-value="run?.id || ''" :options="runOptions" placeholder="选择一次执行" @change="chooseRun" /><button v-if="moreAvailable" class="btn sm" :disabled="loadingOlder" @click="loadEarlier">{{ loadingOlder ? '正在读取…' : '加载更早记录' }}</button></div>
    <p v-if="error" class="report-error" role="alert">{{ error }}</p>
    <article v-if="!run && !error" class="card report-empty"><AppIcon name="file-text" :size="30" /><h3>{{ loading ? '正在读取执行记录…' : '下次执行后，这里会出现简报' }}</h3><p>运行结果保存在服务器。关闭浏览器后仍会继续记录。</p></article>
    <template v-if="run">
      <article class="card report-hero">
        <div class="report-title"><div><span class="report-eyebrow">{{ selectedName }} · 游戏日 {{ run.game_day }}（UTC+8，04:00 刷新）</span><h2>{{ run.retrying ? '正在补做未完成项' : run.state === 'running' ? '本轮执行中' : run.state === 'interrupted' ? '本轮已中断' : '本轮已结束' }}</h2><p>{{ executionTime(run.started_at) }} 开始 · {{ executionDuration(run) }}<span v-if="run.finished_at"> · {{ executionTime(run.finished_at) }} 更新</span></p></div><AppIcon name="file-text" :size="32" /></div>
        <div class="report-chips"><span v-for="status in chips.filter(key => run!.counts[key])" :key="status" class="report-chip" :class="status">{{ outcomeLabels[status] }} <b>{{ run.counts[status] }}</b></span></div>
        <div v-if="run.stale" class="report-note">这份记录已跨过每日刷新时间。重新执行会建立当前游戏日的新记录，并保留原简报。</div>
        <div v-if="run.parent_id" class="report-note">这是跨日重新执行的记录。<button class="report-link" @click="chooseRun(run.parent_id!)">查看原简报</button></div>
        <div class="report-actions"><button class="btn accent" :disabled="retryDisabled" @click="retryTasks()"><AppIcon name="refresh" :size="15" />{{ busy ? '正在加入队列…' : run.stale ? `按当前游戏日执行（${selected.length}）` : `重试所选（${selected.length}）` }}</button><button class="btn" :disabled="!failureTasks.length || busy || run.state === 'running'" @click="selected = failureTasks.map(task => task.command)">选择可重试失败项（{{ failureTasks.length }}）</button><span class="report-action-help">{{ run.state === 'running' ? '本轮结束后可选择重试' : '使用当前配置；已确认完成的步骤会保留' }}</span></div>
      </article>
      <div class="report-filters"><button class="btn sm" :class="{ accent: filter === 'attention' }" @click="filter = 'attention'">需要关注</button><button class="btn sm" :class="{ accent: filter === 'all' }" @click="filter = 'all'">全部结果</button></div>
      <div v-if="filter === 'attention' && !groups[0]?.items.length" class="card report-empty"><h3>没有需要关注的任务</h3><p>已完成与正常跳过的项目可在“全部结果”查看。</p></div>
      <article v-for="group in groups.filter(item => item.items.length)" :key="group.key" class="card report-group">
        <button class="report-group-heading" :aria-expanded="!!openGroups[group.key]" @click="openGroups[group.key] = !openGroups[group.key]"><span>{{ openGroups[group.key] ? '▾' : '▸' }} {{ group.label }} <small>{{ group.items.length }}</small></span></button>
        <div v-if="openGroups[group.key]">
          <div v-for="task in group.items" :key="task.command" class="report-row" :class="{ 'is-selected': selected.includes(task.command) }">
            <input type="checkbox" :aria-label="`选择${task.name}`" :checked="selected.includes(task.command)" :disabled="!task.can_retry || !!task.retry_state || busy || run.state === 'running'" @change="toggleTask(task)">
            <button class="report-task" @click="openDetail(task)"><span class="report-task-title">{{ task.name }} <small v-if="task.attempts > 1 && task.status === 'success'" class="report-recovered">重试后完成</small></span><span class="report-task-summary">{{ task.summary || (task.status === 'pending' ? '本轮尚未执行' : '正在执行…') }}</span><span v-if="task.reason" class="report-task-reason">{{ task.reason }}</span></button>
            <span class="report-chip" :class="task.status">{{ task.retry_state === 'queued' ? '已加入队列' : task.retry_state === 'running' ? '重试中' : outcomeLabels[task.status] }}</span>
            <div class="report-row-actions"><button v-if="task.can_retry && !task.retry_state && run.state !== 'running'" class="btn sm" :disabled="busy" @click="retryTasks([task.command])">{{ task.status === 'blocked' || task.status === 'unknown' ? '检查并重试' : '重试' }}</button><button v-if="task.status === 'blocked'" class="btn sm" @click="configure(task)">去配置</button><button class="btn sm" @click="openDetail(task)">详情</button></div>
          </div>
        </div>
      </article>
    </template>
    <div v-if="detailTask" class="report-drawer-mask" @click.self="closeDetail" @keydown="dialogKey">
      <aside ref="drawer" class="report-drawer" role="dialog" aria-modal="true" aria-labelledby="report-detail-title" tabindex="-1">
        <div class="report-detail-head"><div><p>任务详情</p><h2 id="report-detail-title">{{ detailTask.name }}</h2></div><button class="btn sm" aria-label="关闭详情" @click="closeDetail"><AppIcon name="x" :size="16" /></button></div>
        <p class="report-detail-summary">{{ detailTask.summary || '尚未执行' }}</p><p v-if="detailTask.reason" class="report-note">{{ detailTask.reason }}</p><p v-if="detailTask.retry_hint && !['success', 'skipped'].includes(detailTask.status)" class="report-note">{{ detailTask.retry_hint }}</p><button class="btn sm" @click="configure(detailTask)">打开任务配置</button>
        <h3 class="report-history-title">尝试历史 · {{ detailTask.history?.length || 0 }} 次</h3>
        <article v-for="attempt in detailTask.history" :key="attempt.id" class="report-attempt">
          <div class="report-attempt-title"><b>第 {{ attempt.number }} 次</b><span class="report-chip" :class="attempt.status">{{ outcomeLabels[attempt.status] }}</span></div><small>{{ executionTime(attempt.started_at) }} → {{ executionTime(attempt.finished_at) }}</small><p>{{ attempt.summary }}</p><p v-if="attempt.reason" class="report-task-reason">{{ attempt.reason }}</p>
          <div v-for="(item, key) in attempt.steps" :key="key" class="report-step"><span class="report-chip" :class="item.status">{{ outcomeLabels[item.status] }}</span><div><b>{{ item.label }}</b><p>{{ item.summary }}<span v-if="item.reused"> · 沿用前次结果</span></p><small v-if="item.reason">{{ item.reason }}</small></div></div>
          <details v-if="attempt.has_screenshot"><summary>查看结束时截图</summary><a :href="screenshot(attempt)" target="_blank" rel="noopener"><img class="report-screenshot" :src="screenshot(attempt)" alt="该次尝试结束时的游戏画面" loading="lazy"></a></details>
          <details v-if="attempt.logs.length"><summary>查看相关日志</summary><pre class="report-log">{{ attempt.logs.join('\n') }}</pre></details>
        </article>
      </aside>
    </div>
  </section>
</template>

<style scoped>
.reports-view { max-width: 1400px; margin: auto; }
.reports-heading,.report-title,.report-actions,.report-picker,.report-filters,.report-detail-head,.report-attempt-title { display: flex; align-items: center; gap: 12px; }
.reports-heading,.report-title,.report-detail-head,.report-attempt-title { justify-content: space-between; }
.reports-heading { margin-bottom: 20px; }.reports-heading h2,.report-title h2 { font-size: 22px; }
.reports-heading p,.report-title p { margin-top: 8px; color: var(--text-2); font-size: 13px; line-height: 1.7; }
.report-picker { margin-bottom: 18px; }.report-picker :deep(.app-select) { width: 330px; max-width: 100%; }
.report-hero { padding: 24px; margin-bottom: 20px; }.report-eyebrow { display: block; color: var(--text-2); font-size: 12px; margin-bottom: 10px; }
.report-title > :last-child { color: var(--accent); flex-shrink: 0; }.report-chips { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }
.report-chip { display: inline-flex; gap: 7px; align-items: center; border-radius: 7px; padding: 5px 9px; background: var(--card-2); color: var(--text-2); font-size: 12px; white-space: nowrap; }
.report-chip.success { color: var(--green); background: var(--green-soft); }.report-chip.failed { color: var(--red); background: var(--red-soft); }
.report-chip.partial,.report-chip.blocked { color: var(--yellow); background: var(--yellow-soft); }.report-chip.unknown { color: var(--violet); background: var(--violet-soft); }.report-chip.running { color: var(--accent); background: var(--accent-soft); }
.report-note { padding: 12px 14px; margin: 14px 0; border: 1px solid var(--border); border-radius: 9px; color: var(--text-2); font-size: 13px; line-height: 1.7; }
.report-actions { flex-wrap: wrap; }.report-action-help { color: var(--text-2); font-size: 12px; }.report-filters { margin: 18px 0; }
.report-group { margin-bottom: 16px; overflow: hidden; }.report-group-heading { width: 100%; padding: 16px 20px; text-align: left; background: transparent; color: var(--text); border: 0; font-weight: 600; }.report-group-heading small { color: var(--text-2); margin-left: 8px; }
.report-row { display: grid; grid-template-columns: 18px minmax(150px,1fr) auto auto; align-items: center; gap: 16px; padding: 17px 20px; border-top: 1px solid var(--border); }.report-row.is-selected { background: var(--accent-soft); }.report-row input { width: 16px; height: 16px; accent-color: var(--accent); }
.report-task { min-width: 0; text-align: left; padding: 0; border: 0; background: none; color: var(--text); }.report-task-title { display: block; margin-bottom: 7px; font-weight: 600; font-size: 14px; }
.report-task-summary,.report-task-reason { display: block; color: var(--text-2); font-size: 12px; line-height: 1.7; overflow-wrap: anywhere; }.report-task-reason { margin-top: 4px; }.report-recovered { color: var(--green); font-size: 11px; margin-left: 8px; font-weight: 400; }
.report-row-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; }.report-empty { display: grid; justify-items: center; gap: 12px; text-align: center; padding: 40px 20px; color: var(--text-2); }.report-empty h3 { color: var(--text); font-size: 16px; }.report-empty p { font-size: 13px; }
.report-error { color: var(--red); margin-bottom: 15px; font-size: 14px; }.report-link { border: 0; background: transparent; color: var(--accent); text-decoration: underline; }
.report-drawer-mask { position: fixed; inset: 0; z-index: 50; background: #0008; display: flex; justify-content: flex-end; }.report-drawer { width: min(620px,100vw); height: 100%; overflow: auto; padding: 26px; background: var(--card); border-left: 1px solid var(--border); box-shadow: var(--shadow-hover); outline: none; }
.report-detail-head p { margin-bottom: 6px; color: var(--text-2); font-size: 12px; }.report-detail-summary { margin: 20px 0 10px; line-height: 1.7; }.report-history-title { margin: 28px 0 14px; font-size: 15px; }
.report-attempt { padding: 18px 0; border-top: 1px solid var(--border); }.report-attempt > small { display: block; margin: 10px 0; color: var(--text-2); }.report-attempt > p { font-size: 13px; line-height: 1.7; }
.report-step { display: flex; align-items: flex-start; gap: 10px; padding: 12px 0; border-bottom: 1px solid var(--border); }.report-step b { font-size: 13px; }.report-step p,.report-step small { font-size: 12px; color: var(--text-2); line-height: 1.7; }
.report-attempt details { margin-top: 14px; font-size: 13px; }.report-attempt summary { cursor: pointer; color: var(--accent); padding: 6px 0; }.report-screenshot { display: block; width: min(100%,360px); margin-top: 10px; border-radius: 8px; }
.report-log { margin-top: 8px; padding: 14px; max-height: 320px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; line-height: 1.7; color: var(--text-2); background: var(--log-bg); border-radius: 8px; }
button:focus-visible,input:focus-visible,summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
@media (max-width: 1100px) { .report-row { grid-template-columns: 18px minmax(100px,1fr) auto; gap: 10px; }.report-row-actions { grid-column: 2 / -1; justify-content: flex-start; }.report-hero { padding: 18px; } }
</style>
