export type Outcome = 'success' | 'partial' | 'failed' | 'blocked' | 'skipped' | 'unknown' | 'pending' | 'running'
export const outcomeLabels: Record<Outcome, string> = {
  success: '已完成', partial: '部分完成', failed: '失败', blocked: '待处理',
  skipped: '正常跳过', unknown: '结果未确认', pending: '未执行', running: '执行中',
}
export interface ExecutionAttempt {
  id: string; number: number; status: Outcome; summary: string; reason: string
  started_at: string; finished_at: string | null; has_screenshot: boolean; logs: string[]
  steps: Record<string, { label: string; status: Outcome; summary: string; reason: string; reused?: boolean }>
}
export interface ExecutionTask {
  command: string; name: string; status: Outcome; summary: string; reason: string
  retryable: boolean; can_retry: boolean; retry_hint: string; attempts: number; retry_state: string | null; history?: ExecutionAttempt[]
}
export interface ExecutionRun {
  id: string; instance: string; game_day: string; started_at: string; finished_at: string | null
  state: string; source: string; revision: number; parent_id: string | null
  counts: Record<Outcome, number>; tasks: ExecutionTask[]; retrying: boolean; stale: boolean
  duration_seconds: number
}
export const attentionStatuses: Outcome[] = ['failed', 'partial', 'blocked', 'unknown', 'running', 'pending']
export function executionTime(value: string | null) {
  return value ? new Date(value).toLocaleString(undefined, { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) : '—'
}
export function executionDuration(run: ExecutionRun) {
  const seconds = Math.max(0, Math.round(run.duration_seconds))
  return seconds < 60 ? `${seconds} 秒` : `${Math.floor(seconds / 60)} 分钟`
}
