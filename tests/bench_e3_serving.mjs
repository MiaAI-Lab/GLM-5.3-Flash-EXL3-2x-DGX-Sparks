// Fixed-input cold-prefill benchmark. Restart the server before EACH invocation.
// Uses sparkDash's unchanged request/timing implementation; see docs/e3-activation-validation.md.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const pinned = 'e03b9d624e7135d6e82b4c8fc94ea0ddcf300547';
if (!process.env.SPARKDASH_ROOT || !process.env.RESULT_DIR) {
  throw Error('Set SPARKDASH_ROOT (clean pinned checkout) and a fresh RESULT_DIR');
}
const root = path.resolve(process.env.SPARKDASH_ROOT);
if (execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], {encoding:'utf8'}).trim() !== pinned ||
    execFileSync('git', ['-C', root, 'status', '--porcelain'], {encoding:'utf8'}).trim()) {
  throw Error(`sparkDash must be clean at ${pinned}`);
}
const out = path.resolve(process.env.RESULT_DIR);
fs.mkdirSync(out); // Fail rather than overwrite an earlier measurement.
for (const k of ['BENCH_HISTORY_PATH','BENCH_ACTIVE_PATH','PREFILL_BENCH_HISTORY_PATH','PREFILL_BENCH_ACTIVE_PATH']) {
  process.env[k] = path.join(out, `unused-${k}.json`);
}
const file = process.env.FIXTURE_FILE || fileURLToPath(new URL('./fixtures/e3-prefill.json.gz', import.meta.url));
const bytes = fs.readFileSync(file);
const raw = file.endsWith('.gz') ? gunzipSync(bytes) : bytes;
const sha = createHash('sha256').update(raw).digest('hex');
if (sha !== '90b7f1d9fe2517f0aff3d82e4062b7d1fdbdf72fe95c68c15b61ae8f6ea321fd') throw Error('Fixture hash mismatch');
const fixtures = JSON.parse(raw);
// Preserve the original measurement harness's module initialization order.
await import(pathToFileURL(path.join(root, 'server/collectors/PrefillBench.js')));
const {runStreamingRequest, applyThinkingFlags, median, LLM_STREAM_AGENT} =
  await import(pathToFileURL(path.join(root, 'server/collectors/LlmStreaming.js')));
const base = (process.env.LLM_BASE_URL || 'http://127.0.0.1:8888').replace(/\/$/, '');
const apiKey = process.env.VLLM_API_KEY || null;
const headers = apiKey ? {Authorization:`Bearer ${apiKey}`} : {};
async function get(endpoint) {
  const r = await fetch(`${base}${endpoint}`, {headers, signal:AbortSignal.timeout(10000)});
  if (!r.ok) throw Error(`${endpoint}: HTTP ${r.status}`);
  return r;
}
async function counters() {
  const text = await (await get('/metrics')).text();
  const series = text.split('\n').filter(l => l.startsWith('vllm:prefix_cache_hits_total{'));
  const values = series.map(l => Number(l.trim().split(/\s+/)[1]));
  if (!values.length || values.some(v => !Number.isFinite(v))) throw Error('Missing/invalid prefix-hit counters');
  return {text, hits:values.reduce((a,b) => a+b, 0)};
}
try {
  const model = (await (await get('/v1/models')).json()).data[0].id;
  const before = await counters();
  fs.writeFileSync(path.join(out,'prom-before.txt'), before.text);
  const rows = [];
  for (const f of fixtures) {
    const body = {model, messages:[{role:'user', content:f.prompt}], max_tokens:8,
      temperature:0, top_p:1, stream:true, stream_options:{include_usage:true}};
    applyThinkingFlags(body, model, false);
    const r = await runStreamingRequest(`${base}/v1/chat/completions`, body,
      AbortSignal.timeout(600000), {retryOnThinking400:true, thinking:false, debug:true, apiKey});
    if (r.error || !r.usage?.promptTokens || !r.usage?.completionTokens || !r.ttftMs) throw Error(JSON.stringify(r));
    rows.push({name:f.name, rep:f.rep, kind:f.kind, promptTokens:r.prefillTokens,
      prefillTps:r.prefillTps, ttftMs:r.ttftMs, completionTokens:r.completionTokens,
      usage:r.usage, contentPreview:r.contentPreview});
    console.log(JSON.stringify(rows.at(-1)));
    fs.writeFileSync(path.join(out,'paired-results.json'), JSON.stringify({sha,model,rows},null,2));
  }
  const after = await counters();
  fs.writeFileSync(path.join(out,'prom-after.txt'), after.text);
  if (after.hits !== before.hits) throw Error(`INVALID: prefix hits increased by ${after.hits-before.hits}`);
  const metrics = {};
  for (const size of [4096,16384,65536]) metrics[`prefill_${size}_tps`] =
    median(rows.filter(r => r.kind==='sparkdash' && r.name===String(size) && r.rep>=0).map(r => r.prefillTps));
  metrics.paired_prefill_tps = Math.exp([4096,16384,65536].reduce((s,n) => s+Math.log(metrics[`prefill_${n}_tps`]),0)/3);
  for (const name of ['docs-short','code-medium','mixed-long']) metrics[name.replace('-','_')+'_tps'] =
    median(rows.filter(r => r.kind==='holdout' && r.name===name && r.rep>=0).map(r => r.prefillTps));
  fs.writeFileSync(path.join(out,'metrics.json'), JSON.stringify(metrics,null,2));
  console.log(JSON.stringify(metrics));
} finally {
  await LLM_STREAM_AGENT.close();
}
