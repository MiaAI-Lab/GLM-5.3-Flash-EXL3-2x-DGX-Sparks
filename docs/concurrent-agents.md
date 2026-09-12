# Concurrent agents

Use `GLM53_MIXED_PREFILL_CHUNK=0` for interactive use by multiple agents.
This restores vLLM's stock chunked-prefill scheduler: a decoding peer no
longer explicitly prevents another prompt from using the remaining token
budget in the same step. `MAX_NUM_SEQS` still limits active sequences, and
`MAX_NUM_BATCHED_TOKENS` limits the whole step. A context still needs its full
prefill before it can emit a first token; this setting does not impose a
wall-clock latency guarantee or bypass a full KV cache.

The total token budget can still be exhausted by an already-running long
prefill. If a second prompt waits on capacity while cache and sequence slots
are available, inspect the per-request allocation as well as the mixed-prefill
rule. vLLM's existing `--long-prefill-token-threshold` option can cap that
allocation for both running and waiting prompts. For example, with
`MAX_NUM_BATCHED_TOKENS=1024`, a threshold of `512` leaves budget for a second
request (speculative decoding also consumes budget). Add it to `EXTRA_ARGS`
without removing any existing arguments, then recreate and requalify the
service. A smaller cap can reduce single-request prefill throughput; test a
short peer arriving during a cold long prefill and measure both latencies.

`MAX_MODEL_LEN` is a per-request limit, not a promise that every sequence slot
can hold a context of that size. Compare the engine's startup KV-cache token
capacity and maximum-concurrency estimate with the combined active contexts.
Two million-token windows require capacity for both; enabling interleaving
does not add memory. Agent-side context compression can bound the resident
history while preserving a session across many hours and requests.

The previous `skip` default gives a new prompt zero prefill tokens whenever a
peer is decoding. A long generation can therefore starve another agent for
minutes, even with spare sequence slots and KV memory. A client idle timeout
then cancels and retries that waiting request. Requests already admitted to
the running list can also have their remaining prefill suppressed.

`skip` and `-1` remain explicit options for protecting decode throughput at the
expense of new-prompt latency. A positive number caps mixed prefill tokens.
Interleaving can reduce per-agent generation speed, especially for long
contexts on the older sparse-MLA kernels. Choose that tradeoff explicitly.

Existing `.env` files are not rewritten during updates. Change
`GLM53_MIXED_PREFILL_CHUNK=skip` to `GLM53_MIXED_PREFILL_CHUNK=0` there, then
recreate the service through your approved deployment procedure. A plain
`docker restart` reuses the old container environment and does not apply this
setting. In-flight requests should be drained first.

Before measuring, record the deployed source commit, image digest, model and
drafter revisions, launch arguments, context/sequence/token budgets, and
relevant environment. Compare with that exact versioned recipe. Isolate the
mixed-prefill setting; an upgrade of kernels, weights, or memory geometry
requires separate qualification.

Run the CPU regressions with an installed scheduler file. These operate on
temporary copies and do not import the model or modify the running scheduler:

```bash
GLM53_SCHEDULER_PY_SRC=/path/to/scheduler.py python3 tests/test_prefill_concurrency.py
GLM53_SCHEDULER_PY_SRC=/path/to/scheduler.py python3 tests/test_scheduler_decode_floor.py
```

After the deployment and recipe are approved, run one live canary:

```bash
python3 tests/check_concurrent_agents.py --base-url http://127.0.0.1:8888/v1 --model GLM-5.3-Flash-EXL3 --out logs/concurrent-agents-canary.json
```

The canary starts a long, deterministic counting response with thinking
disabled. After A emits content, it submits B with a distinct 8k-word prompt
and asks for `OK`. It requires valid completed streams, the expected short
answer, and B's first content before A's last content. Two HTTP 200s alone do
not pass. The default overall deadline is 240 seconds; on failure it stops
its own requests, returns nonzero, and does not retry or run a benchmark
matrix. If authentication is enabled, supply `OPENAI_API_KEY` through the
environment. Receipts do not include that credential.

Review the canary's output and latency before wider concurrency testing.
Confirm the model still handles realistic tool calls and long prompts in
your agent client. Increasing a client's idle timeout can accommodate long
prefills but does not repair scheduler starvation.
