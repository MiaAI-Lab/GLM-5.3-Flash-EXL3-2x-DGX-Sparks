# KV continuity findings (2026-08-30) — experiment, no code change needed

## Question
Does the follow-up turn re-prefill the previous turn's GENERATED assistant
tokens, or does prefix caching reuse them? ("KV continuity between turns")

## Method
Controlled A/B on the production 2x DGX Spark serve (fine-grained APC active,
hash grain 64, `partial_hash=True`):

- Turn 1: unique 2.7k-token doc + task asking for a ~700-word fable,
  `max_tokens=1100`, temp 0, thinking off → **910 generated tokens**.
- Turn 2: same messages + the assistant reply + a 6-token instruction.
  Prompt = **3,648 tokens**.

Prefix-cache counters (`vllm:prefix_cache_hits/queries_total`) snapshotted
immediately before/after each request; wall time via curl.

## Result

| Turn | Prompt tok | Wall time |
|---|---:|---:|
| 1 (cold, unique) | ~2,738 | 57.5 s (= 910-token generation at ~20 tok/s, non-streaming) |
| 2 (full history + 910-token reply + new turn) | **3,648** | **0.415 s** |

0.415 s budgets ~200-300 tokens of GPU prefill at ~900 tok/s. Had the 910
generated tokens been re-prefilled, turn 2 would have taken >= 1.3 s.
Counter deltas (+3,648 hits / +3,584 queries = full prompt / one scheduler
block) are consistent with block-level reuse of the generated tokens.

## Conclusion

**Generated tokens are already KV-continuous across turns**: after each
request completes, vLLM's prefix cache retains the generated blocks and the
next turn's prompt (which embeds the assistant reply) hits them at hash
granularity. With fine-grained APC the reuse grain is 64 tokens instead of
the 3584-token scheduler block, so effectively only each turn's genuinely
new tokens are computed.

A KV-connector/session layer would therefore add little on this serve:
the remaining per-turn costs are (a) the turn's own new tokens (irreducible
under the stateless API), (b) client-side dynamic prompt mutations (e.g.
injected context that shifts mid-prompt invalidates the shared prefix from
that point), and (c) O(context) CPU work (tokenize + chat-template render),
which a connector would not remove either.

No code change shipped for this experiment; the fine-grained APC patch
(`fine-grained-apc` branch) is what makes the reuse fine-grained.
