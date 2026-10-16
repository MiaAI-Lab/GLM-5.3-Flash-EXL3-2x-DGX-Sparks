# Contributing

Read the [README](README.md) and [docs/REFERENCE.md](docs/REFERENCE.md) first,
then fill in the pull request template.

## Performance claims

Most pull requests need nothing extra. A pull request that **claims a speed
improvement** brings raw numbers measured against the recipe's baseline, so
maintainers can judge the claim without re-testing it.

**What counts as a claim:** saying the change makes the kit faster (in the title,
description, CHANGELOG or your own comments), changing a number in the README's
Performance table, or changing a default or adding an option because it is
faster. No claim, no measurement. If the numbers don't support the claim, drop
the claim; the pull request can still be opened.

### Baseline

Every run uses the recipe's Default TP2 setup, so numbers from different
contributors compare:

- `.env` copied fresh from the `.env.example` of the checkout you are booting.
  Change only what your machines need: the `nodes`, `weights`, `worker weights`
  and `API auth` sections, plus `NCCL_IB_GID_INDEX` if your fabric needs it.
- A shell with no kit variables exported: exports override `.env`.
- `BUILD=1 ./start.sh`, so the image is built from that checkout.
- Nothing else running on the GPUs.

A claim about another profile or topology uses it on every run instead: for
example `GLM53_MODEL_PRESET=dense-h3`, or `.env.tp3` copied fresh from
`.env.tp3.example` with `BUILD=1 ./start-tp3.sh`. Say so in the declarations.

### Single-request speed claims: the Grill

The Grill measures generation speed for one request at a time: a short counting
prompt, temperature 0, thinking off, 400 output tokens, timed from request to
last token. A claim about that ("faster decode", "+N% tok/s") is measured with
the newest release of [The Grill](https://github.com/plotarmordev/thegrill/releases)
(the highest version listed, pre-releases included). Install it as its
[install guide](https://github.com/plotarmordev/thegrill/blob/main/docs/performance/INSTALL.md#obtain-verify-unpack)
shows, checksum included.

Claims the Grill doesn't measure (cold prefill, long context, concurrency, prose
only) use the Testing section's protocol on the same baseline instead, with the
raw output attached.

Take three runs. Before each one, stop the kit, check out the commit, copy
`.env` fresh and start the kit:

| Run | Checkout | Settings |
|---|---|---|
| A | `main` | baseline |
| B | your pull request | baseline |
| A2 | `main` again | baseline |

For a new opt-in option, all three runs use your pull request: A and A2 without
the option, B with it.

Write one declaration for each side, `a.json` and `b.json`. `runtime` is the
checkout's `git rev-parse HEAD`. `settings` is `baseline` (the checkout's
`.env.example`), plus the option on B for an option claim, plus the profile on
both for a profile claim. The two files differ only in `runtime` (a code or
default change) or only in `settings` (an option).

```json
{"model_revision": "GLM53_MODEL_PRESET from .env.example",
 "runtime": "GLM-5.3-Flash-EXL3-2x-DGX-Sparks@<git rev-parse HEAD>",
 "hardware": "2x DGX Spark GB10, TP2",
 "settings": "baseline"}
```

On the head, with `ENDPOINT=http://127.0.0.1:8888/v1/chat/completions` and
`MODEL=GLM-5.3-Flash-EXL3`:

```sh
# A
grill-perf baseline --workload portable-v1 --client-placement same-host \
  --endpoint "$ENDPOINT" --local-http --model "$MODEL" --deployment a.json --out A
# B
grill-perf check A --change deployment --deployment b.json --endpoint "$ENDPOINT" \
  --local-http --model "$MODEL" --client-placement same-host --out B
# A2 (its own exit code doesn't matter)
grill-perf check A --change none --deployment A/deployment.json --out A2

grill-perf compare A B --reference A2 > qualification.txt
grill-perf compare A B --reference A2 --json > qualification.json
```

Runs must start at least 60 seconds apart; a reboot usually covers that. If a
run needs more than five minutes, pass the same `--seconds N` to all three.

**The claim stands only if `compare` prints `MEASURED FASTER`:** B is faster than
both A and A2. Quote the measured change, not a bigger number. Anything else,
drop the claim. If you push code after measuring, measure again.

### Attach the evidence

```sh
tar czf grill-evidence.tgz A B A2      # add every earlier attempt
jq -r --slurpfile a2 A2/report.json '
  def pct: . * 100 | round / 100 | if . >= 0 then "+\(.)" else "\(.)" end;
  def rounds: map(. * 10 | round / 10) | join(" ");
  def leg: "\(.observed_change_percent | pct)% [\(.model_based_interval_percent | map(pct) | join(", "))]";
  .deployment as $d |
  "grill:    \($d.baseline.collector.version), binary sha256 A \($d.baseline.collector_sha256), B \($d.candidate.collector_sha256)",
  "A:        \($d.baseline.declaration | .runtime + "; " + .settings)",
  "B:        \($d.candidate.declaration | .runtime + "; " + .settings)",
  "result:   \(.result)",
  "B vs A:   \(leg)    B vs A2: \($d.candidate_against_reference | leg)    A2 vs A: \($d.reference_against_baseline | leg)",
  "complete: A \(.baseline_accounting.complete_requests)/32, B \(.candidate_accounting.complete_requests)/32, A2 \($a2[0].candidate_accounting.complete_requests)/32",
  "tok/s A:  \(.baseline_acquisition_medians | rounds)",
  "tok/s B:  \(.candidate_acquisition_medians | rounds)",
  "tok/s A2: \($a2[0].candidate_acquisition_medians | rounds)"' qualification.json
```

Attach `grill-evidence.tgz` to the pull request, and paste the summary and
`qualification.txt` into its Performance claim section. Exact values are in
`qualification.json`. The evidence holds only the counting prompt, the replies,
a loopback address and your declarations.

### Maintainer check

No GPU and no re-run. `compare` re-verifies every byte (one edited byte makes it
`INVALID`), and the `jq` checks the rest. Set `HEAD` to the pull request's head
commit:

```sh
tar xzf grill-evidence.tgz
V=v$(jq -r .collector.version A/capture.json)
START=$(jq '.acquisitions[0].started_unix_ms / 1000 | floor' A/capture.json)
NEWEST=$(gh release list --repo plotarmordev/thegrill --json tagName,publishedAt \
  --jq "[.[] | select((.publishedAt | fromdate) < $START)][0].tagName")
SHA=$(gh release download "$V" --repo plotarmordev/thegrill \
  -p "grill-perf-${V#v}-$(jq -r .collector.target A/capture.json).receipt.json" \
  -O - | jq -r .binary.sha256)
grill-perf compare A B --reference A2 --json | jq --arg sha "$SHA" --arg head "$HEAD" \
  --arg v "$V" --arg newest "$NEWEST" -e '
  .result == "IMPROVED"
  and $v == $newest
  and .deployment.baseline.collector_sha256 == $sha
  and .deployment.candidate.collector_sha256 == $sha
  and (.deployment.candidate.declaration.runtime | endswith($head))
  and ([.deployment.baseline.declaration, .deployment.candidate.declaration]
       | . as [$a, $b] | [$a | keys[] | select($a[.] != $b[.])] | IN(["runtime"], ["settings"]))'
```

`true` means: B is faster than both baseline runs; both sides were measured by
the official binary of the Grill release that was newest when measuring started;
B is the pull request head; and the declarations differ only in `runtime` or only
in `settings`. The evidence is tamper-evident, not signed. Hardware, settings and
the reboots are the contributor's word, so maintainers may still re-run a claim
occasionally.
