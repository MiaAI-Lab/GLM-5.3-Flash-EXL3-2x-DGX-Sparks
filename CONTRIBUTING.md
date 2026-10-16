# Contributing

Read the [README](README.md) and [docs/REFERENCE.md](docs/REFERENCE.md) first,
then fill in the pull request template. Most pull requests need nothing more.

## GPU qualification

A pull request that changes a GPU path, or claims a speed change, is measured on
your own cluster **before it is opened**, and arrives with the full raw numbers.
Maintainers judge it from that evidence instead of re-testing it. If the numbers
don't support the pull request, don't open it.

**GPU paths:** `overlay/`, `extensions/`, `ablit/`, `files/`, `tools/`,
`start*.sh`, `stop.sh`, `download.sh`, `Dockerfile*`, `.env*.example`.
Docs, tests and `kernel_lab/` changes skip qualification.

**Agents:** run these steps yourself. If the result is a stop, do not open the
pull request; tell your human why and keep the evidence.

### 1. Use the newest Grill

Install the newest release of [The Grill](https://github.com/plotarmordev/thegrill/releases)
(the highest version listed, pre-releases included). Verify its `.sha256` before
unpacking. Evidence from an older or modified copy does not qualify: its binary
fingerprint won't match the newest release.

### 2. Measure main, the pull request, then main again

Run everything on the head of the topology your change targets (the TP2
reference unless the change is TP3/TP4-only). Use the same topology, `.env` and
weights for all three runs, and nothing else on the GPUs.

Write two declarations, `main.json` and `pr.json`:

```json
{"model_revision": "pack/preset served, with its revision",
 "runtime": "GLM-5.3-Flash-EXL3-2x-DGX-Sparks@<commit sha>, image <tag or digest>",
 "hardware": "2x DGX Spark GB10, TP2",
 "settings": "every non-default .env line"}
```

`runtime` names the main commit in `main.json` and the pull request head in
`pr.json`. Then, with `ENDPOINT=http://127.0.0.1:8888/v1/chat/completions`:

1. Check out `main`, run `./start.sh`, then:
   `grill-perf baseline --workload portable-v1 --client-placement same-host --endpoint "$ENDPOINT" --local-http --model GLM-5.3-Flash-EXL3 --deployment main.json --out A`
2. `./stop.sh`, check out the pull request, `./start.sh`, then:
   `grill-perf check A --change deployment --deployment pr.json --endpoint "$ENDPOINT" --local-http --model GLM-5.3-Flash-EXL3 --client-placement same-host --out B`
3. `./stop.sh`, check out `main`, `./start.sh` again, then (exit code 2 here is
   normal):
   `grill-perf check A --change none --deployment A/deployment.json --out A2`

Each run takes about four minutes after its boot. The Grill enforces at least
60 seconds between them. If a run needs more than five minutes, add the same
`--seconds N` to all three.

### 3. Decide

```sh
grill-perf compare A B --reference A2 --json > qualification.json
grill-perf compare A B --reference A2 > qualification.txt
```

**Stop** if any of these hold:

- `.result` is `INVALID` (broken, incomplete or edited evidence) or `REGRESSED`;
- `.deployment.candidate_against_reference.result` is `REGRESSED` (slower than
  the main re-run);
- the upper end of `.model_based_interval_percent` is below 0 (slower than main);
- the pull request claims a speedup and `.result` is not `IMPROVED`.

Otherwise it qualifies. `INCONCLUSIVE` without a claim means no measured slowdown.

### 4. Attach the evidence

```sh
tar czf grill-evidence.tgz A B A2 qualification.json
```

Attach `grill-evidence.tgz` to the pull request, fill the **GPU qualification**
block in the template from `qualification.json`, and paste `qualification.txt`
below it. Keep every attempt, including stops. Never edit the evidence folders.

### How maintainers check it

No GPU and no re-run needed:

```sh
tar xzf grill-evidence.tgz
grill-perf compare A B --reference A2   # re-verifies every byte, recomputes the verdict
```

Then confirm the evidence came from the newest Grill. The capture's fingerprint
must equal that release's receipt:

```sh
jq -r .collector_sha256 A/capture.json
gh release download vX.Y.Z --repo plotarmordev/thegrill \
  -p 'grill-perf-X.Y.Z-aarch64-unknown-linux-gnu.receipt.json' -O - | jq -r .binary.sha256
```

The evidence is tamper-evident: one changed byte makes `compare` report
`INVALID`. It is not a signature. Declared hardware and settings are the
contributor's word, so maintainers may still re-run a pull request occasionally.
