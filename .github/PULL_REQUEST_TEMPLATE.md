## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this kit, a good description usually covers:
      * which knob, default, script behavior, overlay or kernel path changes
      * whether the change affects measured numbers (KV pool, TTFT / cold prefill,
        decode throughput, draft acceptance, concurrency) and, if so, the expected
        direction
      * why the change is safe on both nodes of the two-node TP=2 lane
      * for measurement claims: the exact protocol (prompts, concurrency, medians)

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the issue tracker.
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this kit that usually means:

      * `bash -n start.sh stop.sh download.sh` (syntax check) for script changes
      * an actual boot on the cluster with the change, and the resulting boot-log
        facts (pool size, kernel applied, capture sizes)
      * if behavior changed, the measured numbers with the new settings in the
        README's format (e.g. tests/bench_decode.py median-of-5, or the
        cold-prefill table from docs/cold-prefill.md)
      * draft-behavior changes: an acceptance measurement, not just tok/s

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and kept my changes consistent with it.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (a boot and, for behavior changes, measurements).
- [ ] Defaults still work out of the box; a new knob has a sane fallback consistent with the existing ones.
- [ ] I updated the README (and docs/ where relevant) for any knob, default, or measured number I changed.
