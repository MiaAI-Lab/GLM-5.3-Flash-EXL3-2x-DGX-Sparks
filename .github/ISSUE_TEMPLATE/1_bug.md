---
name: Bug report
about: The kit is not behaving as expected, produces an error, or numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's tables list every knob and its default. -->

- Hardware / nodes: <!-- e.g. 2x DGX Spark (GB10, 128 GB unified memory), 4x via start-tp4.sh, ... -->
- Interconnect: <!-- e.g. ConnectX RoCE/IB, 10GbE, ... -->
- Image: <!-- `docker images | grep glm` — the kit image is ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3 -->
- Model (`MODEL` + `MODEL_REVISION`): <!-- e.g. Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw @ 25a44fd -->
- Served name (`SERVED_MODEL_NAME`): <!-- e.g. GLM-5.3-Flash-EXL3 -->
- How you started the serve: <!-- e.g. ./start.sh, ./start.sh restart, ABLIT=1 ./start.sh, custom compose -->
- Relevant settings: <!-- e.g. MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, EXL3_FAT_KERNEL, SPEC_METHOD, DFLASH_DRAFT_TP, LANGUAGE_MODEL_ONLY, ABLIT -->

---

## Steps to Reproduce

<!-- Full steps so that we can reproduce the problem. -->

1. <!-- e.g. `./start.sh` — what it printed up to the failure -->
2. ... <!-- the request or action that shows the bug -->
3. ... <!-- e.g. "curl /v1/models returns a different served name than configured" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add anything else: a minimal failing request, JSON responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.
-->

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.3-Flash-EXL3",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the lines with the backticks, and mention
      whether it came from start.sh, the container logs on either node, or a client.

      Common culprits worth checking before filing:
        * Weights missing or incomplete at boot -> run ./download.sh first and check
          the MODEL / MODEL_REVISION pins.
        * Decode much slower than the README tables -> check MAX_NUM_BATCHED_TOKENS
          (7168 is the validated default), DFLASH_DRAFT_TP, and whether
          EXL3_FAT_KERNEL applied (boot log).
        * Lower context than 1M -> do not reduce MAX_MODEL_LEN; pool size depends on
          MNBT and graph reservations (README "Context").
        * Vision requests rejected -> LANGUAGE_MODEL_ONLY=1 disables image/video;
          check --limit-mm-per-prompt shape {image:4,video:1}.
-->

```

```

</details>

<!--
      Consider also attaching screenshots and/or videos to better illustrate the issue.

      You can upload them directly on GitHub.
      Beware that video file size is limited to 10MB.
-->
