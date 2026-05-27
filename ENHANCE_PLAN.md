# ENHANCE PLAN

## Objective
1. Allow loading/running models with mmap even when full model weights do not fit in currently free RAM.
2. Add Modelfile/API support equivalent to llama.cpp --n-cpu-moe.

## Investigation Summary

### 1) Why mmap does not currently help enough in Ollama
- In load planning, system memory fit is hard-gated in llm/server.go verifyLayout:
  - cpuSize is compared against FreeMemory + FreeSwap and returns an error if exceeded.
  - This blocks loads before runner/model mmap behavior can help.
- The mmap toggle exists in API options (api/types.go Runner.UseMMap) and is sent in llm.LoadRequest.UseMmap.
- For the compatibility runner (runner/llamarunner/runner.go), req.UseMmap is honored and passed to llama.ModelParams.UseMmap.
- For the new engine runner (runner/ollamarunner/runner.go), req.UseMmap is currently ignored.
- Additional heuristic disables mmap by default on Linux when FreeMemory < total model size (llm/server.go), unless user explicitly sets use_mmap=true.

### 2) Why n-cpu-moe equivalent is not currently available
- llama.cpp already implements:
  - --cpu-moe and --n-cpu-moe
  - via tensor_buft_overrides for MoE expert tensor regexes (llama/llama.cpp/vendor/common/arg.cpp).
- Ollama API/Modelfile options currently have no n_cpu_moe field (api/types.go Runner).
- Ollama load request has no n_cpu_moe field (llm/server.go LoadRequest).
- llama Go binding ModelParams currently has no tensor override plumbing (llama/llama.go).

## Implementation Plan

## Phase A: mmap for models larger than free RAM

### A1. Define intended semantics
- Keep existing use_mmap option.
- Add explicit policy for overcommit behavior:
  - Option 1 (recommended): add new option mmap_overcommit (bool), default false.
  - Option 2: treat explicit use_mmap=true as implicit overcommit permission.
- Behavior when enabled:
  - Do not require full model weights to fit in FreeMemory+FreeSwap during pre-load checks.
  - Still enforce minimum safety margins for runtime allocations (KV cache, graph/work buffers).

### A2. Relax scheduler/load memory gating
- Modify llm/server.go verifyLayout:
  - Split cpuSize into:
    - file-backed weight estimate (mmap-eligible)
    - anonymous/runtime memory estimate (must fit)
  - When mmap overcommit is enabled, only hard-gate anonymous/runtime requirement, not full file-backed weight bytes.
- Keep existing strict gating when mmap overcommit is disabled.

### A3. Runner behavior parity
- Compatibility runner path:
  - Already honors UseMmap; keep.
- New engine path:
  - Add mmap signal into ml.BackendParams (or equivalent), and plumb from load request.
  - If full mmap semantics cannot be implemented immediately in new engine, gate feature to compatibility runner with clear warning/log.

### A4. Guardrails
- Keep mmap disabled where known-problematic unless user explicitly forces it:
  - Existing Metal partial-offload behavior.
  - Existing Vulkan behavior.
- Add explicit warning when overcommit mode is enabled:
  - risk of severe paging/thrashing
  - expected degraded throughput/latency

### A5. Tests
- Unit tests in llm/server_test.go:
  - verifyLayout allows overcommit when mmap_overcommit/use_mmap forced.
  - verifyLayout still rejects insufficient anon/runtime memory.
- Integration:
  - load model where file size > free RAM but runnable with paging; ensure load no longer fails at planner stage.
  - confirm runner logs show mmap enabled.

## Phase B: Modelfile equivalent of --n-cpu-moe

### B1. API and model option schema
- Add runner options in api/types.go:
  - n_cpu_moe (int, default 0)
  - (optional) cpu_moe (bool) for all layers parity with llama.cpp --cpu-moe
- Ensure FromMap handles these fields (automatic via struct tags).

### B2. Modelfile support
- No parser grammar changes needed for PARAMETER shape.
- Add docs and examples for:
  - PARAMETER n_cpu_moe <N>
  - (optional) PARAMETER cpu_moe true

### B3. Load request plumbing
- Extend llm.LoadRequest with new field(s).
- Populate from api.Options in llm/server.go.
- Pass through both runners.

### B4. Compatibility runner implementation (llama.cpp path)
- Extend llama.ModelParams in llama/llama.go to support tensor buffer overrides.
- Build regex overrides for first N MoE layers using llama.cpp pattern logic used by --n-cpu-moe.
- Set overrides to ggml_backend_cpu_buffer_type.
- Preserve existing behavior when N=0.

### B5. New engine implementation
- Add equivalent tensor-placement policy in ml/backend/ggml/ggml.go:
  - For first N blocks, force MoE expert tensors to CPU buffer type.
  - Keep non-expert tensors on normal layer assignment.
- Reuse same tensor-name patterns used in llama.cpp for parity.

### B6. Tests
- API/options tests (api/types_test.go): parse/serialize new fields.
- Server load tests: load request includes n_cpu_moe.
- Runner tests:
  - compatibility runner: confirms overrides set when n_cpu_moe>0.
  - new engine: confirms expert tensors for first N blocks route to CPU memory.
- End-to-end on MoE model:
  - compare VRAM footprint and placement logs for n_cpu_moe=0 vs n_cpu_moe=N.

## Documentation Updates
- docs/modelfile.mdx:
  - add n_cpu_moe (and cpu_moe if implemented) to valid parameter table.
  - add cautionary notes for performance tradeoffs.
- docs/api.md and docs/openapi.yaml:
  - add new option fields and examples.
- runner/README.md:
  - describe mmap overcommit mode and operational caveats.

## Rollout Strategy
1. Ship Phase A with feature flag/option off by default.
2. Validate on Linux hosts under memory pressure.
3. Ship Phase B in compatibility runner first (fast parity with llama.cpp).
4. Add new engine parity for n_cpu_moe.
5. Remove temporary caveats once both runners support full behavior.

## Risks and Mitigations
- Thrashing and tail-latency spikes with mmap overcommit:
  - Mitigate via explicit opt-in and warnings.
- Behavioral divergence between runners:
  - Mitigate by implementing compatibility first, then new-engine parity with shared tests.
- MoE tensor pattern drift across architectures:
  - Mitigate by centralizing regex helpers and validating against representative MoE model families.

## Acceptance Criteria
- A model with weights > free RAM can be loaded when mmap overcommit is explicitly enabled, without planner rejection.
- n_cpu_moe in Modelfile/API is accepted, persisted, and applied at load time.
- MoE expert placement matches configured N layers and reduces VRAM usage accordingly.
- Existing default behavior remains unchanged unless explicitly opted in.
