# DenseSpark — thin layer over a vLLM runtime for Qwen3.8-27B on DGX Spark.
#
# The base is either the official vLLM arm64 image (default, no compilation) or
# a vLLM built from source with TORCH_CUDA_ARCH_LIST=12.1a. Both are passed in
# by install.sh; nothing here depends on which one it is.
#
# Everything this layer adds is a text transform of an installed vLLM file,
# applied at build time so container startup goes straight to serving.
#
# Build:
#   docker build --build-arg RUNTIME_BASE=vllm/vllm-openai:v0.27.1 \
#                -t densespark:latest -f docker/Dockerfile .

ARG RUNTIME_BASE=vllm/vllm-openai:v0.27.1
FROM ${RUNTIME_BASE}

ARG RELEASE_VERSION=1.3
ARG RUNTIME_BASE
LABEL org.opencontainers.image.title="DenseSpark" \
      org.opencontainers.image.description="Qwen3.8-27B inference tuning for NVIDIA DGX Spark" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      dev.densespark.base="${RUNTIME_BASE}"

# Patch 1: INT8 LM head. Qwen3.8-27B keeps an unquantized 248320 x 5120 head,
# which is 2.54 GB re-read per decoded token. Per-channel INT8 halves that.
COPY patches/01-int8-lm-head/patch_int8_lmhead.py /opt/densespark/patch_int8_lmhead.py
RUN python3 /opt/densespark/patch_int8_lmhead.py \
 && python3 -c "import pathlib, vllm; \
p = pathlib.Path(vllm.__file__).parent / 'model_executor/layers/logits_processor.py'; \
assert 'DENSESPARK_INT8_LMHEAD' in p.read_text(), 'patch marker missing after apply'; \
compile(p.read_text(), str(p), 'exec'); \
print('INT8 LM head patch verified')"

# Patch 2: hybrid prefill dispatch guard. A short prompt can have the same shape
# as a speculative verification batch; keep it out of the FULL decode graph so
# the GDN recurrent state is initialized. Real decode batches retain FULL graphs.
COPY patches/05-gdn-prefill-dispatch/patch_gdn_prefill_dispatch.py /opt/densespark/patch_gdn_prefill_dispatch.py
RUN python3 /opt/densespark/patch_gdn_prefill_dispatch.py \
 && python3 -c "import pathlib, vllm; \
p = pathlib.Path(vllm.__file__).parent / 'v1/worker/gpu_model_runner.py'; \
assert 'DENSESPARK_GDN_PREFILL_DISPATCH' in p.read_text(), 'dispatch marker missing after apply'; \
compile(p.read_text(), str(p), 'exec'); \
print('hybrid prefill dispatch guard verified')"

# Patch 3: PQ draft head. The MTP proposer's argmax runs over a product
# quantization of the head - a 33.3 MiB code scan plus a gathered deployed-INT8
# rerank of the top candidates - instead of the full 1.184 GiB read. Only the proposal changes; the
# target still verifies it. Inert unless DENSESPARK_PQ_DRAFT=1 and an artifact is
# mounted at /opt/densespark/pq_head_m128.pt.
COPY patches/04-pq-draft-head/densespark_pq.py /opt/densespark/densespark_pq.py
COPY patches/04-pq-draft-head/patch_pq_draft_head.py /opt/densespark/patch_pq_draft_head.py
RUN python3 /opt/densespark/patch_pq_draft_head.py \
 && python3 -c "import pathlib, vllm; \
r = pathlib.Path(vllm.__file__).parent; \
p = r / 'v1/spec_decode/llm_base_proposer.py'; \
assert 'DENSESPARK_PQ_DRAFT' in p.read_text(), 'patch marker missing after apply'; \
compile(p.read_text(), str(p), 'exec'); \
m = r / '_densespark_pq.py'; \
assert m.exists(), 'runtime module not installed'; \
compile(m.read_text(), str(m), 'exec'); \
print('PQ draft head patch verified')"

# Patch 4: Marlin column-block dispatch. GB10 has a 24 MiB L2 and Marlin
# re-reads its packed weight once per row tile, so a projection whose weight
# exceeds the cache streams it from DRAM on every tile. The merged 34,816-wide
# gate_up linear in all 64 layers is 85 MiB and runs at 43% of the bf16 tensor
# ceiling; L2-resident blocks reach 67-70%. Inert unless
# DENSESPARK_MARLIN_NSPLIT is set.
COPY patches/06-marlin-nsplit/densespark_nsplit.py /opt/densespark/densespark_nsplit.py
COPY patches/06-marlin-nsplit/patch_marlin_nsplit.py /opt/densespark/patch_marlin_nsplit.py
RUN python3 /opt/densespark/patch_marlin_nsplit.py \
 && python3 -c "import pathlib, vllm; \
r = pathlib.Path(vllm.__file__).parent; \
p = r / 'model_executor/kernels/linear/mixed_precision/marlin.py'; \
assert 'DENSESPARK_MARLIN_NSPLIT' in p.read_text(), 'patch marker missing after apply'; \
compile(p.read_text(), str(p), 'exec'); \
m = r / '_densespark_nsplit.py'; \
assert m.exists(), 'runtime module not installed'; \
compile(m.read_text(), str(m), 'exec'); \
print('Marlin column-block dispatch patch verified')"

# Patch 5: Humming 0.1.13. Not a text transform - a pinned dependency. vLLM
# 0.27.1 ships Humming 0.1.10, whose SM121 selector falls back to SM89 and whose
# NVML bandwidth probe fails on GB10. 0.1.13 adds a native SM121 heuristic and a
# GB10 bandwidth fallback. The profile serves with --linear-backend humming, so
# this is a build dependency of the composed image rather than an optional layer.
COPY patches/07-humming-kernels/README.md /opt/densespark/humming-kernels.md
RUN python3 -m pip install --no-cache-dir --no-deps humming-kernels==0.1.13 \
 && python3 -c "import humming, inspect, humming.tune, humming.utils.device; \
assert humming.__version__ == '0.1.13', humming.__version__; \
assert humming.tune.heuristics_map[121].__name__ == 'Sm121Heuristics'; \
assert 'return 273.0' in inspect.getsource(humming.utils.device.calculate_gpu_bandwidth); \
print('Humming 0.1.13 SM121 verified')"
ENV HUMMING_CACHE_DIR=/root/.cache/vllm/humming-0.1.13/cache \
    HUMMING_TMP_DIR=/root/.cache/vllm/humming-0.1.13/tmp

# Patch 6: sparse probabilistic PQ draft. The PQ proposer scatters its shortlist
# into a dense vocabulary buffer before sampling; the sparse form samples the
# shortlist directly and reports the same q the rejection sampler consumes, so
# the proposal distribution is unchanged. Inert unless
# DENSESPARK_LAB86_SPARSE_PQ=1.
COPY patches/08-sparse-pq-draft/densespark_sparse_pq.py /opt/densespark/densespark_sparse_pq.py
COPY patches/08-sparse-pq-draft/patch_sparse_pq_draft.py /opt/densespark/patch_sparse_pq_draft.py
RUN python3 /opt/densespark/patch_sparse_pq_draft.py \
      --apply --runtime /opt/densespark/densespark_sparse_pq.py \
 && python3 -c "import pathlib, vllm; \
r = pathlib.Path(vllm.__file__).parent; \
m = r / '_densespark_lab86_sparse_pq.py'; \
assert m.exists(), 'sparse PQ runtime not installed'; \
compile(m.read_text(), str(m), 'exec'); \
print('sparse probabilistic PQ draft verified')"

# Patch 7: three-way linear dispatch. Below M=256 the projection runs Marlin
# column-block; at larger M it runs Humming INT8; at the exact prefill shapes it
# runs a direct CUTLASS block-FP8 route. One composition instead of three
# independent guards. Inert unless DENSESPARK_LAB118_ENABLE=1.
COPY patches/09-three-way-linear/densespark_three_way.py /opt/densespark/densespark_three_way.py
COPY patches/09-three-way-linear/densespark_marlin_component.py /opt/densespark/densespark_marlin_component.py
COPY patches/09-three-way-linear/densespark_cutlass_component.py /opt/densespark/densespark_cutlass_component.py
COPY patches/09-three-way-linear/patch_three_way_linear.py /opt/densespark/patch_three_way_linear.py
RUN python3 /opt/densespark/patch_three_way_linear.py \
      --apply \
      --runtime /opt/densespark/densespark_three_way.py \
      --marlin-component /opt/densespark/densespark_marlin_component.py \
      --cutlass-component /opt/densespark/densespark_cutlass_component.py \
 && python3 -c "import pathlib, vllm; \
r = pathlib.Path(vllm.__file__).parent; \
m = r / '_densespark_lab118_three_way.py'; \
assert m.exists(), 'three-way runtime not installed'; \
compile(m.read_text(), str(m), 'exec'); \
print('three-way linear dispatch verified')"

# Patch 8: FlashInfer GDN prefill on SM12.x. vLLM 0.27.1 admits only family 10.x
# even though the bundled FlashInfer carries an SM120 path. This changes backend
# eligibility only; --gdn-prefill-backend still selects it. Lab 289 measured the
# swap at -0.000487 nat/token against Triton FLA over 23,997 tokens.
COPY patches/10-flashinfer-gdn-sm121/patch_flashinfer_gdn_sm121.py /opt/densespark/patch_flashinfer_gdn_sm121.py
RUN python3 /opt/densespark/patch_flashinfer_gdn_sm121.py \
 && python3 /opt/densespark/patch_flashinfer_gdn_sm121.py --check \
 && echo "FlashInfer GDN SM121 eligibility verified"

# Patch 9: the M=8000 prefill route. A dispatch choice at the forum prompt
# length. Lab 289 measured it at -0.000700 nat/token on top of patch 8. Inert
# unless DENSESPARK_LAB133_M8000=1.
COPY patches/11-m8000-prefill-route/patch_m8000_prefill_route.py /opt/densespark/patch_m8000_prefill_route.py
RUN python3 /opt/densespark/patch_m8000_prefill_route.py \
 && python3 /opt/densespark/patch_m8000_prefill_route.py --check \
 && echo "M=8000 prefill route verified"

# Refuse an image whose torch lost CUDA during dependency resolution: a CPU
# build imports fine and then fails at the first kernel launch.
RUN python3 -c "import torch; assert torch.version.cuda, f'CPU-only torch: {torch.__version__}'; \
print(f'torch {torch.__version__} (CUDA {torch.version.cuda})')"

ENTRYPOINT ["vllm"]
