#!/usr/bin/env bash
#
# install.sh — DenseSpark setup for Qwen3.8-27B on NVIDIA DGX Spark.
#
# Two runtimes, same model and same patches:
#
#   --fast   (default)  Layer DenseSpark on the official vLLM arm64 image.
#                       No compilation. The image carries sm_120 kernels, which
#                       a GB10 (sm_121) runs under CUDA minor-version
#                       compatibility, but not the sm_120a arch-specific
#                       CUTLASS kernels.
#   --sm121             Build vLLM from source with TORCH_CUDA_ARCH_LIST=12.1a
#                       so the arch-specific kernels exist. Takes 30-60+ min.
#
# Steps:
#   1. Preflight: python3, docker, GPU, disk
#   2. Fetch the INT4 checkpoint (~19 GB, skipped if already cached)
#   3. Prepare the base runtime (pull, or build for SM121)
#   4. Build the DenseSpark image: nine verified transforms of the pinned vLLM
#   5. Verify the image
#   6. Train the product-quantization structure the draft head loads (--no-pq
#      skips it, and then only DENSESPARK_PQ_DRAFT=0 profiles can start)
#   7. Ask how many requests should run at once, and record the answer
#   8. Optionally launch configs/launch-densespark.sh
#
# Idempotent: every step skips itself when its output already exists.
# Run from anywhere: ./install.sh, or bash /path/to/install.sh
#
# Flags:
#   --fast            Use the official vLLM image (default)
#   --sm121           Build vLLM for SM121 from source instead
#   --skip-model      Do not fetch the checkpoint
#   --model REPO      Override the checkpoint repository
#   --no-cache        Force a rebuild of the DenseSpark image layer
#   --concurrency N   The maximum number of requests you intend to serve at
#                     once. The profile is tuned per concurrency, so this picks
#                     the speculation depth that measured fastest at N. Asked
#                     interactively when omitted; defaults to 16.
#   --no-pq           Skip the product-quantization structure the PQ draft head
#                     loads. It is built by default because the shipped profile
#                     needs it; training reads the checkpoint's lm_head on the
#                     GPU and takes about ten seconds and roughly 5 GB of
#                     scratch. Without it, only DENSESPARK_PQ_DRAFT=0 profiles
#                     can start.
#   --launch          Start the server when the build finishes
#   --no-launch       Never prompt to launch
#   -h, --help        Show this help

set -euo pipefail

RELEASE_VERSION="1.3"
VLLM_VERSION="0.27.1"
BASE_IMAGE="vllm/vllm-openai:v${VLLM_VERSION}"
SM121_IMAGE="vllm-sm121:${VLLM_VERSION}"
FINAL_IMAGE="densespark"
DEFAULT_MODEL_REPO="Frozenlock/Qwen3.8-27B-int4-AutoRound"
# The launcher reads DENSESPARK_MODEL, so the installer reads it too. Otherwise
# setting it for the launcher alone leaves the draft-head structure built for
# the default checkpoint, and the server quietly serves without a draft head.
MODEL_REPO="${DENSESPARK_MODEL:-$DEFAULT_MODEL_REPO}"
MODEL_GB=19
HF_HOME_DEFAULT="${HOME}/.cache/huggingface"

MODE="fast"
SKIP_MODEL=0
NO_CACHE=0
BUILD_PQ=1
LAUNCH="ask"
CONCURRENCY=""
CONCURRENCY_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/densespark/concurrency"
PQ_ARTIFACT="${DENSESPARK_PQ_ARTIFACT_HOST:-${HOME}/.cache/densespark/pq_head_m128.pt}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_BUILD_INPUTS=(
    docker/Dockerfile
    patches/01-int8-lm-head/patch_int8_lmhead.py
    patches/05-gdn-prefill-dispatch/patch_gdn_prefill_dispatch.py
    patches/04-pq-draft-head/densespark_pq.py
    patches/04-pq-draft-head/patch_pq_draft_head.py
    patches/06-marlin-nsplit/densespark_nsplit.py
    patches/06-marlin-nsplit/patch_marlin_nsplit.py
    patches/07-humming-kernels/README.md
    patches/08-sparse-pq-draft/densespark_sparse_pq.py
    patches/08-sparse-pq-draft/patch_sparse_pq_draft.py
    patches/09-three-way-linear/densespark_three_way.py
    patches/09-three-way-linear/densespark_marlin_component.py
    patches/09-three-way-linear/densespark_cutlass_component.py
    patches/09-three-way-linear/patch_three_way_linear.py
    patches/10-flashinfer-gdn-sm121/patch_flashinfer_gdn_sm121.py
    patches/11-m8000-prefill-route/patch_m8000_prefill_route.py
)
PQ_BUILDER="patches/04-pq-draft-head/build_pq_artifact.py"

# ── output helpers ────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi
STEP_NUM=0
step()      { STEP_NUM=$((STEP_NUM + 1)); CURRENT_STEP="$1"; printf '\n%s── Step %d: %s ──%s\n' "$C_DIM" "$STEP_NUM" "$1" "$C_OFF"; }
step_skip() { STEP_NUM=$((STEP_NUM + 1)); printf '\n%s── Step %d: %s — already done, skipping ──%s\n' "$C_DIM" "$STEP_NUM" "$1" "$C_OFF"; }
note()      { printf '   %s\n' "$1"; }
ok()        { printf '%s   ok%s %s\n' "$C_OK" "$C_OFF" "$1"; }
warn()      { printf '%s   warning%s %s\n' "$C_WARN" "$C_OFF" "$1"; }
# Every run writes a transcript. When something fails, the transcript plus a
# snapshot of the machine becomes one file to send, instead of a screenshot of
# the last three lines and a round of questions about the rest.
DIAG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/densespark"
DIAG_FILE="${DIAG_DIR}/install-diagnostics.txt"
TRANSCRIPT="$(mktemp -t densespark-install.XXXXXX.log 2>/dev/null || echo /tmp/densespark-install.log)"
CURRENT_STEP="starting up"

write_diagnostics() {
    local reason="$1"
    mkdir -p "$DIAG_DIR" 2>/dev/null || return 0
    {
        echo "DenseSpark install diagnostics"
        echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "release:   ${RELEASE_VERSION}   vllm pin: ${VLLM_VERSION}   mode: ${MODE}"
        echo "failed at: ${CURRENT_STEP}"
        echo "reason:    ${reason}"
        echo
        echo "── host ──"
        echo "os:        $( . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" ) $(uname -m)"
        echo "kernel:    $(uname -r)"
        echo "python3:   $(python3 -c 'import platform;print(platform.python_version())' 2>&1)"
        echo "docker:    $(docker version --format '{{.Server.Version}}' 2>&1)"
        echo "git:       $(git --version 2>&1)"
        echo "driver:    $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)"
        echo "gpu:       $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 | head -1)"
        echo "on gpu:    $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | head -5)"
        echo "memory:    $(free -h 2>&1 | awk 'NR==2{print $2" total, "$7" available"}')"
        echo "disk HOME: $(df -h "$HOME" 2>&1 | awk 'NR==2{print $4" free on "$1}')"
        echo "disk dckr: $(df -h "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)" 2>&1 | awk 'NR==2{print $4" free on "$1}')"
        echo "/dev/shm:  $(df -h /dev/shm 2>&1 | awk 'NR==2{print $2" total, "$4" free"}')"
        echo
        echo "── images ──"
        docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | grep -E 'densespark|vllm' | head -10
        echo
        echo "── where to send this ──"
        echo "open an issue at https://github.com/albond/DenseSpark-Qwen3.8-27B/issues"
        echo "and attach this file. For anything security-related see SECURITY.md."
        echo
        echo "── last 200 lines of this run ──"
        tail -n 200 "$TRANSCRIPT" 2>/dev/null \
            | sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g'
    } > "$DIAG_FILE" 2>&1
    printf '%s   diagnostics written to %s%s\n' "$C_DIM" "$DIAG_FILE" "$C_OFF" >&2
    printf '%s   attach that one file to a bug report; it has the failure and the machine%s\n' \
        "$C_DIM" "$C_OFF" >&2
}

abort() {
    printf '\n%serror%s %b\n\n' "$C_ERR" "$C_OFF" "$1" >&2
    write_diagnostics "$1"
    DENSESPARK_ABORTED=1
    exit 1
}

on_unexpected_exit() {
    local code=$?
    [ "$code" -eq 0 ] && return 0
    # abort() already reported and wrote diagnostics.
    [ "${DENSESPARK_ABORTED:-0}" = "1" ] && return 0
    printf '\n%serror%s the installer stopped unexpectedly at: %s (exit %d)\n\n' \
        "$C_ERR" "$C_OFF" "$CURRENT_STEP" "$code" >&2
    write_diagnostics "unexpected exit ${code}"
}

# Mirror the whole run into the transcript the report will quote, and catch any
# exit the script did not choose. Colour still goes to the terminal; the report
# strips it.
trap on_unexpected_exit EXIT
if [ -w "$(dirname "$TRANSCRIPT")" ] 2>/dev/null; then
    exec > >(tee -a "$TRANSCRIPT") 2>&1
fi

# ── arguments ─────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --fast)       MODE="fast" ;;
        --sm121)      MODE="sm121" ;;
        --skip-model) SKIP_MODEL=1 ;;
        --model)      shift; [ $# -gt 0 ] || abort "--model needs a repository id"; MODEL_REPO="$1" ;;
        --no-cache)   NO_CACHE=1 ;;
        --pq)         BUILD_PQ=1 ;;
        --no-pq)      BUILD_PQ=0 ;;
        --concurrency)
                      shift
                      [ $# -gt 0 ] || abort "--concurrency needs a positive integer"
                      case "$1" in ""|*[!0-9]*) abort "--concurrency must be a positive integer, got: $1" ;; esac
                      [ "$1" -ge 1 ] || abort "--concurrency must be at least 1"
                      CONCURRENCY="$1" ;;
        --launch)     LAUNCH="yes" ;;
        --no-launch)  LAUNCH="no" ;;
        -h|--help)    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; exit 0 ;;
        *)            abort "unknown flag: $1  (try --help)" ;;
    esac
    shift
done

printf '%sDenseSpark %s — Qwen3.8-27B on DGX Spark, vLLM %s, %s runtime%s\n' \
    "$C_DIM" "$RELEASE_VERSION" "$VLLM_VERSION" "$MODE" "$C_OFF"

# ── Step 1: preflight ─────────────────────────────────────────────────────────
# Every prerequisite is checked before anything is downloaded or built, and all
# failures are collected before any of them is reported. Stopping at the first
# missing package makes a user install one thing, re-run, and find the next.
step "preflight"

MISSING=()
have() {
    local label="$1" present="$2" fix="$3"
    if [ "$present" = "1" ]; then
        printf '  %s✓%s %s\n' "$C_OK" "$C_OFF" "$label"
    else
        printf '  %s✗%s %s %s— missing%s\n' "$C_ERR" "$C_OFF" "$label" "$C_DIM" "$C_OFF"
        MISSING+=("${label}"$'\t'"${fix}")
    fi
}

present=0; command -v python3 >/dev/null 2>&1 && present=1
have "python3 $( [ "$present" = 1 ] && python3 -c 'import platform;print(platform.python_version())' )" \
    "$present" "sudo apt update && sudo apt install -y python3"

# The checkpoint download installs the Hugging Face CLI into a private
# virtualenv when the system has none, which needs venv and ensurepip.
present=0
if command -v python3 >/dev/null 2>&1 && python3 -c 'import venv, ensurepip' 2>/dev/null; then
    present=1
fi
have "python3-venv and ensurepip" "$present" "sudo apt install -y python3-venv python3-pip"

present=0; command -v git >/dev/null 2>&1 && present=1
have "git" "$present" "sudo apt install -y git"

present=0; command -v sha256sum >/dev/null 2>&1 && command -v df >/dev/null 2>&1 && present=1
have "coreutils (sha256sum, df)" "$present" "sudo apt install -y coreutils"

present=0; command -v docker >/dev/null 2>&1 && present=1
have "docker on PATH" "$present" \
    "install Docker Engine: https://docs.docker.com/engine/install/ubuntu/"

present=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then present=1; fi
have "docker daemon reachable as ${USER} without sudo" "$present" \
    "sudo usermod -aG docker ${USER} && newgrp docker   # then open a new terminal"

present=0; command -v nvidia-smi >/dev/null 2>&1 && present=1
have "nvidia-smi (NVIDIA driver)" "$present" \
    "install the NVIDIA driver for your distribution"

# Only meaningful once docker answers; probed with a small stock image so a
# missing container runtime is reported before anything large is pulled.
present=0
if docker info >/dev/null 2>&1 && docker run --rm --gpus all ubuntu:24.04 true >/dev/null 2>&1; then
    present=1
fi
have "docker can reach the GPU (--gpus all)" "$present" \
    "sudo apt install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"

if [ "${#MISSING[@]}" -gt 0 ]; then
    printf '\n'
    printf '%s%d prerequisite(s) missing. Install them and re-run ./install.sh:%s\n\n' \
        "$C_ERR" "${#MISSING[@]}" "$C_OFF"
    for entry in "${MISSING[@]}"; do
        printf '  %s\n      %s\n' "${entry%%$'\t'*}" "${entry#*$'\t'}"
    done
    printf '\n'
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    ok "GPU ${GPU_NAME}"
    case "$GPU_NAME" in
        *GB10*) : ;;
        *) warn "this project targets DGX Spark (GB10, SM121); ${GPU_NAME} is untested here" ;;
    esac
fi

# The checkpoint lands under $HOME and the image lands in Docker's data root.
# Those are frequently different filesystems, so checking only one of them lets
# a split-mount machine pass here and then fail mid-build with ENOSPC.
IMAGE_GB=40
free_gb() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
[ -d "${DOCKER_ROOT:-}" ] || DOCKER_ROOT="$HOME"

HOME_AVAIL_GB="$(free_gb "$HOME")"
DOCKER_AVAIL_GB="$(free_gb "$DOCKER_ROOT")"
if [ "$(readlink -f "$DOCKER_ROOT" 2>/dev/null || echo "$DOCKER_ROOT")" = \
     "$(readlink -f "$HOME" 2>/dev/null || echo "$HOME")" ] \
   || [ "$(df -P "$HOME" 2>/dev/null | tail -1 | awk '{print $1}')" = \
        "$(df -P "$DOCKER_ROOT" 2>/dev/null | tail -1 | awk '{print $1}')" ]; then
    NEEDED_GB=$((MODEL_GB + IMAGE_GB))
    [ "${HOME_AVAIL_GB:-0}" -ge "$NEEDED_GB" ] \
        || abort "need about ${NEEDED_GB} GB free on the filesystem holding \$HOME and the Docker images, found ${HOME_AVAIL_GB} GB."
    ok "${HOME_AVAIL_GB} GB free (about ${NEEDED_GB} GB needed)"
else
    [ "${HOME_AVAIL_GB:-0}" -ge "$MODEL_GB" ] \
        || abort "need about ${MODEL_GB} GB free under \$HOME for the checkpoint, found ${HOME_AVAIL_GB} GB."
    [ "${DOCKER_AVAIL_GB:-0}" -ge "$IMAGE_GB" ] \
        || abort "need about ${IMAGE_GB} GB free under ${DOCKER_ROOT} for the image, found ${DOCKER_AVAIL_GB} GB."
    ok "${HOME_AVAIL_GB} GB free under \$HOME and ${DOCKER_AVAIL_GB} GB under ${DOCKER_ROOT}"
fi

for required in "${IMAGE_BUILD_INPUTS[@]}"; do
    [ -f "${SCRIPT_DIR}/${required}" ] \
        || abort "run this from a DenseSpark checkout; ${required} is missing."
done
[ -f "${SCRIPT_DIR}/${PQ_BUILDER}" ] \
    || abort "run this from a DenseSpark checkout; ${PQ_BUILDER} is missing."

# ── Step 2: checkpoint ────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
MODEL_MARKER="${HF_HOME}/hub/models--${MODEL_REPO//\//--}"

if [ "$SKIP_MODEL" = "1" ]; then
    step_skip "checkpoint (--skip-model)"
elif [ -d "$MODEL_MARKER" ] && [ -n "$(find "$MODEL_MARKER" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
    step_skip "checkpoint ${MODEL_REPO}"
else
    # The size is only known for the checkpoint this project pins; --model can
    # point anywhere, and quoting 19 GB for someone else's repository is a
    # guess dressed as a fact.
    if [ "$MODEL_REPO" = "$DEFAULT_MODEL_REPO" ]; then
        step "fetch ${MODEL_REPO} (about ${MODEL_GB} GB)"
    else
        step "fetch ${MODEL_REPO}"
    fi

    # Ask the hub about the repository before starting a 19 GB transfer. The
    # three ways this fails need three different answers, and "re-run, it
    # resumes" - the only advice the transfer itself can give - is wrong for two
    # of them.
    HUB_STATUS="$(python3 - "$MODEL_REPO" <<'HFPROBE'
import json, os, sys, urllib.error, urllib.request

repo = sys.argv[1]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
request = urllib.request.Request(f"https://huggingface.co/api/models/{repo}")
if token:
    request.add_header("Authorization", f"Bearer {token}")
try:
    with urllib.request.urlopen(request, timeout=25) as response:
        json.load(response)
except urllib.error.HTTPError as error:
    print(f"http:{error.code}")
except Exception as error:
    print(f"unreachable:{type(error).__name__}")
else:
    print("ok")
HFPROBE
)" || HUB_STATUS="unreachable:python"

    case "$HUB_STATUS" in
        ok)
            ok "${MODEL_REPO} is reachable"
            ;;
        http:401|http:403)
            abort "the hub will not serve ${MODEL_REPO} to this machine (HTTP ${HUB_STATUS#http:}).\n  The hub answers 401 both for a repository that does not exist and for one you\n  cannot read, so it is one of:\n    1. the name is misspelt   — check it, or pass --model <owner>/<name>\n    2. the repository is gated — accept its terms on its Hugging Face page\n    3. it is private          — create a read token and export it:\n                                  export HF_TOKEN=hf_...\n  Re-running without fixing one of those fails the same way."
            ;;
        http:404)
            abort "no repository named ${MODEL_REPO} on the hub.\n  Check the spelling, or pass another one with --model <owner>/<name>."
            ;;
        http:*)
            abort "the hub answered HTTP ${HUB_STATUS#http:} for ${MODEL_REPO}.\n  A 5xx is the hub's side; try again shortly."
            ;;
        *)
            abort "cannot reach huggingface.co (${HUB_STATUS#unreachable:}).\n  This step needs outbound HTTPS. Behind a proxy, export HTTPS_PROXY first.\n  If the checkpoint is already on disk elsewhere, point HF_HOME at that cache\n  and re-run, or pass --skip-model."
            ;;
    esac

    if ! command -v hf >/dev/null && ! command -v huggingface-cli >/dev/null; then
        note "no Hugging Face CLI on PATH; installing one into a private virtualenv"
        python3 -m venv "${SCRIPT_DIR}/.venv-hf" >/dev/null 2>&1 \
            || abort "python3 -m venv failed.\n  sudo apt install -y python3-venv python3-pip"
        "${SCRIPT_DIR}/.venv-hf/bin/pip" install -q --upgrade huggingface_hub \
            || abort "could not install huggingface_hub from PyPI.\n  This step needs outbound HTTPS to pypi.org. Behind a proxy, export HTTPS_PROXY.\n  Or install it yourself and re-run:  pip install --user huggingface_hub"
        HF_CLI="${SCRIPT_DIR}/.venv-hf/bin/hf"
    else
        HF_CLI="$(command -v hf || command -v huggingface-cli)"
    fi
    note "downloading with ${HF_CLI}"

    "$HF_CLI" download "$MODEL_REPO" --quiet \
        || abort "the transfer stopped before it finished.\n  The repository answered a moment ago, so this is the transfer itself:\n  re-run ./install.sh and it resumes, downloads are incremental."
    ok "checkpoint in ${HF_HOME}"
fi

# ── Step 3: base runtime ──────────────────────────────────────────────────────
if [ "$MODE" = "sm121" ]; then
    if docker image inspect "$SM121_IMAGE" >/dev/null 2>&1; then
        step_skip "vLLM ${VLLM_VERSION} built for SM121"
    else
        step "build vLLM ${VLLM_VERSION} for SM121 (30-60+ min)"
        note "arch-specific kernels (12.1a) are only available from a source build"
        SRC_DIR="${SCRIPT_DIR}/.build/vllm-${VLLM_VERSION}"
        if [ ! -d "$SRC_DIR/.git" ]; then
            mkdir -p "$(dirname "$SRC_DIR")"
            git clone --depth 1 --branch "v${VLLM_VERSION}" \
                https://github.com/vllm-project/vllm.git "$SRC_DIR" \
                || abort "could not clone vLLM v${VLLM_VERSION}"
        fi
        # vLLM's setup.py derives the ninja job count as MAX_JOBS // NVCC_THREADS.
        # Passing max_jobs alone is a trap: the upstream Dockerfile defaults
        # nvcc_threads to 8, so max_jobs=8 compiles single-threaded. Set both,
        # and size them so jobs x threads lands near the core count.
        NVCC_THREADS="${DENSESPARK_NVCC_THREADS:-2}"
        BUILD_JOBS="${DENSESPARK_BUILD_JOBS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"
        note "compiling with $(( BUILD_JOBS / NVCC_THREADS )) parallel jobs x ${NVCC_THREADS} nvcc threads on $(nproc) cores"
        docker build \
            --build-arg torch_cuda_arch_list='12.1a' \
            --build-arg CUDA_VERSION=13.0.3 \
            --build-arg max_jobs="$BUILD_JOBS" \
            --build-arg nvcc_threads="$NVCC_THREADS" \
            --target vllm-openai \
            -t "$SM121_IMAGE" \
            -f "${SRC_DIR}/docker/Dockerfile" "$SRC_DIR" \
            || abort "the SM121 build failed. The --fast path needs no compilation."
        ok "built ${SM121_IMAGE}"
    fi
    RUNTIME_BASE="$SM121_IMAGE"
else
    if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
        step_skip "pull ${BASE_IMAGE}"
    else
        step "pull ${BASE_IMAGE}"
        docker pull "$BASE_IMAGE" || abort "could not pull ${BASE_IMAGE}"
        ok "pulled ${BASE_IMAGE}"
    fi
    RUNTIME_BASE="$BASE_IMAGE"
fi

# ── Step 4: DenseSpark image ──────────────────────────────────────────────────
RUNTIME_BASE_ID="$(docker image inspect --format '{{.Id}}' "$RUNTIME_BASE")"
IMAGE_CONTENT_SHA256="$(
    {
        printf 'release=%s\nruntime=%s\nruntime_id=%s\n' \
            "$RELEASE_VERSION" "$RUNTIME_BASE" "$RUNTIME_BASE_ID"
        (
            cd "$SCRIPT_DIR"
            sha256sum "${IMAGE_BUILD_INPUTS[@]}"
        )
    } | sha256sum | cut -d ' ' -f 1
)"

BUILD_FINAL=1
if [ "$NO_CACHE" = "0" ] && docker image inspect "${FINAL_IMAGE}:latest" >/dev/null 2>&1; then
    HAVE_CONTENT_SHA256="$(docker image inspect --format \
        '{{ index .Config.Labels "dev.densespark.content-sha256" }}' \
        "${FINAL_IMAGE}:latest" 2>/dev/null || true)"
    if [ "$HAVE_CONTENT_SHA256" = "$IMAGE_CONTENT_SHA256" ]; then
        BUILD_FINAL=0
        step_skip "${FINAL_IMAGE}:latest (content ${IMAGE_CONTENT_SHA256:0:12})"
    fi
fi

if [ "$BUILD_FINAL" = "1" ]; then
    step "build ${FINAL_IMAGE}:latest"
    note "thin layer: all version-pinned runtime patches are baked in"
    BUILD_ARGS=()
    [ "$NO_CACHE" = "1" ] && BUILD_ARGS+=(--no-cache)
    docker build \
        "${BUILD_ARGS[@]}" \
        --label "dev.densespark.content-sha256=${IMAGE_CONTENT_SHA256}" \
        --build-arg "RUNTIME_BASE=${RUNTIME_BASE}" \
        --build-arg "RELEASE_VERSION=${RELEASE_VERSION}" \
        -t "${FINAL_IMAGE}:latest" \
        -f "${SCRIPT_DIR}/docker/Dockerfile" "$SCRIPT_DIR" \
        || abort "the DenseSpark layer failed to build."
    ok "built ${FINAL_IMAGE}:latest"
fi

# ── Step 5: verify ────────────────────────────────────────────────────────────
step "verify ${FINAL_IMAGE}:latest"
VERIFY="$(docker run --rm --gpus all \
    -e "DENSESPARK_EXPECTED_VLLM=${VLLM_VERSION}" \
    --entrypoint python3 "${FINAL_IMAGE}:latest" -c '
import os, torch, vllm, pathlib
assert torch.version.cuda, f"CPU-only torch: {torch.__version__}"
expected_vllm = os.environ["DENSESPARK_EXPECTED_VLLM"]
assert vllm.__version__ == expected_vllm, \
    f"wrong vLLM version: {vllm.__version__} != {expected_vllm}"
import vllm._custom_ops  # noqa: F401  — loads the native extension, whatever it is named
assert getattr(torch.ops._C, "gptq_gemm", None) is not None, \
    "vLLM native extension did not register torch.ops._C.gptq_gemm"
root = pathlib.Path(vllm.__file__).parent
for relative, marker, name in (
    ("model_executor/layers/logits_processor.py", "DENSESPARK_INT8_LMHEAD", "INT8 LM head"),
    ("v1/worker/gpu_model_runner.py", "DENSESPARK_GDN_PREFILL_DISPATCH", "hybrid prefill dispatch"),
    ("v1/spec_decode/llm_base_proposer.py", "DENSESPARK_PQ_DRAFT", "PQ draft head"),
    ("model_executor/kernels/linear/mixed_precision/marlin.py",
     "DENSESPARK_MARLIN_NSPLIT", "Marlin column-block dispatch"),
):
    assert marker in (root / relative).read_text(), f"{name} patch missing"
for module, name in (("_densespark_pq.py", "PQ"), ("_densespark_nsplit.py", "column-block")):
    assert (root / module).exists(), f"{name} runtime module missing"
print(f"vllm {vllm.__version__}; torch {torch.__version__} (CUDA {torch.version.cuda}); "
      "INT8 head, hybrid dispatch, PQ and Marlin column-block patches present")
' 2>&1)" || abort "image verification failed:\n${VERIFY}"
ok "$VERIFY"

# ── Step 6: the PQ structure (on by default; --no-pq skips) ──────────────────
if [ "$BUILD_PQ" = "1" ]; then
    MODEL_REVISION=""
    MODEL_SNAPSHOT=""
    if [ -s "${MODEL_MARKER}/refs/main" ]; then
        IFS= read -r MODEL_REVISION < "${MODEL_MARKER}/refs/main" \
            || [ -n "$MODEL_REVISION" ]
    else
        MODEL_SNAPSHOTS=()
        if [ -d "${MODEL_MARKER}/snapshots" ]; then
            while IFS= read -r -d '' snapshot; do
                MODEL_SNAPSHOTS+=("$snapshot")
            done < <(find "${MODEL_MARKER}/snapshots" -mindepth 1 -maxdepth 1 \
                -type d -print0)
        fi
        [ "${#MODEL_SNAPSHOTS[@]}" -eq 1 ] \
            || abort "cannot identify one cached revision for ${MODEL_REPO}.\n  Fetch the selected checkpoint without --skip-model and retry."
        MODEL_REVISION="${MODEL_SNAPSHOTS[0]##*/}"
    fi
    case "$MODEL_REVISION" in
        ""|*[!A-Za-z0-9._-]*)
            abort "invalid cached revision for ${MODEL_REPO}: ${MODEL_REVISION:-empty}" ;;
    esac
    MODEL_SNAPSHOT="${MODEL_MARKER}/snapshots/${MODEL_REVISION}"
    [ -d "$MODEL_SNAPSHOT" ] \
        || abort "selected checkpoint snapshot is missing: ${MODEL_SNAPSHOT}"
    [ -n "$(find "$MODEL_SNAPSHOT" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null)" ] \
        || abort "selected checkpoint snapshot has no safetensors: ${MODEL_SNAPSHOT}"

    PQ_DIRECTORY="$(dirname "$PQ_ARTIFACT")"
    PQ_FILENAME="$(basename "$PQ_ARTIFACT")"
    PQ_CONTAINER_SNAPSHOT="/checkpoint/snapshots/${MODEL_REVISION}"
    mkdir -p "$PQ_DIRECTORY"

    validate_pq_artifact() {
        docker run --rm --ipc=host \
            -v "${MODEL_MARKER}:/checkpoint:ro" \
            -v "${SCRIPT_DIR}/patches:/patches:ro" \
            -v "${PQ_DIRECTORY}:/out:ro" \
            --entrypoint python3 "${FINAL_IMAGE}:latest" \
            /patches/04-pq-draft-head/build_pq_artifact.py \
                --validate-artifact "/out/${PQ_FILENAME}" \
                --model-repo "$MODEL_REPO" \
                --snapshot "$PQ_CONTAINER_SNAPSHOT"
    }

    BUILD_PQ_ARTIFACT=1
    if [ -f "$PQ_ARTIFACT" ]; then
        if PQ_VALIDATION="$(validate_pq_artifact 2>&1)"; then
            BUILD_PQ_ARTIFACT=0
            step_skip "${PQ_FILENAME} (${MODEL_REPO}@${MODEL_REVISION:0:12})"
            note "$PQ_VALIDATION"
        else
            warn "${PQ_FILENAME} is stale or has invalid provenance; rebuilding it"
            [ -z "$PQ_VALIDATION" ] || note "$PQ_VALIDATION"
        fi
    fi

    if [ "$BUILD_PQ_ARTIFACT" = "1" ]; then
        step "train the PQ draft-head structure"
        note "one pass of the fixed Lloyd schedule over the checkpoint's lm_head"
        docker run --rm --gpus all --ipc=host \
            -v "${MODEL_MARKER}:/checkpoint:ro" \
            -v "${SCRIPT_DIR}/patches:/patches:ro" \
            -v "${PQ_DIRECTORY}:/out" \
            --entrypoint python3 "${FINAL_IMAGE}:latest" \
            /patches/04-pq-draft-head/build_pq_artifact.py --train \
                --model-repo "$MODEL_REPO" \
                --snapshot "$PQ_CONTAINER_SNAPSHOT" \
                --out "/out/${PQ_FILENAME}" \
            || abort "the PQ structure failed to build."
        PQ_VALIDATION="$(validate_pq_artifact 2>&1)" \
            || abort "the PQ structure failed provenance validation:\n${PQ_VALIDATION}"
        ok "wrote ${PQ_ARTIFACT} for ${MODEL_REPO}@${MODEL_REVISION}"
    fi
    note "serve it with configs/launch-densespark.sh"
fi

printf '\n%sDenseSpark %s is ready.%s\n' "$C_OK" "$RELEASE_VERSION" "$C_OFF"
note "start the server with ./configs/launch-densespark.sh"
note "the other profiles in configs/ are single-change A/B arms"
note "measure with ./bench_densespark.py once a server is up"

# ── Step 7: the concurrency this deployment is tuned for ─────────────────────
# The profile is not one configuration: the speculation depth that maximises
# throughput depends on how many requests are in flight, so the installer asks
# once and the launcher reads the answer. An explicit DENSESPARK_CONCURRENCY in
# the environment still wins at launch time.
if [ -z "$CONCURRENCY" ] && [ -t 0 ]; then
    printf '\nMaximum concurrent requests this server should be tuned for? [16] '
    read -r reply
    case "$reply" in
        "") CONCURRENCY=16 ;;
        *[!0-9]*) abort "concurrency must be a positive integer, got: $reply" ;;
        *) [ "$reply" -ge 1 ] || abort "concurrency must be at least 1"
           CONCURRENCY="$reply" ;;
    esac
fi
[ -n "$CONCURRENCY" ] || CONCURRENCY=16
mkdir -p "$(dirname "$CONCURRENCY_FILE")"
printf '%s\n' "$CONCURRENCY" > "$CONCURRENCY_FILE"
ok "tuned for up to ${CONCURRENCY} concurrent requests (${CONCURRENCY_FILE})"
note "change it any time: re-run with --concurrency N, or set DENSESPARK_CONCURRENCY"

# ── Step 8: launch ────────────────────────────────────────────────────────────
if [ "$LAUNCH" = "ask" ] && [ -t 0 ]; then
    printf '\nStart the DenseSpark server now? [y/N] '
    read -r reply
    case "$reply" in [yY]*) LAUNCH="yes" ;; *) LAUNCH="no" ;; esac
fi

if [ "$LAUNCH" = "yes" ]; then
    # Keep the launch target identical to the checkpoint selected above. The
    # profile defaults are intentionally standalone, so propagate installer
    # choices explicitly across exec.
    export DENSESPARK_MODEL="$MODEL_REPO"
    export DENSESPARK_PQ_ARTIFACT_HOST="$PQ_ARTIFACT"
    export DENSESPARK_CONCURRENCY="$CONCURRENCY"
    exec "${SCRIPT_DIR}/configs/launch-densespark.sh"
fi
