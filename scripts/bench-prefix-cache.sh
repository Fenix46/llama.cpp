#!/usr/bin/env bash
# bench-prefix-cache.sh
#
# Benchmark --kv-prefix-cache against baseline.
#
# Usage:
#   ./scripts/bench-prefix-cache.sh <model.gguf> [n_parallel] [port]
#
# Requirements: llama-server built, curl, jq, bc (standard on macOS/Linux).
#
# What it measures:
#   - Prefill latency (ms) for a long shared prefix processed N times
#   - Average tokens/s for decode following the cached prefix
#   - n_prompt_tokens_cache reported by the server (sanity check)
#
# Two runs: baseline (no flag) then with --kv-prefix-cache.
# Prints a side-by-side summary at the end.

set -euo pipefail

MODEL="${1:?usage: $0 <model.gguf> [n_parallel] [port]}"
N_PARALLEL="${2:-4}"
PORT="${3:-8080}"

BINARY="./build/bin/llama-server"
if [[ ! -x "$BINARY" ]]; then
    BINARY="./llama-server"
fi
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: llama-server not found. Build with cmake first." >&2
    exit 1
fi

# Shared prefix: 256 tokens of lorem ipsum (hard-coded tokens avoided: use text)
SHARED_PREFIX="The history of artificial intelligence is a story of remarkable ambition, \
persistent failure, and eventual triumph. From the earliest symbolic reasoning systems \
of the 1950s to the transformer-based large language models of the 2020s, the field has \
cycled through periods of intense optimism followed by so-called AI winters when funding \
dried up and progress stalled. Yet each winter planted seeds for the next spring. \
The perceptron, backpropagation, convolutional networks, attention mechanisms — each \
breakthrough arrived quietly and then changed everything. Today, language models with \
hundreds of billions of parameters routinely pass bar exams, write production code, and \
hold nuanced conversations. The question is no longer whether machines can think, but \
what thinking even means, and whether silicon substrates can carry meaning the way \
biological neurons do. This question touches philosophy, neuroscience, linguistics, and \
ethics in equal measure. We are only beginning to understand the implications."

# Short divergent suffixes per request (forces different completions)
SUFFIXES=(
    " Discuss the philosophical implications."
    " Summarize the key milestones."
    " What comes next in AI development?"
    " Compare to human intelligence."
)

N_PREDICT=32
LOG_DIR="/tmp/bench-prefix-cache-$$"
mkdir -p "$LOG_DIR"

run_server() {
    local extra_flags="$1"
    local log="$LOG_DIR/server-$2.log"

    "$BINARY" \
        --model "$MODEL" \
        --port "$PORT" \
        --parallel "$N_PARALLEL" \
        --ctx-size $(( N_PARALLEL * 512 )) \
        --n-predict "$N_PREDICT" \
        --log-disable \
        $extra_flags \
        > "$log" 2>&1 &
    echo $!
}

wait_server() {
    local retries=30
    while (( retries-- > 0 )); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: server did not start on port $PORT" >&2
    return 1
}

fire_requests() {
    local out_file="$1"
    local results=()

    # First request: warms up the shared prefix KV
    for i in "${!SUFFIXES[@]}"; do
        local prompt="${SHARED_PREFIX}${SUFFIXES[$i]}"
        local t0
        t0=$(date +%s%3N)
        local resp
        resp=$(curl -sf "http://localhost:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d "{
                \"prompt\": $(jq -Rs . <<< "$prompt"),
                \"n_predict\": $N_PREDICT,
                \"cache_prompt\": true,
                \"temperature\": 0
            }")
        local t1
        t1=$(date +%s%3N)
        local elapsed=$(( t1 - t0 ))

        local cache_tokens
        cache_tokens=$(echo "$resp" | jq -r '.timings.prompt_n_cached // 0')
        local prompt_ms
        prompt_ms=$(echo "$resp" | jq -r '.timings.prompt_ms // 0')

        echo "  request $i: wall=${elapsed}ms  prompt_ms=${prompt_ms}  cached_tokens=${cache_tokens}"
        results+=("$elapsed $cache_tokens $prompt_ms")
    done

    # write results to file for comparison
    printf '%s\n' "${results[@]}" > "$out_file"
}

summarise() {
    local file="$1"
    local tag="$2"
    local total_wall=0 total_cached=0 total_prompt_ms=0 n=0
    while IFS=' ' read -r wall cached pms; do
        total_wall=$(( total_wall + wall ))
        total_cached=$(( total_cached + cached ))
        total_prompt_ms=$(echo "$total_prompt_ms + $pms" | bc)
        (( n++ ))
    done < "$file"
    local avg_wall=$(( total_wall / n ))
    local avg_cached=$(( total_cached / n ))
    local avg_pms
    avg_pms=$(echo "scale=1; $total_prompt_ms / $n" | bc)
    echo "[$tag] avg wall=${avg_wall}ms  avg prompt_ms=${avg_pms}  avg cached_tokens=${avg_cached}  (n=$n)"
}

echo "=== Prefix Cache Benchmark ==="
echo "Model      : $MODEL"
echo "Parallel   : $N_PARALLEL"
echo "Port       : $PORT"
echo "Shared prefix length: ${#SHARED_PREFIX} chars"
echo ""

# ---- BASELINE ----
echo "--- BASELINE (no --kv-prefix-cache) ---"
PID=$(run_server "" "baseline")
wait_server
echo "Server started (pid=$PID)"
sleep 1

echo "Firing requests..."
fire_requests "$LOG_DIR/baseline.txt"

kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
sleep 2

# ---- WITH PREFIX CACHE ----
echo ""
echo "--- WITH --kv-prefix-cache ---"
PID=$(run_server "--kv-prefix-cache" "prefixcache")
wait_server
echo "Server started (pid=$PID)"
sleep 1

echo "Firing requests..."
fire_requests "$LOG_DIR/prefixcache.txt"

kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true

echo ""
echo "=== RESULTS ==="
summarise "$LOG_DIR/baseline.txt"    "baseline     "
summarise "$LOG_DIR/prefixcache.txt" "prefix-cache "

rm -rf "$LOG_DIR"
echo ""
echo "Done."
