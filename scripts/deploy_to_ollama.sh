#!/usr/bin/env bash
set -e

echo "=== WEFT to Ollama Deployment Pipeline ==="

WEFT_INPUT="$1"
OUTPUT_DIR="models/ollama_export"
mkdir -p "$OUTPUT_DIR"

if [ -z "$WEFT_INPUT" ]; then
    WEFT_INPUT="/run/media/akarta/AI/models/qwen2.5-coder-7b-subbit/model-00001-of-00004_subbit.weft"
fi

GGUF_OUT="$OUTPUT_DIR/weft_standard.gguf"
MODELFILE_OUT="$OUTPUT_DIR/Modelfile"

echo "[1/3] Exporting .weft shard to standard GGUF..."
uv run python3 weft/export_standard_gguf.py --inputs "$WEFT_INPUT" --output "$GGUF_OUT" --arch qwen2 --dtype f16

echo "[2/3] Generating Ollama Modelfile..."
cat <<EOF > "$MODELFILE_OUT"
FROM $(pwd)/$GGUF_OUT
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
EOF

echo "[3/3] Registering model in Ollama: weft-test..."
ollama create weft-test -f "$MODELFILE_OUT"
echo "[PASS] Model successfully registered in Ollama. Available models:"
ollama list | grep "weft-test"
