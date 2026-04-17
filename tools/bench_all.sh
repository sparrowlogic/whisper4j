#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

AUDIO="src/test/resources/data/physicsworks.wav"
CP="target/classes"
MOD="com.sparrowlogic.whisper4j/com.sparrowlogic.whisper4j.WhisperCli"
CORRETTO="$HOME/.sdkman/candidates/java/26-amzn/bin/java"
VALHALLA="$HOME/code/jackdpeterson/openjdk-valhalla/build/macosx-aarch64-server-release/jdk/bin/java"
OUT="/tmp/whisper4j-full-bench.txt"

MODELS=(
  "ggml-tiny.en.bin"
  "ggml-base.en.bin"
  "ggml-small.en.bin"
  "ggml-medium.en.bin"
  "ggml-large-v3-turbo.bin"
)

> "$OUT"

run_platform() {
  local label="$1"; shift
  echo "$label" >> "$OUT"
  echo "--- $label ---"
  for m in "${MODELS[@]}"; do
    echo "  $m ..."
    "$@" "/Volumes/RAMDisk/$m" "$AUDIO" 1 2>/dev/null >> "$OUT"
  done
}

run_platform "Corretto26" \
  "$CORRETTO" -XX:+UnlockDiagnosticVMOptions -XX:-UseSuperWord \
  --enable-native-access=ALL-UNNAMED -p "$CP" -m "$MOD"

run_platform "Corretto26+Vec" \
  "$CORRETTO" --enable-preview --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED -p "$CP:$CP/META-INF/versions/26" -m "$MOD"

run_platform "Valhalla" \
  "$VALHALLA" -XX:+UnlockDiagnosticVMOptions -XX:-UseSuperWord \
  --enable-native-access=ALL-UNNAMED -p "$CP" -m "$MOD"

run_platform "Valhalla+Vec" \
  "$VALHALLA" --enable-preview --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED -p "$CP:$CP/META-INF/versions/26" -m "$MOD"

echo "NativeImage" >> "$OUT"
echo "--- NativeImage ---"
for m in "${MODELS[@]}"; do
  echo "  $m ..."
  ./target/whisper4j "/Volumes/RAMDisk/$m" "$AUDIO" 1 2>/dev/null >> "$OUT"
done

echo "=== DONE ==="
cat "$OUT"
