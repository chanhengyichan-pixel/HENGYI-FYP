#!/usr/bin/env bash
# Generate the voice prompts with Piper — an offline neural TTS that
# sounds much more natural than espeak-ng. Needs internet ONCE to download
# the voice; after that everything (and the demo) is fully offline.
#
# Run from the fyp2/ directory, inside the venv:
#   pip install piper-tts
#   bash scripts/make_voice_prompts_piper.sh            # default voice
#   VOICE=en_GB-alan-medium bash scripts/make_voice_prompts_piper.sh
#
# Other voices: https://rhasspy.github.io/piper-samples/  (listen first,
# then use the name, e.g. en_US-amy-medium, en_GB-alba-medium).
# LENGTH=1.15 makes speech slower (1.0 = normal) — slower is clearer
# through a small speaker.
set -euo pipefail
VOICE=${VOICE:-en_US-lessac-medium}
LENGTH=${LENGTH:-1.1}
VOICE_DIR=models/voices
OUT=assets/audio
mkdir -p "$VOICE_DIR" "$OUT"

if [ ! -f "$VOICE_DIR/$VOICE.onnx" ]; then
  python -m piper.download_voices --download-dir "$VOICE_DIR" "$VOICE"
fi

say() {
  echo "$2" | python -m piper -m "$VOICE_DIR/$VOICE.onnx" \
    --length-scale "$LENGTH" --sentence-silence 0.3 -f "$OUT/$1"
  echo "wrote $OUT/$1"
}
say granted.wav    "Safety check complete, you may enter."
say denied.wav           "Access denied. Please wear your safety helmet and safety vest."
say denied_no_helmet.wav "Access denied. Please wear your safety helmet."
say denied_no_vest.wav   "Access denied. Please wear your safety vest."
say hazard_ppe.wav       "Warning. You are in a hazard area. Please put on your helmet and safety vest."
say hazard_no_helmet.wav "Warning. You are in a hazard area. Please put on your safety helmet."
say hazard_no_vest.wav   "Warning. You are in a hazard area. Please put on your safety vest."
say restricted.wav "Stop. This is a restricted area. Please leave immediately."
