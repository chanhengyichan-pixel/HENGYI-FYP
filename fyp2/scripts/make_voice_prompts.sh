#!/usr/bin/env bash
# Generate the voice prompts as WAV files with offline text-to-speech.
# Needs espeak-ng:  sudo apt install espeak-ng
# Run from the fyp2/ directory:  bash scripts/make_voice_prompts.sh
#
# For the final demo you can replace these with your own recordings —
# keep the same file names (or change the paths in config.yaml).
set -euo pipefail
out=assets/audio
mkdir -p "$out"
say() { espeak-ng -s 150 -w "$out/$1" "$2"; echo "wrote $out/$1"; }
say granted.wav    "You are wearing your safety helmet and safety vest. You may enter."
say denied.wav           "Access denied. Please wear your safety helmet and safety vest."
say denied_no_helmet.wav "Access denied. Please wear your safety helmet."
say denied_no_vest.wav   "Access denied. Please wear your safety vest."
say hazard_ppe.wav       "Warning. You are in a hazard area. Put on your helmet and safety vest."
say hazard_no_helmet.wav "Warning. You are in a hazard area. Put on your safety helmet."
say hazard_no_vest.wav   "Warning. You are in a hazard area. Put on your safety vest."
say restricted.wav "Stop. This is a restricted area. Please leave now."
