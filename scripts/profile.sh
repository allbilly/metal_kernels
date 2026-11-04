#!/usr/bin/env bash
# Record a Metal trace, export the tables used by kernels.py/timeline.py, delete it.
#
# usage:
#   scripts/profile.sh --attach PID [--seconds 5] [--out DIR]
#   scripts/profile.sh --launch [--seconds 5] [--out DIR] -- command [args...]
#
# options:
#   --counters     also export gpu-counter-value (large)
#   --samples      also export gpu-shader-profiler-sample + shader-list
#   --kernels      print the per-kernel interval table after export
#   --keep         keep run.trace
#   --template P   patched template (default $MST_TEMPLATE or ./MST-shader.tracetemplate)
#
# Build the template once, as in README2 §1.4.

set -euo pipefail

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "profile.sh: $*" >&2
  exit 2
}

MODE=""
PID=""
DURATION=5
OUT=""
COUNTERS=0
SAMPLES=0
KERNELS=0
KEEP=0
TEMPLATE="${MST_TEMPLATE:-./MST-shader.tracetemplate}"
CMD=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --attach)
      [ "$#" -ge 2 ] || die "--attach needs a PID"
      MODE=attach
      PID="$2"
      shift 2
      ;;
    --launch)
      MODE=launch
      shift
      ;;
    --)
      shift
      CMD=("$@")
      break
      ;;
    --seconds)
      [ "$#" -ge 2 ] || die "--seconds needs a value"
      DURATION="$2"
      shift 2
      ;;
    --out)
      [ "$#" -ge 2 ] || die "--out needs a directory"
      OUT="$2"
      shift 2
      ;;
    --counters)
      COUNTERS=1
      shift
      ;;
    --samples)
      SAMPLES=1
      shift
      ;;
    --kernels)
      KERNELS=1
      shift
      ;;
    --keep)
      KEEP=1
      shift
      ;;
    --template)
      [ "$#" -ge 2 ] || die "--template needs a path"
      TEMPLATE="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (options must come before the command)"
      ;;
  esac
done

[ -n "$MODE" ] || { usage >&2; exit 2; }
[ -e "$TEMPLATE" ] || die "missing $TEMPLATE; create it with scripts/trace_template.py (README2 §1.4)"
if [ "$MODE" = attach ]; then
  [ -n "$PID" ] || die "--attach needs a PID"
else
  [ "${#CMD[@]}" -gt 0 ] || die "--launch needs a command"
fi

if [ -z "$OUT" ]; then
  OUT="traces/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT"
TRACE="$OUT/run.trace"
rm -rf "$TRACE"

REC=(xcrun xctrace record --template "$TEMPLATE" --instrument 'Metal GPU Counters'
     --time-limit "${DURATION}s" --output "$TRACE")
if [ "$MODE" = attach ]; then
  REC+=(--attach "$PID")
else
  REC+=(--launch -- "${CMD[@]}")
fi

echo "recording ${DURATION}s -> $TRACE"
if ! "${REC[@]}"; then
  if [ ! -e "$TRACE" ]; then
    die "recording failed and no trace was written"
  fi
  echo "warn: xctrace returned nonzero; using the saved trace" >&2
fi

export_table() {
  schema="$1"
  file="$2"
  if ! xcrun xctrace export --input "$TRACE" \
      --xpath "//trace-toc/run[@number=\"1\"]/data/table[@schema=\"$schema\"]" \
      >"$OUT/$file" 2>"$OUT/export.err"; then
    echo "warn: export failed for $schema" >&2
    rm -f "$OUT/$file"
    return
  fi
  if ! grep -q '<row' "$OUT/$file"; then
    echo "warn: no rows for $schema" >&2
  fi
}

export_table metal-application-command-buffer-submissions command-buffers.xml
export_table metal-application-encoders-list encoders.xml
export_table metal-shader-profiler-intervals shaders.xml
if [ "$COUNTERS" = 1 ]; then
  echo "exporting counters (large)"
  export_table gpu-counter-value counters.xml
fi
if [ "$SAMPLES" = 1 ]; then
  echo "exporting PC samples (large)"
  export_table gpu-shader-profiler-sample samples.xml
  export_table metal-shader-profiler-shader-list shader-list.xml
fi
rm -f "$OUT/export.err"

if ! grep -q '<row' "$OUT/encoders.xml" 2>/dev/null &&
   ! grep -q '<row' "$OUT/shaders.xml" 2>/dev/null; then
  echo "warn: no encoder or shader rows; xctrace --launch misses some targets, retry with --attach" >&2
fi

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(sw_vers -productName) $(sw_vers -productVersion) $(uname -m)"
  echo "xcode: $(xcodebuild -version 2>/dev/null | head -1)"
  echo "template: $TEMPLATE"
  echo "seconds: $DURATION"
  if [ "$MODE" = attach ]; then
    echo "attach: $PID"
  else
    echo "launch: ${CMD[*]}"
  fi
} >"$OUT/meta.txt"

if [ "$KEEP" = 0 ]; then
  rm -rf "$TRACE"
fi

echo "wrote $OUT"
if [ "$KERNELS" = 1 ]; then
  echo
  python3 "$(dirname "$0")/kernels.py" "$OUT"
else
  echo "  scripts/kernels.py $OUT"
fi
echo "  scripts/timeline.py $OUT --out timeline.html"
if [ "$COUNTERS" = 1 ] && [ "$KEEP" = 1 ]; then
  echo "  scripts/counters.py $OUT"
fi
if [ "$SAMPLES" = 1 ]; then
  echo "  scripts/samples.py $OUT"
fi
