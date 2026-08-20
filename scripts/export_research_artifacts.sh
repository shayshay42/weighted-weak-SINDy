#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s SOURCE_ROOT DEST_ROOT [core|full]\n' "$0" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

source_root="${1%/}"
dest_root="${2%/}"
mode="${3:-core}"

[[ -d "${source_root}" ]] || {
  printf 'Source root does not exist: %s\n' "${source_root}" >&2
  exit 1
}

case "${mode}" in
  core|full) ;;
  *) usage ;;
esac

core_paths=(
  data/v2
  tuning/v2
  tuning/v2_few_shot
  results/v2
)

full_paths=(
  data/v2_context_matched
  data/v2_few_shot
  runs/v2
  runs/v2_panda_comparison
  runs/v2_context_matched
  runs/v2_few_shot
)

mkdir -p "${dest_root}"
paths=("${core_paths[@]}")
if [[ "${mode}" == "full" ]]; then
  paths+=("${full_paths[@]}")
fi

copied=0
for relative_path in "${paths[@]}"; do
  source_path="${source_root}/${relative_path}"
  if [[ ! -e "${source_path}" ]]; then
    printf 'Skipping missing path: %s\n' "${source_path}" >&2
    continue
  fi
  mkdir -p "${dest_root}/$(dirname "${relative_path}")"
  rsync -a "${source_path}/" "${dest_root}/${relative_path}/"
  copied=$((copied + 1))
done

[[ ${copied} -gt 0 ]] || {
  printf 'No artifact directories were copied.\n' >&2
  exit 1
}

checksum_file="${dest_root}/SHA256SUMS"
: > "${checksum_file}"

if command -v sha256sum >/dev/null 2>&1; then
  hash_file() { sha256sum "$1"; }
elif command -v shasum >/dev/null 2>&1; then
  hash_file() { shasum -a 256 "$1"; }
else
  printf 'Neither sha256sum nor shasum is available.\n' >&2
  exit 1
fi

(
  cd "${dest_root}"
  while IFS= read -r artifact; do
    hash_file "${artifact}"
  done < <(find . -type f ! -name SHA256SUMS | LC_ALL=C sort)
) > "${checksum_file}"

printf 'Copied %d artifact directories to %s (%s mode).\n' \
  "${copied}" "${dest_root}" "${mode}"
printf 'Checksums: %s\n' "${checksum_file}"
