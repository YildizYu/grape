#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv-thor"
KNOWN_ASSET_DIR="/home/nvidia/thor/grape/grape_stem_3d_deploy_complete"

find_asset() {
  local name="$1"
  if [[ -f "${PROJECT_ROOT}/${name}" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/${name}"
    return 0
  fi
  if [[ -f "${KNOWN_ASSET_DIR}/${name}" ]]; then
    printf '%s\n' "${KNOWN_ASSET_DIR}/${name}"
    return 0
  fi
  printf 'Missing required file: %s\n' "${name}" >&2
  printf 'Place it in: %s\n' "${PROJECT_ROOT}" >&2
  return 1
}

TORCH_WHEEL="$(find_asset torch-2.10.0-cp312-cp312-linux_aarch64.whl)"
TORCHVISION_WHEEL="$(find_asset torchvision-0.25.0-cp312-cp312-linux_aarch64.whl)"
ORBBEC_WHEEL="$(find_asset pyorbbecsdk2-2.1.2-cp312-cp312-manylinux_2_27_aarch64.whl)"

python3 -m venv --system-site-packages "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install "${TORCH_WHEEL}" "${TORCHVISION_WHEEL}"
"${VENV_PATH}/bin/python" -m pip install --no-deps "${ORBBEC_WHEEL}"
"${VENV_PATH}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
"${VENV_PATH}/bin/python" -m pip install onnx onnx2torch scikit-image

echo "Thor vision environment ready: ${VENV_PATH}"
echo "Activate it with: source ${VENV_PATH}/bin/activate"
echo "Install the Orbbec USB rule once (commands are in THOR_ORBBEC_MIGRATION.md)."
