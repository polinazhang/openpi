#!/usr/bin/env bash
set -euo pipefail
config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
openpi_root="$(cd -- "$config_dir/.." && pwd)"
mode="${1:-}"
[[ $# -eq 0 ]] || shift
unset PYTHONPATH
export PYTHONNOUSERSITE=1
case "$mode" in
  server)
    export TORCHDYNAMO_DISABLE=1
    exec "$openpi_root/.venv/bin/python" "$openpi_root/examples/franka_real/inference_server.py" "$@"
    ;;
  robot)
    openteach_root="$(python3 - "$config_dir/config.py" <<'PYCONFIG'
import runpy, sys
print(runpy.run_path(sys.argv[1])["openteach_root"])
PYCONFIG
)"
    exec "$openteach_root/.conda-env/bin/python" "$openpi_root/examples/franka_real/robot_communicator.py" "$@"
    ;;
  *) echo 'Usage: bash repo-configs/run_franka.bash server|robot [arguments]' >&2; exit 2 ;;
esac
