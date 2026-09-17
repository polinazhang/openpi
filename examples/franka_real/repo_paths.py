"""Load repository locations relative to this checkout, independent of cwd."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_config_file = Path(__file__).resolve().parents[2] / "repo-configs" / "config.py"
_spec = spec_from_file_location("_openpi_repo_config", _config_file)
_config = module_from_spec(_spec)
_spec.loader.exec_module(_config)
openpi_root = Path(_config.openpi_root)
openteach_root = Path(_config.openteach_root)
checkpoint_dir = _config.checkpoint_dir
evaluation_suite_name = _config.evaluation_suite_name
task_prompt = _config.task_prompt


def use_robot_repositories():
    """Resolve OpenTeach and its sibling Deoxys from explicit central config."""
    config_path = openteach_root / "repo-configs" / "config.py"
    spec = spec_from_file_location("_robot_repo_config", config_path)
    config = module_from_spec(spec)
    spec.loader.exec_module(config)
    franka_root = Path(config.repo_root) / "franka-control"
    for directory in (franka_root / "deoxys", openteach_root):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        sys.path.insert(0, str(directory))
    # Do not silently reuse an already-imported package from another checkout.
    import deoxys
    import openteach.components.operators.franka as operator
    if not Path(deoxys.__file__).resolve().is_relative_to(franka_root.resolve()):
        raise RuntimeError("Deoxys was imported from a different checkout; restart in the configured environment.")
    if not Path(operator.__file__).resolve().is_relative_to(openteach_root.resolve()):
        raise RuntimeError("OpenTeach was imported from a different checkout; restart in the configured environment.")
