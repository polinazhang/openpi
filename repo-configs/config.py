"""Editable locations for the Franka inference workflow."""
from pathlib import Path

openpi_root = str(Path(__file__).resolve().parents[1])
openteach_root = "/home/ripl/Desktop/openteach"
# Absolute directory for a Cartesian-tag Pi0.5 checkpoint. None disables startup.
checkpoint_dir = "/home/ripl/openpi/checkpoints/task10_demo30"
# Normalization is ALWAYS read from checkpoint_dir/assets/franka/norm_stats.json.

# Franka evaluation settings.
evaluation_suite_name = "franka_task10_demo30"
task_prompt = "pick up the plate, then put it down on the platform"
