"""
Render videos for completed recordings using playback.py.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .run_dir import PROJECT_ROOT


def render_videos(
    perturbation_info: List[Dict[str, Any]],
    results_dir: Path,
    config_dir: Path,
    project_root: Optional[Path] = None,
) -> int:
    """
    Render videos for unperturbed and each perturbed recording that has HDF5 + config.
    Returns the number of videos rendered.
    """
    project_root = project_root or PROJECT_ROOT
    videos_dir = results_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    rendered_count = 0

    unperturbed_hdf5 = results_dir / "unperturbed.hdf5"
    if unperturbed_hdf5.exists():
        unperturbed_config = config_dir / "unperturbed.yaml"
        if unperturbed_config.exists():
            print("[INFO] Rendering unperturbed video...")
            try:
                cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "playback.py"),
                    "--config",
                    str(unperturbed_config),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  ✓ Rendered: {videos_dir / 'unperturbed.mp4'}")
                rendered_count += 1
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to render unperturbed video: {e}")

    for pert_info in perturbation_info:
        if pert_info["id"] == "unperturbed":
            continue
        pert_id = pert_info["id"]
        pert_hdf5 = results_dir / f"{pert_id}.hdf5"
        pert_config = Path(pert_info["config_file"])
        if pert_hdf5.exists() and pert_config.exists():
            print(f"[INFO] Rendering {pert_id} video...")
            try:
                cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "playback.py"),
                    "--config",
                    str(pert_config),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  ✓ Rendered: {videos_dir / f'{pert_id}.mp4'}")
                rendered_count += 1
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to render {pert_id} video: {e}")

    print(f"\n[INFO] Rendered {rendered_count} video(s) to {videos_dir}")
    return rendered_count
