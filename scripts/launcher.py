#!/usr/bin/env python3
"""
Launcher script for pipelined perturbed dataset generation in PACE-ICE environment.

This script:
1. Creates a run directory structure in scratch folder
2. Generates perturbation files (BDDL and config YAMLs)
3. Dispatches SLURM jobs in a queue-like fashion
4. Runs evaluation scripts after all jobs complete

Usage:
    # Full pipeline (generate + SLURM jobs + videos + evaluation):
    python scripts/launcher.py --config configs/main.yaml

    # Local tryout: only generate perturbations, then record one config by hand:
    python scripts/launcher.py --config configs/main.yaml --generate-only --run-dir ./local_run
    python scripts/record.py --config ./local_run/configs/perturbed_0.yaml
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import copy

# Add project root to path (resolve to absolute path)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import perturbation utilities
perturbation_utils_path = project_root / "libero" / "libero" / "utils" / "generate_perturbation_bddl.py"
if perturbation_utils_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_perturbation_bddl", perturbation_utils_path)
    pert_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pert_utils)
    read_bddl = pert_utils.read_bddl
    fix_init_ranges = getattr(pert_utils, 'fix_init_ranges', lambda t, **kw: t)
    generate_move_spec_dict = getattr(pert_utils, 'generate_move_spec_dict', lambda *a, **k: {})
    apply_perturbations_kitchen = pert_utils.apply_perturbations_kitchen
    apply_perturbations = getattr(pert_utils, 'apply_perturbations', pert_utils.apply_perturbations_kitchen)
    validate_bddl = pert_utils.validate_bddl
    # resolve_z_overrides: present in updated bddl_perturbation; graceful fallback
    # for older versions that don't have it yet.
    resolve_z_overrides = getattr(pert_utils, 'resolve_z_overrides', None)
else:
    raise ImportError(f"Could not find perturbation utilities at {perturbation_utils_path}")


class Launcher:
    """Main launcher class for managing the pipeline."""
    
    def __init__(self, config_path: str, run_dir_override: Optional[str] = None):
        """Initialize launcher with config file.

        Args:
            config_path: Path to main.yaml.
            run_dir_override: If set, use this directory as run_dir (e.g. ./local_run).
                             Otherwise use run_base_dir from config + timestamped folder.
        """
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get scratch directory
        scratch = os.environ.get('SCRATCH', os.path.expanduser('~/scratch'))
        self.scratch_dir = Path(scratch)
        
        # Set up run directory
        if run_dir_override is not None:
            self.run_dir = Path(run_dir_override)
        else:
            run_base = self.config.get('run_base_dir')
            if run_base is None:
                run_base = self.scratch_dir / 'vla-explainability-runs'
            else:
                run_base = Path(run_base)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_name = self.config.get('task_suite_name', 'libero')
            self.run_dir = run_base / f"{task_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Run directory: {self.run_dir}")
        
        # Set up subdirectories
        self.bddl_dir = self.run_dir / "bddl_files"
        self.config_dir = self.run_dir / "configs"
        self.results_dir = self.run_dir / "results"
        self.logs_dir = self.run_dir / "logs"
        self.jobs_dir = self.run_dir / "jobs"
        
        for d in [self.bddl_dir, self.config_dir, self.results_dir, 
                  self.logs_dir, self.jobs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Save config for reference
        with open(self.run_dir / "main_config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        # Track jobs
        self.job_status = {}  # job_id -> status
        self.pending_jobs = []
        self.running_jobs = []
        self.completed_jobs = []
        self.failed_jobs = []
        
        # Track perturbation files
        self.perturbation_info = []
    
    def generate_perturbations(self):
        """Generate all perturbation files (BDDL and config YAMLs)."""
        print("\n[INFO] Generating perturbation files...")
        
        base_bddl = Path(self.config['base_bddl_file'])
        if not base_bddl.is_absolute():
            base_bddl = project_root / base_bddl
        
        if not base_bddl.exists():
            raise FileNotFoundError(f"Base BDDL file not found: {base_bddl}")
        
        # Read base BDDL
        base_bddl_text = read_bddl(str(base_bddl))
        
        # Apply init range to all regions
        pert_config = self.config.get('perturbations', {})
        bddl_spatial = pert_config.get('bddl_spatial', {})
        init_object_range_m = self.config.get('init_object_range_m', bddl_spatial.get('init_object_range_m', 0.0))
        max_init_range_m = bddl_spatial.get('max_init_range_m', 0.001)
        base_bddl_text = fix_init_ranges(base_bddl_text, init_object_range_m=init_object_range_m, max_init_range_m=max_init_range_m)
        
        # Copy base BDDL (unperturbed)
        unperturbed_bddl_path = self.bddl_dir / "unperturbed.bddl"
        with open(unperturbed_bddl_path, 'w') as f:
            f.write(base_bddl_text)
        
        # Create unperturbed config (no z_overrides for the baseline)
        unperturbed_config = self._create_record_config(
            perturbation_id="unperturbed",
            bddl_file=str(unperturbed_bddl_path),
            prompt=self.config['base_prompt'],
            z_overrides_file=None,
        )
        unperturbed_config_path = self.config_dir / "unperturbed.yaml"
        with open(unperturbed_config_path, 'w') as f:
            yaml.dump(unperturbed_config, f)
        
        self.perturbation_info.append({
            'id': 'unperturbed',
            'bddl_file': str(unperturbed_bddl_path),
            'config_file': str(unperturbed_config_path),
            'prompt': self.config['base_prompt'],
            'type': 'baseline',
            'description': 'Baseline unperturbed task',
            'z_overrides_file': None,
        })
        
        # Generate perturbed versions
        pert_config = self.config.get('perturbations', {})
        pert_types = pert_config.get('types', [])
        
        pert_id = 0
        
        if 'bddl_spatial' in pert_types:
            pert_id = self._generate_bddl_spatial_perturbations(
                base_bddl_text, pert_id,
                init_object_range_m=init_object_range_m,
                max_init_range_m=max_init_range_m,
            )
        
        if 'language' in pert_types:
            pert_id = self._generate_language_perturbations(base_bddl_text, pert_id)
        
        print(f"[INFO] Generated {len(self.perturbation_info)} perturbation files")
        
        # Save perturbation manifest
        manifest_path = self.run_dir / "perturbation_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.perturbation_info, f, indent=2)
    
    def _generate_bddl_spatial_perturbations(
        self,
        base_bddl_text: str,
        start_id: int,
        init_object_range_m: float = 0.0,
        max_init_range_m: float = 0.001,
    ) -> int:
        """Generate BDDL spatial perturbations."""
        pert_config = self.config['perturbations']['bddl_spatial']
        specs = pert_config.get('perturbation_specs', [])
        max_move_m = pert_config.get('max_move_m', 0.05)
        
        pert_id = start_id
        pert_id_copy = pert_id
        
        for spec in specs:
            pert_type = spec['type']
            spec_max_move_m = spec.get('max_move_m', max_move_m)
            
            # Build perturbation dict for apply_perturbations
            perturbations = {}
            
            if pert_type == 'distractor':
                count = spec.get('count', 1)
                perturbations['distractor'] = [None] * count
            elif pert_type == 'control':
                pert_id_copy = pert_id
                pert_id = "control"
            else:
                objects = spec.get('objects', [])
                if pert_type not in perturbations:
                    perturbations[pert_type] = []
                perturbations[pert_type].extend(objects)
            
            # Build perturbation_spec_dict for move perturbations
            perturbation_spec_dict = None
            if pert_type == 'move':
                objects = spec.get('objects', [])
                perturbation_spec_dict = generate_move_spec_dict(
                    base_bddl_text, objects, max_move_m=spec_max_move_m
                )

            try:
                # ----------------------------------------------------------------
                # apply_perturbations returns (bddl_text, z_overrides).
                # z_overrides is a dict {obj_name: sentinel_or_stack_tuple} that
                # records any collision-driven Z adjustments needed after
                # env.reset().  It is empty when no collisions were detected.
                # ----------------------------------------------------------------
                perturbed_bddl, z_overrides = apply_perturbations(
                    copy.deepcopy(base_bddl_text),
                    perturbations,
                    init_object_range_m=init_object_range_m,
                    max_move_m=spec_max_move_m,
                    max_init_range_m=max_init_range_m,
                    perturbation_spec_dict=perturbation_spec_dict,
                )
                
                # Validate BDDL structure
                if not validate_bddl(perturbed_bddl):
                    print(f"[WARN] Perturbation {pert_id} failed validation, skipping")
                    continue
                
                # Save BDDL
                pert_bddl_path = self.bddl_dir / f"perturbed_{pert_id}.bddl"
                with open(pert_bddl_path, 'w') as f:
                    f.write(perturbed_bddl)

                # ----------------------------------------------------------
                # Persist z_overrides as a JSON sidecar alongside the BDDL.
                # record.py will load this after env.reset() and call
                # resolve_z_overrides(sim, z_overrides) to get the final
                # (cx, cy, z) for each object that needs a height correction.
                # ----------------------------------------------------------
                z_overrides_file = None
                if z_overrides:
                    z_overrides_path = self.bddl_dir / f"perturbed_{pert_id}_z_overrides.json"
                    with open(z_overrides_path, 'w') as f:
                        # Tuples are not JSON-serialisable directly; convert to lists.
                        json.dump(
                            {k: list(v) for k, v in z_overrides.items()},
                            f, indent=2,
                        )
                    z_overrides_file = str(z_overrides_path)
                    print(f"[INFO] Saved z_overrides for {pert_id} → {z_overrides_path}")
                
                # Create per-perturbation record config
                pert_record_config = self._create_record_config(
                    perturbation_id=f"perturbed_{pert_id}",
                    bddl_file=str(pert_bddl_path),
                    prompt=self.config['base_prompt'],
                    z_overrides_file=z_overrides_file,
                )
                config_path = self.config_dir / f"perturbed_{pert_id}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(pert_record_config, f)
                
                # Build human-readable description
                if pert_type == 'distractor':
                    count = spec.get('count', 1)
                    description = f"Added {count} distractor object(s) to the scene"
                else:
                    objects = spec.get('objects', [])
                    pert_type_names = {
                        'move': 'moved',
                        'reorient': 'reoriented',
                        'color': 'changed color of',
                        'replace': 'replaced',
                    }
                    action = pert_type_names.get(pert_type, pert_type)
                    description = f"{action.capitalize()} {', '.join(objects)}"
                
                self.perturbation_info.append({
                    'id': f'perturbed_{pert_id}',
                    'bddl_file': str(pert_bddl_path),
                    'config_file': str(config_path),
                    'prompt': self.config['base_prompt'],
                    'type': f'bddl_spatial_{pert_type}',
                    'perturbations': perturbations,
                    'description': description,
                    'z_overrides_file': z_overrides_file,
                })
                
                if pert_id == "control":
                    pert_id = pert_id_copy
                else:
                    pert_id += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to generate perturbation {pert_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return pert_id
    
    def _generate_language_perturbations(self, base_bddl_text: str, start_id: int) -> int:
        """Generate language perturbations."""
        sys.path.insert(0, str(project_root / "explainability" / "perturbations" / "language"))
        from generate_perturbations import generate_perturbations
        
        pert_id = start_id
        base_prompt = self.config['base_prompt']
        
        pert_dict = generate_perturbations(base_prompt)
        
        for pert_name, pert_prompt in pert_dict.items():
            # Language perturbations use the unperturbed BDDL — no z_overrides
            pert_bddl_path = self.bddl_dir / "unperturbed.bddl"
            
            pert_config = self._create_record_config(
                perturbation_id=f"perturbed_{pert_id}",
                bddl_file=str(pert_bddl_path),
                prompt=pert_prompt,
                z_overrides_file=None,
            )
            config_path = self.config_dir / f"perturbed_{pert_id}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(pert_config, f)
            
            pert_descriptions = {
                'keyboard': 'Keyboard typo',
                'ocr': 'OCR error simulation',
                'ci': 'Concatenation/insertion',
                'cr': 'Character replacement',
                'cs': 'Character swap',
                'cd': 'Character deletion',
                'ws': 'Word swap',
                'wd': 'Word deletion',
                'ip': 'Insert punctuation',
                'paraphrase0': 'Paraphrase variant 0',
                'paraphrase1': 'Paraphrase variant 1',
                'paraphrase2': 'Paraphrase variant 2',
                'paraphrase3': 'Paraphrase variant 3',
                'paraphrase4': 'Paraphrase variant 4',
            }
            
            if pert_name.startswith('wd_all_'):
                idx = pert_name.split('_')[-1]
                description = f'Word deletion (removed word at position {idx})'
            else:
                description = pert_descriptions.get(pert_name, f'Language perturbation: {pert_name}')
            
            self.perturbation_info.append({
                'id': f'perturbed_{pert_id}',
                'bddl_file': str(pert_bddl_path),
                'config_file': str(config_path),
                'prompt': pert_prompt,
                'type': f'language_{pert_name}',
                'original_prompt': base_prompt,
                'description': description,
                'z_overrides_file': None,
            })
            
            pert_id += 1
        
        return pert_id
    
    def _create_record_config(
        self,
        perturbation_id: str,
        bddl_file: str,
        prompt: str,
        z_overrides_file: Optional[str] = None,
    ) -> Dict:
        """
        Create a record config YAML for a perturbation.

        Args:
            perturbation_id : Unique string ID for this perturbation.
            bddl_file       : Absolute path to the BDDL file to use.
            prompt          : Language prompt for the task.
            z_overrides_file: Path to the JSON sidecar produced by
                              apply_perturbations() that contains collision-
                              driven Z corrections.  None when no collisions
                              were detected (most cases).  record.py reads this
                              after env.reset() and calls resolve_z_overrides()
                              to apply the correct object heights.
        """
        bddl_path = Path(bddl_file)
        if not bddl_path.is_absolute():
            bddl_path = self.bddl_dir / bddl_path.name
        
        out_file = self.results_dir / f"{perturbation_id}.hdf5"
        
        videos_dir = self.results_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        config = {
            'model': self.config['model'],
            'task_suite_name': self.config['task_suite_name'],
            'device': self.config['device'],
            'cache_dir': self.config['cache_dir'],
            'bddl_file': str(bddl_path),
            'prompt': prompt,
            'out_file': str(out_file),
            'record_path': str(videos_dir / f"{perturbation_id}.mp4"),
            'action_scale': self.config.get('action_scale', 1.0),
            'num_demos': self.config.get('num_demos', 1),
            'noise_std': self.config.get('noise_std', 0.0),
            # Path to JSON sidecar with collision Z corrections.
            # None (written as null in YAML) when no collisions were detected.
            'z_overrides_file': z_overrides_file,
        }
        
        return config
    
    def create_slurm_job(self, perturbation_id: str, config_file: str) -> str:
        """Create a SLURM job script for recording."""
        job_script = self.jobs_dir / f"{perturbation_id}.sh"
        
        slurm_config = self.config['slurm']
        job_params = slurm_config['job_params']
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_config['job_name_prefix']}_{perturbation_id}
#SBATCH --account={job_params['account']}
#SBATCH --time={job_params['time']}
#SBATCH --nodes={job_params['nodes']}
#SBATCH --ntasks-per-node={job_params['ntasks_per_node']}
#SBATCH --cpus-per-task={job_params['cpus_per_task']}
"""
        
        if job_params.get('partition'):
            script_content += f"#SBATCH --partition={job_params['partition']}\n"
        
        if job_params.get('mem'):
            script_content += f"#SBATCH --mem={job_params['mem']}\n"
        
        if job_params.get('gpus', 0) > 0:
            gpu_type = job_params.get('gpu_type', 'V100')
            num_gpus = job_params['gpus']
            script_content += f"#SBATCH --gres=gpu:{gpu_type}:{num_gpus}\n"
        
        if job_params.get('constraint'):
            script_content += f"#SBATCH -C {job_params['constraint']}\n"
            
        blacklisted_nodes = job_params.get('blacklisted_nodes', [])
        if blacklisted_nodes:
            node_list = ','.join(str(n) for n in blacklisted_nodes) \
                if isinstance(blacklisted_nodes, list) else str(blacklisted_nodes)
            script_content += f"#SBATCH --exclude={node_list}\n"
        
        script_content += f"""#SBATCH --output={self.logs_dir}/{perturbation_id}_%j.out
#SBATCH --error={self.logs_dir}/{perturbation_id}_%j.err

cd $SLURM_SUBMIT_DIR

"""
        
        for module in slurm_config.get('module_load', []):
            script_content += f"module load {module}\n"
        
        script_content += f"""
conda activate {slurm_config['conda_env']}

cd {project_root}

if ! python -c "import libero" 2>/dev/null; then
    echo "Installing libero package..."
    pip install -e . || {{ echo "ERROR: Failed to install libero package"; exit 1; }}
else
    echo "libero package already installed, skipping installation"
fi

python scripts/record.py --config {config_file}

echo "Job completed: {perturbation_id}"
"""
        
        with open(job_script, 'w') as f:
            f.write(script_content)
        
        os.chmod(job_script, 0o755)
        return str(job_script)
    
    def submit_job(self, job_script: str) -> Optional[str]:
        """Submit a SLURM job and return job ID."""
        try:
            result = subprocess.run(
                ['sbatch', job_script],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip().split()[-1]
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to submit job {job_script}: {e.stderr}")
            return None
    
    def check_job_status(self, job_id: str) -> str:
        """Check status of a SLURM job."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True, text=True, check=True,
            )
            status = result.stdout.strip()
            return status if status else "COMPLETED"
        except subprocess.CalledProcessError:
            return "COMPLETED"
    
    def dispatch_jobs(self):
        """Dispatch SLURM jobs in queue-like fashion."""
        print("\n[INFO] Dispatching SLURM jobs...")
        
        max_concurrent = self.config['slurm']['max_concurrent_jobs']
        
        job_scripts = {}
        for pert_info in self.perturbation_info:
            pert_id = pert_info['id']
            config_file = pert_info['config_file']
            job_script = self.create_slurm_job(pert_id, config_file)
            job_scripts[pert_id] = job_script
            self.pending_jobs.append(pert_id)
        
        print(f"[INFO] Created {len(job_scripts)} job scripts")
        print(f"[INFO] Max concurrent jobs: {max_concurrent}")
        
        while self.pending_jobs or self.running_jobs:
            for pert_id in list(self.running_jobs):
                job_id = self.job_status.get(pert_id)
                if job_id:
                    status = self.check_job_status(job_id)
                    if status == "COMPLETED":
                        out_file = Path(self.results_dir / f"{pert_id}.hdf5")
                        if out_file.exists():
                            print(f"[INFO] Job {pert_id} completed successfully")
                            self.running_jobs.remove(pert_id)
                            self.completed_jobs.append(pert_id)
                        else:
                            print(f"[WARN] Job {pert_id} completed but output file missing")
                            self.running_jobs.remove(pert_id)
                            self.failed_jobs.append(pert_id)
                        del self.job_status[pert_id]
            
            while len(self.running_jobs) < max_concurrent and self.pending_jobs:
                pert_id = self.pending_jobs.pop(0)
                job_script = job_scripts[pert_id]
                job_id = self.submit_job(job_script)
                
                if job_id:
                    print(f"[INFO] Submitted job {pert_id} (SLURM ID: {job_id})")
                    self.job_status[pert_id] = job_id
                    self.running_jobs.append(pert_id)
                else:
                    print(f"[ERROR] Failed to submit job {pert_id}")
                    self.failed_jobs.append(pert_id)
            
            if self.running_jobs:
                time.sleep(self.config['slurm']['poll_interval'])
        
        print(f"\n[INFO] All jobs dispatched")
        print(f"  Completed: {len(self.completed_jobs)}")
        print(f"  Failed: {len(self.failed_jobs)}")
        
        with open(self.run_dir / "job_summary.json", 'w') as f:
            json.dump({
                'completed': self.completed_jobs,
                'failed': self.failed_jobs,
                'total': len(self.perturbation_info),
            }, f, indent=2)
    
    def render_videos(self):
        """Render videos for all completed recordings using playback.py."""
        print("\n[INFO] Rendering videos for completed recordings...")
        
        videos_dir = self.results_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        rendered_count = 0
        
        all_items = [{'id': 'unperturbed', 'config_file': str(self.config_dir / 'unperturbed.yaml')}] \
                    + [p for p in self.perturbation_info if p['id'] != 'unperturbed']
        
        for item in all_items:
            pert_id = item['id']
            hdf5 = self.results_dir / f"{pert_id}.hdf5"
            cfg = Path(item['config_file'])
            if not hdf5.exists() or not cfg.exists():
                continue
            print(f"[INFO] Rendering {pert_id}...")
            try:
                subprocess.run(
                    [sys.executable, str(project_root / "scripts" / "playback.py"),
                     "--config", str(cfg)],
                    check=True, capture_output=True,
                )
                print(f"  ✓ {videos_dir / f'{pert_id}.mp4'}")
                rendered_count += 1
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed: {e}")
        
        print(f"\n[INFO] Rendered {rendered_count} video(s) to {videos_dir}")
    
    def run_evaluation(self):
        """Run evaluation scripts after all jobs complete."""
        eval_config = self.config.get('evaluation', {})
        if not eval_config.get('enabled', False):
            print("[INFO] Evaluation disabled, skipping")
            return
        
        print("\n[INFO] Running evaluation...")
        
        unperturbed_file = self.results_dir / "unperturbed.hdf5"
        if not unperturbed_file.exists():
            print("[ERROR] Unperturbed file not found, cannot run evaluation")
            return
        
        perturbed_files = [
            str(self.results_dir / f"{p['id']}.hdf5")
            for p in self.perturbation_info
            if p['id'] != 'unperturbed'
            and (self.results_dir / f"{p['id']}.hdf5").exists()
        ]
        
        if not perturbed_files:
            print("[WARN] No perturbed files found for evaluation")
            return

        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root) + ":" + env.get('PYTHONPATH', '')
        
        if 'json' in eval_config.get('output_formats', []):
            json_output = self.results_dir / "trajectories.json"
            cmd = [
                sys.executable,
                str(project_root / "utils" / "hdf5_to_json.py"),
                str(unperturbed_file), "-p",
            ] + perturbed_files + ["-o", str(json_output)]
            print(f"[INFO] Converting to JSON: {json_output}")
            subprocess.run(cmd, check=True, env=env)
        
        self._run_analysis(unperturbed_file, perturbed_files, eval_config)
    
    def _run_analysis(self, unperturbed_file: Path, perturbed_files: List[str], eval_config: Dict):
        """Run analysis using episodic_explanation and vla_metrics."""
        print("[INFO] Running trajectory analysis...")
        
        analysis_module_path = project_root / "explainability" / "run_analysis.py"
        if not analysis_module_path.exists():
            raise FileNotFoundError(f"Analysis module not found at {analysis_module_path}")
        
        output_file = self.run_dir / "analysis_results.json"
        
        cmd = [
            sys.executable, str(analysis_module_path),
            "--unperturbed", str(unperturbed_file),
            "--perturbed",
        ] + [str(p) for p in perturbed_files] + [
            "--output", str(output_file),
            "--metric-weights", json.dumps(eval_config['metric_weights']),
            "--trajectory-weights", json.dumps(eval_config['trajectory_weights']),
            "--project-root", str(project_root),
        ]
        
        subprocess.run(cmd, check=True)
    
    def run(self, generate_only: bool = False):
        """Run the complete pipeline (or only perturbation generation).

        Args:
            generate_only: If True, only generate perturbations and print instructions
                           for running record.py locally; do not submit SLURM jobs.
        """
        print("=" * 80)
        print("VLA Explainability Pipeline Launcher")
        print("=" * 80)
        print(f"Run directory: {self.run_dir}")
        
        self.generate_perturbations()
        
        if generate_only:
            self._print_local_recording_instructions()
            return
        
        self.dispatch_jobs()
        self.render_videos()
        self.run_evaluation()
        
        print("\n" + "=" * 80)
        print("Pipeline complete!")
        print(f"Results available in: {self.run_dir}")
        print("=" * 80)

    def _print_local_recording_instructions(self):
        """Print how to record one perturbation locally."""
        print("\n" + "=" * 80)
        print("Generate-only complete. To record one perturbation locally:")
        print("=" * 80)
        print(f"\n1. Run record for a single config (e.g. unperturbed or perturbed_0):")
        print(f"   python scripts/record.py --config {self.config_dir / 'unperturbed.yaml'}")
        print(f"\n   Or for a specific perturbation:")
        for info in self.perturbation_info:
            if info["id"] != "unperturbed":
                print(f"   python scripts/record.py --config {info['config_file']}")
                break
        print(f"\n2. From project root, ensure device/cache_dir in the generated config")
        print(f"   (in {self.config_dir}) match your machine, or override in the YAML.")
        print(f"\n3. After recording, render video (optional):")
        print(f"   python scripts/playback.py --config <same_config.yaml>")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Launch pipelined perturbed dataset generation"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to main.yaml configuration file")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate perturbation files. Do not submit SLURM jobs.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Override run directory (e.g. ./local_run).")
    
    args = parser.parse_args()
    
    launcher = Launcher(args.config, run_dir_override=args.run_dir)
    launcher.run(generate_only=args.generate_only)


if __name__ == "__main__":
    main()