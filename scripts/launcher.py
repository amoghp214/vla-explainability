#!/usr/bin/env python3
"""
Launcher script for pipelined perturbed dataset generation in PACE-ICE environment.

This script:
1. Creates a run directory structure in scratch folder
2. Generates perturbation files (BDDL and config YAMLs)
3. Dispatches SLURM jobs in a queue-like fashion
4. Runs evaluation scripts after all jobs complete

Usage:
    python scripts/launcher.py --config configs/main.yaml
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
    apply_perturbations_kitchen = pert_utils.apply_perturbations_kitchen
    apply_perturbations = getattr(pert_utils, 'apply_perturbations', pert_utils.apply_perturbations_kitchen)  # Use generic if available
    validate_bddl = pert_utils.validate_bddl
    fix_init_ranges = getattr(pert_utils, 'fix_init_ranges', lambda t, r=0: t)
else:
    raise ImportError(f"Could not find perturbation utilities at {perturbation_utils_path}")


class Launcher:
    """Main launcher class for managing the pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize launcher with config file."""
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get scratch directory
        scratch = os.environ.get('SCRATCH', os.path.expanduser('~/scratch'))
        self.scratch_dir = Path(scratch)
        
        # Set up run directory
        run_base = self.config.get('run_base_dir')
        if run_base is None:
            run_base = self.scratch_dir / 'vla-explainability-runs'
        else:
            run_base = Path(run_base)
        
        # Create timestamped run directory
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

        # Init object range (m): size of spawn region. 0 = point; >0 = box. Used for unperturbed and perturbed.
        init_object_range_m = self.config.get('init_object_range_m', 0.0)
        pert_config = self.config.get('perturbations', {})
        if 'bddl_spatial' in pert_config:
            init_object_range_m = pert_config['bddl_spatial'].get('init_object_range_m', init_object_range_m)

        # Fix init ranges (sets region size for unperturbed; perturbed move uses same size)
        base_bddl_text = fix_init_ranges(base_bddl_text, init_object_range_m=init_object_range_m)

        # Copy base BDDL (unperturbed) with fixed ranges
        unperturbed_bddl_path = self.bddl_dir / "unperturbed.bddl"
        with open(unperturbed_bddl_path, 'w') as f:
            f.write(base_bddl_text)
        
        # Create unperturbed config
        unperturbed_config = self._create_record_config(
            perturbation_id="unperturbed",
            bddl_file=str(unperturbed_bddl_path),
            prompt=self.config['base_prompt']
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
            'description': 'Baseline unperturbed task'
        })
        
        # Generate perturbed versions
        pert_config = self.config.get('perturbations', {})
        pert_types = pert_config.get('types', [])
        
        pert_id = 0
        
        if 'bddl_spatial' in pert_types:
            pert_id = self._generate_bddl_spatial_perturbations(
                base_bddl_text, pert_id
            )
        
        if 'language' in pert_types:
            pert_id = self._generate_language_perturbations(
                base_bddl_text, pert_id
            )
        
        print(f"[INFO] Generated {len(self.perturbation_info)} perturbation files")
        
        # Save perturbation manifest
        manifest_path = self.run_dir / "perturbation_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.perturbation_info, f, indent=2)
    
    def _generate_bddl_spatial_perturbations(self, base_bddl_text: str, start_id: int) -> int:
        """Generate BDDL spatial perturbations."""
        pert_config = self.config['perturbations']['bddl_spatial']
        init_object_range_m = pert_config.get('init_object_range_m', 0.0)
        specs = pert_config.get('perturbation_specs', [])
        
        pert_id = start_id
        
        for spec in specs:
            pert_type = spec['type']
            max_move_m = pert_config.get('max_move_m', 0.05)

            # Build perturbation dict for apply_perturbations
            perturbations = {}
            
            if pert_type == 'distractor':
                count = spec.get('count', 1)
                perturbations['distractor'] = [None] * count  # List of None values for distractor count
            else:
                objects = spec.get('objects', [])
                if pert_type not in perturbations:
                    perturbations[pert_type] = []
                perturbations[pert_type].extend(objects)
            
            # Per-spec override for max_move_m (only applies to type: move)
            if pert_type == 'move':
                max_move_m = spec.get('max_move_m', max_move_m)
            
            # Apply perturbations (use generic function for all scene types)
            try:
                perturbed_bddl = apply_perturbations(
                    copy.deepcopy(base_bddl_text),
                    perturbations,
                    init_object_range_m=init_object_range_m,
                    max_move_m=max_move_m,
                )
                
                # Validate
                if not validate_bddl(perturbed_bddl):
                    print(f"[WARN] Perturbation {pert_id} failed validation, skipping")
                    continue
                
                # Save BDDL
                pert_bddl_path = self.bddl_dir / f"perturbed_{pert_id}.bddl"
                with open(pert_bddl_path, 'w') as f:
                    f.write(perturbed_bddl)
                
                # Create config
                pert_config_path = self._create_record_config(
                    perturbation_id=f"perturbed_{pert_id}",
                    bddl_file=str(pert_bddl_path),
                    prompt=self.config['base_prompt']  # Keep same prompt for spatial
                )
                config_path = self.config_dir / f"perturbed_{pert_id}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(pert_config_path, f)
                
                # Generate description
                if pert_type == 'distractor':
                    count = spec.get('count', 1)
                    description = f"Added {count} distractor object(s) to the scene"
                else:
                    objects = spec.get('objects', [])
                    pert_type_names = {
                        'move': 'moved',
                        'reorient': 'reoriented',
                        'color': 'changed color of',
                        'replace': 'replaced'
                    }
                    action = pert_type_names.get(pert_type, pert_type)
                    obj_list = ', '.join(objects)
                    description = f"{action.capitalize()} {obj_list}"
                
                self.perturbation_info.append({
                    'id': f'perturbed_{pert_id}',
                    'bddl_file': str(pert_bddl_path),
                    'config_file': str(config_path),
                    'prompt': self.config['base_prompt'],
                    'type': f'bddl_spatial_{pert_type}',
                    'perturbations': perturbations,
                    'description': description
                })
                
                pert_id += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to generate perturbation {pert_id}: {e}")
                continue
        
        return pert_id
    
    def _generate_language_perturbations(self, base_bddl_text: str, start_id: int) -> int:
        """Generate language perturbations."""
        # Import language perturbation generator
        sys.path.insert(0, str(project_root / "explainability" / "perturbations" / "language"))
        from generate_perturbations import generate_perturbations
        
        pert_id = start_id
        base_prompt = self.config['base_prompt']
        
        # Generate all language perturbations
        pert_dict = generate_perturbations(base_prompt)
        
        for pert_name, pert_prompt in pert_dict.items():
            # Use same BDDL file (no spatial changes)
            pert_bddl_path = self.bddl_dir / "unperturbed.bddl"  # Same as baseline
            
            # Create config with perturbed prompt
            pert_config = self._create_record_config(
                perturbation_id=f"perturbed_{pert_id}",
                bddl_file=str(pert_bddl_path),
                prompt=pert_prompt
            )
            config_path = self.config_dir / f"perturbed_{pert_id}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(pert_config, f)
            
            # Generate description for language perturbation
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
                'paraphrase4': 'Paraphrase variant 4'
            }
            
            # Handle word deletion variants
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
                'description': description
            })
            
            pert_id += 1
        
        return pert_id
    
    def _create_record_config(self, perturbation_id: str, bddl_file: str, prompt: str) -> Dict:
        """Create a record config YAML for a perturbation."""
        # Make paths absolute
        bddl_path = Path(bddl_file)
        if not bddl_path.is_absolute():
            bddl_path = self.bddl_dir / bddl_path.name
        
        # Output file path
        out_file = self.results_dir / f"{perturbation_id}.hdf5"
        
        # Videos directory for playback
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
        }
        
        return config
    
    def create_slurm_job(self, perturbation_id: str, config_file: str) -> str:
        """Create a SLURM job script for recording."""
        job_script = self.jobs_dir / f"{perturbation_id}.sh"
        
        slurm_config = self.config['slurm']
        job_params = slurm_config['job_params']
        
        # Build job script
        script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_config['job_name_prefix']}_{perturbation_id}
#SBATCH --account={job_params['account']}
#SBATCH --time={job_params['time']}
#SBATCH --nodes={job_params['nodes']}
#SBATCH --ntasks-per-node={job_params['ntasks_per_node']}
#SBATCH --cpus-per-task={job_params['cpus_per_task']}
"""
        
        # Add partition if specified (optional for PACE-ICE)
        if job_params.get('partition'):
            script_content += f"#SBATCH --partition={job_params['partition']}\n"
        
        # Add memory specification
        if job_params.get('mem'):
            script_content += f"#SBATCH --mem={job_params['mem']}\n"
        
        # Add GPU specification (PACE-ICE format: --gres=gpu:<type>:<number>)
        if job_params.get('gpus', 0) > 0:
            gpu_type = job_params.get('gpu_type', 'V100')
            num_gpus = job_params['gpus']
            script_content += f"#SBATCH --gres=gpu:{gpu_type}:{num_gpus}\n"
        
        # Add constraint if specified (for specific GPU models)
        if job_params.get('constraint'):
            script_content += f"#SBATCH -C {job_params['constraint']}\n"
            
        # Add blacklisted nodes if specified
        blacklisted_nodes = job_params.get('blacklisted_nodes', [])
        if blacklisted_nodes:
            # Convert list to comma-separated string
            if isinstance(blacklisted_nodes, list):
                node_list = ','.join(str(node) for node in blacklisted_nodes)
            else:
                node_list = str(blacklisted_nodes)
            script_content += f"#SBATCH --exclude={node_list}\n"
        
        # Add output files
        script_content += f"""#SBATCH --output={self.logs_dir}/{perturbation_id}_%j.out
#SBATCH --error={self.logs_dir}/{perturbation_id}_%j.err

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Load modules if specified
"""
        
        for module in slurm_config.get('module_load', []):
            script_content += f"module load {module}\n"
        
        script_content += f"""
# Initialize conda (try common locations)
# if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#     source $HOME/miniconda3/etc/profile.d/conda.sh
# elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#     source $HOME/anaconda3/etc/profile.d/conda.sh
# elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#     source /opt/conda/etc/profile.d/conda.sh
# fi

# Activate conda environment
conda activate {slurm_config['conda_env']}

# Change to project directory
cd {project_root}

# Install/update package in editable mode
# Try to import first - if it fails, install. If install fails, the job should fail.
if ! python -c "import libero" 2>/dev/null; then
    echo "Installing libero package..."
    pip install -e . || {{ echo "ERROR: Failed to install libero package"; exit 1; }}
else
    echo "libero package already installed, skipping installation"
fi

# Run record script
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
                capture_output=True,
                text=True,
                check=True
            )
            # Extract job ID from output: "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to submit job {job_script}: {e.stderr}")
            return None
    
    def check_job_status(self, job_id: str) -> str:
        """Check status of a SLURM job."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True,
                text=True,
                check=True
            )
            status = result.stdout.strip()
            if not status:
                # Job not in queue, check if it completed successfully
                # Check if output file exists
                return "COMPLETED"
            return status
        except subprocess.CalledProcessError:
            # Job might have finished
            return "COMPLETED"
    
    def dispatch_jobs(self):
        """Dispatch SLURM jobs in queue-like fashion."""
        print("\n[INFO] Dispatching SLURM jobs...")
        
        max_concurrent = self.config['slurm']['max_concurrent_jobs']
        
        # Create job scripts for all perturbations
        job_scripts = {}
        for pert_info in self.perturbation_info:
            pert_id = pert_info['id']
            config_file = pert_info['config_file']
            job_script = self.create_slurm_job(pert_id, config_file)
            job_scripts[pert_id] = job_script
            self.pending_jobs.append(pert_id)
        
        print(f"[INFO] Created {len(job_scripts)} job scripts")
        print(f"[INFO] Max concurrent jobs: {max_concurrent}")
        
        # Dispatch jobs
        while self.pending_jobs or self.running_jobs:
            # Check status of running jobs
            for pert_id in list(self.running_jobs):
                job_id = self.job_status.get(pert_id)
                if job_id:
                    status = self.check_job_status(job_id)
                    if status == "COMPLETED":
                        # Check if output file exists
                        pert_info = next(p for p in self.perturbation_info if p['id'] == pert_id)
                        out_file = Path(self.results_dir / f"{pert_id}.hdf5")
                        if out_file.exists():
                            print(f"[INFO] Job {pert_id} completed successfully")
                            self.running_jobs.remove(pert_id)
                            self.completed_jobs.append(pert_id)
                            del self.job_status[pert_id]
                        else:
                            print(f"[WARN] Job {pert_id} completed but output file missing")
                            self.running_jobs.remove(pert_id)
                            self.failed_jobs.append(pert_id)
                            del self.job_status[pert_id]
            
            # Submit new jobs if we have capacity
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
            
            # Wait before next check
            if self.running_jobs:
                time.sleep(self.config['slurm']['poll_interval'])
        
        print(f"\n[INFO] All jobs dispatched")
        print(f"  Completed: {len(self.completed_jobs)}")
        print(f"  Failed: {len(self.failed_jobs)}")
        
        # Save job summary
        summary = {
            'completed': self.completed_jobs,
            'failed': self.failed_jobs,
            'total': len(self.perturbation_info)
        }
        with open(self.run_dir / "job_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def render_videos(self):
        """Render videos for all completed recordings using playback.py."""
        print("\n[INFO] Rendering videos for completed recordings...")
        
        videos_dir = self.results_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        rendered_count = 0
        
        # Render unperturbed video
        unperturbed_hdf5 = self.results_dir / "unperturbed.hdf5"
        if unperturbed_hdf5.exists():
            unperturbed_config = self.config_dir / "unperturbed.yaml"
            if unperturbed_config.exists():
                print(f"[INFO] Rendering unperturbed video...")
                try:
                    cmd = [
                        sys.executable,
                        str(project_root / "scripts" / "playback.py"),
                        "--config", str(unperturbed_config)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    output_video = videos_dir / "unperturbed.mp4"
                    print(f"  ✓ Rendered: {output_video}")
                    rendered_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Failed to render unperturbed video: {e}")
        
        # Render perturbed videos
        for pert_info in self.perturbation_info:
            if pert_info['id'] == 'unperturbed':
                continue
            
            pert_id = pert_info['id']
            pert_hdf5 = self.results_dir / f"{pert_id}.hdf5"
            pert_config = Path(pert_info['config_file'])
            
            if pert_hdf5.exists() and pert_config.exists():
                print(f"[INFO] Rendering {pert_id} video...")
                try:
                    cmd = [
                        sys.executable,
                        str(project_root / "scripts" / "playback.py"),
                        "--config", str(pert_config)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    output_video = videos_dir / f"{pert_id}.mp4"
                    print(f"  ✓ Rendered: {output_video}")
                    rendered_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Failed to render {pert_id} video: {e}")
        
        print(f"\n[INFO] Rendered {rendered_count} video(s) to {videos_dir}")
    
    def run_evaluation(self):
        """Run evaluation scripts after all jobs complete."""
        eval_config = self.config.get('evaluation', {})
        if not eval_config.get('enabled', False):
            print("[INFO] Evaluation disabled, skipping")
            return
        
        print("\n[INFO] Running evaluation...")
        
        # Find unperturbed file
        unperturbed_file = self.results_dir / "unperturbed.hdf5"
        if not unperturbed_file.exists():
            print("[ERROR] Unperturbed file not found, cannot run evaluation")
            return
        
        # Find all perturbed files
        perturbed_files = []
        for pert_info in self.perturbation_info:
            if pert_info['id'] != 'unperturbed':
                pert_file = self.results_dir / f"{pert_info['id']}.hdf5"
                if pert_file.exists():
                    perturbed_files.append(str(pert_file))
        
        if not perturbed_files:
            print("[WARN] No perturbed files found for evaluation")
            return

        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root) + ":" + env.get('PYTHONPATH', '')
        
        # Convert to JSON if needed
        if 'json' in eval_config.get('output_formats', []):
            json_output = self.results_dir / "trajectories.json"
            
            # Use hdf5_to_json utility
            cmd = [
                sys.executable,
                str(project_root / "utils" / "hdf5_to_json.py"),
                str(unperturbed_file),
                "-p"
            ] + perturbed_files + [
                "-o",
                str(json_output)
            ]
            
            print(f"[INFO] Converting to JSON: {json_output}")
            subprocess.run(cmd, check=True, env=env)
        
        # Run analysis using episodic_explanation functions
        self._run_analysis(unperturbed_file, perturbed_files, eval_config)
    
    def _run_analysis(self, unperturbed_file: Path, perturbed_files: List[str], eval_config: Dict):
        """Run analysis using episodic_explanation and vla_metrics."""
        print("[INFO] Running trajectory analysis...")
        
        # Import analysis module
        analysis_module_path = project_root / "explainability" / "run_analysis.py"
        if not analysis_module_path.exists():
            raise FileNotFoundError(f"Analysis module not found at {analysis_module_path}")
        
        # Prepare output file
        output_file = self.run_dir / "analysis_results.json"
        
        # Build command to run analysis
        cmd = [
            sys.executable,
            str(analysis_module_path),
            "--unperturbed", str(unperturbed_file),
            "--perturbed"
        ] + [str(p) for p in perturbed_files] + [
            "--output", str(output_file),
            "--metric-weights", json.dumps(eval_config['metric_weights']),
            "--trajectory-weights", json.dumps(eval_config['trajectory_weights']),
            "--project-root", str(project_root)
        ]
        
        # Run analysis
        subprocess.run(cmd, check=True)
    
    def run(self):
        """Run the complete pipeline."""
        print("=" * 80)
        print("VLA Explainability Pipeline Launcher")
        print("=" * 80)
        print(f"Run directory: {self.run_dir}")
        
        # Step 1: Generate perturbations
        self.generate_perturbations()
        
        # Step 2: Dispatch jobs
        self.dispatch_jobs()
        
        # Step 3: Render videos
        self.render_videos()
        
        # Step 4: Run evaluation
        self.run_evaluation()
        
        print("\n" + "=" * 80)
        print("Pipeline complete!")
        print(f"Results available in: {self.run_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Launch pipelined perturbed dataset generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to main.yaml configuration file"
    )
    
    args = parser.parse_args()
    
    launcher = Launcher(args.config)
    launcher.run()


if __name__ == "__main__":
    main()

