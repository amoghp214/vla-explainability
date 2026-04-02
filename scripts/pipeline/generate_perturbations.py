"""
Config-driven perturbation generation: build list of (unperturbed + perturbed) BDDL/configs.

Reads main config, generates unperturbed BDDL + record config, then BDDL spatial
and/or language perturbations, writing all BDDL and YAML files. Returns
perturbation_info list for use by launcher (manifest, dispatch, evaluation).
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable

from .run_dir import PROJECT_ROOT
from .perturbation import (
    read_bddl,
    fix_init_ranges,
    apply_perturbations,
    validate_bddl,
    generate_move_spec_dict,
)
from .configs import write_record_config


def _generate_bddl_spatial_perturbations(
    base_bddl_text: str,
    config: Dict,
    bddl_dir: Path,
    config_dir: Path,
    create_record_config_fn: Callable[[str, str, str], Dict],
    perturbation_info: List[Dict],
    start_id: int,
    init_range_m: float,
) -> int:
    """Generate BDDL spatial perturbations; append to perturbation_info; return next pert_id."""
    pert_config = config["perturbations"]["bddl_spatial"]
    specs = pert_config.get("perturbation_specs", [])
    max_move_m = pert_config.get("max_move_m", 0.05)
    base_prompt = config["base_prompt"]

    pert_id = start_id
    pert_id_copy = pert_id

    for spec in specs:
        pert_type = spec["type"]
        spec_max_move_m = spec.get("max_move_m", max_move_m)
        perturbations = {}

        if pert_type == "distractor":
            count = spec.get("count", 1)
            perturbations["distractor"] = [None] * count
        elif pert_type == "control":
            pert_id_copy = pert_id
            pert_id = "control"
        else:
            objects = spec.get("objects", [])
            if pert_type not in perturbations:
                perturbations[pert_type] = []
            perturbations[pert_type].extend(objects)

        perturbation_spec_dict = None
        if pert_type == "move":
            objects = spec.get("objects", [])
            perturbation_spec_dict = generate_move_spec_dict(
                base_bddl_text, objects, max_move_m=spec_max_move_m
            )

        try:
            perturbed_bddl, _z_overrides = apply_perturbations(
                copy.deepcopy(base_bddl_text),
                perturbations,
                init_range_m=init_range_m,
                max_move_m=spec_max_move_m,
                perturbation_spec_dict=perturbation_spec_dict,
            )
            if not validate_bddl(perturbed_bddl):
                print(f"[WARN] Perturbation {pert_id} failed validation, skipping")
                continue

            pert_bddl_path = bddl_dir / f"perturbed_{pert_id}.bddl"
            with open(pert_bddl_path, "w") as f:
                f.write(perturbed_bddl)

            record_config = create_record_config_fn(
                perturbation_id=f"perturbed_{pert_id}",
                bddl_file=str(pert_bddl_path),
                prompt=base_prompt,
            )
            config_path = config_dir / f"perturbed_{pert_id}.yaml"
            write_record_config(record_config, config_path)

            if pert_type == "distractor":
                count = spec.get("count", 1)
                description = f"Added {count} distractor object(s) to the scene"
            else:
                objects = spec.get("objects", [])
                pert_type_names = {
                    "move": "moved",
                    "reorient": "reoriented",
                    "color": "changed color of",
                    "replace": "replaced",
                }
                action = pert_type_names.get(pert_type, pert_type)
                obj_list = ", ".join(objects)
                description = f"{action.capitalize()} {obj_list}"

            perturbation_info.append({
                "id": f"perturbed_{pert_id}",
                "bddl_file": str(pert_bddl_path),
                "config_file": str(config_path),
                "prompt": base_prompt,
                "type": f"bddl_spatial_{pert_type}",
                "perturbations": perturbations,
                "description": description,
            })

            if pert_id == "control":
                pert_id = pert_id_copy
            else:
                pert_id += 1

        except Exception as e:
            print(f"[ERROR] Failed to generate perturbation {pert_id}: {e}")
            continue

    return pert_id


def _generate_language_perturbations(
    config: Dict,
    bddl_dir: Path,
    config_dir: Path,
    create_record_config_fn: Callable[[str, str, str], Dict],
    perturbation_info: List[Dict],
    start_id: int,
) -> int:
    """Generate language perturbations; append to perturbation_info; return next pert_id."""
    lang_path = PROJECT_ROOT / "explainability" / "perturbations" / "language"
    if str(lang_path) not in sys.path:
        sys.path.insert(0, str(lang_path))
    from generate_perturbations import generate_perturbations as generate_language_perturbations

    base_prompt = config["base_prompt"]
    pert_dict = generate_language_perturbations(base_prompt)
    pert_bddl_path = bddl_dir / "unperturbed.bddl"

    pert_descriptions = {
        "keyboard": "Keyboard typo",
        "ocr": "OCR error simulation",
        "ci": "Concatenation/insertion",
        "cr": "Character replacement",
        "cs": "Character swap",
        "cd": "Character deletion",
        "ws": "Word swap",
        "wd": "Word deletion",
        "ip": "Insert punctuation",
        "paraphrase0": "Paraphrase variant 0",
        "paraphrase1": "Paraphrase variant 1",
        "paraphrase2": "Paraphrase variant 2",
        "paraphrase3": "Paraphrase variant 3",
        "paraphrase4": "Paraphrase variant 4",
    }

    pert_id = start_id
    for pert_name, pert_prompt in pert_dict.items():
        record_config = create_record_config_fn(
            perturbation_id=f"perturbed_{pert_id}",
            bddl_file=str(pert_bddl_path),
            prompt=pert_prompt,
        )
        config_path = config_dir / f"perturbed_{pert_id}.yaml"
        write_record_config(record_config, config_path)

        if pert_name.startswith("wd_all_"):
            idx = pert_name.split("_")[-1]
            description = f"Word deletion (removed word at position {idx})"
        else:
            description = pert_descriptions.get(pert_name, f"Language perturbation: {pert_name}")

        perturbation_info.append({
            "id": f"perturbed_{pert_id}",
            "bddl_file": str(pert_bddl_path),
            "config_file": str(config_path),
            "prompt": pert_prompt,
            "type": f"language_{pert_name}",
            "original_prompt": base_prompt,
            "description": description,
        })
        pert_id += 1

    return pert_id


def generate_perturbations_from_config(
    config: Dict,
    bddl_dir: Path,
    config_dir: Path,
    results_dir: Path,
    create_record_config_fn: Callable[[str, str, str], Dict],
) -> List[Dict[str, Any]]:
    """
    Generate all perturbation files (BDDL + record YAMLs) from main config.
    Writes unperturbed.bddl, unperturbed.yaml, then spatial and/or language perturbed files.
    Returns list of perturbation_info dicts (id, bddl_file, config_file, prompt, type, description, ...).
    """
    perturbation_info = []

    base_bddl = Path(config["base_bddl_file"])
    if not base_bddl.is_absolute():
        base_bddl = PROJECT_ROOT / base_bddl
    if not base_bddl.exists():
        raise FileNotFoundError(f"Base BDDL file not found: {base_bddl}")

    base_bddl_text = read_bddl(str(base_bddl))
    pert_config = config.get("perturbations", {})
    bddl_spatial = pert_config.get("bddl_spatial", {})
    init_range_m = config.get("init_range_m", bddl_spatial.get("init_range_m", 0.001))
    base_bddl_text = fix_init_ranges(
        base_bddl_text,
        init_range_m=init_range_m,
    )

    unperturbed_bddl_path = bddl_dir / "unperturbed.bddl"
    with open(unperturbed_bddl_path, "w") as f:
        f.write(base_bddl_text)

    unperturbed_config = create_record_config_fn(
        perturbation_id="unperturbed",
        bddl_file=str(unperturbed_bddl_path),
        prompt=config["base_prompt"],
    )
    unperturbed_config_path = config_dir / "unperturbed.yaml"
    write_record_config(unperturbed_config, unperturbed_config_path)

    perturbation_info.append({
        "id": "unperturbed",
        "bddl_file": str(unperturbed_bddl_path),
        "config_file": str(unperturbed_config_path),
        "prompt": config["base_prompt"],
        "type": "baseline",
        "description": "Baseline unperturbed task",
    })

    pert_types = pert_config.get("types", [])
    pert_id = 0

    if "bddl_spatial" in pert_types:
        pert_id = _generate_bddl_spatial_perturbations(
            base_bddl_text,
            config,
            bddl_dir,
            config_dir,
            create_record_config_fn,
            perturbation_info,
            pert_id,
            init_range_m=init_range_m,
        )

    if "language" in pert_types:
        pert_id = _generate_language_perturbations(
            config,
            bddl_dir,
            config_dir,
            create_record_config_fn,
            perturbation_info,
            pert_id,
        )

    return perturbation_info


def save_perturbation_manifest(perturbation_info: List[Dict], run_dir: Path) -> None:
    """Write perturbation_manifest.json to run_dir."""
    manifest_path = run_dir / "perturbation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(perturbation_info, f, indent=2)
