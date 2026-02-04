import os
import random
import datetime
import re

# --------------------------
# Utilities
# --------------------------

def read_bddl(path):
    with open(path, "r") as f:
        return f.read()

def save_bddl(content, base_name="perturbed_scene", folder="perturbed_bddl"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.bddl"
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"[INFO] Saved perturbed BDDL → {path}")
    return path

# --------------------------
# BDDL Structure Extraction
# --------------------------

def extract_declared_objects(bddl_text):
    """Extract all objects declared in :objects section."""
    obj_pattern = r":objects\s*\n((?:.*\n)*?)\s*\)"
    match = re.search(obj_pattern, bddl_text, re.MULTILINE)
    if not match:
        return set()
    
    objects = set()
    content = match.group(1)
    # Match lines like "akita_black_bowl_1 - akita_black_bowl"
    for line in content.strip().split('\n'):
        obj_match = re.match(r'\s*(\w+)\s*-\s*(\w+)', line)
        if obj_match:
            objects.add(obj_match.group(1))
    return objects

def extract_fixture_objects(bddl_text):
    """Extract all fixtures declared in :fixtures section."""
    fixture_pattern = r":fixtures\s*\n((?:.*\n)*?)\s*\)"
    match = re.search(fixture_pattern, bddl_text, re.MULTILINE)
    if not match:
        return set()
    
    fixtures = set()
    content = match.group(1)
    for line in content.strip().split('\n'):
        fixture_match = re.match(r'\s*(\w+)\s*-\s*(\w+)', line)
        if fixture_match:
            fixtures.add(fixture_match.group(1))
    return fixtures

# --------------------------
# Region parser with better boundary detection
# --------------------------

def find_region_blocks(bddl_text):
    """
    Return dict {region_name: (start_idx, end_idx)} for all *_init_region blocks.
    Only finds TOP-LEVEL regions within :regions block, not nested ones.
    """
    region_blocks = {}
    
    regions_start = bddl_text.find("(:regions")
    if regions_start == -1:
        return region_blocks
    
    depth = 0
    regions_end = -1
    for i in range(regions_start, len(bddl_text)):
        if bddl_text[i] == "(":
            depth += 1
        elif bddl_text[i] == ")":
            depth -= 1
            if depth == 0:
                regions_end = i + 1
                break
    
    if regions_end == -1:
        return region_blocks
    
    pattern = re.compile(r"\((\w+_init_region)\b")
    
    for m in pattern.finditer(bddl_text[regions_start:regions_end]):
        actual_start = regions_start + m.start()
        region_name = m.group(1)
        
        check_depth = 0
        for i in range(regions_start, actual_start):
            if bddl_text[i] == "(":
                check_depth += 1
            elif bddl_text[i] == ")":
                check_depth -= 1
        
        if check_depth == 1:
            depth = 0
            for i in range(actual_start, regions_end):
                if bddl_text[i] == "(":
                    depth += 1
                elif bddl_text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        region_blocks[region_name] = (actual_start, i + 1)
                        break
    
    return region_blocks

def parse_object_region_map(bddl_text, region_blocks):
    """
    Returns dict: {object_name: region_name}
    """
    pattern = re.compile(r"\(On\s+(\w+)\s+([\w_]+_init_region)\)")
    obj_region_map = {}
    available_regions = set(region_blocks.keys())
    
    for match in pattern.finditer(bddl_text):
        obj_name = match.group(1)
        full_region_name = match.group(2)
        
        if full_region_name in available_regions:
            obj_region_map[obj_name] = full_region_name
        else:
            for available_region in available_regions:
                if full_region_name.endswith(available_region):
                    obj_region_map[obj_name] = available_region
                    break
            else:
                obj_region_map[obj_name] = full_region_name
    
    return obj_region_map

# --------------------------
# Workspace Detection
# --------------------------

def extract_target_workspace(bddl_text):
    """Extract the target workspace name from BDDL file."""
    # Look for (:target workspace_name) in region definitions
    target_pattern = re.compile(r"\(:target\s+(\w+)\)")
    matches = target_pattern.findall(bddl_text)
    
    if matches:
        # Return the most common target (usually all regions share the same target)
        from collections import Counter
        target_counts = Counter(matches)
        return target_counts.most_common(1)[0][0]
    
    # Fallback: try to detect from problem name or default
    if "kitchen" in bddl_text.lower():
        return "kitchen_table"
    elif "living_room" in bddl_text.lower():
        return "living_room_table"
    elif "study" in bddl_text.lower():
        return "study_table"
    elif "LIBERO_Kitchen" in bddl_text:
        return "kitchen_table"
    elif "LIBERO_Living_Room" in bddl_text:
        return "living_room_table"
    elif "LIBERO_Study" in bddl_text:
        return "study_table"
    
    return "kitchen_table"  # Default fallback


# --------------------------
# Attribute Support Check
# --------------------------

def supports_attributes(region_name, bddl_text):
    """
    Check if a region supports color/texture attributes.
    Fixture regions (like white_cabinet_init_region) typically don't support these.
    """
    # Extract fixtures
    fixtures = extract_fixture_objects(bddl_text)
    
    # Check if this region is associated with a fixture
    for fixture in fixtures:
        if region_name.startswith(fixture.replace('_', '')):
            return False
        if fixture in region_name:
            return False
    
    return True

# --------------------------
# Init range utilities
# --------------------------

RANGES_PATTERN = re.compile(
    r":ranges\s*\(\s*\(\s*([-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+)\s*\)",
    re.DOTALL,
)


def _collapse_ranges_to_center(coords):
    """Collapse rectangle (x_min, y_min, x_max, y_max) to center point for exact init."""
    x_min, y_min, x_max, y_max = coords
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return [center_x, center_y, center_x, center_y]


def fix_init_ranges(bddl_text, init_object_range_m=0.0):
    """
    Fix region ranges for deterministic initialization.

    When init_object_range_m == 0: collapse all region ranges to their center point
    (exact initialization). When init_object_range_m > 0: collapse to center and
    optionally expand by tolerance (currently same as 0 for deterministic base).

    Args:
        bddl_text: BDDL file content
        init_object_range_m: Valid init range in meters. 0 = exact init (collapse to center).

    Returns:
        Modified BDDL text with fixed ranges.
    """
    region_blocks = find_region_blocks(bddl_text)
    result = bddl_text

    for region_name, (start, end) in region_blocks.items():
        block = result[start:end]
        match = RANGES_PATTERN.search(block)
        if match:
            coords = list(map(float, match.group(1).split()))
            if init_object_range_m <= 0:
                new_coords = _collapse_ranges_to_center(coords)
            else:
                new_coords = _collapse_ranges_to_center(coords)
                half = init_object_range_m / 2
                new_coords = [
                    new_coords[0] - half, new_coords[1] - half,
                    new_coords[0] + half, new_coords[1] + half,
                ]
            new_range = " ".join(f"{x:.6g}" for x in new_coords)
            block = block[: match.start(1)] + new_range + block[match.end(1) :]
            result = result[:start] + block + result[end:]
    return result


# --------------------------
# Perturbation functions
# --------------------------

def move_object(bddl_text, obj_name, obj_region_map, region_blocks, init_object_range_m=0.0, max_move_m=0.05):
    """
    Move object's init region. Shifts center in X/Z (table plane); no Y (vertical).

    - init_object_range_m: Size of init region. 0 = point; >0 = box. Used for both unperturbed and perturbed.
    - max_move_m: Max distance (m) from unperturbed center that the object can be shifted (x/z, diagonal ok).

    The perturbed init region is: center = unperturbed_center + (dx, dz), where |dx|,|dz| <= max_move_m;
    region size = init_object_range_m.
    """
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]

    match = RANGES_PATTERN.search(block)
    if not match:
        return bddl_text

    coords = list(map(float, match.group(1).split()))
    unperturbed_center_x = (coords[0] + coords[2]) / 2
    unperturbed_center_z = (coords[1] + coords[3]) / 2

    # Shift center by up to max_move_m in x and z (diagonal ok)
    delta_x = round(random.uniform(-max_move_m, max_move_m), 4)
    delta_z = round(random.uniform(-max_move_m, max_move_m), 4)
    new_center_x = unperturbed_center_x + delta_x
    new_center_z = unperturbed_center_z + delta_z

    # Init region size: 0 = point; >0 = box
    if init_object_range_m <= 0:
        new_coords = [new_center_x, new_center_z, new_center_x, new_center_z]
    else:
        half = init_object_range_m / 2
        new_coords = [
            new_center_x - half, new_center_z - half,
            new_center_x + half, new_center_z + half,
        ]
    print(f"[MOVE] {obj_name} shifted center by (dx={delta_x}, dz={delta_z}) m, init_range={init_object_range_m}m")

    new_range = " ".join(f"{x:.6g}" for x in new_coords)
    block = block[: match.start(1)] + new_range + block[match.end(1) :]
    bddl_text = bddl_text[:start] + block + bddl_text[end:]
    return bddl_text

def reorient_object(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]

    match = re.search(r":yaw_rotation\s*\(\s*\(\s*([-+]?[0-9]*\.?[0-9]+(?:\s+[-+]?[0-9]*\.?[0-9]+)?)\s*\)", block, re.DOTALL)
    if match:
        vals = list(map(float, match.group(1).split()))
        rotation_type = random.choice(["clockwise", "anticlockwise"])
        angle = round(random.uniform(5, 30), 2)
        delta = angle if rotation_type == "clockwise" else -angle
        vals = [v + delta for v in vals]
        new_yaw = " ".join(map(lambda x: f"{x:.2f}", vals))
        block = block[:match.start(1)] + new_yaw + block[match.end(1):]
        print(f"[REORIENT] {obj_name} rotated {rotation_type} by {angle}°")
        bddl_text = bddl_text[:start] + block + bddl_text[end:]
    return bddl_text

def change_color(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    # Check if this region supports color attributes
    if not supports_attributes(region_name, bddl_text):
        print(f"[SKIP] {obj_name} (region: {region_name}) doesn't support color attributes (likely a fixture)")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]
    
    # Check if rgba already exists and remove it first
    rgba_match = re.search(r'\s*\(:rgba\s*\([^)]+\)\s*\)\s*\n?', block)
    if rgba_match:
        block = block[:rgba_match.start()] + block[rgba_match.end():]
    
    # Define colors with their RGBA values (R, G, B, Alpha)
    colors = {
        "red": [1.0, 0.0, 0.0, 1.0],
        "blue": [0.0, 0.0, 1.0, 1.0],
        "green": [0.0, 1.0, 0.0, 1.0],
        "yellow": [1.0, 1.0, 0.0, 1.0],
        "purple": [0.5, 0.0, 0.5, 1.0],
        "orange": [1.0, 0.5, 0.0, 1.0],
        "white": [1.0, 1.0, 1.0, 1.0],
        "black": [0.0, 0.0, 0.0, 1.0],
        "cyan": [0.0, 1.0, 1.0, 1.0],
        "magenta": [1.0, 0.0, 1.0, 1.0],
    }
    
    color_name = random.choice(list(colors.keys()))
    rgba_values = colors[color_name]
    
    # Find insertion point before the final closing parenthesis
    last_close = block.rfind(')')
    depth = 0
    second_last_close = -1
    for i in range(last_close - 1, -1, -1):
        if block[i] == ')':
            if depth == 0:
                second_last_close = i
                break
            depth += 1
        elif block[i] == '(':
            depth -= 1
    
    if second_last_close > 0:
        insert_pos = second_last_close + 1
        while insert_pos < len(block) and block[insert_pos] in ' \t':
            insert_pos += 1
        if insert_pos < len(block) and block[insert_pos] == '\n':
            insert_pos += 1
        
        prev_line_start = block.rfind('\n', 0, second_last_close)
        if prev_line_start >= 0:
            indent_match = re.match(r'^(\s*)', block[prev_line_start+1:])
            indent = indent_match.group(1) if indent_match else '          '
        else:
            indent = '          '
        
        # Format RGBA values as space-separated floats
        rgba_str = " ".join(str(v) for v in rgba_values)
        rgba_line = f"{indent}(:rgba ({rgba_str}))\n"
        block = block[:insert_pos] + rgba_line + block[insert_pos:]
    
    bddl_text = bddl_text[:start] + block + bddl_text[end:]
    print(f"[COLOR] {obj_name} color changed to {color_name} (RGBA: {rgba_values})")
    return bddl_text

def replace_object(bddl_text, obj_name, target_workspace=None):
    """Replace an object with a new one, maintaining BDDL consistency."""
    # Valid LIBERO object categories (based on common LIBERO objects)
    # These match the object types available in libero/libero/envs/objects/__init__.py
    valid_objects = [
        "akita_black_bowl",
        "white_yellow_mug",
        "wine_bottle",
        "plate",
        "alphabet_soup",
        "cream_cheese",
        "tomato_sauce",
        "ketchup",
        "butter",
        "milk",
        "chocolate_pudding",
        "orange_juice",
        "bbq_sauce",
        "salad_dressing",
        "black_book",
        "moka_pot",
        "chefmate_8_frypan"
    ]
    
    # Auto-detect workspace if not provided
    if target_workspace is None:
        target_workspace = extract_target_workspace(bddl_text)
    
    new_obj_type = random.choice(valid_objects)
    new_obj = f"{new_obj_type}_{random.randint(1,999)}"
    
    # Get the old object type for region name mapping
    old_obj_type_match = re.search(rf"{obj_name}\s*-\s*(\w+)", bddl_text)
    if not old_obj_type_match:
        print(f"[WARN] Could not find object type for {obj_name}")
        return bddl_text
    
    old_obj_type = old_obj_type_match.group(1)
    
    # Create old and new region names
    old_region_name = f"{obj_name}_init_region"
    new_region_name = f"{new_obj}_init_region"
    
    # 1. Replace the region definition name
    bddl_text = re.sub(
        rf"\({old_region_name}\b",
        f"({new_region_name}",
        bddl_text
    )
    
    # 2. Replace in :objects declaration (object_name - object_type)
    bddl_text = re.sub(
        rf"(\s*){obj_name}(\s*-\s*){old_obj_type}\b",
        rf"\1{new_obj}\2{new_obj_type}",
        bddl_text
    )
    
    # 3. Replace in :init statements (On object_name ...)
    bddl_text = re.sub(
        rf"\(On\s+{obj_name}\b",
        f"(On {new_obj}",
        bddl_text
    )
    
    # 4. Replace in region references (target_workspace_old_region_name)
    bddl_text = re.sub(
        rf"\b{target_workspace}_{old_region_name}\b",
        f"{target_workspace}_{new_region_name}",
        bddl_text
    )
    
    # 5. Replace in :goal statements (if the object appears there)
    bddl_text = re.sub(
        rf"\(In\s+{obj_name}\b",
        f"(In {new_obj}",
        bddl_text
    )
    
    # 6. Replace in :obj_of_interest (if present)
    bddl_text = re.sub(
        rf"(\s+){obj_name}(\s*\n)",
        rf"\1{new_obj}\2",
        bddl_text
    )
    
    print(f"[REPLACE] {obj_name} replaced with {new_obj}")
    return bddl_text

def add_distractor(bddl_text, target_workspace=None):
    """Add a distractor object to the scene."""
    # Valid LIBERO object categories (expanded for all scene types)
    valid_objects = [
        "akita_black_bowl",
        "white_yellow_mug",
        "wine_bottle",
        "plate",
        "alphabet_soup",
        "cream_cheese",
        "tomato_sauce",
        "ketchup",
        "butter",
        "milk",
        "chocolate_pudding",
        "orange_juice",
        "bbq_sauce",
        "salad_dressing",
        "black_book",
        "moka_pot",
        "chefmate_8_frypan"
    ]
    
    # Auto-detect workspace if not provided
    if target_workspace is None:
        target_workspace = extract_target_workspace(bddl_text)
    
    obj_type = random.choice(valid_objects)
    new_obj = f"{obj_type}_{random.randint(100,999)}"
    region_name = f"{new_obj}_init_region"

    # Generate ranges: (x_min, y_min, x_max, y_max)
    # Ensure x_max >= x_min and y_max >= y_min
    x_vals = sorted([round(random.uniform(-0.2, 0.2), 3) for _ in range(2)])
    y_vals = sorted([round(random.uniform(-0.2, 0.2), 3) for _ in range(2)])
    x_min, x_max = x_vals[0], x_vals[1]
    y_min, y_max = y_vals[0], y_vals[1]
    
    # Format: (x_min, y_min, x_max, y_max)
    ranges_str = f"{x_min} {y_min} {x_max} {y_max}"

    region_def = f"""      ({region_name}
          (:target {target_workspace})
          (:ranges (
              ({ranges_str})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )"""

    regions_start = bddl_text.find("(:regions")
    if regions_start == -1:
        print("[WARN] Could not find :regions block")
        return bddl_text
    
    depth = 0
    regions_end = -1
    for i in range(regions_start, len(bddl_text)):
        if bddl_text[i] == "(":
            depth += 1
        elif bddl_text[i] == ")":
            depth -= 1
            if depth == 0:
                regions_end = i
                break
    
    if regions_end == -1:
        print("[WARN] Could not find end of :regions block")
        return bddl_text
    
    line_before = bddl_text[:regions_end].rfind('\n')
    indent_match = re.match(r'^(\s*)', bddl_text[line_before+1:regions_end])
    base_indent = indent_match.group(1) if indent_match else "    "
    
    bddl_text = bddl_text[:regions_end] + "\n" + region_def + "\n" + base_indent + bddl_text[regions_end:]

    # Add to :objects
    obj_pattern = r"(\(:objects\s*\n)((?:.*\n)*?)(\s*\))"
    obj_match = re.search(obj_pattern, bddl_text)
    if obj_match:
        obj_content = obj_match.group(2)
        last_line = obj_content.rstrip().split('\n')[-1] if obj_content.strip() else ""
        indent = re.match(r'^(\s*)', last_line).group(1) if last_line else "    "
        new_content = obj_content + f"{indent}{new_obj} - {obj_type}\n"
        bddl_text = bddl_text[:obj_match.start()] + obj_match.group(1) + new_content + obj_match.group(3) + bddl_text[obj_match.end():]

    # Add to :init
    init_pattern = r"(\(:init\s*\n)((?:.*\n)*?)(\s*\))"
    init_match = re.search(init_pattern, bddl_text)
    if init_match:
        init_content = init_match.group(2)
        last_line = init_content.rstrip().split('\n')[-1] if init_content.strip() else ""
        indent = re.match(r'^(\s*)', last_line).group(1) if last_line else "    "
        new_content = init_content + f"{indent}(On {new_obj} {target_workspace}_{region_name})\n"
        bddl_text = bddl_text[:init_match.start()] + init_match.group(1) + new_content + init_match.group(3) + bddl_text[init_match.end():]

    print(f"[DISTRACTOR] Added new object {new_obj} at ({x_min}, {y_min}, {x_max}, {y_max}) on {target_workspace}")
    return bddl_text

# --------------------------
# Apply perturbations
# --------------------------

def apply_perturbations_kitchen(bddl_text, perturbations, init_object_range_m=0.0, max_move_m=0.05):
    """Apply perturbations to kitchen scenes (deprecated, use apply_perturbations instead)."""
    return apply_perturbations(bddl_text, perturbations, init_object_range_m, max_move_m)


def apply_perturbations(bddl_text, perturbations, init_object_range_m=0.0, max_move_m=0.05):
    """
    Apply perturbations to any LIBERO scene type.

    Args:
        bddl_text: BDDL file content as string
        perturbations: Dictionary of perturbations to apply
            - "move": list of object names to move
            - "reorient": list of object names to reorient
            - "color": list of object names to change color
            - "replace": list of object names to replace
            - "distractor": list of None values (count determines number of distractors)
        init_object_range_m: Size of init region (m). 0 = point; >0 = box. Used for both unperturbed and perturbed.
        max_move_m: Max distance (m) from unperturbed center that move can shift the object.

    Returns:
        Modified BDDL text
    """
    region_blocks = find_region_blocks(bddl_text)
    obj_region_map = parse_object_region_map(bddl_text, region_blocks)
    target_workspace = extract_target_workspace(bddl_text)

    print(f"[DEBUG] Detected workspace: {target_workspace}")
    print(f"[DEBUG] Object-Region mapping: {obj_region_map}")
    print(f"[DEBUG] Available regions: {list(region_blocks.keys())}")
    print(f"[DEBUG] init_object_range_m: {init_object_range_m}, max_move_m: {max_move_m}")

    for key, obj_list in perturbations.items():
        for obj_name in obj_list:
            if key == "move":
                bddl_text = move_object(bddl_text, obj_name, obj_region_map, region_blocks, init_object_range_m, max_move_m)
                region_blocks = find_region_blocks(bddl_text)
                obj_region_map = parse_object_region_map(bddl_text, region_blocks)
            elif key == "reorient":
                bddl_text = reorient_object(bddl_text, obj_name, obj_region_map, region_blocks)
                region_blocks = find_region_blocks(bddl_text)
                obj_region_map = parse_object_region_map(bddl_text, region_blocks)
            elif key == "color":
                bddl_text = change_color(bddl_text, obj_name, obj_region_map, region_blocks)
                region_blocks = find_region_blocks(bddl_text)
                obj_region_map = parse_object_region_map(bddl_text, region_blocks)
            elif key == "replace":
                bddl_text = replace_object(bddl_text, obj_name, target_workspace)
                region_blocks = find_region_blocks(bddl_text)
                obj_region_map = parse_object_region_map(bddl_text, region_blocks)

    if "distractor" in perturbations:
        for _ in perturbations["distractor"]:
            bddl_text = add_distractor(bddl_text, target_workspace)

    return bddl_text

# --------------------------
# Validation
# --------------------------

def validate_bddl(bddl_text):
    """Comprehensive BDDL validation."""
    errors = []
    
    # 1. Check parentheses balance
    stack = []
    for i, ch in enumerate(bddl_text):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                errors.append(f"Extra ')' at index {i}")
                return False
            stack.pop()
    if stack:
        errors.append(f"{len(stack)} unmatched '(' remaining")
        return False
    
    # 2. Check all objects in :init are declared in :objects or :fixtures
    declared_objects = extract_declared_objects(bddl_text)
    declared_fixtures = extract_fixture_objects(bddl_text)
    all_declared = declared_objects | declared_fixtures
    
    init_pattern = r"\(On\s+(\w+)\s+"
    for match in re.finditer(init_pattern, bddl_text):
        obj_name = match.group(1)
        if obj_name not in all_declared:
            errors.append(f"Object '{obj_name}' used in :init but not declared in :objects or :fixtures")
    
    # 3. Check all regions referenced in :init exist
    region_blocks = find_region_blocks(bddl_text)
    available_regions = set(region_blocks.keys())
    
    # Pattern: (On object_name target_region_name)
    # The region_name format is: target_regionname or just regionname
    init_region_pattern = r"\(On\s+\w+\s+([\w_]+)\)"
    for match in re.finditer(init_region_pattern, bddl_text):
        full_region_ref = match.group(1)
        # The reference might be like "kitchen_table_akita_black_bowl_init_region"
        # We need to extract just the "akita_black_bowl_init_region" part
        
        # Check if it's a direct match first
        if full_region_ref in available_regions:
            continue
            
        # Otherwise, try to extract the region name from a composite like "target_regionname"
        found = False
        for region in available_regions:
            if full_region_ref.endswith(region):
                found = True
                break
        
        if not found:
            errors.append(f"Region reference '{full_region_ref}' in :init doesn't match any defined region")
    
    if errors:
        print("[VALIDATION ERRORS]")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("[VALID] BDDL structure is correct")
    return True

# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    input_file = "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl"
    bddl_text = read_bddl(input_file)

    perturbations = {
        "move": ["akita_black_bowl_1", "wine_bottle_1"],
        "reorient": ["wine_bottle_1"],
        "color": ["wine_bottle_1"],  # Now uses :rgba attribute
        "replace": ["wine_bottle_1"],
        "distractor": [1]
    }

    perturbed_bddl = apply_perturbations_kitchen(bddl_text, perturbations)
    
    if validate_bddl(perturbed_bddl):
        save_bddl(perturbed_bddl, base_name="LIBERO_Kitchen_Tabletop_Manipulation_perturbed")
    else:
        print("[ERROR] Generated BDDL failed validation. Not saving.")
