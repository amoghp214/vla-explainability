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
# Region parser
# --------------------------

def find_region_blocks(bddl_text):
    """
    Returns dict {region_name: (start_idx, end_idx)} for all *_init_region blocks
    """
    region_blocks = {}
    pattern = re.compile(r"\((\w+_init_region)")
    for m in pattern.finditer(bddl_text):
        start = m.start()
        region_name = m.group(1)
        # Parse until matching parenthesis
        idx = start
        stack = 0
        while idx < len(bddl_text):
            if bddl_text[idx] == "(":
                stack += 1
            elif bddl_text[idx] == ")":
                stack -= 1
                if stack == 0:
                    end = idx + 1
                    region_blocks[region_name] = (start, end)
                    break
            idx += 1
    return region_blocks

def parse_object_region_map(bddl_text, region_blocks):
    """
    Returns dict: {object_name: region_name}
    Strips prefixes from region names in :init to match actual region definitions.
    E.g., "kitchen_table_akita_black_bowl_init_region" → "akita_black_bowl_init_region"
    """
    pattern = re.compile(r"\(On\s+(\w+)\s+([\w_]+_init_region)\)")
    obj_region_map = {}
    available_regions = set(region_blocks.keys())

    for match in pattern.finditer(bddl_text):
        obj_name = match.group(1)
        full_region_name = match.group(2)

        # First check if the full name exists as-is
        if full_region_name in available_regions:
            obj_region_map[obj_name] = full_region_name
        else:
            # Try to find a matching region by removing prefix
            # Look for any available region that is a suffix of the full name
            found = False
            for available_region in available_regions:
                if full_region_name.endswith(available_region):
                    obj_region_map[obj_name] = available_region
                    found = True
                    break

            if not found:
                # Fallback: use the full name anyway
                obj_region_map[obj_name] = full_region_name

    return obj_region_map

# --------------------------
# Perturbation functions
# --------------------------

def move_object(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]

    match = re.search(r":ranges\s*\(\s*\((.*?)\)\s*\)", block, re.DOTALL)
    if match:
        coords = list(map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+", match.group(1))))
        direction = random.choice(["left", "right", "up", "down"])
        offset = round(random.uniform(0.01, 0.05), 3)
        if direction in ["left", "right"]:
            delta = offset if direction == "right" else -offset
            coords = [coords[i] + delta if i % 2 == 0 else coords[i] for i in range(4)]
        else:
            delta = offset if direction == "up" else -offset
            coords = [coords[i] + delta if i % 2 == 1 else coords[i] for i in range(4)]
        new_range = " ".join(map(lambda x: f"{x:.3f}", coords))
        block = block[:match.start(1)] + new_range + block[match.end(1):]
        print(f"[MOVE] {obj_name} moved {direction} by {offset}")
        bddl_text = bddl_text[:start] + block + bddl_text[end:]
        # Update region_blocks for subsequent operations
        region_blocks[region_name] = (start, start + len(block))
    return bddl_text

def reorient_object(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]

    match = re.search(r":yaw_rotation\s*\(\s*\((.*?)\)\s*\)", block, re.DOTALL)
    if match:
        vals = list(map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+", match.group(1))))
        rotation_type = random.choice(["clockwise", "anticlockwise"])
        angle = round(random.uniform(5, 30), 2)
        delta = angle if rotation_type == "clockwise" else -angle
        vals = [v + delta for v in vals]
        new_yaw = " ".join(map(lambda x: f"{x:.2f}", vals))
        block = block[:match.start(1)] + new_yaw + block[match.end(1):]
        print(f"[REORIENT] {obj_name} rotated {rotation_type} by {angle}°")
        bddl_text = bddl_text[:start] + block + bddl_text[end:]
        region_blocks[region_name] = (start, start + len(block))
    return bddl_text

def change_color(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    new_color = random.choice(colors)
    insert_idx = block.rfind(")")
    block = block[:insert_idx] + f"\n      (:color {new_color})" + block[insert_idx:]
    bddl_text = bddl_text[:start] + block + bddl_text[end:]
    print(f"[COLOR] {obj_name} color changed to {new_color}")
    region_blocks[region_name] = (start, start + len(block))
    return bddl_text

def change_texture(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]
    textures = ["matte", "metallic", "wood", "plastic"]
    new_texture = random.choice(textures)
    insert_idx = block.rfind(")")
    block = block[:insert_idx] + f"\n      (:texture {new_texture})" + block[insert_idx:]
    bddl_text = bddl_text[:start] + block + bddl_text[end:]
    print(f"[TEXTURE] {obj_name} texture changed to {new_texture}")
    region_blocks[region_name] = (start, start + len(block))
    return bddl_text

def replace_object(bddl_text, obj_name):
    kitchen_objs = ["mug", "plate", "can", "bottle", "jar", "bowl"]
    new_obj = random.choice(kitchen_objs) + f"_{random.randint(1,999)}"
    bddl_text = re.sub(rf"\b{obj_name}\b", new_obj, bddl_text)
    print(f"[REPLACE] {obj_name} replaced with {new_obj}")
    return bddl_text

def add_distractor(bddl_text):
    kitchen_objs = ["mug", "plate", "can", "bottle", "jar", "bowl"]
    new_obj = random.choice(kitchen_objs) + f"_{random.randint(100,999)}"
    target = "kitchen_table"
    region_name = f"{new_obj}_init_region"
    x1, y1, x2, y2 = [round(random.uniform(-0.2, 0.2),3) for _ in range(4)]
    region_def = f"""
      ({region_name}
          (:target {target})
          (:ranges (
              ({x1} {y1} {x2} {y2})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )"""
    # Add to regions
    bddl_text = re.sub(r"(\(:regions[\s\S]*?)\)\s*\)", rf"\1{region_def}\n    )\n  )", bddl_text)
    # Add to objects
    bddl_text = re.sub(r"(\(:objects[\s\S]*?)\)", rf"\1\n    {new_obj} - {new_obj.split('_')[0]}\n  )", bddl_text)
    # Add to init
    bddl_text = re.sub(r"(\(:init[\s\S]*?)\)", rf"\1\n    (On {new_obj} {target}_{region_name})\n  )", bddl_text)
    print(f"[DISTRACTOR] Added new object {new_obj} on {target}")
    return bddl_text

# --------------------------
# Apply perturbations
# --------------------------

def apply_perturbations_kitchen(bddl_text, perturbations):
    region_blocks = find_region_blocks(bddl_text)
    obj_region_map = parse_object_region_map(bddl_text, region_blocks)

    print(f"[DEBUG] Object-Region mapping: {obj_region_map}")
    print(f"[DEBUG] Available regions: {list(region_blocks.keys())}")

    for key, obj_list in perturbations.items():
        for obj_name in obj_list:
            if key == "move":
                bddl_text = move_object(bddl_text, obj_name, obj_region_map, region_blocks)
            elif key == "reorient":
                bddl_text = reorient_object(bddl_text, obj_name, obj_region_map, region_blocks)
            elif key == "color":
                bddl_text = change_color(bddl_text, obj_name, obj_region_map, region_blocks)
            elif key == "texture":
                bddl_text = change_texture(bddl_text, obj_name, obj_region_map, region_blocks)
            elif key == "replace":
                bddl_text = replace_object(bddl_text, obj_name)

    if "distractor" in perturbations:
        for _ in perturbations["distractor"]:
            bddl_text = add_distractor(bddl_text)

    return bddl_text

# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    input_file = "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl"
    bddl_text = read_bddl(input_file)

    perturbations = {
        "move": ["akita_black_bowl_1", "wine_bottle_1"],
        "reorient": ["wine_bottle_1"],
        "color": ["white_cabinet_1"],
        "texture": ["wine_bottle_1"],
        "replace": ["wine_bottle_1"],
        "distractor": [1]  # just indicates add 1 distractor
    }

    perturbed_bddl = apply_perturbations_kitchen(bddl_text, perturbations)
    save_bddl(perturbed_bddl, base_name="LIBERO_Kitchen_Tabletop_Manipulation_perturbed")
