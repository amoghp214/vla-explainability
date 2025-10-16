# Quick Start Guide - Fixed `record.py`

## ✅ What Was Fixed

Your `record.py` script has been completely rewritten to match the **official OpenVLA LIBERO implementation**.

**Main issues fixed:**
1. ❌ Wrong action unnormalization (`bridge_orig` → ✅ task suite name)
2. ❌ Incorrect manual scaling (removed entirely)
3. ❌ Missing gripper processing (added normalize + invert)
4. ❌ Wrong image resolution (512→256)

---

## 🚀 Quick Test (Do This First!)

```bash
# 1. Record a demo (MUST specify task_suite_name now!)
python scripts/record.py \
    --model openvla \
    --bddl_file <path/to/your.bddl> \
    --out_file test_demo.hdf5 \
    --prompt "pick up the mug" \
    --task_suite_name libero_spatial \
    --cache_dir <path/to/cache>

# 2. Play it back
python scripts/playback.py \
    --demo_file test_demo.hdf5 \
    --bddl_file <path/to/your.bddl> \
    --out_video test_playback.mp4

# 3. Watch the video
# The robot should now move properly! 🎉
```

---

## 📋 Required Arguments (CHANGED!)

### New Required Argument:
- `--task_suite_name`: Which LIBERO task suite your task belongs to

### Choices:
- `libero_spatial` - Spatial reasoning tasks (default)
- `libero_object` - Object manipulation tasks
- `libero_goal` - Goal-oriented tasks
- `libero_10` - LIBERO-10 benchmark
- `libero_90` - LIBERO-90 benchmark

**⚠️ CRITICAL**: You MUST specify the correct task suite, or actions will be in the wrong scale!

---

## 🎯 Example Usage

### Example 1: LIBERO Spatial Task
```bash
python scripts/record.py \
    --model openvla \
    --bddl_file libero/libero/bddl_files/libero_spatial/put_the_black_bowl_on_top_of_the_cabinet.bddl \
    --out_file demos/spatial_demo.hdf5 \
    --prompt "put the black bowl on top of the cabinet" \
    --task_suite_name libero_spatial \
    --cache_dir ~/models/openvla
```

### Example 2: LIBERO Object Task
```bash
python scripts/record.py \
    --model openvla \
    --bddl_file libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup.bddl \
    --out_file demos/object_demo.hdf5 \
    --prompt "pick up the alphabet soup" \
    --task_suite_name libero_object \
    --cache_dir ~/models/openvla
```

### Example 3: LIBERO Goal Task
```bash
python scripts/record.py \
    --model openvla \
    --bddl_file libero/libero/bddl_files/libero_goal/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl \
    --out_file demos/goal_demo.hdf5 \
    --prompt "put both the alphabet soup and the cream cheese box in the basket" \
    --task_suite_name libero_goal \
    --cache_dir ~/models/openvla
```

---

## 🔍 How to Know Which Task Suite to Use

### Method 1: Check the BDDL file path
The path usually contains the suite name:
- `libero/libero/bddl_files/libero_spatial/...` → use `libero_spatial`
- `libero/libero/bddl_files/libero_object/...` → use `libero_object`
- `libero/libero/bddl_files/libero_goal/...` → use `libero_goal`

### Method 2: Check the LIBERO benchmark documentation
Refer to the LIBERO paper or documentation to see which suite your task belongs to.

### Method 3: When in doubt, try `libero_spatial` first
It's the default and most common suite.

---

## 🐛 Troubleshooting

### Robot still barely moves:
1. **Check task_suite_name**: Make sure it matches your actual task
2. **Check model**: Are you using base `openvla-7b` or a LIBERO-finetuned version?
3. **Check prompt**: Should be natural language, e.g., "pick up the mug"

### Error: `Action un-norm key not found`:
```
AssertionError: Action un-norm key libero_spatial not found in VLA `norm_stats`!
```

**Solution**: Your model doesn't have statistics for LIBERO tasks. You need:
- A LIBERO-finetuned OpenVLA model, OR
- Train/fine-tune the model on LIBERO first

### Wrong gripper behavior:
- Should be fixed now with proper `normalize_gripper_action` + `invert_gripper_action`
- If still wrong, check if your BDDL file has special gripper requirements

---

## 📊 Verify the Fix Worked

### Quick check - Action magnitudes:
```python
import h5py
import numpy as np

with h5py.File("test_demo.hdf5", "r") as f:
    actions = f["data/demo_0/actions"][()]
    
    print(f"Number of steps: {len(actions)}")
    print(f"Action shape: {actions.shape}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"Position range: [{actions[:, :3].min():.3f}, {actions[:, :3].max():.3f}]")
    print(f"Rotation range: [{actions[:, 3:6].min():.3f}, {actions[:, 3:6].max():.3f}]")
    print(f"Gripper range: [{actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}]")
```

**Expected output** (approximately):
```
Number of steps: 50-200
Action shape: (N, 7)
Action range: [-1.0, 1.0] or similar
Position range: [-0.5, 0.5] or similar
Rotation range: [-0.5, 0.5] or similar
Gripper range: [-1.0, 1.0] (should be binarized to -1 or 1)
```

**Bad output** (what you had before):
```
Action range: [-0.1, 0.1]  # Too small!
Position range: [-0.05, 0.05]  # Way too small!
```

---

## 📚 Documentation Files

1. **QUICK_START.md** (this file) - Start here
2. **FIXES_SUMMARY.md** - Detailed before/after comparison
3. **RECORDING_TROUBLESHOOTING.md** - Advanced debugging guide

---

## 🎉 You're Done!

Your `record.py` script now:
- ✅ Uses correct action normalization for LIBERO
- ✅ Properly processes gripper actions  
- ✅ Matches OpenVLA's official LIBERO implementation
- ✅ Saves actions that can be replayed correctly

**Test it and enjoy watching your robot actually move! 🤖**

