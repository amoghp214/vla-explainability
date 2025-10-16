# Recording & Playback Troubleshooting Guide

## 🐛 Critical Bugs Fixed (Updated with OpenVLA Reference Implementation)

After reviewing the official OpenVLA LIBERO evaluation script, the following critical issues were found and fixed:

### Bug 1: Wrong Action Unnormalization Key ❌ → ✅
**The Problem:**
- Used `unnorm_key="bridge_orig"` (Bridge dataset statistics)
- OpenVLA has **different action statistics for each LIBERO task suite**
- Using wrong statistics resulted in completely wrong action scales

**The Fix:**
```python
# BEFORE (WRONG):
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# AFTER (CORRECT):
action = vla.predict_action(**inputs, unnorm_key=task_suite_name, do_sample=False)
# where task_suite_name is "libero_spatial", "libero_object", "libero_goal", etc.
```

### Bug 2: Incorrect Manual Scaling ❌ → ✅
**The Problem:**
- Applied manual 10x scaling to actions
- OpenVLA doesn't need this when using correct unnorm_key!
- This was a workaround for using the wrong unnorm_key

**The Fix:**
```python
# BEFORE (WRONG):
scaled_action = action.copy()
scaled_action[:3] *= LINEAR_SCALE_FACTOR  # 10x
scaled_action[3:6] *= ANGULAR_SCALE_FACTOR  # 10x
env.step(scaled_action)

# AFTER (CORRECT):
# No scaling needed! Just process gripper and execute
action = normalize_gripper_action(action, binarize=True)
action = invert_gripper_action(action)
env.step(action.tolist())
```

### Bug 3: Missing Gripper Action Processing ❌ → ✅
**The Problem:**
- Gripper action wasn't normalized or inverted
- LIBERO expects gripper in range [-1, +1], not [0, 1]
- OpenVLA flips gripper sign during training (0=close, 1=open) but LIBERO expects (-1=open, +1=close)

**The Fix:**
```python
# Added two functions:
action = normalize_gripper_action(action, binarize=True)  # [0,1] -> [-1,+1]
action = invert_gripper_action(action)  # Flip sign
```

### Bug 4: Wrong Image Resolution ❌ → ✅
**The Problem:**
- Used 512x512 resolution
- OpenVLA uses 256x256 for LIBERO

**The Fix:**
```python
# BEFORE (WRONG):
env_args = {
    "bddl_file_name": bddl_file,
    "camera_heights": 512,
    "camera_widths": 512,
}

# AFTER (CORRECT):
env_args = {
    "bddl_file_name": bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
}
```

---

## ⚙️ How the Fixed Script Works

### 1. Correct Action Unnormalization
The script now uses the **task suite name** as the unnorm_key:
```python
# Must specify which LIBERO task suite you're using:
python record.py \
    --task_suite_name libero_spatial \  # or libero_object, libero_goal, etc.
    ...
```

**Available task suites:**
- `libero_spatial` - Spatial reasoning tasks
- `libero_object` - Object manipulation tasks  
- `libero_goal` - Goal-oriented tasks
- `libero_10` - LIBERO-10 benchmark
- `libero_90` - LIBERO-90 benchmark

**Important:** The unnorm_key MUST match the actual task suite, otherwise actions will be in the wrong range!

### 2. No Manual Scaling Needed
The script no longer applies manual scaling factors. OpenVLA's unnormalization handles this automatically when you use the correct task suite name.

### 3. Proper Gripper Processing
The script now correctly processes gripper actions:
```python
# Step 1: Normalize from [0,1] to [-1,+1]
action = normalize_gripper_action(action, binarize=True)

# Step 2: Invert sign (OpenVLA convention -> LIBERO convention)
action = invert_gripper_action(action)
```

### 4. Correct Image Resolution
Now uses 256x256 resolution, matching OpenVLA's LIBERO setup.

---

## 🔍 Debugging Steps

### Step 1: Check Action Values
Add this debug code to see what the model is outputting:
```python
print(f"Raw action from model: {action}")
print(f"Scaled action: {scaled_action}")
print(f"Action shape: {action.shape}")
print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
```

### Step 2: Verify Action Execution
In `playback.py`, add:
```python
print(f"Action min/max: [{action.min():.3f}, {action.max():.3f}]")
```

### Step 3: Compare with Human Demonstrations
Look at a human-collected LIBERO demonstration to see the typical action ranges:
```python
import h5py
with h5py.File("path/to/human_demo.hdf5", "r") as f:
    human_actions = f["data/demo_0/actions"][()]
    print(f"Human action range: [{human_actions.min():.3f}, {human_actions.max():.3f}]")
    print(f"Human action shape: {human_actions.shape}")
```

### Step 4: Test with a LIBERO-finetuned Model
If available, use a model specifically finetuned on LIBERO:
```python
# Check Hugging Face for OpenVLA LIBERO-finetuned models
# Example: "openvla/openvla-7b-finetuned-libero-spatial"
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial",  # if it exists
    trust_remote_code=True
)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)
```

---

## 📊 Expected Behavior After Fix

After fixing the action saving bug, you should see:
- ✅ Robot actually moves during playback (not just minimal jitter)
- ✅ Playback trajectory roughly matches what was recorded
- ✅ Action magnitudes in the HDF5 file match what was executed

If the robot still barely moves, the issue is likely with the **scaling factors** or **unnorm_key**.

---

## 🔗 Useful Resources

1. **OpenVLA GitHub**: https://github.com/openvla/openvla
2. **OpenVLA LIBERO Experiments**: https://github.com/openvla/openvla/tree/main/experiments/robot/libero
3. **LIBERO Benchmark**: https://github.com/Lifelong-Robot-Learning/LIBERO

---

## 💡 Quick Test

After the fixes, try this test:
```bash
# Record a short demo with correct task suite specified
python scripts/record.py \
    --model openvla \
    --bddl_file <your_bddl_file> \
    --out_file test_demo.hdf5 \
    --prompt "pick up the mug" \
    --task_suite_name libero_spatial \  # IMPORTANT: Must match your task!
    --cache_dir <cache_dir>

# Play it back
python scripts/playback.py \
    --demo_file test_demo.hdf5 \
    --bddl_file <your_bddl_file> \
    --out_video test_playback.mp4

# Check the video - the robot should move properly now!
```

### Common Issues After Fix

**If the robot still doesn't move well:**

1. **Wrong task_suite_name**: Make sure you're using the correct task suite for your BDDL file
   - Check which suite your task belongs to
   - The unnorm_key must match the actual task suite

2. **Model not trained on LIBERO**: If using the base `openvla/openvla-7b` model, it may not perform well on LIBERO without fine-tuning
   - Try using a LIBERO-finetuned checkpoint if available
   - Check Hugging Face for models like `openvla-7b-finetuned-libero-spatial`

3. **Wrong prompt format**: The prompt should match the task description
   - Use natural language descriptions like "pick up the mug"
   - Not: "move the phone to the left of the table" (too abstract)

