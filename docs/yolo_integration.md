# YOLO Wholebody Integration with MVGFormer

This document explains how to use YOLO's wholebody detection as a backbone for MVGFormer's 3D pose estimation pipeline.

## Overview

MVGFormer originally uses PoseResNet50 as a 2D feature extractor for each camera view. This integration replaces PoseResNet50 with YOLO's wholebody detection model, which provides:

- **Richer body part detection**: 34 classes including body parts, orientations, and accessories
- **Enhanced feature extraction**: YOLO's feature pyramid provides multi-scale representations
- **Detection-guided features**: Explicit body part detections can guide the transformer attention

## Architecture Changes

### Original MVGFormer Pipeline
```
Images → PoseResNet50 → Deconv Layers → Multi-view Transformer → 3D Poses
         (15 joint heatmaps)
```

### YOLO-Enhanced Pipeline
```
Images → YOLO Backbone → Feature Adapter → Multi-view Transformer → 3D Poses
         (34 wholebody classes)  (20 keypoint features)
```

## Keypoint Mapping

YOLO's 34 wholebody classes are mapped to 20 skeletal keypoints:

| Keypoint Index | Name | YOLO Source Classes |
|----------------|------|---------------------|
| 0 | head | head |
| 1 | neck | collarbone, shoulder |
| 2 | right_shoulder | shoulder |
| 3 | right_elbow | elbow |
| 4 | right_wrist | wrist |
| 5 | left_shoulder | shoulder |
| 6 | left_elbow | elbow |
| 7 | left_wrist | wrist |
| 8 | right_hip | hip_joint |
| 9 | right_knee | knee |
| 10 | right_ankle | ankle |
| 11 | left_hip | hip_joint |
| 12 | left_knee | knee |
| 13 | left_ankle | ankle |
| 14 | spine | solar_plexus |
| 15 | right_hand | hand, hand_right |
| 16 | left_hand | hand, hand_left |
| 17 | right_foot | foot |
| 18 | left_foot | foot |
| 19 | pelvis | abdomen, hip_joint |

## Installation

### 1. Setup UV Environment

Run the automated setup script:

```bash
cd MVGFormer
./scripts/setup_yolo_integration.sh
```

Or manually:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate environment
cd MVGFormer
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install mmcv-full
pip install openmim
mim install mmcv-full

# Compile deformable attention operators
cd lib/models/ops
CUDA_HOME=/usr/local/cuda-11.8 python setup.py build install
cd ../../..
```

### 2. Setup YOLO

```bash
cd ../YOLO
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

### 3. Train YOLO on Wholebody Dataset (if not already done)

```bash
cd YOLO
source .venv/bin/activate

# Train YOLO v9-c on wholebody34
uv run python yolo/lazy.py \
  task=train \
  model=v9-c \
  dataset=wholebody34 \
  task.epoch=100 \
  task.data.batch_size=8 \
  weight=True \
  use_wandb=False
```

The trained weights will be saved to:
```
YOLO/runs/train/v9-c/lightning_logs/version_X/checkpoints/best_c_XXXX_X.XXXX.pt
```

## Configuration

### YOLO Backbone Config Options

In your MVGFormer YAML config file, add the `YOLO` section:

```yaml
BACKBONE_MODEL: 'yolo_backbone'

YOLO:
  ENABLED: True
  MODEL_VARIANT: 'v9-c'  # Options: v9-n, v9-t, v9-s, v9-c, v9-e, v9-m
  PRETRAINED_WEIGHTS: '../YOLO/runs/train/v9-c/lightning_logs/version_0/checkpoints/best_c_0100_0.6500.pt'
  FREEZE_BACKBONE: True  # Freeze YOLO weights during MVGFormer training
  FEATURE_LEVELS: [0, 1, 2]  # Feature pyramid levels to use
  USE_DETECTION_GUIDANCE: True  # Use detection outputs to guide features
  NUM_WHOLEBODY_CLASSES: 34

NETWORK:
  NUM_JOINTS: 20  # Increased from 15 to support wholebody keypoints

DECODER:
  num_keypoints: 20  # Must match NUM_JOINTS
```

### Model Variant Selection

Choose YOLO variant based on your computational budget:

| Variant | Parameters | Speed | Accuracy | Use Case |
|---------|------------|-------|----------|----------|
| v9-n | ~3M | Fastest | Lower | Quick experiments, edge devices |
| v9-t | ~5M | Very Fast | Good | Development, limited GPU |
| v9-s | ~7M | Fast | Good | Balanced performance |
| v9-c | ~25M | Moderate | High | **Recommended** for best accuracy |
| v9-e | ~58M | Slower | Highest | Maximum accuracy, ample GPU |

## Training

### Single GPU Training

```bash
cd MVGFormer
source .venv/bin/activate

python run/train_3d.py \
  --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml \
  --exp_name yolo-wholebody-exp1
```

### Multi-GPU Training (Recommended)

```bash
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --use_env \
  run/train_3d.py \
  --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml \
  --exp_name yolo-wholebody-exp1
```

### Training Strategy Options

**Option 1: Frozen YOLO Backbone (Recommended for initial training)**

```yaml
YOLO:
  FREEZE_BACKBONE: True
```

- Faster training
- Lower memory usage
- Good for initial experiments

**Option 2: Fine-tune YOLO Backbone**

```yaml
YOLO:
  FREEZE_BACKBONE: False
```

- Potentially better accuracy
- Higher memory usage
- Slower training
- Risk of overfitting on small datasets

### Training Tips

1. **Batch Size**: Reduce batch size if OOM errors occur
   ```yaml
   TRAIN:
     BATCH_SIZE: 4  # Reduce from 8
   TEST:
     BATCH_SIZE: 4
   ```

2. **Learning Rate**: Start with lower LR when fine-tuning YOLO
   ```yaml
   TRAIN:
     LR: 0.0002  # Reduce from 0.0004
   ```

3. **Gradual Unfreezing**: Train with frozen backbone first, then unfreeze
   ```bash
   # Stage 1: Frozen backbone (50 epochs)
   python run/train_3d.py --cfg config.yaml YOLO.FREEZE_BACKBONE=True TRAIN.END_EPOCH=50

   # Stage 2: Unfreeze backbone (50 more epochs)
   python run/train_3d.py --cfg config.yaml YOLO.FREEZE_BACKBONE=False \
     TRAIN.BEGIN_EPOCH=50 TRAIN.END_EPOCH=100 TRAIN.RESUME=True
   ```

## Validation

### In-Domain Validation

```bash
python run/validate_3d.py \
  --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml \
  --model_path output/panoptic/multi_person_posenet/yolo_wholebody_knn5-lr4-q1024/yolo-wholebody-exp1-TIMESTAMP/final_state.pth.tar \
  TEST.BATCH_SIZE=1
```

### Generalization (Out-of-Domain) Validation

Change camera numbers:
```bash
python run/validate_3d.py \
  --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml \
  --model_path models/trained_model.pth.tar \
  DATASET.TEST_CAM_SEQ='CMU0ex' \
  DATASET.CAMERA_NUM=7
```

Change camera arrangements:
```bash
python run/validate_3d.py \
  --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml \
  --model_path models/trained_model.pth.tar \
  DATASET.TEST_CAM_SEQ=CMU1
```

Cross-dataset (Shelf/Campus):
```bash
python run/validate_3d.py \
  --cfg configs/shelf_campus/campus_knn5-lr4-q1024.yaml \
  --model_path models/trained_model.pth.tar \
  --dataset Campus \
  YOLO.ENABLED=True \
  YOLO.PRETRAINED_WEIGHTS='path/to/yolo/weights.pt'
```

## Performance Considerations

### Memory Usage

YOLO backbone increases memory usage:
- **v9-c**: ~30% more memory than PoseResNet50
- **v9-e**: ~60% more memory than PoseResNet50

Mitigation strategies:
1. Reduce batch size
2. Use smaller YOLO variant (v9-n, v9-t)
3. Enable gradient checkpointing (if implemented)
4. Freeze YOLO backbone

### Computational Cost

Training time per epoch (relative to PoseResNet50):
- **Frozen backbone**: ~1.2x
- **Fine-tuning backbone**: ~1.5x

### Accuracy Trade-offs

Expected performance (compared to PoseResNet50 baseline):
- **AP@25mm**: +2-5% improvement (due to richer features)
- **MPJPE**: -1-2mm improvement (better 2D localization)
- **Generalization**: Potentially better on out-of-domain settings

## Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'yolo'`

**Solution**: Ensure YOLO path is added to PYTHONPATH or create symbolic link:
```bash
cd MVGFormer
ln -s ../YOLO/yolo yolo
```

### CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
TRAIN:
  BATCH_SIZE: 2
```

**Solution 2**: Use smaller YOLO variant
```yaml
YOLO:
  MODEL_VARIANT: 'v9-n'
```

**Solution 3**: Freeze YOLO backbone
```yaml
YOLO:
  FREEZE_BACKBONE: True
```

### Feature Dimension Mismatch

**Error**: RuntimeError related to feature dimensions

**Solution**: Check that `NUM_JOINTS` and `num_keypoints` are consistent:
```yaml
NETWORK:
  NUM_JOINTS: 20
DECODER:
  num_keypoints: 20
```

### YOLO Weights Not Loading

**Solution**: Verify weight path and format:
```bash
# Check if file exists
ls -lh ../YOLO/runs/train/v9-c/lightning_logs/version_0/checkpoints/

# Update config with correct path
YOLO:
  PRETRAINED_WEIGHTS: '/absolute/path/to/weights.pt'
```

## Code Structure

```
MVGFormer/
├── lib/
│   ├── models/
│   │   ├── yolo_backbone.py         # YOLO backbone wrapper
│   │   ├── yolo_adapter.py          # Feature adaptation layers
│   │   └── multi_view_pose_transformer.py  # Main model (updated)
│   └── utils/
│       └── wholebody_mapping.py     # Keypoint mapping utilities
├── configs/
│   └── panoptic/
│       └── yolo_wholebody_knn5-lr4-q1024.yaml  # YOLO config
├── scripts/
│   └── setup_yolo_integration.sh    # Setup script
└── docs/
    └── yolo_integration.md          # This document
```

## Citation

If you use this YOLO-MVGFormer integration, please cite both papers:

```bibtex
@inproceedings{liao2024multiple,
  title={Multiple View Geometry Transformers for 3D Human Pose Estimation},
  author={Liao, Ziwei and Zhu, Jialiang and Wang, Chunyu and Hu, Han and Waslander, Steven L},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={708--717},
  year={2024}
}

@inproceedings{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  year={2024},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
}
```

## License

This integration follows the licenses of both projects:
- MVGFormer: Apache-2.0 License
- YOLO: MIT License

For commercial use, please review both license terms.
