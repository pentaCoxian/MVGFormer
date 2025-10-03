#!/usr/bin/env python
"""
Test script for YOLO-MVGFormer integration.

This script verifies that all components are correctly installed and
can be imported without errors.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing YOLO-MVGFormer Integration")
    print("=" * 60)
    print()

    errors = []

    # Test basic dependencies
    print("1. Testing basic dependencies...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA version: {torch.version.cuda}")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        print(f"   ✗ PyTorch: {e}")

    try:
        import torchvision
        print(f"   ✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        errors.append(f"TorchVision: {e}")
        print(f"   ✗ TorchVision: {e}")

    try:
        import mmcv
        print(f"   ✓ mmcv {mmcv.__version__}")
    except ImportError as e:
        errors.append(f"mmcv: {e}")
        print(f"   ✗ mmcv (required for deformable attention): {e}")

    print()

    # Test MVGFormer modules
    print("2. Testing MVGFormer modules...")
    try:
        import lib.models.pose_resnet as pose_resnet
        print("   ✓ PoseResNet module")
    except ImportError as e:
        errors.append(f"PoseResNet: {e}")
        print(f"   ✗ PoseResNet: {e}")

    try:
        import lib.utils.wholebody_mapping as wholebody_mapping
        print("   ✓ Wholebody mapping module")
        print(f"     - {wholebody_mapping.NUM_SKELETAL_JOINTS} skeletal keypoints")
    except ImportError as e:
        errors.append(f"Wholebody mapping: {e}")
        print(f"   ✗ Wholebody mapping: {e}")

    try:
        import lib.models.yolo_adapter as yolo_adapter
        print("   ✓ YOLO adapter module")
    except ImportError as e:
        errors.append(f"YOLO adapter: {e}")
        print(f"   ✗ YOLO adapter: {e}")

    print()

    # Test YOLO integration
    print("3. Testing YOLO integration...")

    # Add YOLO to path
    yolo_path = Path(__file__).parent.parent.parent / 'YOLO'
    if yolo_path.exists():
        sys.path.insert(0, str(yolo_path))
        print(f"   ✓ YOLO directory found: {yolo_path}")

        try:
            from yolo.model.yolo import create_model, YOLO
            print("   ✓ YOLO model module")

            try:
                import lib.models.yolo_backbone as yolo_backbone
                print("   ✓ YOLO backbone wrapper")
            except ImportError as e:
                errors.append(f"YOLO backbone: {e}")
                print(f"   ✗ YOLO backbone: {e}")

        except ImportError as e:
            errors.append(f"YOLO model: {e}")
            print(f"   ✗ YOLO model: {e}")
            print(f"     Please ensure YOLO is installed: cd {yolo_path} && uv sync")
    else:
        errors.append(f"YOLO directory not found: {yolo_path}")
        print(f"   ✗ YOLO directory not found: {yolo_path}")

    print()

    # Test deformable attention ops
    print("4. Testing deformable attention operators...")
    try:
        from lib.models.ops.modules import ProjAttn
        print("   ✓ Deformable attention compiled and importable")
    except ImportError as e:
        errors.append(f"Deformable attention: {e}")
        print(f"   ✗ Deformable attention: {e}")
        print("     Please compile: cd lib/models/ops && CUDA_HOME=/usr/local/cuda-11.8 python setup.py build install")

    print()

    # Test configuration
    print("5. Testing configuration...")
    try:
        from lib.core.config import config
        print("   ✓ Base configuration loaded")
        if hasattr(config, 'YOLO'):
            print(f"   ✓ YOLO config section present")
            print(f"     - MODEL_VARIANT: {config.YOLO.MODEL_VARIANT}")
            print(f"     - NUM_WHOLEBODY_CLASSES: {config.YOLO.NUM_WHOLEBODY_CLASSES}")
        else:
            errors.append("YOLO config section missing")
            print("   ✗ YOLO config section missing")
    except Exception as e:
        errors.append(f"Configuration: {e}")
        print(f"   ✗ Configuration: {e}")

    print()

    # Test config file
    config_file = Path(__file__).parent.parent / 'configs' / 'panoptic' / 'yolo_wholebody_knn5-lr4-q1024.yaml'
    if config_file.exists():
        print(f"   ✓ YOLO config file exists: {config_file.name}")
    else:
        errors.append(f"YOLO config file not found: {config_file}")
        print(f"   ✗ YOLO config file not found: {config_file}")

    print()
    print("=" * 60)

    if errors:
        print(f"FAILED: {len(errors)} error(s) found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print()
        print("Please run: ./scripts/setup_yolo_integration.sh")
        return False
    else:
        print("SUCCESS: All integration components are working!")
        print()
        print("Next steps:")
        print("  1. Train YOLO on wholebody dataset (if not done):")
        print("     cd ../YOLO && uv run python yolo/lazy.py task=train model=v9-c dataset=wholebody34")
        print()
        print("  2. Train MVGFormer with YOLO backbone:")
        print("     python -m torch.distributed.launch --nproc_per_node=8 --use_env \\")
        print("       run/train_3d.py --cfg configs/panoptic/yolo_wholebody_knn5-lr4-q1024.yaml")
        return True


def test_simple_forward():
    """Test a simple forward pass through YOLO backbone."""
    print()
    print("=" * 60)
    print("Testing Simple Forward Pass")
    print("=" * 60)
    print()

    try:
        import torch
        from lib.core.config import config, update_config
        from omegaconf import OmegaConf

        # Update config for YOLO
        config.YOLO.ENABLED = True
        config.YOLO.MODEL_VARIANT = 'v9-c'
        config.YOLO.FREEZE_BACKBONE = True
        config.NETWORK.NUM_JOINTS = 20

        print("Creating dummy input...")
        batch_size = 2
        channels = 3
        height, width = 512, 960
        dummy_input = torch.randn(batch_size, channels, height, width)
        print(f"  Input shape: {dummy_input.shape}")

        print()
        print("Attempting to load YOLO backbone...")
        print("  Note: This will fail if YOLO weights are not available")
        print("  Set YOLO.PRETRAINED_WEIGHTS=False to test without weights")

        try:
            import lib.models.yolo_backbone as yolo_backbone

            # Try to create backbone (may fail without weights)
            config.YOLO.PRETRAINED_WEIGHTS = ''  # Don't load weights for test
            backbone = yolo_backbone.get_pose_net(config, is_train=True)
            print("  ✓ YOLO backbone created")

            print()
            print("Running forward pass...")
            with torch.no_grad():
                features = backbone(dummy_input, use_feat_level=[0, 1, 2])
                print(f"  ✓ Forward pass successful")
                print(f"  Number of feature levels: {len(features)}")
                for i, feat in enumerate(features):
                    print(f"    Level {i}: {feat.shape}")

            print()
            print("✓ Simple forward pass test PASSED")
            return True

        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            print()
            print("This is expected if:")
            print("  - YOLO is not fully installed")
            print("  - YOLO weights are not available")
            print("  - CUDA is not available")
            return False

    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        return False


if __name__ == '__main__':
    # Change to MVGFormer directory
    script_dir = Path(__file__).parent
    mvgformer_dir = script_dir.parent
    os.chdir(mvgformer_dir)

    print(f"Working directory: {os.getcwd()}")
    print()

    # Run import tests
    import_success = test_imports()

    # Optionally run forward pass test
    if import_success and '--forward-test' in sys.argv:
        test_simple_forward()

    sys.exit(0 if import_success else 1)
