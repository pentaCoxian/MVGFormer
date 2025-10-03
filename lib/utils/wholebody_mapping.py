"""
Wholebody Detection to Skeletal Keypoint Mapping

Maps YOLO's 34 wholebody classes to skeletal keypoints compatible with
MVGFormer's pose estimation pipeline.

YOLO wholebody34 classes:
['body', 'adult', 'child', 'male', 'female', 'body_with_wheelchair',
 'body_with_crutches', 'head', 'front', 'right-front', 'right-side',
 'right-back', 'back', 'left-back', 'left-side', 'left-front', 'face',
 'eye', 'nose', 'mouth', 'ear', 'collarbone', 'shoulder', 'solar_plexus',
 'elbow', 'wrist', 'hand', 'hand_left', 'hand_right', 'abdomen',
 'hip_joint', 'knee', 'ankle', 'foot']
"""

import numpy as np
import torch

# YOLO wholebody34 class names and indices
WHOLEBODY_CLASSES = [
    'body', 'adult', 'child', 'male', 'female', 'body_with_wheelchair',
    'body_with_crutches', 'head', 'front', 'right-front', 'right-side',
    'right-back', 'back', 'left-back', 'left-side', 'left-front', 'face',
    'eye', 'nose', 'mouth', 'ear', 'collarbone', 'shoulder', 'solar_plexus',
    'elbow', 'wrist', 'hand', 'hand_left', 'hand_right', 'abdomen',
    'hip_joint', 'knee', 'ankle', 'foot'
]

# Map YOLO class indices to semantic body part types
WHOLEBODY_CLASS_TO_IDX = {name: idx for idx, name in enumerate(WHOLEBODY_CLASSES)}

# Skeletal keypoint definition (compatible with MVGFormer)
# Extended from 15 joints to 20 joints to accommodate wholebody parts
SKELETAL_KEYPOINTS = [
    'head',           # 0
    'neck',           # 1 (derived from collarbone/shoulder)
    'right_shoulder', # 2
    'right_elbow',    # 3
    'right_wrist',    # 4
    'left_shoulder',  # 5
    'left_elbow',     # 6
    'left_wrist',     # 7
    'right_hip',      # 8
    'right_knee',     # 9
    'right_ankle',    # 10
    'left_hip',       # 11
    'left_knee',      # 12
    'left_ankle',     # 13
    'spine',          # 14 (solar_plexus)
    'right_hand',     # 15
    'left_hand',      # 16
    'right_foot',     # 17
    'left_foot',      # 18
    'pelvis',         # 19 (abdomen/hip_joint center)
]

NUM_SKELETAL_JOINTS = len(SKELETAL_KEYPOINTS)

# Mapping from skeletal keypoints to YOLO wholebody classes
# Format: {keypoint_idx: [list of yolo class indices that can represent this keypoint]}
KEYPOINT_TO_WHOLEBODY_CLASSES = {
    0: [WHOLEBODY_CLASS_TO_IDX['head']],  # head
    1: [WHOLEBODY_CLASS_TO_IDX['collarbone'], WHOLEBODY_CLASS_TO_IDX['shoulder']],  # neck (derived)
    2: [WHOLEBODY_CLASS_TO_IDX['shoulder']],  # right_shoulder (need laterality info)
    3: [WHOLEBODY_CLASS_TO_IDX['elbow']],     # right_elbow
    4: [WHOLEBODY_CLASS_TO_IDX['wrist']],     # right_wrist
    5: [WHOLEBODY_CLASS_TO_IDX['shoulder']],  # left_shoulder
    6: [WHOLEBODY_CLASS_TO_IDX['elbow']],     # left_elbow
    7: [WHOLEBODY_CLASS_TO_IDX['wrist']],     # left_wrist
    8: [WHOLEBODY_CLASS_TO_IDX['hip_joint']], # right_hip
    9: [WHOLEBODY_CLASS_TO_IDX['knee']],      # right_knee
    10: [WHOLEBODY_CLASS_TO_IDX['ankle']],    # right_ankle
    11: [WHOLEBODY_CLASS_TO_IDX['hip_joint']], # left_hip
    12: [WHOLEBODY_CLASS_TO_IDX['knee']],     # left_knee
    13: [WHOLEBODY_CLASS_TO_IDX['ankle']],    # left_ankle
    14: [WHOLEBODY_CLASS_TO_IDX['solar_plexus']], # spine
    15: [WHOLEBODY_CLASS_TO_IDX['hand'], WHOLEBODY_CLASS_TO_IDX['hand_right']], # right_hand
    16: [WHOLEBODY_CLASS_TO_IDX['hand'], WHOLEBODY_CLASS_TO_IDX['hand_left']],  # left_hand
    17: [WHOLEBODY_CLASS_TO_IDX['foot']],     # right_foot
    18: [WHOLEBODY_CLASS_TO_IDX['foot']],     # left_foot
    19: [WHOLEBODY_CLASS_TO_IDX['abdomen'], WHOLEBODY_CLASS_TO_IDX['hip_joint']], # pelvis
}

# CMU Panoptic 15-joint skeleton for backward compatibility
CMU_PANOPTIC_JOINTS = 15  # Original MVGFormer joint count

# Mapping from 20 wholebody joints to 15 CMU Panoptic joints (if needed for compatibility)
WHOLEBODY_TO_CMU_MAPPING = [
    0,   # head -> head
    1,   # neck -> neck
    2,   # right_shoulder -> right_shoulder
    3,   # right_elbow -> right_elbow
    4,   # right_wrist -> right_wrist
    5,   # left_shoulder -> left_shoulder
    6,   # left_elbow -> left_elbow
    7,   # left_wrist -> left_wrist
    8,   # right_hip -> right_hip
    9,   # right_knee -> right_knee
    10,  # right_ankle -> right_ankle
    11,  # left_hip -> left_hip
    12,  # left_knee -> left_knee
    13,  # left_ankle -> left_ankle
    14,  # spine -> spine
    # Extended joints (15-19) are excluded in CMU compatibility mode
]


def yolo_detections_to_keypoint_heatmap(detections, image_size, heatmap_size, sigma=3):
    """
    Convert YOLO wholebody detections to keypoint heatmaps.

    Args:
        detections: YOLO detection output
            Format: [N, 7] where each row is [batch_idx, class_id, score, x1, y1, x2, y2]
            or dict with 'boxes', 'scores', 'labels' keys
        image_size: tuple (H, W) of original image size
        heatmap_size: tuple (H, W) of output heatmap size
        sigma: Gaussian sigma for heatmap generation

    Returns:
        heatmap: torch.Tensor of shape [num_joints, heatmap_h, heatmap_w]
    """
    heatmap_h, heatmap_w = heatmap_size
    image_h, image_w = image_size

    # Initialize heatmap for all skeletal joints
    heatmap = torch.zeros(NUM_SKELETAL_JOINTS, heatmap_h, heatmap_w)

    # Parse detections
    if isinstance(detections, dict):
        boxes = detections['boxes']  # [N, 4] in x1,y1,x2,y2 format
        labels = detections['labels']  # [N]
        scores = detections['scores']  # [N]
    else:
        # Assume tensor format [N, 7]: [batch_idx, class_id, score, x1, y1, x2, y2]
        labels = detections[:, 1].long()
        scores = detections[:, 2]
        boxes = detections[:, 3:7]

    # For each skeletal keypoint, find corresponding YOLO detections
    for joint_idx in range(NUM_SKELETAL_JOINTS):
        yolo_class_indices = KEYPOINT_TO_WHOLEBODY_CLASSES[joint_idx]

        # Find all detections matching this keypoint
        for yolo_class_idx in yolo_class_indices:
            mask = labels == yolo_class_idx
            if mask.sum() == 0:
                continue

            matching_boxes = boxes[mask]
            matching_scores = scores[mask]

            # Use the highest-confidence detection for this keypoint
            best_idx = matching_scores.argmax()
            box = matching_boxes[best_idx]

            # Convert box center to heatmap coordinates
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # Scale to heatmap size
            heatmap_x = center_x * heatmap_w / image_w
            heatmap_y = center_y * heatmap_h / image_h

            # Generate Gaussian heatmap
            heatmap[joint_idx] = add_gaussian_to_heatmap(
                heatmap[joint_idx], heatmap_x, heatmap_y, sigma
            )

    return heatmap


def add_gaussian_to_heatmap(heatmap, center_x, center_y, sigma):
    """
    Add a 2D Gaussian centered at (center_x, center_y) to the heatmap.

    Args:
        heatmap: torch.Tensor of shape [H, W]
        center_x, center_y: float coordinates of Gaussian center
        sigma: float, standard deviation of Gaussian

    Returns:
        heatmap: Updated heatmap with Gaussian added
    """
    h, w = heatmap.shape

    # Create coordinate grids
    x = torch.arange(0, w, dtype=torch.float32)
    y = torch.arange(0, h, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Compute 2D Gaussian
    gaussian = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))

    # Take maximum to preserve existing activations
    heatmap = torch.maximum(heatmap, gaussian)

    return heatmap


def convert_wholebody_joints_to_cmu(joints_3d):
    """
    Convert 20-joint wholebody representation to 15-joint CMU Panoptic format.

    Args:
        joints_3d: torch.Tensor or np.ndarray of shape [..., 20, 3]

    Returns:
        joints_3d_cmu: Tensor/array of shape [..., 15, 3]
    """
    if isinstance(joints_3d, torch.Tensor):
        return joints_3d[..., WHOLEBODY_TO_CMU_MAPPING, :]
    else:
        return joints_3d[..., WHOLEBODY_TO_CMU_MAPPING, :]


def get_wholebody_skeleton_definition():
    """
    Get skeleton connectivity for visualization.

    Returns:
        bones: List of (joint_idx1, joint_idx2) tuples defining bone connections
    """
    bones = [
        # Head to torso
        (0, 1),   # head - neck
        (1, 14),  # neck - spine
        (14, 19), # spine - pelvis

        # Right arm
        (1, 2),   # neck - right_shoulder
        (2, 3),   # right_shoulder - right_elbow
        (3, 4),   # right_elbow - right_wrist
        (4, 15),  # right_wrist - right_hand

        # Left arm
        (1, 5),   # neck - left_shoulder
        (5, 6),   # left_shoulder - left_elbow
        (6, 7),   # left_elbow - left_wrist
        (7, 16),  # left_wrist - left_hand

        # Right leg
        (19, 8),  # pelvis - right_hip
        (8, 9),   # right_hip - right_knee
        (9, 10),  # right_knee - right_ankle
        (10, 17), # right_ankle - right_foot

        # Left leg
        (19, 11), # pelvis - left_hip
        (11, 12), # left_hip - left_knee
        (12, 13), # left_knee - left_ankle
        (13, 18), # left_ankle - left_foot
    ]
    return bones


def infer_laterality_from_spatial_context(detections, person_bbox):
    """
    Infer left/right laterality for symmetric body parts based on spatial position.

    Args:
        detections: YOLO detections for body parts
        person_bbox: Bounding box [x1, y1, x2, y2] of the person

    Returns:
        laterality_map: Dict mapping detection indices to 'left' or 'right'
    """
    laterality_map = {}

    # Person center
    person_center_x = (person_bbox[0] + person_bbox[2]) / 2.0

    if isinstance(detections, dict):
        boxes = detections['boxes']
        labels = detections['labels']
    else:
        labels = detections[:, 1].long()
        boxes = detections[:, 3:7]

    # Symmetric body parts that need laterality inference
    symmetric_classes = ['shoulder', 'elbow', 'wrist', 'hand', 'hip_joint', 'knee', 'ankle', 'foot']

    for det_idx in range(len(labels)):
        class_name = WHOLEBODY_CLASSES[labels[det_idx]]

        if class_name in symmetric_classes:
            box = boxes[det_idx]
            part_center_x = (box[0] + box[2]) / 2.0

            # Determine laterality based on position relative to person center
            # (assuming person is facing camera)
            if part_center_x < person_center_x:
                laterality_map[det_idx] = 'right'  # From person's perspective
            else:
                laterality_map[det_idx] = 'left'

    return laterality_map
