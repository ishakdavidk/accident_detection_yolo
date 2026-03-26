import os
import sys
import shutil
import random
from wo_opticalFlow.load_data.load_DoTA import DoTABinaryDataset

frames_root = '../../dataset/DoTA/frames'
annotations_root = '../../dataset/DoTA/annotations'
split_file = '../../dataset/DoTA/Detection-of-Traffic-Anomaly-master/dataset/train_split.txt'
segment_len = 16
stride = 1

dataset = DoTABinaryDataset(
    frames_root,
    annotations_root,
    split_file,
    segment_len=segment_len,
    stride=stride,
    transform=None
)

frame_set = set()
for segment_frames, _ in dataset.samples:
    frame_set.update(segment_frames)

frame_list = list(frame_set)
print(f'Found {len(frame_list)} unique frame images.')
print("Sample frame_list entries:", frame_list[:5])  # Debug: see the structure

quant_folder = 'quant_images'
dataset_file = 'dataset.txt'
num_images_to_copy = 2000

os.makedirs(quant_folder, exist_ok=True)

# Randomly sample frames
if len(frame_list) > num_images_to_copy:
    sampled_frames = random.sample(frame_list, num_images_to_copy)
else:
    sampled_frames = frame_list
print(f"Preparing to copy {len(sampled_frames)} images...")

new_paths = []
for i, src_path in enumerate(sampled_frames):
    # Always treat src_path as relative to frames_root unless it's absolute
    src_path_norm = os.path.normpath(src_path)
    if os.path.isabs(src_path_norm):
        src_full_path = src_path_norm
    else:
        # Remove leading frames_root from src_path if present
        if src_path_norm.startswith(os.path.normpath(frames_root)):
            rel_path = os.path.relpath(src_path_norm, frames_root)
            src_full_path = os.path.join(frames_root, rel_path)
        else:
            src_full_path = os.path.join(frames_root, src_path_norm)

    if not os.path.exists(src_full_path):
        print(f"Warning: Skipping missing file {src_full_path}")
        continue

    dst_filename = f"{i:05d}_" + os.path.basename(src_path_norm)
    dst_full_path = os.path.join(quant_folder, dst_filename)
    shutil.copy(src_full_path, dst_full_path)

    # Always write relative paths with forward slashes for RKNN
    rel_out_path = os.path.relpath(dst_full_path).replace('\\', '/')
    new_paths.append(rel_out_path)

with open(dataset_file, 'w') as f:
    for path in new_paths:
        f.write(path + '\n')

print(f"Successfully copied {len(new_paths)} images to '{quant_folder}' and created '{dataset_file}'.")
