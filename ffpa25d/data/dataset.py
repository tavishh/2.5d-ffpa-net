"""Patient-slice dataset and dataloader factory.

2.5D modification: __getitem__ returns a 3-channel image tensor formed by
stacking the previous, current, and next slice from the SAME patient volume.
Edge slices are padded by repeating the boundary slice (replicate padding).

Key invariant: slices within each patient are sorted by filename before the
flat sample list is built. A per-patient index map (self._patient_slice_idx)
records the position of each sample within its patient so that neighbor
lookups never cross patient boundaries.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PatientSliceDataset(Dataset):
    """Dataset for patient-slice structured segmentation data.

    Args:
        three_slice (bool): If True, return a multi-channel image formed by
            stacking neighbouring slices from the same patient. If False
            (default), return the original single-channel image.
        num_slices (int): Only used when three_slice=True. Number of slices
            to stack (must be odd): 3 = [prev, curr, next], 5 = [prev2, prev1,
            curr, next1, next2], etc. Defaults to 3 for backward compatibility.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=256,
        num_classes=1,
        selected_classes=None,
        remap_classes=True,
        subset_ratio=1.0,
        verbose=False,
        three_slice=False,
        num_slices=3,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.selected_classes = selected_classes
        self.remap_classes = remap_classes
        self.subset_ratio = subset_ratio
        self.verbose = verbose
        self.is_binary = num_classes == 1
        self.three_slice = three_slice

        if three_slice:
            if num_slices < 1 or num_slices % 2 == 0:
                raise ValueError(f"num_slices must be odd and >= 1, got {num_slices}")
        self.num_slices = num_slices if three_slice else 1
        self._half_window = self.num_slices // 2  # e.g. 3->1, 5->2, 7->3

        print(f"\nInitializing {split} dataset from {root_dir}")
        if three_slice:
            print(f"  Mode: {self.num_slices}-slice stacking (2.5D)")

        if not self.is_binary and selected_classes and remap_classes:
            # start=1 so foreground classes remap to 1..N and background
            # pixels (value 0, absent from selected_classes) stay unambiguously
            # at index 0 via the torch.zeros_like initialisation below.
            self.class_mapping = {
                class_id: idx for idx, class_id in enumerate(selected_classes, start=1)
            }
            self.inverse_mapping = {
                idx: class_id for class_id, idx in self.class_mapping.items()
            }
            # +1 for background at index 0
            self.num_classes = len(selected_classes) + 1
            if self.verbose:
                print(f"  Class mapping: {self.class_mapping}")
        else:
            self.class_mapping = None
            self.inverse_mapping = None
            self.num_classes = num_classes

        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"

        # ── Build flat sample list, strictly sorted by patient then filename ──
        # Sorting by filename gives correct anatomical slice order as long as
        # filenames are zero-padded numerics (e.g. 0001.png, 0002.png).
        self.samples = []
        for patient_dir in sorted(self.image_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            for slice_path in sorted(patient_dir.glob("*.png")):
                slice_name = slice_path.name
                self.samples.append(
                    {
                        "patient_id": patient_id,
                        "slice_name": slice_name,
                        "image_path": slice_path,
                        "mask_path": self.mask_dir / patient_id / slice_name,
                    }
                )

        # ── Build per-patient position index for safe neighbor lookup ────────
        # _patient_slice_idx[i] = position of sample i within its patient.
        # _patient_slice_count[patient_id] = total slices for that patient.
        # These are computed ONCE here so __getitem__ can find neighbors in O(1)
        # without any scanning, and crucially without ever crossing boundaries.
        self._build_patient_index()

        original_count = len(self.samples)
        if subset_ratio < 1.0:
            random.seed(hash(split) % 10000)
            subset_size = int(len(self.samples) * subset_ratio)
            self.samples = random.sample(self.samples, subset_size)
            # Rebuild index after subsetting because flat positions changed.
            self._build_patient_index()
            print(
                f"  Using subset: {len(self.samples)}/{original_count} "
                f"samples ({subset_ratio * 100:.0f}%)"
            )

        print(f"  Loaded {len(self.samples)} samples")

    # ── Index helpers ─────────────────────────────────────────────────────────

    def _build_patient_index(self):
        """Record each sample's position within its patient volume.

        After this call:
          self._patient_slice_idx[i]   -> int, 0-based position in patient
          self._patient_boundaries[i]  -> (first_flat_idx, last_flat_idx)
                                          both inclusive, for patient of sample i
        """
        # First pass: group flat indices by patient (preserving sorted order).
        from collections import defaultdict
        patient_to_indices = defaultdict(list)
        for flat_idx, sample in enumerate(self.samples):
            patient_to_indices[sample["patient_id"]].append(flat_idx)

        self._patient_slice_idx = [0] * len(self.samples)
        self._patient_boundaries = [(0, 0)] * len(self.samples)

        for flat_indices in patient_to_indices.values():
            first = flat_indices[0]
            last = flat_indices[-1]
            for pos, flat_idx in enumerate(flat_indices):
                self._patient_slice_idx[flat_idx] = pos
                self._patient_boundaries[flat_idx] = (first, last)

    def _get_neighbor_idx(self, flat_idx, offset):
        """Return flat index of neighbor at +/- offset, clamped within patient.

        Implements replicate (boundary) padding: asking for a slice before
        the first slice of a patient returns the first slice; asking for a
        slice after the last returns the last. Works for ANY offset magnitude
        (not just +/-1), so this needs no change to support wider windows.

        Args:
            flat_idx: flat index of the current sample in self.samples
            offset: signed integer offset (e.g. -2, -1, +1, +2 for a 5-slice
                window)

        Returns:
            flat index of the neighbor (or boundary slice if edge case)
        """
        first, last = self._patient_boundaries[flat_idx]
        neighbor = flat_idx + offset
        # Clamp to [first, last] - replicate padding at patient boundaries.
        neighbor = max(first, min(last, neighbor))
        return neighbor

    # ── Core I/O ──────────────────────────────────────────────────────────────

    def _load_image(self, path):
        """Load and resize a single grayscale slice. Returns (H, W) ndarray."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        return np.ascontiguousarray(img)

    def _load_mask(self, path):
        """Load and resize a mask. Returns (H, W) ndarray."""
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.ascontiguousarray(mask)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.three_slice:
            # ── 2.5D path: stack [prev...curr...next] along channel dim ──────
            # Generalized to any odd window size via _half_window. For
            # num_slices=3, offsets are [-1, 0, +1] (unchanged from before).
            # For num_slices=5, offsets are [-2, -1, 0, +1, +2]. Neighbors are
            # clamped to patient boundaries (replicate padding) regardless of
            # window size.
            imgs = []
            for offset in range(-self._half_window, self._half_window + 1):
                if offset == 0:
                    neighbor_idx = idx
                else:
                    neighbor_idx = self._get_neighbor_idx(idx, offset)
                imgs.append(self._load_image(self.samples[neighbor_idx]["image_path"]))

            # Stack to (num_slices, H, W) and normalise to [0, 1].
            image = np.stack(imgs, axis=0).astype(np.float32)
            image = torch.from_numpy(image) / 255.0
            # image shape: (num_slices, H, W)
        else:
            # ── 2D baseline path: single-channel, unchanged behaviour ─────────
            curr_img = self._load_image(sample["image_path"])
            image = torch.from_numpy(curr_img).float().unsqueeze(0) / 255.0
            # image shape: (1, H, W)

        # ── Mask (same regardless of window size - we only segment the centre slice) ──
        mask = self._load_mask(sample["mask_path"])

        if self.is_binary:
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            mask = (mask > 0.5).float()
        else:
            mask = torch.from_numpy(mask.astype(np.int64))
            if self.class_mapping is not None:
                new_mask = torch.zeros_like(mask)
                for original_id, remapped_id in self.class_mapping.items():
                    mask_locations = mask == original_id
                    if mask_locations.any():
                        new_mask[mask_locations] = remapped_id
                mask = new_mask

        return {
            "image": image,
            "mask": mask,
            "patient_id": sample["patient_id"],
            "slice_name": sample["slice_name"],
        }

    def get_class_info(self):
        if self.class_mapping:
            return {
                "num_classes": self.num_classes,
                "class_mapping": self.class_mapping,
                "inverse_mapping": self.inverse_mapping,
                "original_classes": self.selected_classes,
            }
        return {
            "num_classes": self.num_classes,
            "class_mapping": None,
            "original_classes": self.selected_classes,
        }


def create_dataloaders(config):
    """Create train, validation, and test dataloaders."""
    print("\nCreating dataloaders...")

    selected_classes = config["model"].get("selected_classes")
    remap_classes = config["model"].get("remap_classes", True)

    debug_mode = config.get("debug", {}).get("enabled", False)
    subset_ratio = config.get("debug", {}).get("subset_ratio", 1.0)
    verbose = config.get("debug", {}).get("verbose", False)

    # three_slice flag read from config; defaults to False (2D baseline).
    # num_slices only matters when three_slice=True; defaults to 3 so every
    # existing config (which doesn't set this key) is unaffected.
    three_slice = config.get("data", {}).get("three_slice", False)
    num_slices = config.get("data", {}).get("num_slices", 3)

    if selected_classes and remap_classes:
        # +1 for background at index 0 (foreground remapped to 1..N)
        num_classes = len(selected_classes) + 1
    else:
        num_classes = config["model"].get("num_classes", 1)

    config["model"]["actual_num_classes"] = num_classes
    print(f"  Number of classes: {num_classes}")
    if selected_classes:
        print(f"  Selected classes: {selected_classes}")
    print(f"  3-slice mode: {three_slice}" + (f"  (num_slices={num_slices})" if three_slice else ""))

    common_kwargs = {
        "root_dir": config["data"]["root_dir"],
        "image_size": config["data"]["image_size"],
        "num_classes": num_classes,
        "selected_classes": selected_classes,
        "remap_classes": remap_classes,
        "verbose": verbose,
        "three_slice": three_slice,
        "num_slices": num_slices,
    }

    train_dataset = PatientSliceDataset(
        split="train",
        subset_ratio=subset_ratio if debug_mode else 1.0,
        **common_kwargs,
    )
    val_dataset = PatientSliceDataset(
        split="val",
        subset_ratio=subset_ratio if debug_mode else 1.0,
        **common_kwargs,
    )
    test_dataset = PatientSliceDataset(split="test", subset_ratio=1.0, **common_kwargs)

    config["model"]["class_info"] = train_dataset.get_class_info()

    num_workers = config["data"].get("num_workers", 4)
    batch_size = config["data"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader