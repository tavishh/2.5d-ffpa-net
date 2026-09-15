"""Patient-slice dataset and dataloader factory.

2.5D modification: __getitem__ returns a multi-channel image tensor formed by
stacking neighbouring slices from the SAME patient volume. Edge slices are
padded by repeating the boundary slice (replicate padding).

Positional encoding (SC-UNet style, Henson et al. 2024): optionally returns a
"position" field -- an integer bucket (0-99) representing this slice's
percentage position along its patient's volume. See ffpa_net_25d.py for how
this is consumed (a separate parallel pathway, not mixed into the image).

Slice-consistency support (added 2026-09): optionally returns a SECOND
stacked window, "image_next", centered one slice further along the same
patient's volume (clamped at patient boundaries, same replicate-padding rule
as everything else). This lets the training loop run the model on both the
current and the immediately-following slice, then penalize abrupt prediction
changes between them -- a gentle regularizer targeting the discontinuous
"flip to 0 then back" failure pattern found in the failure-zone diagnosis,
without needing the model itself to change at all (this is purely a data +
training-loop addition).

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
        use_position_encoding (bool): If True, each sample's dict includes a
            "position" field -- an integer in [0, 99] giving this slice's
            percentage position along its patient's volume (SC-UNet style).
            Defaults to False. Independent of three_slice/num_slices.
        use_slice_consistency (bool): If True, each sample's dict ALSO
            includes an "image_next" field -- the same kind of stacked window
            as "image", but centered one slice further along this patient's
            volume. Defaults to False. Requires three_slice=True (a
            single-slice model has no "window" to shift). Independent of
            use_position_encoding.
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
        use_position_encoding=False,
        use_slice_consistency=False,
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
        self.use_position_encoding = use_position_encoding

        if use_slice_consistency and not three_slice:
            raise ValueError(
                "use_slice_consistency=True requires three_slice=True -- a "
                "single-slice (2D) model has no stacked window to shift."
            )
        self.use_slice_consistency = use_slice_consistency

        if three_slice:
            if num_slices < 1 or num_slices % 2 == 0:
                raise ValueError(f"num_slices must be odd and >= 1, got {num_slices}")
        self.num_slices = num_slices if three_slice else 1
        self._half_window = self.num_slices // 2  # e.g. 3->1, 5->2, 7->3

        print(f"\nInitializing {split} dataset from {root_dir}")
        if three_slice:
            print(f"  Mode: {self.num_slices}-slice stacking (2.5D)")
        if use_position_encoding:
            print(f"  Position encoding: ON (SC-UNet style, 100-bucket % along limb)")
        if use_slice_consistency:
            print(f"  Slice-consistency pairing: ON (also returns the next slice's window)")

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
        # The SAME index is reused for the positional-encoding bucket AND the
        # slice-consistency "next slice" lookup -- no extra bookkeeping needed.
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

    def _get_position_bucket(self, flat_idx, num_buckets=100):
        """Return this slice's percentage-along-the-limb bucket, in [0, num_buckets-1].

        SC-UNet style (Henson et al. 2024): a percentage-along-the-limb value
        converted to one of num_buckets discrete positions. Bucket 0 = first
        slice of this patient, bucket num_buckets-1 = last slice.
        """
        first, last = self._patient_boundaries[flat_idx]
        total_minus_one = max(last - first, 1)  # avoid div-by-zero for a
        # single-slice patient (degenerate case, shouldn't occur in practice)
        normalized = (flat_idx - first) / total_minus_one  # in [0.0, 1.0]
        bucket = int(normalized * num_buckets)
        return min(bucket, num_buckets - 1)  # clamp: normalized==1.0 must map
        # to the last valid bucket (num_buckets-1), not num_buckets (out of range)

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

    def _build_slice_window(self, center_idx):
        """Build the (num_slices, H, W) stacked image tensor centered at
        center_idx. Extracted as its own method (rather than inlined in
        __getitem__) so it can be called for the CURRENT sample's window and,
        when slice-consistency is enabled, for the NEXT slice's window too --
        identical logic, just a different centre.

        Neighbors are clamped to patient boundaries (replicate padding),
        exactly as for the primary window -- center_idx is assumed to already
        be a valid flat index within some patient (the caller is responsible
        for computing it via _get_neighbor_idx if it is not the sample's own
        index).
        """
        imgs = []
        for offset in range(-self._half_window, self._half_window + 1):
            if offset == 0:
                neighbor_idx = center_idx
            else:
                neighbor_idx = self._get_neighbor_idx(center_idx, offset)
            imgs.append(self._load_image(self.samples[neighbor_idx]["image_path"]))
        image = np.stack(imgs, axis=0).astype(np.float32)
        return torch.from_numpy(image) / 255.0  # (num_slices, H, W)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.three_slice:
            # ── 2.5D path: stack [prev...curr...next] along channel dim ──────
            image = self._build_slice_window(idx)
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

        item = {
            "image": image,
            "mask": mask,
            "patient_id": sample["patient_id"],
            "slice_name": sample["slice_name"],
        }

        # ── Positional encoding: a SEPARATE field, not mixed into the image ──
        if self.use_position_encoding:
            item["position"] = torch.tensor(self._get_position_bucket(idx), dtype=torch.long)

        # ── Slice-consistency: the NEXT slice's window, built the identical
        # way as the current one, just re-centred. Clamped to the patient's
        # own last slice (replicate padding) -- if idx is already the last
        # slice, image_next will be nearly identical to image, which is the
        # correct degenerate behaviour (no real "next slice" exists, so the
        # consistency loss should contribute ~0 there, not error out).
        if self.use_slice_consistency:
            next_idx = self._get_neighbor_idx(idx, +1)
            item["image_next"] = self._build_slice_window(next_idx)

        return item

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

    # Data-shape flags read from config; all default to False/3 so every
    # existing config (which doesn't set them) behaves exactly as before.
    three_slice = config.get("data", {}).get("three_slice", False)
    num_slices = config.get("data", {}).get("num_slices", 3)
    use_position_encoding = config.get("data", {}).get("use_position_encoding", False)
    use_slice_consistency = config.get("data", {}).get("use_slice_consistency", False)

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
    print(f"  Position encoding: {use_position_encoding}")
    print(f"  Slice-consistency pairing: {use_slice_consistency}")

    common_kwargs = {
        "root_dir": config["data"]["root_dir"],
        "image_size": config["data"]["image_size"],
        "num_classes": num_classes,
        "selected_classes": selected_classes,
        "remap_classes": remap_classes,
        "verbose": verbose,
        "three_slice": three_slice,
        "num_slices": num_slices,
        "use_position_encoding": use_position_encoding,
        "use_slice_consistency": use_slice_consistency,
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