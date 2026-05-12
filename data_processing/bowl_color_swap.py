"""
Color-swap data augmentation for videos of three colored bowls.

Pipeline:
  1. HSV-threshold each bowl per frame to get a mask.
  2. Sample frames evenly across the video and compute each bowl's actual
     (circular-mean H, mean S, mean V) in HSV. Circular mean handles the
     H=0/180 wrap; without it, "red" averages incorrectly to cyan.
  3. For each swap src->tgt: rotate H by the short circular distance between
     the measured means, shift S by their mean delta, leave V untouched
     (so shading and highlights survive).

The per-bowl stats are measured from the actual scene, so the recolored
pixels look like the real target bowl in this video -- no manual hue offsets
or saturation knobs.

Assumes a uniformly white/gray background; pixels matching the bowl HSV
ranges anywhere else in the frame will also be recolored.

Usage (single video, all 5 permutations -> /tmp/aug/clip_grb.mp4 etc.):
    python bowl_color_swap.py --video clip.mp4 --out-dir /tmp/aug --all-swaps

Batch (writes alongside each input video):
    for v in /data/clips/*.mp4; do
        python bowl_color_swap.py --video "$v" --all-swaps
    done
"""

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# OpenCV HSV: H in [0,180], S/V in [0,255]. Red wraps around 0/180.
HSV_RANGES = {
    "red":   [((0,   70, 50), (10,  255, 255)),
              ((170, 70, 50), (180, 255, 255))],
    "green": [((40,  60, 40), (85,  255, 255))],
    "blue":  [((95,  80, 40), (130, 255, 255))],
}

COLORS = ("red", "green", "blue")


def _episode_boundaries(dataset_name):
    """Returns [start_0, start_1, ..., start_N, total_frames] for a HF
    LeRobot dataset. Cached so repeated calls don't re-download."""
    if not hasattr(_episode_boundaries, "_cache"):
        _episode_boundaries._cache = {}
    if dataset_name in _episode_boundaries._cache:
        return _episode_boundaries._cache[dataset_name]
    from datasets import load_dataset
    ds = load_dataset(dataset_name)["train"]
    ep = np.asarray(ds["episode_index"])
    boundaries = [0]
    for i in range(1, len(ep)):
        if ep[i] != ep[i - 1]:
            boundaries.append(i)
    boundaries.append(len(ep))
    _episode_boundaries._cache[dataset_name] = boundaries
    return boundaries


# ---------- masks ----------
def color_mask(hsv, ranges):
    m = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def bowl_region_from_mask(mask_u8):
    """Largest connected blob -> convex hull, filled. Used to get a clean
    bowl-shaped region from frame 0 where bowls are unoccluded. Bowls are
    roughly circular so the hull approximates the outline well."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return mask_u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8) * 255
    ys, xs = np.where(blob > 0)
    if len(xs) == 0:
        return blob
    points = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(points)
    out = np.zeros_like(mask_u8)
    cv2.fillPoly(out, [hull], 255)
    return out


def save_mask_viz(frame, masks, path):
    """Save a high-contrast visualization: per-bowl masks get tinted with
    a *complementary* color (so they don't blend into the actual bowl
    color) and a thick outline. Labels make each region unambiguous."""
    viz = frame.copy()
    # Complementary tints (NOT the bowl's own color, so the overlay pops):
    tints = {
        "red":   (255,   0, 255),   # magenta over red bowl
        "green": (255, 255,   0),   # cyan    over green bowl
        "blue":  (  0, 255, 255),   # yellow  over blue bowl
    }
    for name, m in masks.items():
        layer = np.zeros_like(frame)
        layer[m] = tints.get(name, (255, 255, 255))
        viz = cv2.addWeighted(viz, 0.4, layer, 0.6, 0)  # heavy tint where masked
        # Thick outline + label
        m_u8 = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz, contours, -1, (0, 0, 0), 4)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(viz, name.upper(), (cx - 30, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
            cv2.putText(viz, name.upper(), (cx - 30, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(path), viz)


def build_static_masks(video_path, sat_boost, ref_idx, viz_path=None):
    """Compute per-bowl masks once, from a reference frame: HSV threshold
    plus a small dilation to recover anti-aliased rim pixels. No hole
    filling, no convex hull -- the mask is exactly what the HSV threshold
    catches. Holes from specular highlights are intentional (they stay
    bright in the output, like real plastic). Reused for every frame; apply
    `apply_arm_exclusion` per frame to drop dark arm pixels."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read reference frame {ref_idx} from {video_path}")
    if sat_boost > 1.0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0, 255)
        hsv = hsv.astype(np.uint8)
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    out = {}
    for name, ranges in HSV_RANGES.items():
        m = color_mask(hsv, ranges)
        m = cv2.dilate(m, kernel)
        out[name] = m.astype(bool)
    if viz_path is not None:
        save_mask_viz(frame, out, viz_path)
        print(f"saved static-mask viz to {viz_path}")
    return out


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def sam3_track_occluders(video_path, start_frame, num_frames, prompts, model_id):
    """Use SAM3's video tracker to produce per-frame masks for each text
    prompt across the given frame range. Returns a list (one entry per
    frame in the range) where each entry is a bool mask = union of all
    detections for that frame (subtract from bowl masks before recoloring)."""
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    import torch

    device = _pick_device()
    print(f"loading SAM3 video tracker ({model_id}) on {device}...")
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    model = Sam3VideoModel.from_pretrained(model_id).to(device).eval()

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    ok, f0 = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"could not read start frame {start_frame}")
    h, w = f0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    session = processor.init_video_session(
        inference_device=torch.device(device),
        inference_state_device=torch.device(device),
        video_storage_device=torch.device("cpu"),
        dtype=torch.float32,
    )
    processor.add_text_prompt(session, list(prompts))
    print(f"  text prompts: {list(prompts)}")

    per_frame = []
    pbar = tqdm(total=num_frames, desc="sam3 track")
    for i in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            output = model(inference_session=session, frame=pixel_values)
        combined = np.zeros((h, w), dtype=bool)
        for oid, mask_t in (output.obj_id_to_mask or {}).items():
            m = mask_t
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            m = np.asarray(m).squeeze()
            if m.ndim == 0:
                continue
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST) > 0.5
            combined |= (m > 0)
        per_frame.append(combined)
        pbar.update(1)
    pbar.close()
    cap.release()
    return per_frame


def _sam3_segment(processor, model, device, frame_rgb, text, h, w):
    """Run SAM3 on one frame with one text prompt, return the highest-scoring
    boolean mask (or None) plus the score."""
    import torch
    inputs = processor(images=frame_rgb, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)]
    )
    if not results or len(results[0]["masks"]) == 0:
        return None, 0.0
    scores = results[0]["scores"]
    best = int(scores.argmax())
    return results[0]["masks"][best].cpu().numpy().astype(bool), float(scores[best])


def _hsv_signature(frame_bgr, mask, h_tol=15, sv_tol=60):
    """Return HSV ranges (for cv2.inRange via color_mask) describing the
    pixels in `mask`. Used to find similar-colored pixels in other frames."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[mask]
    if len(pix) < 50:
        return None
    h_rad = pix[:, 0].astype(np.float32) * (np.pi / 90.0)
    h_mean = (np.arctan2(np.sin(h_rad).mean(), np.cos(h_rad).mean()) * 90.0 / np.pi) % 180
    s_lo = max(0, int(pix[:, 1].mean()) - sv_tol)
    v_lo = max(0, int(pix[:, 2].mean()) - sv_tol)
    h_lo = (h_mean - h_tol) % 180
    h_hi = (h_mean + h_tol) % 180
    if h_lo <= h_hi:
        ranges = [((int(h_lo), s_lo, v_lo), (int(h_hi), 255, 255))]
    else:                                              # wraps across 0/180
        ranges = [((0, s_lo, v_lo), (int(h_hi), 255, 255)),
                  ((int(h_lo), s_lo, v_lo), (180, 255, 255))]
    return ranges, float(h_mean)


def sam3_static_masks(video_path, ref_idx, model_id, viz_path=None,
                      exclude_prompts=None):
    """Use SAM3 to segment each bowl ('red bowl', 'green bowl', 'blue bowl')
    plus optional `exclude_prompts` (e.g. ['yellow banana']). For exclude
    objects, returns their HSV signature so per-frame we can find similar
    pixels anywhere in the frame and keep them un-recolored."""
    from transformers import Sam3Model, Sam3Processor

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref_idx))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {ref_idx} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    device = _pick_device()
    print(f"loading SAM3 ({model_id}) on {device}...")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()

    masks = {}
    for name in COLORS:
        text = f"{name} bowl"
        mask, score = _sam3_segment(processor, model, device, frame_rgb, text, h, w)
        if mask is None:
            print(f"  WARN: SAM3 found no '{text}'")
            continue
        masks[name] = mask
        print(f"  {text}: score={score:.3f}, mask pixels={int(mask.sum())}")

    exclude_hsv = {}
    viz_extra = {}
    for prompt in (exclude_prompts or []):
        mask, score = _sam3_segment(processor, model, device, frame_rgb, prompt, h, w)
        if mask is None:
            print(f"  WARN: SAM3 found no '{prompt}'")
            continue
        sig = _hsv_signature(frame_bgr, mask)
        if sig is None:
            print(f"  WARN: '{prompt}' mask too small for HSV signature")
            continue
        ranges, h_mean = sig
        exclude_hsv[prompt] = ranges
        viz_extra[prompt] = mask
        print(f"  {prompt}: score={score:.3f}, H~{h_mean:.0f}, "
              f"mask pixels={int(mask.sum())}, HSV ranges={ranges}")

    if viz_path is not None:
        save_mask_viz(frame_bgr, {**masks, **viz_extra}, viz_path)
        print(f"saved SAM3 mask viz to {viz_path}")
    return masks, exclude_hsv


def apply_arm_exclusion(frame_bgr, static_masks, v_threshold):
    """Legacy: drop pixels whose V is below `v_threshold`. Catches the
    dark robot arm but not bright foreign objects (e.g. a yellow banana
    passing through the bowl region)."""
    v = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    not_arm = v > v_threshold
    return {name: m & not_arm for name, m in static_masks.items()}


def apply_color_filter(frame_bgr, static_masks):
    """Per frame, restrict each static mask to pixels currently matching
    the bowl's expected HSV color range. Excludes anything that passes
    over the bowl region but isn't actually a bowl pixel: the dark arm,
    a yellow banana, the gripper, fingers, etc."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    out = {}
    for name, m in static_masks.items():
        ranges = HSV_RANGES.get(name)
        if ranges is None:
            out[name] = m
            continue
        cm = color_mask(hsv, ranges)
        cm = cv2.dilate(cm, kernel)               # keep anti-aliased rim
        out[name] = m & cm.astype(bool)
    return out


def color_masks_for_frame(frame_bgr, dilate_kernel, sat_boost=1.0):
    """HSV masks per bowl. With sat_boost > 1, mask thresholding runs on a
    saturation-boosted copy of the frame so highlight gradients on shiny
    bowls get captured. The boosted copy is used only for mask computation;
    the original frame is recolored unchanged elsewhere."""
    if sat_boost > 1.0:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0, 255)
        hsv = hsv.astype(np.uint8)
    else:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out = {}
    for name, ranges in HSV_RANGES.items():
        m = color_mask(hsv, ranges)
        m = cv2.dilate(m, dilate_kernel)         # recover anti-aliased rim
        out[name] = m.astype(bool)
    return out


# ---------- measured per-bowl color stats ----------
def compute_bowl_means(frames_and_masks):
    """Returns dict bowl -> (H_circular_mean, S_mean, V_mean) in OpenCV scale."""
    h_cos = {}; h_sin = {}; s_sum = {}; v_sum = {}; counts = {}
    for frame, masks in frames_and_masks:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for name, m in masks.items():
            if m is None or not m.any():
                continue
            h_rad = hsv[..., 0][m].astype(np.float32) * (np.pi / 90.0)
            h_cos[name] = h_cos.get(name, 0.0) + np.cos(h_rad).sum()
            h_sin[name] = h_sin.get(name, 0.0) + np.sin(h_rad).sum()
            s_sum[name] = s_sum.get(name, 0.0) + hsv[..., 1][m].astype(np.float32).sum()
            v_sum[name] = v_sum.get(name, 0.0) + hsv[..., 2][m].astype(np.float32).sum()
            counts[name] = counts.get(name, 0) + int(m.sum())
    out = {}
    for n, c in counts.items():
        if c == 0:
            continue
        h_mean = (np.arctan2(h_sin[n] / c, h_cos[n] / c) * (90.0 / np.pi)) % 180
        out[n] = (h_mean, s_sum[n] / c, v_sum[n] / c)
    return out


# ---------- recoloring ----------
def apply_swap(frame, masks, swap, means):
    """For each src->tgt pair: rotate H by the short circular distance between
    measured means, shift S by (tgt_S_mean - src_S_mean). Keep V."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    out = hsv.copy()
    for src, tgt in swap.items():
        if src == tgt or src not in masks or src not in means or tgt not in means:
            continue
        sm = masks[src]
        if not sm.any():
            continue
        src_h, src_s, _ = means[src]
        tgt_h, tgt_s, _ = means[tgt]
        # signed shortest-arc circular shift, in (-90, 90]
        h_shift = int(round(((tgt_h - src_h + 90) % 180) - 90))
        s_shift = int(round(tgt_s - src_s))
        out[..., 0] = np.where(sm, (hsv[..., 0] + h_shift) % 180, out[..., 0])
        out[..., 1] = np.where(sm,
                               np.clip(hsv[..., 1] + s_shift, 0, 255),
                               out[..., 1])
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------- swap plan ----------
def all_permutation_swaps():
    """All 5 non-identity permutations of (red, green, blue).
    suffix is the target ordering, e.g. 'grb' = R->G, G->R, B->B."""
    out = {}
    for perm in itertools.permutations(COLORS):
        if perm == COLORS:
            continue
        suffix = "".join(c[0] for c in perm)
        out[suffix] = {COLORS[i]: perm[i] for i in range(3) if COLORS[i] != perm[i]}
    return out


def suffix_for_swap(swap):
    """Target-color ordering for the (red, green, blue) bowls, e.g. 'grb'."""
    return "".join(swap.get(c, c)[0] for c in COLORS)


def build_plan(args):
    """List of (suffix, swap_dict, output_path), one per output video.
    Output paths are derived from the input video's stem so batch runs over
    many videos don't collide."""
    video = Path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.all_swaps:
        swaps = all_permutation_swaps()
    else:
        swap = dict(p.split(":") for p in args.swap.split(","))
        swaps = {suffix_for_swap(swap): swap}
    stem = video.stem
    if args.episode_tag:
        stem = f"{stem}_{args.episode_tag}"
    return [(s, sw, str(out_dir / f"{stem}_{s}.mp4")) for s, sw in swaps.items()]


# ---------- driver ----------
def run(args, plan):
    cap = cv2.VideoCapture(str(args.video))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kernel = np.ones((3, 3), np.uint8)

    # Frame range we'll actually process: [start_frame, start_frame + n_to_process)
    start_frame = max(0, int(args.start_frame))
    available = max(0, total - start_frame)
    n_to_process = available if args.max_frames is None else min(args.max_frames, available)
    if n_to_process <= 0:
        raise SystemExit(f"no frames to process (start={start_frame}, total={total})")
    end_frame = start_frame + n_to_process
    print(f"processing frames [{start_frame}, {end_frame})")

    tracked_occluder_per_frame = None
    if args.sam3 or args.static_masks:
        ep_stem = Path(plan[0][2]).stem.split("_")[0]
        tag = ("_sam3" if args.sam3 else "") + (("_" + args.episode_tag) if args.episode_tag else "")
        viz_stem = f"{ep_stem}{tag}_static_mask_viz"
        viz_path = Path(plan[0][2]).parent / f"{viz_stem}.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        exclude_hsv = {}
        exclude_prompts = []
        if args.sam3:
            exclude_prompts = [p.strip() for p in args.exclude.split(",") if p.strip()]
            static_masks, exclude_hsv = sam3_static_masks(
                args.video, args.ref_frame, args.sam3_model, viz_path, exclude_prompts)
        else:
            static_masks = build_static_masks(args.video, args.sat_boost, args.ref_frame, viz_path)
        print(f"static masks (pixels): "
              f"{ {n: int(m.sum()) for n, m in static_masks.items()} }")

        # Optional: precompute per-frame occluder masks via SAM3 video tracker.
        if args.track_occluder and args.sam3 and exclude_prompts:
            tracked_occluder_per_frame = sam3_track_occluders(
                args.video, start_frame, n_to_process, exclude_prompts, args.sam3_model)
            exclude_hsv = {}                       # tracker supersedes HSV signature

        def masks_fn(frame, frame_offset):
            m = apply_arm_exclusion(frame, static_masks, args.arm_v)
            exc = None
            if tracked_occluder_per_frame is not None and frame_offset < len(tracked_occluder_per_frame):
                exc = tracked_occluder_per_frame[frame_offset]
            elif exclude_hsv:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                exc = np.zeros(hsv.shape[:2], dtype=bool)
                for ranges in exclude_hsv.values():
                    exc |= color_mask(hsv, ranges).astype(bool)
            if exc is not None:
                m = {name: bm & ~exc for name, bm in m.items()}
            return m
    else:
        def masks_fn(frame, frame_offset):
            return color_masks_for_frame(frame, kernel, args.sat_boost)

    # Compute per-bowl HSV stats from the reference frame only -- frame 0
    # has all three bowls unoccluded. Using HSV-thresholded pixels (not
    # SAM3) keeps the stats clean: only saturated bowl pixels contribute,
    # so the means stay vivid instead of getting pulled toward the
    # shaded/highlighted interior that SAM3 includes.
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.ref_frame))
    ok, ref_f = cap.read()
    if not ok:
        raise SystemExit(f"could not read ref frame {args.ref_frame} for means")
    ref_masks = color_masks_for_frame(ref_f, kernel, args.sat_boost)
    means = compute_bowl_means([(ref_f, ref_masks)])
    print("measured per-bowl (H, S, V) from ref frame:",
          {k: tuple(round(float(x), 1) for x in v) for k, v in means.items()})
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    snap_idx = int(round(args.snapshot_time * fps)) if args.snapshot_time is not None else None
    writers = []
    for label, swap, out_path in plan:
        writers.append((label, swap, out_path,
                        cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps, (w, h))))

    i = 0
    pbar = tqdm(desc=f"swapping x{len(writers)}", total=n_to_process)
    while i < n_to_process:
        ok, frame = cap.read()
        if not ok:
            break
        masks = masks_fn(frame, i)
        for _label, swap, out_path, writer in writers:
            swapped = apply_swap(frame, masks, swap, means)
            writer.write(swapped)
            if snap_idx is not None and i == snap_idx:
                cv2.imwrite(str(Path(out_path).with_suffix(".png")), swapped)
        i += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    for _label, _swap, out_path, writer in writers:
        writer.release()
        print(f"wrote {i} frames to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Files are named "
                         "{video_stem}_{suffix}.mp4 where suffix is the target "
                         "ordering for (red,green,blue) e.g. 'grb'. "
                         "Defaults to the input video's directory.")
    ap.add_argument("--swap", default="red:green,green:red",
                    help="Comma-separated src:tgt pairs for single-output mode. "
                         "Ignored when --all-swaps is set.")
    ap.add_argument("--all-swaps", action="store_true",
                    help="Produce all 5 non-identity color permutations.")
    ap.add_argument("--snapshot-time", type=float, default=4.5,
                    help="Also save a PNG snapshot at this timestamp (seconds) "
                         "next to each output video. Set <0 to disable.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap number of frames processed (debugging).")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="Global frame index to start from. Use this to "
                         "process one episode of a LeRobot-style file where "
                         "many episodes are concatenated.")
    ap.add_argument("--episode-tag", default="",
                    help="String inserted into the output filename "
                         "(e.g. 'ep00') so per-episode runs don't collide.")
    ap.add_argument("--dataset", default=None,
                    help="HuggingFace dataset name (e.g. rslxcvg/banana_blue). "
                         "With --episode, overrides --start-frame, --max-frames, "
                         "--episode-tag and --ref-frame from the dataset's "
                         "episode boundaries.")
    ap.add_argument("--episode", type=int, default=None,
                    help="Episode index (0-based) to process. Requires --dataset.")
    ap.add_argument("--sat-boost", type=float, default=1.0,
                    help="Multiply saturation by this factor on a copy of "
                         "each frame before computing HSV masks. >1 pulls "
                         "shiny gradient pixels around specular highlights "
                         "into the bowl's mask. Output pixels are unchanged. "
                         "Try 1.5-2.0 for shiny plastic bowls.")
    ap.add_argument("--static-masks", action="store_true",
                    help="Compute one mask per bowl from a single reference "
                         "frame, fill highlight holes, and reuse for every "
                         "frame. Per-frame, exclude dark pixels (the robot "
                         "arm). Requires a stationary camera and bowls.")
    ap.add_argument("--ref-frame", type=int, default=None,
                    help="Frame index used to build static masks. Defaults "
                         "to --start-frame (i.e. the first frame of the "
                         "episode being processed).")
    ap.add_argument("--arm-v", type=int, default=40,
                    help="V threshold (0-255) below which a pixel is treated "
                         "as the arm and excluded from recoloring. Default 40.")
    ap.add_argument("--sam3", action="store_true",
                    help="Use SAM3 (with text prompts 'red/green/blue bowl') "
                         "to build the static masks instead of HSV thresholds. "
                         "Implies --static-masks. Needs torch + transformers.")
    ap.add_argument("--sam3-model", default="facebook/sam3",
                    help="HuggingFace model id for SAM3.")
    ap.add_argument("--exclude", default="yellow banana",
                    help="Comma-separated SAM3 prompts whose color should be "
                         "preserved (kept un-recolored) everywhere in every "
                         "frame. SAM3 finds the object in the ref frame, "
                         "records its HSV signature, and per-frame pixels "
                         "matching that signature are subtracted from the "
                         "bowl masks. Set '' to disable.")
    ap.add_argument("--track-occluder", action="store_true",
                    help="Use SAM3 video tracker to get a per-frame mask for "
                         "each --exclude prompt (more precise than the HSV "
                         "signature, but adds a slow pre-pass).")
    args = ap.parse_args()
    if args.snapshot_time is not None and args.snapshot_time < 0:
        args.snapshot_time = None
    if args.episode is not None:
        if args.dataset is None:
            raise SystemExit("--episode requires --dataset")
        boundaries = _episode_boundaries(args.dataset)
        if args.episode >= len(boundaries) - 1:
            raise SystemExit(
                f"--episode {args.episode} out of range; dataset has "
                f"{len(boundaries) - 1} episodes")
        args.start_frame = boundaries[args.episode]
        ep_len = boundaries[args.episode + 1] - boundaries[args.episode]
        args.max_frames = ep_len if args.max_frames is None else min(args.max_frames, ep_len)
        if not args.episode_tag:
            args.episode_tag = f"ep{args.episode:02d}"
        print(f"episode {args.episode}: frames [{args.start_frame}, "
              f"{args.start_frame + ep_len}), tag '{args.episode_tag}'")
    if args.ref_frame is None:
        args.ref_frame = args.start_frame

    plan = build_plan(args)
    print(f"plan: {len(plan)} output(s)")
    for label, swap, out_path in plan:
        print(f"  {label or '-'}: {swap}  ->  {out_path}")
    run(args, plan)


if __name__ == "__main__":
    main()
