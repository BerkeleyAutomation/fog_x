import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CAMERAS = (
    ("observation.images.top", "observation/images/top"),
    ("observation.images.wrist_left", "observation/images/wrist_left"),
    ("observation.images.wrist_right", "observation/images/wrist_right"),
)


def perf():
    return time.perf_counter()


def fastwam_compose_video(cam_tensors):
    """Match FastWAM robotwin path: 33 raw frames -> 9 video frames -> 384x320 canvas."""
    video_indices = list(range(0, 33, 4))
    cams = []
    for x in cam_tensors:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # LeRobot gives [T,C,H,W] float in [0,1]; RoboDM gives [T,H,W,C] uint8.
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
        else:
            x = x.float()
        x = x[video_indices]
        cams.append(x)

    cam_top = F.interpolate(cams[0], size=(256, 320), mode="bilinear", align_corners=False)
    cam_left = F.interpolate(cams[1], size=(128, 160), mode="bilinear", align_corners=False)
    cam_right = F.interpolate(cams[2], size=(128, 160), mode="bilinear", align_corners=False)
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    video = torch.cat([cam_top, bottom], dim=-2)
    video = video.mul_(2.0).sub_(1.0)
    return video.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]


def episode_bounds(ds):
    bounds = []
    for ep in range(int(ds.meta.total_episodes)):
        row = ds.meta.episodes[ep]
        bounds.append((int(row["dataset_from_index"]), int(row["dataset_to_index"])))
    return bounds


def make_window_indices(bounds, num_windows, mode):
    valid = [(ep, start, end) for ep, (start, end) in enumerate(bounds) if end - start >= 33]
    out = []
    if mode == "episode0":
        ep, start, end = valid[0]
        for idx in range(start, min(start + num_windows, end - 32)):
            out.append((ep, idx))
    elif mode == "roundrobin":
        offsets = {ep: start for ep, start, end in valid}
        ends = {ep: end for ep, start, end in valid}
        while len(out) < num_windows:
            progressed = False
            for ep, start, end in valid:
                idx = offsets[ep]
                if idx <= ends[ep] - 33:
                    out.append((ep, idx))
                    offsets[ep] += 1
                    progressed = True
                    if len(out) >= num_windows:
                        break
            if not progressed:
                break
    else:
        raise ValueError(mode)
    return out


def bench_lerobot(ds, windows):
    t0 = perf()
    checksum = 0.0
    for _, idx in windows:
        s = ds[idx]
        video = fastwam_compose_video([s[k] for k, _ in CAMERAS])
        action = s["action"]
        proprio = s["observation.state"][:-1]
        checksum += float(video[:, 0, 0, 0].sum() + action[0].sum() + proprio[0].sum())
        del s, video, action, proprio
    elapsed = perf() - t0
    return {
        "seconds": elapsed,
        "windows_per_s": len(windows) / elapsed,
        "raw_camera_frames_per_s": len(windows) * 33 * 3 / elapsed,
        "output_video_frames_per_s": len(windows) * 9 / elapsed,
        "checksum": checksum,
    }


def load_robodm_episode(path):
    from robodm.trajectory import Trajectory

    traj = Trajectory(str(path), mode="r")
    try:
        return traj.load(return_type="numpy")
    finally:
        traj.close()


def bench_robodm(robodm_root, windows, include_cache_load):
    robodm_root = Path(robodm_root)
    cache = {}
    t_cache0 = perf()
    if include_cache_load:
        for ep, _ in windows:
            if ep not in cache:
                cache[ep] = load_robodm_episode(robodm_root / f"episode_{ep:03d}.vla")
    cache_s = perf() - t_cache0

    t0 = perf()
    checksum = 0.0
    for ep, idx in windows:
        if ep not in cache:
            cache[ep] = load_robodm_episode(robodm_root / f"episode_{ep:03d}.vla")
        local_idx = idx - int(cache[ep]["frame_index"][0, 0])
        cam_tensors = [cache[ep][rk][local_idx : local_idx + 33] for _, rk in CAMERAS]
        video = fastwam_compose_video(cam_tensors)
        action = torch.from_numpy(cache[ep]["action"][local_idx : local_idx + 32])
        proprio = torch.from_numpy(cache[ep]["observation/state"][local_idx : local_idx + 32])
        checksum += float(video[:, 0, 0, 0].sum() + action[0].sum() + proprio[0].sum())
        del video, action, proprio
    elapsed = perf() - t0
    frames = len(windows)
    result = {
        "cache_load_s": cache_s,
        "window_seconds": elapsed,
        "total_seconds": cache_s + elapsed,
        "windows_per_s_hot": frames / elapsed,
        "windows_per_s_with_cache": frames / (cache_s + elapsed),
        "raw_camera_frames_per_s_hot": frames * 33 * 3 / elapsed,
        "output_video_frames_per_s_hot": frames * 9 / elapsed,
        "episodes_cached": len(cache),
        "checksum": checksum,
    }
    cache.clear()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", default="/Users/pfb30/lute/quality_processor/flatten_box")
    parser.add_argument("--repo-id", default="pfb30/flatten_box")
    parser.add_argument("--robodm-root", default="/Users/pfb30/lute/robodm/flatten_box_robodm")
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--mode", choices=["episode0", "roundrobin"], default="episode0")
    parser.add_argument("--out", default="/Users/pfb30/lute/robodm/flatten_box_robodm/fastwam_window_benchmark.json")
    args = parser.parse_args()

    sys.path.insert(0, "/Users/pfb30/lute/robodm")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = 30
    delta_timestamps = {k: [i / fps for i in range(33)] for k, _ in CAMERAS}
    delta_timestamps["observation.state"] = [i / fps for i in range(33)]
    delta_timestamps["action"] = [i / fps for i in range(32)]

    t0 = perf()
    ds = LeRobotDataset(args.repo_id, root=args.lerobot_root, video_backend="pyav", delta_timestamps=delta_timestamps)
    init_s = perf() - t0
    windows = make_window_indices(episode_bounds(ds), args.windows, args.mode)
    print(json.dumps({"event": "setup", "init_s": init_s, "windows": len(windows), "mode": args.mode}), flush=True)

    print(json.dumps({"event": "lerobot_start"}), flush=True)
    lerobot = bench_lerobot(ds, windows)
    print(json.dumps({"event": "lerobot_done", **lerobot}), flush=True)

    print(json.dumps({"event": "robodm_cold_start"}), flush=True)
    robodm_cold = bench_robodm(args.robodm_root, windows, include_cache_load=True)
    print(json.dumps({"event": "robodm_cold_done", **robodm_cold}), flush=True)

    print(json.dumps({"event": "robodm_hot_start"}), flush=True)
    # Load once in the benchmark function, then report hot window slicing independently.
    robodm_hot = bench_robodm(args.robodm_root, windows, include_cache_load=True)
    print(json.dumps({"event": "robodm_hot_done", **robodm_hot}), flush=True)

    summary = {
        "fastwam_reference": {
            "num_frames": 33,
            "action_video_freq_ratio": 4,
            "video_frames_after_subsample": 9,
            "robotwin_canvas": [384, 320],
            "cameras": [k for k, _ in CAMERAS],
        },
        "mode": args.mode,
        "windows": len(windows),
        "lerobot_init_s": init_s,
        "lerobot": lerobot,
        "robodm_cold": robodm_cold,
        "robodm_hot": {
            "window_seconds": robodm_hot["window_seconds"],
            "windows_per_s_hot": robodm_hot["windows_per_s_hot"],
            "raw_camera_frames_per_s_hot": robodm_hot["raw_camera_frames_per_s_hot"],
            "output_video_frames_per_s_hot": robodm_hot["output_video_frames_per_s_hot"],
            "episodes_cached": robodm_hot["episodes_cached"],
        },
        "notes": [
            "LeRobot benchmark uses delta_timestamps exactly like FastWAM: 33 image/state frames and 32 action frames per sample.",
            "FastWAM then keeps every fourth image frame, giving 9 video frames, but LeRobot still decodes all 33 frames per camera.",
            "RoboDM cold includes full episode materialization for episodes touched by the sampled windows; hot reports slicing/resize after that cache exists.",
        ],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
