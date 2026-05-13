import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


def perf():
    return time.perf_counter()


def tensor_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def image_tensor_to_uint8_hwc(x):
    arr = tensor_to_numpy(x)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def vector_to_float32(x):
    return np.ascontiguousarray(tensor_to_numpy(x).astype(np.float32))


def scalar(x):
    arr = tensor_to_numpy(x)
    return arr.item() if hasattr(arr, "item") else x


def episode_bounds(ds, episode_idx):
    ep = ds.meta.episodes[int(episode_idx)]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def sample_to_robodm_dict(sample):
    return {
        "observation/images/top": image_tensor_to_uint8_hwc(sample["observation.images.top"]),
        "observation/images/wrist_left": image_tensor_to_uint8_hwc(sample["observation.images.wrist_left"]),
        "observation/images/wrist_right": image_tensor_to_uint8_hwc(sample["observation.images.wrist_right"]),
        "observation/state": vector_to_float32(sample["observation.state"]),
        "observation/state_ee": vector_to_float32(sample["observation.state_ee"]),
        "action": vector_to_float32(sample["action"]),
        "action_ee": vector_to_float32(sample["action_ee"]),
        "frame_index": np.asarray([scalar(sample["frame_index"])], dtype=np.int64),
        "episode_index": np.asarray([scalar(sample["episode_index"])], dtype=np.int64),
        "task_index": np.asarray([scalar(sample["task_index"])], dtype=np.int64),
    }


def validate_vla(path, expected_frames):
    from robodm.trajectory import Trajectory

    traj = Trajectory(str(path), mode="r")
    try:
        data = traj.load(return_type="numpy")
    finally:
        traj.close()

    required = {
        "observation/images/top": [expected_frames, 480, 640, 3],
        "observation/images/wrist_left": [expected_frames, 480, 640, 3],
        "observation/images/wrist_right": [expected_frames, 480, 640, 3],
        "observation/state": [expected_frames, 14],
        "action": [expected_frames, 14],
        "observation/state_ee": [expected_frames, 20],
        "action_ee": [expected_frames, 20],
    }
    shapes = {k: list(v.shape) for k, v in data.items() if hasattr(v, "shape")}
    for key, shape in required.items():
        if shapes.get(key) != shape:
            raise RuntimeError(f"{path}: {key} shape {shapes.get(key)} != {shape}")
    return shapes


def completed(meta_path, expected):
    if not meta_path.exists():
        return False
    try:
        actual = json.loads(meta_path.read_text())
    except Exception:
        return False
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def convert_episode(ds, episode_idx, out_dir, codec, compact, force=False):
    from robodm.trajectory import Trajectory

    start_idx, end_idx = episode_bounds(ds, episode_idx)
    n = end_idx - start_idx
    out_path = out_dir / f"episode_{episode_idx:03d}.vla"
    meta_path = out_dir / f"episode_{episode_idx:03d}.json"
    expected_meta = {
        "episode_index": int(episode_idx),
        "frames": int(n),
        "codec": codec,
        "compact": bool(compact),
    }
    if not force and out_path.exists() and completed(meta_path, expected_meta):
        return json.loads(meta_path.read_text()) | {"skipped": True}

    tmp_path = out_dir / f"episode_{episode_idx:03d}.vla.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    t0 = perf()
    traj = Trajectory(
        str(tmp_path),
        mode="w",
        video_codec=codec,
        codec_options={"preset": "ultrafast"} if codec == "libx264" else None,
        time_unit="ms",
        enforce_monotonic=True,
    )
    try:
        for idx in range(start_idx, end_idx):
            sample = ds[idx]
            timestamp_ms = int(round(float(scalar(sample["timestamp"])) * 1000.0))
            traj.add_by_dict(sample_to_robodm_dict(sample), timestamp=timestamp_ms, time_unit="ms")
    finally:
        traj.close(compact=compact)
    write_s = perf() - t0

    t0 = perf()
    shapes = validate_vla(tmp_path, n)
    validate_s = perf() - t0

    if out_path.exists():
        out_path.unlink()
    tmp_path.rename(out_path)

    result = expected_meta | {
        "path": str(out_path),
        "size_mb": out_path.stat().st_size / (1024 * 1024),
        "write_s": write_s,
        "write_fps": n / write_s,
        "validate_load_s": validate_s,
        "validate_load_fps": n / validate_s,
        "shapes": shapes,
        "skipped": False,
    }
    meta_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def benchmark_lerobot(ds, frame_limit=None):
    total_frames = int(ds.meta.total_frames if frame_limit is None else min(frame_limit, ds.meta.total_frames))
    t0 = perf()
    bytes_seen = 0
    for idx in range(total_frames):
        sample = ds[idx]
        for key in CAMERA_KEYS:
            bytes_seen += tensor_to_numpy(sample[key]).nbytes
    elapsed = perf() - t0
    return {
        "frames": total_frames,
        "seconds": elapsed,
        "fps": total_frames / elapsed,
        "camera_frames_per_s": (total_frames * len(CAMERA_KEYS)) / elapsed,
        "decoded_gb_float_tensors": bytes_seen / 1e9,
    }


def benchmark_robodm(out_dir, frame_limit=None):
    import gc

    from robodm.trajectory import Trajectory

    paths = sorted(out_dir.glob("episode_*.vla"))
    total_frames = 0
    episodes_loaded = 0
    t0 = perf()
    for path in paths:
        traj = Trajectory(str(path), mode="r")
        try:
            data = traj.load(return_type="numpy")
        finally:
            traj.close()
        n = int(data["action"].shape[0])
        if frame_limit is not None and total_frames + n > frame_limit:
            # Full trajectory loads only; report at whole-episode granularity.
            del data
            gc.collect()
            break
        total_frames += n
        episodes_loaded += 1
        del data
        gc.collect()
    elapsed = perf() - t0
    return {
        "episodes_loaded": episodes_loaded,
        "frames": total_frames,
        "seconds": elapsed,
        "fps": total_frames / elapsed if elapsed else 0.0,
        "camera_frames_per_s": (total_frames * len(CAMERA_KEYS)) / elapsed if elapsed else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/pfb30/lute/quality_processor/flatten_box")
    parser.add_argument("--repo-id", default="pfb30/flatten_box")
    parser.add_argument("--out-dir", default="/Users/pfb30/lute/robodm/flatten_box_robodm")
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--compact", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-lerobot-bench", action="store_true")
    parser.add_argument("--bench-frame-limit", type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, "/Users/pfb30/lute/robodm")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf()
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    init_s = perf() - t0
    print(json.dumps({
        "event": "lerobot_loaded",
        "init_s": init_s,
        "episodes": ds.meta.total_episodes,
        "frames": ds.meta.total_frames,
        "out_dir": str(out_dir),
    }), flush=True)

    episode_results = []
    if not args.skip_convert:
        for episode_idx in range(int(ds.meta.total_episodes)):
            print(json.dumps({"event": "convert_start", "episode": episode_idx}), flush=True)
            result = convert_episode(ds, episode_idx, out_dir, args.codec, args.compact, force=args.force)
            episode_results.append(result)
            print(json.dumps({"event": "convert_done", **result}), flush=True)

    summary = {
        "event": "summary",
        "root": args.root,
        "repo_id": args.repo_id,
        "out_dir": str(out_dir),
        "codec": args.codec,
        "compact": args.compact,
        "dataset_frames": int(ds.meta.total_frames),
        "dataset_episodes": int(ds.meta.total_episodes),
        "robodm_total_size_mb": sum(p.stat().st_size for p in out_dir.glob("episode_*.vla")) / (1024 * 1024),
        "episode_results": episode_results,
    }

    if not args.skip_lerobot_bench:
        print(json.dumps({"event": "benchmark_lerobot_start"}), flush=True)
        summary["lerobot_benchmark"] = benchmark_lerobot(ds, frame_limit=args.bench_frame_limit)
        print(json.dumps({"event": "benchmark_lerobot_done", **summary["lerobot_benchmark"]}), flush=True)

    print(json.dumps({"event": "benchmark_robodm_start"}), flush=True)
    summary["robodm_benchmark"] = benchmark_robodm(out_dir, frame_limit=args.bench_frame_limit)
    print(json.dumps({"event": "benchmark_robodm_done", **summary["robodm_benchmark"]}), flush=True)

    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
