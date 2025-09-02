# -*- coding: utf-8 -*-
import os, sys, time, argparse, subprocess as sp, warnings, shutil
warnings.filterwarnings("ignore")

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import freeze_support

# -------------------------------
# Optional UI bits (only if --display)
# -------------------------------
def try_import_display():
    try:
        from IPython.display import Image as Image_colab, display, SVG, clear_output  # noqa
        from ipywidgets import IntSlider, Output, IntProgress  # noqa
        return True
    except Exception:
        return False


def ensure_file(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def download_u2net(dest_dir):
    """
    CLIPasso expects U2Net_/saved_models/u2net.pth.
    The original code called `gdown` on a folder; do it explicitly to a file
    and overwrite any stale .part files.
    """
    ensure_file(os.path.join(dest_dir, "u2net.pth"))
    # Clean .part leftovers if any
    for f in os.listdir(dest_dir):
        if f.endswith(".part"):
            try:
                os.remove(os.path.join(dest_dir, f))
            except Exception:
                pass
    # Install gdown if missing
    try:
        import gdown  # noqa
    except Exception:
        sp.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    # Download (id is from the original script)
    url_id = "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
    out_path = os.path.join(dest_dir, "u2net.pth")
    # Use CLI to stay faithful to original behavior
    sp.check_call(["gdown", "--id", url_id, "--fuzzy", "-O", out_path])
    return out_path


def run_once(seed, wandb_name, params, shared_losses):
    """
    Runs painterly_rendering.py once and records the best loss in shared dict.
    """
    target         = params["target"]
    output_dir     = params["output_dir"]
    num_strokes    = params["num_strokes"]
    num_iter       = params["num_iter"]
    save_interval  = params["save_interval"]
    use_gpu        = params["use_gpu"]
    fix_scale      = params["fix_scale"]
    mask_object    = params["mask_object"]
    colab_mode     = params["colab"]
    display_mode   = params["display"]

    cmd = [
        sys.executable, "painterly_rendering.py", target,
        "--num_paths", str(num_strokes),
        "--output_dir", output_dir,
        "--wandb_name", wandb_name,
        "--num_iter", str(num_iter),
        "--save_interval", str(save_interval),
        "--seed", str(seed),
        "--use_gpu", str(int(use_gpu)),
        "--fix_scale", str(fix_scale),
        "--mask_object", str(mask_object),
        "--mask_object_attention", str(mask_object),
        "--display_logs", str(int(colab_mode)),
        "--display", str(int(display_mode)),
    ]
    rc = sp.run(cmd).returncode
    if rc:
        raise SystemExit(f"Subprocess failed (seed={seed}) with code {rc}")

    cfg_path = os.path.join(output_dir, wandb_name, "config.npy")
    config = np.load(cfg_path, allow_pickle=True).item()
    loss_eval = np.array(config["loss_eval"])
    best = loss_eval.min()
    shared_losses[wandb_name] = float(best)


def main():
    freeze_support()
    # Safer on Windows
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_file", type=str, required=True,
                        help="filename under target_images/")
    parser.add_argument("--num_strokes", type=int, default=16)
    parser.add_argument("--num_iter", type=int, default=2001)
    parser.add_argument("--fix_scale", type=int, default=0)
    parser.add_argument("--mask_object", type=int, default=0)
    parser.add_argument("--num_sketches", type=int, default=3)
    parser.add_argument("--multiprocess", type=int, default=0)
    parser.add_argument("-colab", action="store_true")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-display", action="store_true")
    parser.add_argument("--gpunum", type=int, default=0)
    args = parser.parse_args()

    abs_path = os.path.abspath(os.getcwd())
    target = os.path.join(abs_path, "target_images", args.target_file)
    if not os.path.isfile(target):
        raise FileNotFoundError(f"{target} does not exist!")

    # Prepare U2Net weights
    u2_dir = os.path.join(abs_path, "U2Net_", "saved_models")
    if not os.path.isfile(os.path.join(u2_dir, "u2net.pth")):
        os.makedirs(u2_dir, exist_ok=True)
        print("Downloading U2Net weights ...")
        download_u2net(u2_dir)

    test_name = os.path.splitext(args.target_file)[0]
    output_dir = os.path.join(abs_path, "output_sketches", test_name)
    os.makedirs(output_dir, exist_ok=True)

    num_iter = args.num_iter
    save_interval = 10
    use_gpu = (not args.cpu) and torch.cuda.is_available()

    if not use_gpu:
        print("CUDA not available; running on CPU (will be slow).")

    if args.display or args.colab:
        print("=" * 50)
        print(f"Processing [{args.target_file}] ...")
        print(f"GPU: {use_gpu}")
        print(f"Results -> {output_dir}")
        print("=" * 50)
        if args.display and not try_import_display():
            print("`--display` requested but IPython/ipywidgets not found; continuing headless.")

    multiprocess = (not args.colab) and (args.num_sketches > 1) and bool(args.multiprocess)

    # Pack parameters to pass to workers
    params = dict(
        target=target,
        output_dir=output_dir,
        num_strokes=args.num_strokes,
        num_iter=num_iter,
        save_interval=save_interval,
        use_gpu=use_gpu,
        fix_scale=args.fix_scale,
        mask_object=args.mask_object,
        colab=args.colab,
        display=args.display,
    )

    seeds = list(range(0, args.num_sketches * 1000, 1000))
    manager = mp.Manager()
    losses_all = manager.dict()

    if multiprocess:
        # Reasonable pool size; cap by number of sketches
        ncpus = min(len(seeds), max(1, os.cpu_count() or 1))
        with mp.Pool(processes=ncpus) as pool:
            for seed in seeds:
                wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
                pool.apply_async(run_once, (seed, wandb_name, params, losses_all))
            pool.close()
            pool.join()
    else:
        for seed in seeds:
            wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
            run_once(seed, wandb_name, params, losses_all)

    # Pick best
    if not len(losses_all):
        raise RuntimeError("No runs completed; check earlier logs.")

    sorted_final = dict(sorted(losses_all.items(), key=lambda kv: kv[1]))
    best_key = list(sorted_final.keys())[0]
    src_svg = os.path.join(output_dir, best_key, "best_iter.svg")
    dst_svg = os.path.join(output_dir, f"{best_key}_best.svg")
    if not os.path.isfile(src_svg):
        # fallback: choose last saved SVG if best_iter.svg missing
        svg_logs = os.path.join(output_dir, best_key, "svg_logs")
        if os.path.isdir(svg_logs):
            svgs = [os.path.join(svg_logs, f) for f in os.listdir(svg_logs) if f.endswith(".svg")]
            if svgs:
                src_svg = max(svgs, key=os.path.getmtime)
    if os.path.isfile(src_svg):
        shutil.copyfile(src_svg, dst_svg)
        print(f"Best sketch: {dst_svg}")
    else:
        print("WARNING: Could not locate best SVG to copy; check output folders.")


if __name__ == "__main__":
    main()
