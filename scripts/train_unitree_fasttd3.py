#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
import runpy
import site
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FASTTD3_DIR = ROOT / "fast_td3"
for path in (ROOT, FASTTD3_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from unitree_bridge import (
    add_fasttd3_script_path,
    add_unitree_source_path,
    FASTTD3_UNITREE_ALIAS,
    repo_root,
    UNITREE_LOG_EXPERIMENT_NAME,
    UNITREE_TASK,
)


FASTTD3_EXP_NAME = "UnitreeFastTD3"
FASTTD3_PROJECT = "UnitreeFastTD3"


def _prioritize_venv_site_packages() -> None:
    """Keep Isaac Sim's pip_prebundle from shadowing installed Python deps."""
    site_paths: list[str] = []
    for path in site.getsitepackages():
        site_paths.append(str(Path(path).resolve()))
    user_site = site.getusersitepackages()
    if user_site:
        site_paths.append(str(Path(user_site).resolve()))

    insert_at = 0
    protected_paths = {str(ROOT), str(FASTTD3_DIR)}
    for idx, path in enumerate(sys.path):
        if str(Path(path or ".").resolve()) in protected_paths:
            insert_at = idx + 1

    for site_path in reversed(site_paths):
        for existing in list(sys.path):
            if str(Path(existing or ".").resolve()) == site_path:
                sys.path.remove(existing)
                sys.path.insert(insert_at, existing)

    typing_extensions = sys.modules.get("typing_extensions")
    if typing_extensions is None:
        return
    module_file = getattr(typing_extensions, "__file__", "")
    if "pip_prebundle" in module_file:
        del sys.modules["typing_extensions"]


def _extract_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--unitree_rl_lab_path", default=None)
    parser.add_argument("--task", default=UNITREE_TASK)
    parser.add_argument("--experiment_name", default=UNITREE_LOG_EXPERIMENT_NAME)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_root_path", default=None)
    return parser.parse_known_args(argv)


def _remove_task_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == "--task":
            skip = True
            continue
        if arg.startswith("--task="):
            continue
        cleaned.append(arg)
    return cleaned


def _has_arg(argv: list[str], *names: str) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in argv)


def _append_default(argv: list[str], name: str, value: str) -> list[str]:
    alt_name = name.replace("_", "-")
    if not _has_arg(argv, name, alt_name):
        argv.append(name)
        if value:
            argv.append(value)
    return argv


def _append_bool_default(
    argv: list[str],
    enabled_name: str,
    disabled_name: str,
    enabled: bool,
    base_default: bool,
) -> list[str]:
    names = (
        enabled_name,
        enabled_name.replace("_", "-"),
        disabled_name,
        disabled_name.replace("_", "-"),
    )
    if _has_arg(argv, *names) or enabled == base_default:
        return argv
    argv.append(enabled_name if enabled else disabled_name)
    return argv


def _set_env_name(argv: list[str], env_name: str) -> list[str]:
    out: list[str] = []
    i = 0
    changed = False
    while i < len(argv):
        arg = argv[i]
        if arg in ("--env_name", "--env-name"):
            out.extend([arg, env_name])
            i += 2
            changed = True
            continue
        if arg.startswith("--env_name="):
            out.append(f"--env_name={env_name}")
            changed = True
            i += 1
            continue
        if arg.startswith("--env-name="):
            out.append(f"--env-name={env_name}")
            changed = True
            i += 1
            continue
        out.append(arg)
        i += 1
    if not changed:
        out.extend(["--env_name", env_name])
    return out


def main() -> None:
    launcher_args, train_args = _extract_launcher_args(sys.argv[1:])
    if launcher_args.task != UNITREE_TASK:
        raise ValueError(f"Only {UNITREE_TASK!r} is wired for this launcher")

    _prioritize_venv_site_packages()
    add_fasttd3_script_path()
    unitree_root = add_unitree_source_path(launcher_args.unitree_rl_lab_path)
    os.environ["UNITREE_RL_LAB_PATH"] = str(unitree_root)

    log_root_path = (
        Path(launcher_args.log_root_path).expanduser().resolve()
        if launcher_args.log_root_path
        else unitree_root / "logs" / "rsl_rl" / launcher_args.experiment_name
    )
    run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if launcher_args.run_name:
        run_dir_name += f"_{launcher_args.run_name}"
    log_dir = log_root_path / run_dir_name

    train_args = _remove_task_args(train_args)
    train_args = _set_env_name(train_args, FASTTD3_UNITREE_ALIAS)
    if not _has_arg(train_args, "--exp_name", "--exp-name"):
        train_args.extend(["--exp_name", FASTTD3_EXP_NAME])
    if not _has_arg(train_args, "--project"):
        train_args.extend(["--project", FASTTD3_PROJECT])
    train_args = _append_default(train_args, "--save_dir", str(log_dir))
    train_args = _append_default(train_args, "--checkpoint_prefix", "model")
    train_args = _append_bool_default(
        train_args,
        "--save_final_as_step",
        "--no_save_final_as_step",
        True,
        False,
    )
    train_args = _append_bool_default(
        train_args,
        "--export_unitree_params",
        "--no_export_unitree_params",
        True,
        False,
    )

    print(f"[INFO] FastTD3 Unitree log directory: {log_dir}")
    sys.argv = [str(repo_root() / "fast_td3" / "train.py"), *train_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
