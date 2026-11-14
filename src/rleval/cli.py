"""Local command-line entry points. No paid requests occur in doctor, install-core, or smoke."""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from .config import load_config

CORE_VERSION = "v5.0.0-rc18"
CORE_DIR = Path("runtime") / f"rlbot-{CORE_VERSION}"
CORE_EXE = CORE_DIR / "RLBotServer.exe"


def install_core():
    if sys.platform != "win32":
        raise RuntimeError(
            "This installer targets Windows. Install RLBot v5 for your OS separately."
        )
    url = f"https://api.github.com/repos/RLBot/core/releases/tags/{CORE_VERSION}"
    req = urllib.request.Request(url, headers={"User-Agent": "rocket-league-frontier-eval/0.1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        release = json.load(response)
    asset = next(x for x in release["assets"] if x["name"] == "RLBotServer.exe")
    download_url = asset["browser_download_url"]
    expected_url = f"https://github.com/RLBot/core/releases/download/{CORE_VERSION}/RLBotServer.exe"
    if download_url != expected_url:
        raise RuntimeError("Unexpected RLBot asset URL")
    expected_hash = asset.get("digest")
    if CORE_EXE.exists():
        digest = "sha256:" + hashlib.sha256(CORE_EXE.read_bytes()).hexdigest()
        if expected_hash and digest == expected_hash:
            print(f"Already installed and verified: {CORE_EXE.resolve()}")
            return
        raise RuntimeError(
            f"{CORE_EXE} already exists and could not be verified; leave it intact and inspect it manually."
        )
    print(f"Downloading official RLBot core {CORE_VERSION} ({asset['size']:,} bytes)...")
    with urllib.request.urlopen(download_url, timeout=30) as response:
        data = response.read(64 * 1024 * 1024)
    if len(data) != asset["size"]:
        raise RuntimeError("RLBot download size does not match the release metadata")
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if expected_hash and digest != expected_hash:
        raise RuntimeError("RLBot download SHA-256 does not match the release metadata")
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    CORE_EXE.write_bytes(data)
    (CORE_DIR / "download.json").write_text(
        json.dumps(
            {
                "version": CORE_VERSION,
                "url": download_url,
                "sha256": digest,
                "verified_against_github_digest": bool(expected_hash),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Installed: {CORE_EXE.resolve()}\n{digest}")


def doctor():
    from importlib.metadata import PackageNotFoundError, version

    import psutil

    packages = {}
    for name in ["rlbot", "rlbot-flatbuffers", "httpx", "mss", "Pillow"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "MISSING"
    processes = []
    for process in psutil.process_iter(["name", "pid", "cmdline"]):
        try:
            if (process.info["name"] or "").lower() == "rocketleague.exe":
                processes.append(
                    {
                        "pid": process.pid,
                        "offline_rlbot_flag": "-rlbot"
                        in [s.lower() for s in process.info["cmdline"] or []],
                    }
                )
        except (psutil.Error, OSError):
            pass
    candidates = [
        CORE_EXE.resolve(),
        Path(os.environ.get("LOCALAPPDATA", "")) / "RLBot5/bin/RLBotServer.exe",
    ]
    installations = []
    manifest_dir = (
        Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
        / "Epic/EpicGamesLauncher/Data/Manifests"
    )
    for path in manifest_dir.glob("*.item"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if "rocket league" in manifest.get("DisplayName", "").lower():
                installations.append(manifest.get("InstallLocation"))
        except (OSError, ValueError):
            pass
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "packages": packages,
                "core_candidates": {str(p): p.is_file() for p in candidates},
                "rocket_league_epic_installs": installations,
                "rocket_league_processes": processes,
                "keys_present": {
                    k: bool(os.environ.get(k))
                    for k in [
                        "OPENAI_API_KEY",
                        "ANTHROPIC_API_KEY",
                        "GEMINI_API_KEY",
                        "JEV_API_KEY",
                    ]
                },
                "next": "Run install-core if needed; use smoke to test without Rocket League or API usage.",
            },
            indent=2,
        )
    )


def compare(directory: str):
    rows = []
    for file in sorted(Path(directory).glob("*/summary.json")):
        row = json.loads(file.read_text(encoding="utf-8"))
        rows.append(
            {
                k: row.get(k)
                for k in [
                    "provider",
                    "model",
                    "synthetic_environment",
                    "observation_mode",
                    "control_scheduling",
                    "action_format",
                    "planner",
                    "controls_with_strategy",
                    "combined_usage_totals",
                    "status",
                    "elapsed_s",
                    "accepted_actions",
                    "mean_latency_s",
                    "p95_latency_s",
                    "achieved_decisions_hz",
                    "accepted_actions_hz",
                    "final_telemetry",
                ]
            }
        )
    print(json.dumps(rows, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate vision and decision models in offline Rocket League using RLBot v5."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="Check local installation and key presence without contacting models"
    )
    sub.add_parser(
        "install-core", help=f"Download official RLBot core {CORE_VERSION} into ./runtime"
    )
    sub.add_parser(
        "smoke", help="Run a short synthetic pipeline check; no game, credentials or API calls"
    )
    comparison = sub.add_parser("compare", help="Print summaries from a runs directory")
    comparison.add_argument("directory", nargs="?", default="runs")
    calibration = sub.add_parser(
        "calibrate", help="Run specified-start motor tests before full matches"
    )
    calibration.add_argument(
        "--config", type=Path, default=Path("configs/control-calibration.toml")
    )
    calibration.add_argument(
        "--policies",
        nargs="+",
        choices=["reference", "model", "motor"],
        default=["reference", "model"],
    )
    calibration.add_argument("--repeats", type=int, default=1)
    calibration.add_argument(
        "--scenario", action="append", help="Exact scenario name; repeat to select several"
    )
    calibration.add_argument("--output")
    calibration.add_argument("--server-path")
    calibration.add_argument("--launcher", choices=["Epic", "Steam"])
    calibration.add_argument("--model")
    calibration.add_argument("--live-window", action="store_true", default=None)
    run = sub.add_parser("run", help="Launch an offline evaluation episode")
    run.add_argument("--config", type=Path)
    run.add_argument(
        "--provider",
        choices=[
            "mock",
            "openai",
            "openai-decisions",
            "jev",
            "anthropic",
            "gemini",
            "openai-compatible",
        ],
    )
    run.add_argument(
        "--model", help="Exact model ID supported by your provider and observation mode"
    )
    run.add_argument("--api-key-env", help="Name of the environment variable holding the API key")
    run.add_argument("--base-url", help="Provider API base URL; never embed credentials")
    run.add_argument("--duration", type=float, help="Wall-clock episode duration in seconds")
    run.add_argument("--max-decisions", type=int)
    run.add_argument("--observation-mode", choices=["pixels", "pixels_and_state", "state"])
    run.add_argument("--control-scheduling", choices=["bounded", "continuous"])
    run.add_argument(
        "--max-decision-hz", type=float, help="Maximum request rate; one request stays in flight"
    )
    run.add_argument("--output", help="Runs directory")
    run.add_argument("--server-path", help="RLBot v5 core executable or directory")
    run.add_argument("--launcher", choices=["Epic", "Steam"])
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument(
        "--live-window",
        action="store_true",
        default=None,
        help="Open a separate live window showing model actions and match telemetry",
    )
    run.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic observations to test a provider; API calls still occur",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            doctor()
            return 0
        if args.command == "install-core":
            install_core()
            return 0
        if args.command == "compare":
            compare(args.directory)
            return 0
        if args.command == "calibrate":
            from .calibration import run_calibration
            from .calibration_tasks import default_scenarios

            config = load_config(
                args.config,
                {
                    "model": {"model": args.model},
                    "recording": {"output_dir": args.output, "live_window": args.live_window},
                    "rlbot": {"server_path": args.server_path, "launcher": args.launcher},
                },
            )
            if not config["rlbot"].get("server_path") and CORE_EXE.exists():
                config["rlbot"]["server_path"] = str(CORE_EXE.resolve())
            scenarios = default_scenarios()
            if args.scenario:
                known = {s.name for s in scenarios}
                if not set(args.scenario) <= known:
                    raise ValueError(f"Choose scenarios from: {', '.join(sorted(known))}")
                scenarios = [s for s in scenarios if s.name in args.scenario]
            _, summary = run_calibration(
                config, policies=args.policies, repeats=args.repeats, scenarios=scenarios
            )
            return int(summary["status"] != "completed")
        from .runner import run_episode

        if args.command == "smoke":
            config = load_config(overrides={"episode": {"duration_s": 3.0, "max_decisions": 3}})
            _, summary = run_episode(config, synthetic=True)
            return int(summary["status"] == "failed")
        if not 1 <= args.episodes <= 1000:
            raise ValueError("episodes must be in [1, 1000]")
        config = load_config(
            args.config,
            {
                "model": {
                    "provider": args.provider,
                    "model": args.model,
                    "api_key_env": args.api_key_env,
                    "base_url": args.base_url,
                },
                "episode": {"duration_s": args.duration, "max_decisions": args.max_decisions},
                "recording": {"output_dir": args.output, "live_window": args.live_window},
                "rlbot": {"server_path": args.server_path, "launcher": args.launcher},
                "observation": {"mode": args.observation_mode},
                "control": {
                    "scheduling": args.control_scheduling,
                    "max_decision_hz": args.max_decision_hz,
                },
            },
        )
        if config["model"]["provider"] != "mock" and config["model"]["model"] in {
            "deterministic-smoke",
            "YOUR_VISION_MODEL_ID",
        }:
            raise ValueError("Set --model to your exact model ID, or edit [model] in the config.")
        if not config["rlbot"].get("server_path") and CORE_EXE.exists():
            config["rlbot"]["server_path"] = str(CORE_EXE.resolve())
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}")
            _, summary = run_episode(config, synthetic=args.synthetic)
            if summary["status"] in {"failed", "interrupted"}:
                return 1 if summary["status"] == "failed" else 130
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
