# RL (Rocket League) Eval

A Windows evaluation harness for models playing **offline Rocket League through RLBot v5**. Give a model screenshots, measured game state, or both; record its decisions, controls, latency, usage, and game telemetry in a local replay.

Start with the **strategy-assisted track**: a model chooses typed tactical goals while a local controller handles steering, throttle, boost, short jumps, and basic recovery. A scripted-tactics baseline uses the same controller without model API calls. Experimental direct-control tracks let models choose the actual controls themselves.

This is an experimental evaluation tool with a modest scripted controller. It does not establish model playing strength, provide expert mechanics, or implement an online-play bot. The public source release contains no recorded matches, screenshots, private run results, or credentials.

## Setup

Live gameplay requires Windows, Python 3.11 or newer, Rocket League installed through Epic Games Launcher or Steam, and a compatible RLBot v5 server. Sign in and complete the game's normal first-run setup before using the harness. `uv` is recommended.

Open PowerShell in the repository:

```powershell
.\setup.ps1
.\.venv\Scripts\rleval.exe doctor
.\.venv\Scripts\rleval.exe smoke
.\.venv\Scripts\rleval.exe install-core
```

Or install the locked environment directly:

```powershell
uv sync --extra dev
uv run rleval doctor
uv run rleval smoke
uv run rleval install-core
```

`doctor` checks dependencies, game installation, RLBot server candidates, and whether key environment variables are present. It does not print key values. `smoke` runs the pipeline with generated images and mock actions; it needs no game or API key and makes no model requests. Synthetic reports are labeled as such.

`install-core` downloads the pinned official RLBot server to `runtime/`, checks its published size, and verifies its SHA-256 digest when the release supplies one. The Python interface is pinned to `rlbot==2.0.0b56`. The game, server, and interface versions must remain compatible; game updates can require a dependency update.

Before a live run, close ordinary Rocket League and other RLBot sessions. The harness launches its own offline match and rejects an existing game process that it cannot verify was started with `-rlbot`. Add `--launcher Steam` for Steam, or `--server-path "PATH_TO_RLBOT_SERVER_EXE"` to select a compatible existing server.

Without `uv`, replace `uv run rleval` in the commands below with `.\.venv\Scripts\rleval.exe` after running `setup.ps1`.

## Start with local motor assistance

First check the controller and try an offline match without model requests:

```powershell
uv run rleval calibrate --policies motor --live-window
uv run rleval run --config configs\scripted-assisted-full-match.toml
```

To use a model strategist, make `OPENAI_API_KEY` available in the terminal environment and set **`[planner] model`** in `configs/sol-assisted-full-match.toml` to an exact model ID available to your account. Set or remove `reasoning_effort` according to that model's supported options. Then run:

```powershell
uv run rleval run --config configs\sol-assisted-full-match.toml
```

The bundled preset selects GPT-6.1 Sol; model IDs and API access depend on your account. The strategist currently uses OpenAI Responses and structured output. The generic `--model` flag selects a direct controller model, so change the assisted strategist in its `[planner]` configuration instead.

Both assisted presets use measured state, a 60 Hz local control target, a built-in Rookie opponent, normal game speed, and a five-minute match. A 900-second wall-clock ceiling allows goal pauses and overtime. The model preset makes billable planner requests; the scripted baseline does not call a model. The game keeps moving during inference.

The live window separates strategy from local controls. Its **Stop run** button requests shutdown, releases controls, and saves the result. `Ctrl+C` in the run terminal also stops the episode. Closing only the outputs window leaves the episode running.

Reports label this track `strategy_assisted_controls`. Local tick counts and compute times are separate from planner request counts, latency, and token usage. The replay samples local controls at approximately 10 Hz; it does not store every motor tick. See [ASSISTED.md](ASSISTED.md) for skills, fallbacks, timing, and fair baseline comparisons.

## Experimental direct model controls

Direct-control tracks use screenshots or state observations and ask a model for controller axes/buttons. They include optional hierarchical planning, where a strategist supplies a plan to another model that chooses the controls. These tracks have different action spaces and assistance levels; compare them separately.

| Configuration | Interface | Credential environment variable |
| --- | --- | --- |
| `configs/openai.toml` | OpenAI Responses; screenshot input | `OPENAI_API_KEY` |
| `configs/anthropic.toml` | Anthropic Messages; screenshot input | `ANTHROPIC_API_KEY` |
| `configs/gemini.toml` | Gemini generateContent; screenshot input | `GEMINI_API_KEY` |
| `configs/compatible.toml` | OpenAI-compatible Chat Completions | `COMPATIBLE_API_KEY` |
| `configs/state-openai.toml` | OpenAI Responses; measured state | `OPENAI_API_KEY` |
| `configs/openai-decisions.toml` | OpenAI Decisions; finite controller choices | `OPENAI_API_KEY` |
| `configs/jev.toml` | Jev; finite controller choices | `JEV_API_KEY` |
| `configs/offline-smoke.toml` | Deterministic mock driving in the actual game | None |

Use an exact model ID supported by your provider and observation mode. Screenshot input needs a vision-capable model. Presets are examples, not promises of API or model availability.

```powershell
uv run rleval run --config configs\openai.toml --model YOUR_OPENAI_VISION_MODEL_ID
uv run rleval run --config configs\anthropic.toml --model YOUR_CLAUDE_VISION_MODEL_ID
uv run rleval run --config configs\gemini.toml --model YOUR_GEMINI_VISION_MODEL_ID
uv run rleval run --config configs\compatible.toml --model YOUR_VISION_MODEL_ID --base-url http://localhost:8000/v1
```

Compatible endpoints must support JSON-schema response formatting and, for screenshot modes, image inputs. Set `api_key_env = ""` for a local endpoint that needs no key. Remote endpoints require HTTPS. Supply an API root rather than a full completion URL, and never embed credentials in URLs or TOML files. No `.env` file is loaded automatically.

The experimental full-match presets are `openai-decisions-full-match.toml` and `openai-hierarchical-full-match.toml`. The latter uses a separate strategist and a Decisions controller; unlike the assisted track, it does not execute plans through the local motor controller. Inspect their request budgets before running: full matches can make thousands of billable requests.

For basic turning, alignment, waypoint, and braking diagnostics:

```powershell
uv run rleval calibrate --policies reference
uv run rleval calibrate --config configs\control-calibration.toml --model YOUR_DECISIONS_MODEL_ID
```

The reference-only command makes no model requests. The second runs reference and model trials using the same discrete ground-control interface. See [CALIBRATION.md](CALIBRATION.md) for exact scenarios and interpretation.

## Observations and timing

| Option | Behavior |
| --- | --- |
| `--observation-mode pixels` | Game-window screenshots; measured state stays with the evaluator |
| `--observation-mode pixels_and_state` | Screenshots and allowlisted measured state |
| `--observation-mode state` | Measured state without screenshot capture |
| `--control-scheduling bounded` | Wait for an action's hold to finish before requesting another |
| `--control-scheduling continuous` | Allow a new accepted decision to replace an active action |
| `--max-decision-hz 5` | Cap model request frequency; does not guarantee response frequency |

Screenshot modes require Rocket League visible and in the foreground, preferably Windowed or Borderless. Capture targets the game window and has no desktop fallback. The default screenshot observation includes up to three recent JPEG frames, task context, prior actions, and a short model memory note. State observations include current positions, velocities, orientation, and other selected gameplay fields.

Direct JSON adapters return continuous axes in `[-1, 1]`, button values, a hold of 50–1,000 milliseconds, and an optional short memory note. Typed match adapters select from a 35-entry direct-controller catalog that includes boost and jump. Calibration uses a separate, smaller ground-control catalog. The local motor has its own continuous controller interface.

Inputs expire to neutral, and invalid or stale responses are rejected. Only one request per model worker is in flight. API latency remains part of the experiment. Planner requests run separately from controls; local motor assistance uses a separate control loop and explicitly labeled fallback behavior.

Ordinary episodes start fresh offline matches. The harness does not provide deterministic pause/step simulation, audio input, menu control, or arbitrary complete-match snapshot restoration. Calibration supports its own specified car/ball resets. Keep game speed and all other experimental settings fixed across comparisons.

## Outputs and privacy

Each run prints a local directory under `runs/`; calibration uses `calibration-runs/`. Open its `report.html` for a screenshot or state replay. No web server is needed.

| File | Contents |
| --- | --- |
| `report.html` | Replay, decisions, modes, and configuration |
| `summary.json` | Completion status, counters, latency, usage, and final telemetry |
| `events.jsonl` | Timestamped requests, decisions, samples, telemetry, and termination |
| `metadata.json` | Resolved configuration, package versions, source hash, and timing modes |
| `frames/` | Sampled screenshots when enabled |
| Generated TOML files and server log | Local match configuration and diagnostics |

Generated files can contain screenshots, game/player information, model notes, local paths, configuration, and timestamps. Review and sanitize them before sharing. API keys belong in environment variables; do not place secrets in task text, model notes, configuration, or custom endpoints. Run directories, runtime downloads, virtual environments, and secret files are excluded from version control.

To inspect multiple run summaries:

```powershell
uv run rleval compare runs
```

Compare matching tasks, observation modes, action interfaces, assistance, opponent settings, game speed, timing, memory, and model configuration. Repeat trials: starting conditions and game trajectories are not guaranteed deterministic. Calibration success and synthetic checks do not establish match strength. The harness does not compute a cross-run skill rating or confidence intervals.

Duration, request, and output-token caps limit activity rather than dollar spend. Token accounting depends on provider responses, and an in-flight request may remain billable after a run stops. `smoke` is free of model requests; `run --synthetic` still calls the selected provider and can incur charges.

## Development and scope

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

`src/rleval/` contains the CLI, configuration, RLBot backend, capture, provider adapters, planners, controllers, runners, and recorder. `tests/` covers these components with synthetic data and mocked requests. A new direct-control provider implements `decide(frames, context) -> ModelReply` and `close()`, and is registered in `build_client`.

This project uses RLBot's offline integration. Online private matches are a different environment; the harness provides no online-play support or anti-cheat bypass. Consult the current game and RLBot guidance for your installation. No game assets or RLBot server binaries are bundled in the source release.

References: [RLBot documentation](https://wiki.rlbot.org/v5/), [Python interface](https://github.com/RLBot/python-interface), [core releases](https://github.com/RLBot/core/releases), [OpenAI Responses](https://developers.openai.com/api/reference/resources/responses/methods/create), [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions), [Jev API](https://api.typesafe.ai/redoc), [Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create), [Gemini generateContent](https://ai.google.dev/api/generate-content).

Released under the [MIT License](LICENSE). See [PRIVACY.md](PRIVACY.md) for what runs record and send to model providers, and how to review artifacts before sharing them.
