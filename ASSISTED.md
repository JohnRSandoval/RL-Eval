# Strategy evaluation with local motor assistance

This track asks a model to choose tactics while a **scripted local controller** handles steering, speed, boost, and basic recovery from fresh RLBot state. It measures strategy with motor assistance. It does not measure whether the model can produce precise real-time button presses on its own.

The local controller targets **60 Hz** and uses continuous axis values. OpenAI Decisions and GPT-6 Luna are not called in this track. GPT-6.1 Sol, when enabled, uses the OpenAI Responses API for periodic tactical plans.

This is the recommended starting track for strategy experiments. The controller provides limited ground mechanics; the presets do not establish playing strength or a strategic advantage. Private run results are not included in the source release.

## Run the two presets

From the project directory, with the offline RLBot installation ready:

```powershell
# Sol chooses tactics; the local motor controller executes them.
uv run rleval run --config configs\sol-assisted-full-match.toml

# No model API calls: scripted tactics use the same motor controller.
uv run rleval run --config configs\scripted-assisted-full-match.toml
```

The Sol preset reads `OPENAI_API_KEY` from the environment. Check that its `[planner] model` is available to your account, or change it before running. The scripted preset needs no model credential. Both start a five-minute offline match against a built-in Rookie opponent at normal game speed, show the blue player's camera for the human viewer, and open the live outputs window. The model observation is measured state without screenshots.

The 900-second wall-clock limit allows goal pauses and overtime; Rocket League's match-ended signal normally stops the run first. The Sol preset allows at most 300 planner requests, each with a 2,048-output-token cap. Those limits bound activity, not dollar spend. Only the strategist makes model API requests in this track.

The live window's **Stop run** button writes a stop request that the runner handles by releasing controls and saving the result. It displays “Stop requested” until termination is recorded. `Ctrl+C` in the run terminal also stops the episode. Closing just the outputs window leaves the match running.

## Swap the strategist independently

Copy `configs/sol-assisted-full-match.toml` and edit its `[planner]` section:

```toml
[planner]
enabled = true
provider = "openai"
model = "YOUR_OPENAI_MODEL_ID"
control_mode = "local_motor"
api_key_env = "OPENAI_API_KEY"
interval_s = 5.0
max_age_s = 15.0
max_requests = 300
```

Use an exact model ID supported by your account and the planner's structured-output request. Keep or remove `reasoning_effort` according to that model's supported options. The strategist adapter currently uses OpenAI Responses. The run command's generic `--model` flag selects the direct controller model; edit **`[planner] model`** to change the assisted strategist.

Keep the following controller settings identical when comparing strategists:

```toml
[control]
mode = "local_motor"
scheduling = "continuous"
motor_hz = 60.0
motor_max_packet_age_s = 0.1
```

Sol receives current measured state and recent motor feedback. It returns a typed skill, optional target position, desired speed, and boost/jump policies, plus an objective and guidance text. The motor executes the typed fields. The objective and guidance remain available for review; the motor does not interpret arbitrary prose into new mechanics.

The planner requests a new plan on a five-second cadence, with one request in flight. A separate local loop keeps controlling the car while the network request runs. Plans expire 15 seconds after their observation and are invalidated by gameplay interruptions. Without a valid plan, the motor uses a labeled conservative defensive fallback. The scripted baseline instead selects its own tactics from current state using the same motor implementation.

## What the local controller provides

| Skill | Implemented behavior |
| --- | --- |
| `drive_to` | Steer toward a supplied ground waypoint and reduce speed near the target |
| `intercept` | Recompute a ground target using current ball position and velocity |
| `shoot` | Aim a ground approach relative to the ball and opposing goal |
| `defend` | Recompute a defensive ground position between the ball and own goal |
| `recover` | Attempt to leave a wall or level the car in the air; brake when back on the floor |
| `stop` | Brake toward a stop |

Ball-related targets are recomputed every local tick. Their prediction is a short **constant-velocity XY estimate**, without bounce simulation. Steering, speed limiting, goal-mouth routing, short ground jumps, and recovery are explicit mathematical rules. Boost/jump policies permit or restrict those rules; they do not force an action regardless of geometry.

This is a modest scripted controller, not a trained Rocket League expert. It does not solve advanced aerial shots, wall play, or dodges. Recovery attempts and approximate intercepts can fail. The local loop neutralizes inputs on inactive gameplay, stale state, or invalid inputs; a 60 Hz target is not a guarantee of 60 achieved ticks per second.

## Check basic motor behavior first

```powershell
uv run rleval calibrate --policies motor --live-window
```

This runs the calibration scenarios with the local motor controller and no model API calls. Use `--repeats` for repeated trials, or `--scenario NAME` to select a known scenario. Read the per-task outcomes and trajectory evidence before interpreting match results.

The model/reference calibration uses a nine-choice ground-control interface at a 5 Hz ceiling. The local motor uses continuous axes at a 60 Hz target. Differences in success rates between those tracks are **not a controlled model comparison**: action space, control frequency, and scripted assistance differ.

To estimate the strategist's contribution, compare the Sol-assisted preset with the scripted-assisted baseline while keeping the motor source, controller rate, tasks, opponent, game speed, and stopping rules fixed. Repeat matches because starts and subsequent game trajectories are not guaranteed deterministic. No playing-strength result is established by these presets alone.

## Read the outputs

The live window identifies **local controller** output separately from the **strategy model**. It shows the accepted strategy, active local skill, policy source, actual controls, target, score, and match clock. It does not display confidence or API latency as if a local controller tick were a model decision.

Reports and metadata label the track as `strategy_assisted_controls` and the action format as `local_motor_controller`. The recorded streams are separate:

| Evidence | Meaning |
| --- | --- |
| `planner_request`, `planner_decision` | Actual model requests, accepted/rejected plans, latency, and returned token usage |
| `planner_invalidated` | A plan expired or was cleared after gameplay changed |
| `motor_sample` | Sampled local controls, skill, target, status, plan version, and policy source |
| `motor_summary` / `summary.motor` | Actual tick count, active ticks, target/achieved rate, compute time, source counts, and errors |
| `telemetry` / `final_telemetry` | Recorded game state and available goals, touches, shots, saves, and score |

`motor_sample` is recorded at approximately **10 Hz** for the replay and live display. It is a sampled action trace, **not every 60 Hz controller tick**. The motor summary records the full loop's tick totals and measured rate. Planner latency and motor compute time use different clocks and remain separate metrics.

Active match samples label their policy source as `planner`, `scripted_baseline`, or `fallback`; paused/countdown samples are `inactive`, and motor calibration uses `calibration`. Check these sources when attributing an outcome to Sol: time spent in fallback is part of the assisted system's result. Controller ticks do not inflate `completed_decisions`, `accepted_actions`, or model token usage. Existing direct-model reports retain their original meanings.
