# Basic control calibration

Use this suite to find out whether the model can turn, straighten, reach a target, and stop before asking it to play a match. Each trial starts from a specified car position, velocity, and heading in a solo offline RLBot Soccar session with calibration mode enabled. There is no opponent, and the ball is reset far from the test area. Success comes from measured game telemetry, not from the model's description of its behavior.

Calibration is an explicitly state-assisted evaluation. It does not measure screenshot perception, ball play, aerial control, or match strategy. The source release includes task definitions and scoring code, without private run results.

For the recommended strategy-assisted track, start with the continuous local motor controller:

```powershell
uv run rleval calibrate --policies motor --live-window
```

This uses the same eight starting scenarios and scoring rules described below, with no model requests. It targets 60 Hz and uses continuous axes; it is a separate track from the discrete reference/model diagnostic. Use `motor` alone rather than mixing it with other policies. See [ASSISTED.md](ASSISTED.md) for its mechanics and comparison limits.

## Run it

Complete the setup and RLBot installation in [README.md](README.md). Close an ordinary Rocket League session and other RLBot sessions first. The harness launches its own offline game. Open PowerShell in the project directory:

```powershell
uv run rleval calibrate --policies reference
```

This runs all eight scenarios with a local, hand-written reference controller and makes no model API requests. It checks whether reset, coordinates, controller delivery, and scoring work together. A reference failure is a reason to inspect the trace before drawing a conclusion about a model.

With `OPENAI_API_KEY` available in the terminal environment:

```powershell
uv run rleval calibrate --config configs\control-calibration.toml
```

The default policy list is `reference model`. For each scenario, the reference runs first and the model runs next, each with its own reset. The supplied preset uses OpenAI Decisions with `gpt-6-luna`, compact state, a maximum request rate of 5 Hz, and a 300 ms input lease. Check model and API availability for your account, or supply `--model YOUR_DECISIONS_MODEL_ID`. A larger strategist is disabled: these trials isolate the fast controller. The model phase makes billable API requests.

The preset opens the live outputs window. It displays choices and controls as the trial runs; it does not invent reasoning text. Use its **Stop run** button or **Ctrl+C in the terminal** to stop the suite. Closing only the outputs window does not stop the suite.

Useful variations:

```powershell
# Test only the model, using another Decisions model available to your account.
uv run rleval calibrate --policies model --model YOUR_DECISIONS_MODEL_ID

# Repeat the complete suite three times.
uv run rleval calibrate --repeats 3

# Focus on mirrored turns or on braking.
uv run rleval calibrate --scenario turn_left --scenario turn_right
uv run rleval calibrate --scenario brake_forward --scenario brake_reverse

# Use Steam or an existing compatible RLBot server.
uv run rleval calibrate --launcher Steam
uv run rleval calibrate --server-path "PATH_TO_RLBOT_SERVER_EXE"
```

Without `uv`, replace `uv run rleval` with `.\.venv\Scripts\rleval.exe`. Repeats must be between 1 and 10. A complete set has at most 54 seconds of trial time per controller per repeat, plus game startup, resets, and cleanup. Passing trials finish early. The game keeps moving during API requests; 5 Hz is a request-rate ceiling, not a guaranteed response rate.

To try Jev, copy `configs/control-calibration.toml` to another TOML file and change `[model] provider` to `"jev"`, `model` to an available Jev model ID, and `api_key_env` to `"JEV_API_KEY"`. Run `calibrate --config` with that file. Keep `action_interface = "ground_axes_v1"`. Both typed adapters support the same two questions; ordinary JSON providers are not interchangeable with this calibration interface.

## Eight scenarios

All pass conditions must remain true for **250 ms** while the car is on the flat floor and the game is ready. Success requires distinct game packets no more than **100 ms old**; duplicate packets do not advance the success dwell, and a gap longer than 100 ms without a fresh packet resets it. A frozen successful-looking observation cannot establish a pass. Distances are Rocket League units (`uu`); speeds are units per second. Turn and alignment trials also require an absolute yaw rate of at most 0.6 radians/second (about 34.4 degrees/second), so briefly swinging through the correct heading is insufficient.

| Scenario | Starting condition and goal | Pass condition | Time limit |
| --- | --- | --- | --- |
| `turn_left` | Moving forward at 600 uu/s; target initially 30° left | Absolute heading error ≤8° and yaw-rate limit | 6 s |
| `turn_right` | Moving forward at 600 uu/s; target initially 30° right | Absolute heading error ≤8° and yaw-rate limit | 6 s |
| `align_left` | Moving forward at 350 uu/s; target initially 60° left; requested cruise speed 450 uu/s | Absolute heading error ≤5° and yaw-rate limit | 8 s |
| `align_right` | Moving forward at 350 uu/s; target initially 60° right; requested cruise speed 450 uu/s | Absolute heading error ≤5° and yaw-rate limit | 8 s |
| `waypoint_left` | Stationary; target 2,000 uu ahead and 600 uu left; requested cruise speed 900 uu/s | Within 250 uu of target, with total speed <250 uu/s | 8 s |
| `waypoint_right` | Stationary; target 2,000 uu ahead and 600 uu right; requested cruise speed 900 uu/s | Within 250 uu of target, with total speed <250 uu/s | 8 s |
| `brake_forward` | Moving forward at 1,000 uu/s; no target | Total speed <80 uu/s | 5 s |
| `brake_reverse` | Moving backward at 800 uu/s; no target | Total speed <80 uu/s | 5 s |

The requested speed is guidance, not a separate pass criterion. In waypoint trials, the task explicitly asks the controller to brake near the target; the supplied cruise-speed value stays constant, and the prompt gives the stopping criterion priority. Turn and alignment success is about stable direction, not reaching the distant waypoint. Exact initial vectors and task descriptions are stored in the run metadata.

## What the model sees and controls

The `compact-motor-v1` observation contains:

- The task, requested speed, target distance, and target position relative to the car. The bearing is in degrees: **positive is right, negative is left**. Braking trials have no target.
- Signed forward and lateral speed, vertical speed, and yaw rate in degrees/second.
- Wheel contact, height, the up-vector's vertical component, and a derived `floor`, `wall_or_ramp`, `ceiling`, or `air` label. Wheel contact alone does not mean the car is on the floor.
- The last input acknowledged in a game packet, the current command lease, packet age, and up to three recent measured state transitions.

The fast model receives this compact observation without screenshots or the full game-state object. These are measured current values and recent history, not a future trajectory or a driving policy.

`ground_axes_v1` asks two choice questions in the same request:

| Question | Choice | Direct controller value |
| --- | --- | --- |
| Steering | `left` | `steer = -0.5` |
| Steering | `straight` | `steer = 0` |
| Steering | `right` | `steer = +0.5` |
| Longitudinal | `accelerate` | `throttle = +1` |
| Longitudinal | `coast` | `throttle = 0` |
| Longitudinal | `brake` | Throttle opposite the observed forward motion; zero when its magnitude is below 30 uu/s |

The two choices are independent: `right / coast` keeps steering right while releasing throttle. Steering signs describe forward driving; reversing changes the turning response. There is no automatic target pursuit or steering correction after the choices. Pitch, yaw, roll, jump, boost, and handbrake remain neutral. Existing full-match configurations keep their larger catalog, including boost and jump.

Each command expires to neutral after its lease, unless a newer accepted command replaces it sooner. Invalid, refused, incomplete, or stale replies are rejected. Each question retains its own confidence and probability distribution. Any combined confidence shown is the smaller of the two confidences, not their joint probability or a prediction of trial success.

## Reading the feedback and results

The run prints a directory under `calibration-runs/`. Open `report.html` for the trial-by-trial pass/fail table and follow its link to `replay.html` for telemetry and action traces. `summary.json` preserves per-policy totals and per-trial metrics; `events.jsonl` records requests, decisions, trial boundaries, and sampled observations. `metadata.json` records the configuration and initial scenarios.

Inspect final heading error, target distance, speed, time to success, accepted actions, and the rotation metrics together. The `circling` flag means the measured net yaw change reached at least one full rotation. It does not detect every oscillation or every curved path. `total_absolute_yaw_rad` also records accumulated turning in either direction.

Braking passes measure a stable stop, not the complete absence of reversal along the way. Inspect `min_forward_speed_uu_s`, `max_forward_speed_uu_s`, and `reverse_overshoot_speed_uu_s` alongside pass/fail. The last metric is the peak speed opposite the trial's initial direction: zero means no measured reversal; a positive value quantifies how far braking overshot into reverse. `fresh_samples` records distinct fresh packets used by the scorer, and `max_success_packet_age_s` records its freshness limit.

The feedback distinguishes three things:

- **Selected command:** what the controller requested and how long the harness intended to hold it.
- **Command lease:** whether the harness still considers that command active, how long it was commanded, and whether expiry or replacement ended it. These are wall-clock measurements, not engine acknowledgements.
- **Observed transition:** displacement, heading change, speed before/after, and acknowledged inputs across a measured interval. One interval can contain several commands; its change must not be attributed to a single action without checking the timeline.

The reference controller is a separate diagnostic policy with hand-written steering damping and stopping-distance logic. That logic is never used to adjust a model's choices. Both use the same discrete inputs, specified resets, request-rate ceiling, and hold limit. **The reference has no model API latency**, however, so a better reference score does not isolate reasoning quality from timing. The summary's `mixed_policy_pipeline` combines both streams and should not be treated as a model-only latency statistic.

## What a result can tell us

Mirrored left/right trials help expose a direction-sign error. Successful turns followed by failed waypoints point toward speed regulation or stopping behavior. Reference success with model failure narrows the problem toward the model's decisions, observations, or request latency; it does not prove which one is responsible. Compare traces and repeat the trials before making that attribution.

The reduced interface can still overshoot. Steering has only three values, API replies arrive after their observation, and an input can remain active for 300 ms. Braking chooses its throttle direction from the supplied observation; momentum can cross zero before a fresh command arrives. These effects remain visible in the logs rather than being corrected by a hidden navigation controller.

Specified resets make trials more repeatable, but this is a live game, not a deterministic pause-and-step simulator. Match versions, physics timing, packet delivery, API timing, and model aliases can change outcomes. Retain the exact resolved model, configuration, request timing, repeated-trial results, and run artifacts when comparing systems. Passing these tasks is evidence of basic control under the recorded conditions, not evidence that the model can play a full Rocket League match.
