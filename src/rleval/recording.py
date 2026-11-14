"""Append-only run evidence and a portable, local HTML replay."""

import hashlib
import json
import math
import re
import statistics
import uuid
from datetime import UTC, datetime
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)


def _usage_totals(usages: list[dict]) -> dict:
    return {
        key: sum(u[key] for u in usages if type(u.get(key)) is int and u[key] >= 0)
        for key in USAGE_KEYS
    }


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


def _measurements(events: list[dict], key: str) -> list[float]:
    return [
        float(e[key])
        for e in events
        if type(e.get(key)) in (int, float) and math.isfinite(e[key]) and e[key] >= 0
    ]


def _metric(values: list[float], kind: str = "mean") -> float | None:
    if not values:
        return None
    if kind == "p95":
        value = sorted(values)[math.ceil(len(values) * 0.95) - 1]
    elif kind == "median":
        value = statistics.median(values)
    else:
        value = statistics.mean(values)
    return round(value, 6)


class Recorder:
    def __init__(self, output_dir: str | Path, config: dict, *, synthetic: bool):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(output_dir).resolve() / f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True)
        (self.path / "frames").mkdir()
        self._file = (self.path / "events.jsonl").open("w", encoding="utf-8")
        self.events = []
        packages = {}
        for name in ["rlbot", "rlbot-flatbuffers", "httpx", "mss", "Pillow"]:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "not installed"
        digest = hashlib.sha256()
        for source in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
        observation_mode = config.get("observation", {}).get("mode", "pixels")
        scheduling = config.get("control", {}).get("scheduling", "bounded")
        typed = config["model"]["provider"] in {"openai-decisions", "jev"}
        planner = config.get("planner", {})
        hierarchical = bool(planner.get("enabled", False))
        motor = config.get("control", {}).get("mode", "model") == "local_motor"
        self.metadata = {
            "schema_version": 2,
            "started_at": datetime.now(UTC).isoformat(),
            "synthetic_environment": synthetic,
            "config": config,
            "packages": packages,
            "source_sha256": digest.hexdigest(),
            "hierarchical": hierarchical,
            "planner_model": planner.get("model") if hierarchical else None,
            "evaluation_track": "strategy_assisted_controls" if motor else "model_controls",
            "control_mode": "local_motor" if motor else "model",
            "motor_control_hz": config.get("control", {}).get("motor_hz", 60) if motor else None,
            "observation_mode": observation_mode,
            "observation_representation": config.get("observation", {}).get(
                "representation", "full"
            ),
            "control_scheduling": scheduling,
            "max_decision_hz": config.get("control", {}).get("max_decision_hz", 10.0),
            "action_format": (
                "local_motor_controller"
                if motor
                else "ground_axes_v1"
                if config["model"].get("action_interface") == "ground_axes_v1"
                else "finite_direct_controller_catalog"
                if typed
                else "controller_json"
            ),
            "timing_mode": "real_time"
            if config["rlbot"].get("game_speed", 1.0) == 1
            else "slowed_real_time",
            "control_policy": (
                "scripted local motor controller executes low-level mechanics; planner directives are optional"
                if motor
                else "refresh decisions while an action is active; neutral on expiry; no scripted mechanics"
                if scheduling == "continuous"
                else "wait for action completion before deciding; neutral on expiry; no scripted mechanics"
            ),
        }
        (self.path / "metadata.json").write_text(json_text(self.metadata), encoding="utf-8")

    def event(self, kind: str, elapsed_s: float, **fields):
        event = {"type": kind, "elapsed_s": round(elapsed_s, 6), **fields}
        self._file.write(json_text(event) + "\n")
        self._file.flush()
        self.events.append(event)

    def frame(self, number: int, jpeg: bytes) -> str | None:
        if not self.metadata["config"]["recording"]["save_frames"]:
            return None
        name = f"frames/{number:06d}.jpg"
        (self.path / name).write_bytes(jpeg)
        return name

    def finish(self, status: str, elapsed_s: float, final: dict, error: str | None = None) -> dict:
        decisions = [e for e in self.events if e["type"] == "decision"]
        latencies = _measurements(decisions, "latency_s")
        ages = _measurements(decisions, "observation_age_s")
        accepted = sum(e.get("accepted", False) for e in decisions)
        usages = [e["usage"] if isinstance(e.get("usage"), dict) else {} for e in decisions]
        planner_decisions = [e for e in self.events if e["type"] == "planner_decision"]
        planner_usages = [
            e["usage"] if isinstance(e.get("usage"), dict) else {} for e in planner_decisions
        ]
        controller_usage = _usage_totals(usages)
        planner_usage = _usage_totals(planner_usages)
        details = [
            e["model_decision"] for e in decisions if isinstance(e.get("model_decision"), dict)
        ]
        catalog_versions = sorted(
            {str(d["catalog_version"]) for d in details if d.get("catalog_version")}
        )
        resolved_models = sorted(
            {str(d["resolved_model"]) for d in details if d.get("resolved_model")}
        )
        motor_summaries = [e for e in self.events if e["type"] == "motor_summary"]
        motor = motor_summaries[-1] if motor_summaries else {}
        motor_metrics = {
            key: motor.get(key)
            for key in (
                "ticks",
                "elapsed_s",
                "worker_elapsed_s",
                "active_ticks",
                "mean_compute_ms",
                "max_compute_ms",
                "achieved_hz",
                "source_counts",
                "errors",
            )
        }
        motor_metrics.update(
            enabled=self.metadata["control_mode"] == "local_motor",
            target_hz=self.metadata["motor_control_hz"],
            recorded_samples=sum(e["type"] == "motor_sample" for e in self.events),
        )
        summary = {
            "status": status,
            "elapsed_s": round(elapsed_s, 3),
            "error": error,
            "synthetic_environment": self.metadata["synthetic_environment"],
            "provider": self.metadata["config"]["model"]["provider"],
            "model": self.metadata["config"]["model"]["model"],
            "hierarchical": self.metadata["hierarchical"],
            "evaluation_track": self.metadata["evaluation_track"],
            "control_mode": self.metadata["control_mode"],
            "motor": motor_metrics,
            "planner": {
                "enabled": self.metadata["hierarchical"],
                "model": self.metadata["planner_model"],
                "requests": sum(e["type"] == "planner_request" for e in self.events),
                "completed": len(planner_decisions),
                "accepted": sum(e.get("accepted", False) for e in planner_decisions),
                "mean_latency_s": _metric(_measurements(planner_decisions, "latency_s")),
                "usage_totals": planner_usage,
            },
            "controls_with_strategy": sum(
                e["type"] == "request"
                and isinstance(e.get("context"), dict)
                and isinstance(e["context"].get("strategy"), dict)
                for e in self.events
            ),
            "observation_mode": self.metadata["observation_mode"],
            "control_scheduling": self.metadata["control_scheduling"],
            "max_decision_hz": self.metadata["max_decision_hz"],
            "action_format": self.metadata["action_format"],
            "catalog_versions": catalog_versions,
            "resolved_models": resolved_models,
            "requests": sum(e["type"] == "request" for e in self.events),
            "completed_decisions": len(decisions),
            "accepted_actions": accepted,
            "rejected_or_failed_decisions": len(decisions) - accepted,
            "mean_latency_s": _metric(latencies),
            "median_latency_s": _metric(latencies, "median"),
            "p95_latency_s": _metric(latencies, "p95"),
            "achieved_decisions_hz": round(len(decisions) / elapsed_s, 6) if elapsed_s > 0 else 0.0,
            "accepted_actions_hz": round(accepted / elapsed_s, 6) if elapsed_s > 0 else 0.0,
            "mean_observation_age_s": _metric(ages),
            "p95_observation_age_s": _metric(ages, "p95"),
            "final_telemetry": final,
            "usage_by_request": usages,
            "usage_totals": controller_usage,
            "combined_usage_totals": {
                key: controller_usage[key] + planner_usage[key] for key in USAGE_KEYS
            },
        }
        self.event("finished", elapsed_s, status=status, error=error)
        self._file.close()
        (self.path / "summary.json").write_text(json_text(summary), encoding="utf-8")
        write_report(self.path, summary, self.events, self.metadata)
        return summary


def write_report(path: Path, summary: dict, events: list, metadata: dict):
    frames = [e for e in events if e["type"] == "frame" and e.get("path")]
    telemetry = [e for e in events if e["type"] == "telemetry" and isinstance(e.get("state"), dict)]
    use_arena = metadata.get("observation_mode") == "state" or not frames
    motor_enabled = metadata.get("control_mode") == "local_motor"
    payload = json_text(
        {
            "frames": frames,
            "telemetry": telemetry,
            "decisions": [e for e in events if e["type"] == "decision"],
            "motor_enabled": motor_enabled,
            "motor_samples": [e for e in events if e["type"] == "motor_sample"],
            "planner_enabled": metadata.get("hierarchical", False),
            "planner_events": [
                e for e in events if e["type"] in {"planner_decision", "planner_invalidated"}
            ],
            "timeline_kind": "telemetry" if use_arena else "frames",
        }
    )
    payload = (
        payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )
    title = f"{summary['provider']} / {summary['model']}"
    if motor_enabled:
        title = f"{metadata.get('planner_model') or 'Scripted baseline'} + local motor controller"
    motor = summary.get("motor", {})
    run_metrics = (
        f"{motor.get('ticks', 'unavailable')} local controller ticks · "
        f"{motor.get('achieved_hz', 'unavailable')} measured Hz · "
        f"{summary.get('planner', {}).get('accepted', 0)} accepted strategy plans"
        if motor_enabled
        else f"{summary['accepted_actions']} accepted actions · mean model latency {summary['mean_latency_s']} s"
    )
    rate_description = (
        f"local controller target {motor.get('target_hz', 'unspecified')} Hz · "
        f"measured {motor.get('achieved_hz', 'unavailable')} Hz · "
        f"{motor.get('recorded_samples', 0)} recorded control samples"
        if motor_enabled
        else f"configured maximum {metadata.get('max_decision_hz', 'unspecified')} decisions/s · "
        f"achieved {summary.get('achieved_decisions_hz', 'unavailable')} completed decisions/s"
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Rocket League evaluation</title><style>
:root{color-scheme:dark;font:16px system-ui;background:#101316;color:#e9edf0}body{max-width:1120px;margin:40px auto;padding:0 24px}
h1{font-size:30px;margin:8px 0}p{color:#b8c1c7}.eyebrow{color:#79c2b9;letter-spacing:.12em;font-size:12px}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:24px}img,canvas{width:100%;border:1px solid #343d44;border-radius:6px;min-height:180px;background:#161d22}
input{width:100%}pre{white-space:pre-wrap;word-break:break-word;background:#192025;padding:16px;border-radius:6px;font:12px monospace}
button{padding:9px 20px;background:#79c2b9;border:0;border-radius:4px;cursor:pointer}a{color:#79c2b9}
.warning{color:#ffd08a}.modes{padding:12px 16px;background:#192025;border-left:3px solid #79c2b9}.muted{font-size:13px}details{margin-top:24px}[hidden]{display:none!important}@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style><div class="eyebrow">OFFLINE ROCKET LEAGUE / EVALUATION RUN</div><h1>__TITLE__</h1>
<p>__STATUS__ · __DURATION__ seconds · __RUN_METRICS__</p>__SYNTHETIC__
__ASSISTANCE__
<p class="modes">Observation: <strong>__OBSERVATION__</strong> · Scheduling: <strong>__SCHEDULING__</strong><br><span class="muted">Action format: __ACTION_FORMAT__ · __RATE_DESCRIPTION__</span></p>
<div class="grid"><section><img id="frame" alt="Recorded game observation" hidden><canvas id="arena" width="720" height="560" aria-label="Top-down replay of recorded game state" hidden></canvas><input id="scrub" type="range" min="0" value="0" aria-label="Replay time">
<button id="play">Play</button> <span id="time"></span><p id="replay-note"></p><p id="height-note" class="muted"></p></section>
<section><div id="planner-panel" hidden><div class="eyebrow">PLANNER DIRECTIVE · __PLANNER_MODEL__</div><pre id="planner">No accepted plan at this time.</pre></div><div class="eyebrow">__ACTION_HEADING__</div><pre id="decision">No controller output at this time.</pre><details><summary>Recorded state at this time</summary><pre id="telemetry">No state available.</pre></details></section></div>
<details open><summary>Run summary</summary><pre>__SUMMARY__</pre></details>
<details><summary>Configuration and reproducibility</summary><pre>__META__</pre></details>
<p><a href="events.jsonl">Event log</a> · <a href="summary.json">Summary JSON</a> · <a href="metadata.json">Run metadata</a></p>
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),s=document.getElementById('scrub');let timer=null;
const arenaMode=data.timeline_kind==='telemetry',timeline=arenaMode?data.telemetry:data.frames;
const canvas=document.getElementById('arena'),ctx=canvas.getContext('2d'),frame=document.getElementById('frame');
canvas.hidden=!arenaMode;frame.hidden=arenaMode;
document.getElementById('planner-panel').hidden=!data.planner_enabled;
document.getElementById('replay-note').textContent=arenaMode?'Recorded telemetry, top-down view. Blue is your car; orange symbols are opponents; white is the ball. Positions are sampled, without physics interpolation. This is not a screenshot observation.':'Recorded screenshot observations. Playback follows recorded wall-clock timing.';
s.max=Math.max(0,timeline.length-1);s.disabled=timeline.length===0;document.getElementById('play').disabled=timeline.length<2;
function vector(v){return v&&typeof v.x==='number'&&Number.isFinite(v.x)&&typeof v.y==='number'&&Number.isFinite(v.y);}
const scale=Math.min((canvas.width-110)/8192,(canvas.height-90)/10240),cx=canvas.width/2,cy=canvas.height/2;
function project(v){return [cx+v.x*scale,cy-v.y*scale];}
function line(points,color,width=2){if(!points.length)return;ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;points.forEach((p,i)=>{const q=project(p);if(i)ctx.lineTo(q[0],q[1]);else ctx.moveTo(q[0],q[1]);});ctx.stroke();}
function symbol(position,forward,color,radius){if(!vector(position))return;const [x,y]=project(position);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,radius,0,2*Math.PI);ctx.fill();
if(vector(forward)){const length=Math.hypot(forward.x,forward.y);if(length>0.001){const dx=forward.x/length,dy=-forward.y/length,tx=x+dx*22,ty=y+dy*22;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(tx,ty);ctx.moveTo(tx-dx*7-dy*5,ty-dy*7+dx*5);ctx.lineTo(tx,ty);ctx.lineTo(tx-dx*7+dy*5,ty-dy*7-dx*5);ctx.stroke();}}}
function drawArena(state,time){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#14261f';const [left,top]=project({x:-4096,y:5120});ctx.fillRect(left,top,8192*scale,10240*scale);ctx.strokeStyle='#667c71';ctx.lineWidth=2;ctx.strokeRect(left,top,8192*scale,10240*scale);
line([{x:-4096,y:0},{x:4096,y:0}],'#52695c',1);ctx.beginPath();ctx.arc(cx,cy,800*scale,0,2*Math.PI);ctx.stroke();ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillStyle='#f2a45f';ctx.fillText('ORANGE  +Y',cx,top-12);ctx.fillStyle='#6bb7ff';ctx.fillText('BLUE  −Y',cx,top+10240*scale+24);
const recent=data.telemetry.filter(e=>e.elapsed_s<=time&&e.elapsed_s>=time-3);for(const [key,color] of [['car_position','#356e9d'],['ball_position','#738179']]){let segment=[];for(const sample of recent){const p=sample.state[key];if(vector(p))segment.push(p);else{line(segment,color,1);segment=[];}}line(segment,color,1);}
for(const opponent of (Array.isArray(state.opponents)?state.opponents:[]))symbol(opponent.position,opponent.forward,'#ee9a50',6);symbol(state.car_position,state.car_forward,'#63b2ff',7);symbol(state.ball_position,null,'#f6f2db',7);
if(!vector(state.car_position)&&!vector(state.ball_position)){ctx.fillStyle='#d9e1dd';ctx.fillText('No recorded position data for this sample',cx,cy);}
const z=p=>p&&typeof p.z==='number'&&Number.isFinite(p.z)?p.z.toFixed(0):'unavailable';document.getElementById('height-note').textContent='Recorded height: car '+z(state.car_position)+' · ball '+z(state.ball_position)+' game units. Top-down positions do not show vertical motion.';}
function showPlanner(time){let latest=null,reason='No accepted plan at this time.';for(const e of data.planner_events){if(e.elapsed_s>time)continue;if(e.type==='planner_invalidated'){latest=null;reason='Plan cleared: '+(e.reason||'gameplay changed');}else if(e.accepted){latest=e;}}
document.getElementById('planner').textContent=latest?JSON.stringify({number:latest.number,accepted_at_s:latest.elapsed_s,latency_s:latest.latency_s,model:latest.resolved_model,plan:latest.plan},null,2):reason;}
function show(){const current=timeline[Number(s.value)];if(!current){document.getElementById('time').textContent='No replay samples saved';if(arenaMode)drawArena({},0);showPlanner(0);return;}
const time=current.elapsed_s;document.getElementById('time').textContent=time.toFixed(2)+' s';const sample=arenaMode?current:data.telemetry.filter(e=>e.elapsed_s<=time).at(-1);
showPlanner(time);
if(arenaMode)drawArena(current.state,time);else frame.src=current.path;
document.getElementById('telemetry').textContent=sample?JSON.stringify(sample.state,null,2):'No state available.';
const trace=data.motor_enabled?data.motor_samples:data.decisions,d=trace.filter(x=>x.elapsed_s<=time).at(-1);document.getElementById('decision').textContent=d?JSON.stringify(d,null,2):(data.motor_enabled?'No local controller sample yet':'No decision yet');}
function stop(){clearTimeout(timer);timer=null;document.getElementById('play').textContent='Play';}
function tick(){let i=Number(s.value);if(i>=timeline.length-1){stop();return;}let delay=(timeline[i+1].elapsed_s-timeline[i].elapsed_s)*1000;
timer=setTimeout(()=>{s.value=i+1;show();tick();},Math.max(10,delay));}
s.oninput=()=>{stop();show();};document.getElementById('play').onclick=()=>{if(timer){stop();return;}if(Number(s.value)>=timeline.length-1)s.value=0;show();document.getElementById('play').textContent='Pause';tick();};show();
</script></html>"""
    substitutions = {
        "__TITLE__": escape(title),
        "__RUN_METRICS__": escape(run_metrics),
        "__RATE_DESCRIPTION__": escape(rate_description),
        "__ACTION_HEADING__": "LOCAL CONTROLLER OUTPUT · SCRIPTED ASSISTANCE"
        if motor_enabled
        else "MOST RECENT MODEL DECISION",
        "__ASSISTANCE__": (
            '<p class="warning">STRATEGY-ASSISTED CONTROLS: a scripted local motor controller '
            "executes low-level mechanics. Strategy plans, scripted baseline, and fallback control "
            "are labeled separately in the trace. Controller ticks are not model decisions. "
            "This track does not measure unassisted model control.</p>"
            if motor_enabled
            else ""
        ),
        "__PLANNER_MODEL__": escape(str(metadata.get("planner_model") or "disabled")),
        "__STATUS__": escape(summary["status"]),
        "__DURATION__": str(summary["elapsed_s"]),
        "__COUNT__": str(summary["accepted_actions"]),
        "__LATENCY__": str(summary["mean_latency_s"]),
        "__OBSERVATION__": escape(metadata.get("observation_mode", "pixels")),
        "__SCHEDULING__": escape(metadata.get("control_scheduling", "bounded")),
        "__ACTION_FORMAT__": escape(metadata.get("action_format", "controller_json")),
        "__MAX_HZ__": str(metadata.get("max_decision_hz", "unspecified")),
        "__ACHIEVED_HZ__": str(summary.get("achieved_decisions_hz", "unavailable")),
        "__SYNTHETIC__": '<p class="warning">SYNTHETIC SMOKE TEST — not a Rocket League performance result.</p>'
        if metadata["synthetic_environment"]
        else "",
        "__SUMMARY__": escape(json.dumps(summary, indent=2)),
        "__META__": escape(json.dumps(metadata, indent=2)),
        "__DATA__": payload,
    }
    # One pass prevents a model name or note containing a placeholder from being expanded.
    page = re.sub(r"__[A-Z_]+__", lambda match: substitutions.get(match[0], match[0]), page)
    (path / "report.html").write_text(page, encoding="utf-8")
