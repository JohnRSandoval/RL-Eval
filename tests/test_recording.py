import json
import re
import shutil
import subprocess

import pytest

from rleval.config import load_config
from rleval.recording import Recorder


def config_for(mode="state", scheduling="continuous", provider="openai-decisions"):
    config = load_config()
    config["observation"] = {"mode": mode}
    config["control"] = {"scheduling": scheduling, "max_decision_hz": 10.0}
    config["model"].update(provider=provider, model="test-model")
    return config


def telemetry(x=0):
    return {
        "car_position": {"x": x, "y": -1000, "z": 17},
        "car_forward": {"x": 0, "y": 1, "z": 0},
        "ball_position": {"x": 500, "y": 1000, "z": 93},
        "opponents": [
            {"position": {"x": -700, "y": 1500, "z": 17}, "forward": {"x": 0, "y": -1, "z": 0}}
        ],
    }


def report_payload(recorder):
    html = (recorder.path / "report.html").read_text(encoding="utf-8")
    raw = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.DOTALL)
    return html, json.loads(raw[1])


def test_state_only_report_has_telemetry_replay_and_metrics(tmp_path):
    recorder = Recorder(tmp_path, config_for(), synthetic=True)
    for at in (0.0, 0.2, 0.4):
        recorder.event("telemetry", at, state=telemetry(at * 100))
    recorder.event("request", 0, number=1)
    recorder.event(
        "decision",
        0.1,
        accepted=True,
        latency_s=0.1,
        observation_age_s=0.12,
        usage={"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
        model_decision={
            "choice": "forward",
            "confidence": 0.75,
            "probabilities": {"forward": 0.75, "neutral": 0.25},
            "catalog_version": "controller-v1",
        },
    )
    recorder.event("request", 0.2, number=2)
    recorder.event(
        "decision",
        0.4,
        accepted=False,
        latency_s=0.2,
        observation_age_s=0.22,
        usage=None,
        error="timeout",
    )
    summary = recorder.finish("time_limit", 0.5, telemetry())
    html, payload = report_payload(recorder)
    assert list((recorder.path / "frames").iterdir()) == []
    assert payload["timeline_kind"] == "telemetry"
    assert payload["frames"] == []
    assert len(payload["telemetry"]) == 3
    assert payload["decisions"][0]["model_decision"]["confidence"] == 0.75
    assert 'id="arena"' in html
    assert "Observation: <strong>state</strong>" in html
    assert "Scheduling: <strong>continuous</strong>" in html
    assert summary["observation_mode"] == "state"
    assert summary["action_format"] == "finite_direct_controller_catalog"
    assert summary["catalog_versions"] == ["controller-v1"]
    assert summary["median_latency_s"] == 0.15
    assert summary["p95_latency_s"] == 0.2
    assert summary["achieved_decisions_hz"] == 4.0
    assert summary["accepted_actions_hz"] == 2.0
    assert summary["mean_observation_age_s"] == 0.17
    assert summary["p95_observation_age_s"] == 0.22
    assert summary["usage_totals"]["total_tokens"] == 14
    assert summary["usage_by_request"][1] == {}


@pytest.mark.parametrize("mode", ["pixels", "pixels_and_state"])
def test_pixel_reports_keep_saved_screenshot_timeline(tmp_path, mode):
    recorder = Recorder(tmp_path, config_for(mode, "bounded", "openai"), synthetic=False)
    frame = recorder.frame(0, b"saved jpeg")
    recorder.event("frame", 0, path=frame)
    recorder.event("telemetry", 0, state=telemetry())
    summary = recorder.finish("completed", 1, telemetry())
    html, payload = report_payload(recorder)
    assert payload["timeline_kind"] == "frames"
    assert payload["frames"][0]["path"] == "frames/000000.jpg"
    assert summary["observation_mode"] == mode
    assert summary["control_scheduling"] == "bounded"
    assert summary["action_format"] == "controller_json"
    assert f"Observation: <strong>{mode}</strong>" in html


def test_failed_empty_run_has_finite_metrics_and_no_fake_trajectory(tmp_path):
    recorder = Recorder(tmp_path, config_for(), synthetic=False)
    summary = recorder.finish("failed", 0, {}, "connection unavailable")
    _, payload = report_payload(recorder)
    assert summary["achieved_decisions_hz"] == 0
    assert summary["accepted_actions_hz"] == 0
    assert summary["median_latency_s"] is None
    assert summary["mean_observation_age_s"] is None
    assert payload["telemetry"] == []
    assert payload["decisions"] == []
    assert summary["usage_totals"]["total_tokens"] == 0
    assert summary["planner"]["enabled"] is False
    assert summary["planner"]["requests"] == 0
    assert summary["combined_usage_totals"] == summary["usage_totals"]


def test_hierarchy_reports_separate_and_combined_usage_and_strategy_coverage(tmp_path):
    config = config_for()
    config["planner"] = {"enabled": True, "provider": "openai", "model": "planner-model"}
    recorder = Recorder(tmp_path, config, synthetic=True)
    recorder.event("planner_request", 0, number=1, context={})
    recorder.event(
        "planner_decision",
        0.2,
        number=1,
        latency_s=0.2,
        accepted=True,
        plan={"objective": "Defend the blue goal"},
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        resolved_model="planner-version",
    )
    recorder.event("request", 0, number=1, context={})
    recorder.event("request", 0.3, number=2, context={"strategy": {"version": 1}})
    recorder.event("decision", 0.4, accepted=True, latency_s=0.1, usage={"total_tokens": 7})
    recorder.event("planner_request", 0.5, number=2, context={})
    recorder.event(
        "planner_decision",
        0.9,
        number=2,
        latency_s=0.4,
        accepted=False,
        error="Stale planner response",
        usage={"total_tokens": 5, "input_tokens": True, "output_tokens": -1},
    )
    summary = recorder.finish("completed", 1, {})
    assert recorder.metadata["hierarchical"] is True
    assert recorder.metadata["planner_model"] == "planner-model"
    planner = summary["planner"]
    assert planner["enabled"] and planner["model"] == "planner-model"
    assert (planner["requests"], planner["completed"], planner["accepted"]) == (2, 2, 1)
    assert planner["mean_latency_s"] == 0.3
    assert planner["usage_totals"]["total_tokens"] == 35
    assert planner["usage_totals"]["input_tokens"] == 10
    assert planner["usage_totals"]["output_tokens"] == 20
    assert summary["usage_totals"]["total_tokens"] == 7
    assert summary["combined_usage_totals"]["total_tokens"] == 42
    assert summary["controls_with_strategy"] == 1
    assert summary["requests"] == 2 and summary["completed_decisions"] == 1


def test_report_escapes_model_output_and_does_not_expand_nested_placeholders(tmp_path):
    config = config_for()
    attack = '__DATA__</script><script>throw Error("injected")</script>'
    config["model"]["model"] = attack
    recorder = Recorder(tmp_path, config, synthetic=False)
    recorder.event(
        "decision",
        0.1,
        accepted=False,
        error=attack,
        latency_s=0.1,
        model_decision={"choice": attack, "confidence": 0},
    )
    recorder.finish("failed", 0.2, {}, attack)
    html, payload = report_payload(recorder)
    assert html.count("<script") == 2
    assert "__DATA__&lt;/script&gt;" in html
    assert payload["decisions"][0]["error"] == attack
    assert "</script><script>throw" not in html


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node validates portable report JavaScript"
)
def test_state_replay_javascript_scrubs_and_draws_coordinate_mapping(tmp_path):
    recorder = Recorder(tmp_path, config_for(), synthetic=True)
    recorder.event("telemetry", 0, state=telemetry())
    recorder.event("telemetry", 0.2, state=telemetry(100))
    recorder.event(
        "decision",
        0.1,
        accepted=True,
        latency_s=0.1,
        model_decision={"choice": "jump", "confidence": 0.8},
    )
    recorder.finish("completed", 0.3, telemetry(100))
    html, payload = report_payload(recorder)
    script = re.search(r"<script>(.*?)</script>", html, re.DOTALL)[1]
    wrapper = """
const fs=require('fs'),vm=require('vm'),input=JSON.parse(fs.readFileSync(0,'utf8'));
const elements={};const calls=[];
const context=new Proxy({}, {get:(obj,key)=>obj[key]??((...args)=>calls.push([key,...args]))});
const document={getElementById:id=>elements[id]??=(id==='data'?{textContent:JSON.stringify(input.payload)}:{value:0,width:720,height:560,textContent:'',getContext:()=>context})};
vm.runInNewContext(input.script+`
s.value=1;show();console.log(JSON.stringify({imageHidden:frame.hidden,arenaHidden:canvas.hidden,time:document.getElementById('time').textContent,decision:document.getElementById('decision').textContent,corner:project({x:4096,y:5120}),car:project({x:100,y:-1000}),sampleCount:timeline.length}));`,{document,console,setTimeout,clearTimeout});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", wrapper],
        input=json.dumps({"script": script, "payload": payload}),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    result = json.loads(result.stdout)
    assert result["imageHidden"] is True
    assert result["arenaHidden"] is False
    assert result["sampleCount"] == 2
    assert result["time"] == "0.20 s"
    assert json.loads(result["decision"])["model_decision"]["choice"] == "jump"
    assert result["corner"] == [548, 45]
    assert result["car"][0] > 360
    assert result["car"][1] > 280


@pytest.mark.skipif(shutil.which("node") is None, reason="Node validates planner replay")
def test_planner_replay_retains_accepted_directive_until_invalidated(tmp_path):
    config = config_for()
    config["planner"] = {"enabled": True, "provider": "openai", "model": "planner-model"}
    recorder = Recorder(tmp_path, config, synthetic=True)
    recorder.event("telemetry", 0, state=telemetry())
    recorder.event("planner_decision", 0.2, number=1, accepted=True, plan={"objective": "Defend"})
    recorder.event(
        "planner_decision", 0.5, number=2, accepted=False, plan={"objective": "Rejected"}
    )
    recorder.event("planner_invalidated", 0.7, reason="kickoff reset")
    recorder.event("planner_decision", 0.9, number=3, accepted=True, plan={"objective": "Attack"})
    recorder.finish("completed", 1, telemetry())
    html, payload = report_payload(recorder)
    script = re.search(r"<script>(.*?)</script>", html, re.DOTALL)[1]
    wrapper = """
const fs=require('fs'),vm=require('vm'),input=JSON.parse(fs.readFileSync(0,'utf8'));
const elements={},context=new Proxy({}, {get:(obj,key)=>obj[key]??(()=>{})});
const document={getElementById:id=>elements[id]??=(id==='data'?{textContent:JSON.stringify(input.payload)}:{value:0,width:720,height:560,textContent:'',getContext:()=>context})};
vm.runInNewContext(input.script+`
const snapshots=[0.1,0.3,0.6,0.8,1].map(t=>{showPlanner(t);return document.getElementById('planner').textContent;});
console.log(JSON.stringify({hidden:document.getElementById('planner-panel').hidden,snapshots}));`,{document,console,setTimeout,clearTimeout});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", wrapper],
        input=json.dumps({"script": script, "payload": payload}),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    result = json.loads(result.stdout)
    assert result["hidden"] is False
    first, accepted, rejected, cleared, resumed = result["snapshots"]
    assert first == "No accepted plan at this time."
    assert json.loads(accepted)["plan"]["objective"] == "Defend"
    assert rejected == accepted
    assert cleared == "Plan cleared: kickoff reset"
    assert json.loads(resumed)["plan"]["objective"] == "Attack"


@pytest.mark.parametrize("planner_enabled", [False, True])
def test_assisted_control_metrics_and_replay_are_separate_from_model_decisions(
    tmp_path, planner_enabled
):
    config = config_for(provider="local-motor")
    config["model"]["model"] = "local-motor-v1"
    config["control"].update(mode="local_motor", motor_hz=60)
    config["planner"] = {"enabled": planner_enabled, "model": "gpt-6.1-sol"}
    recorder = Recorder(tmp_path, config, synthetic=True)
    recorder.event("telemetry", 0, state=telemetry())
    for at in (0.1, 0.2, 0.3):
        recorder.event(
            "motor_sample",
            at,
            controls={"throttle": 1},
            target={"x": 1, "y": 2, "z": 17},
            skill="drive",
            status="active",
            metrics={},
            plan_version=1 if planner_enabled else None,
            policy_source="planner" if planner_enabled else "scripted_baseline",
            tick_count=int(at * 60),
        )
    recorder.event(
        "motor_summary",
        1,
        ticks=60,
        worker_elapsed_s=0.99,
        active_ticks=50,
        mean_compute_ms=0.2,
        max_compute_ms=1.5,
        achieved_hz=60.6,
        source_counts={"planner": 40, "fallback": 20}
        if planner_enabled
        else {"scripted_baseline": 60},
        errors=0,
    )
    summary = recorder.finish("completed", 1, telemetry())
    html, payload = report_payload(recorder)
    assert recorder.metadata["evaluation_track"] == "strategy_assisted_controls"
    assert summary["action_format"] == "local_motor_controller"
    assert summary["control_mode"] == "local_motor"
    assert summary["motor"]["recorded_samples"] == 3
    assert summary["motor"]["ticks"] == 60
    assert summary["motor"]["target_hz"] == 60
    assert summary["motor"]["achieved_hz"] == 60.6
    assert (summary["requests"], summary["completed_decisions"], summary["accepted_actions"]) == (
        0,
        0,
        0,
    )
    assert summary["mean_latency_s"] is None
    assert payload["motor_enabled"] is True
    assert len(payload["motor_samples"]) == 3
    assert payload["decisions"] == []
    assert "LOCAL CONTROLLER OUTPUT · SCRIPTED ASSISTANCE" in html
    assert "Controller ticks are not model decisions" in html
    assert "mean model latency" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node validates motor replay")
def test_motor_replay_scrubs_actual_control_samples(tmp_path):
    config = config_for(provider="local-motor")
    config["control"]["mode"] = "local_motor"
    recorder = Recorder(tmp_path, config, synthetic=True)
    recorder.event("telemetry", 0, state=telemetry())
    recorder.event(
        "motor_sample",
        0.1,
        controls={"steer": -0.4},
        target={},
        skill="drive",
        status="active",
        policy_source="fallback",
        tick_count=6,
    )
    recorder.event("telemetry", 0.2, state=telemetry(100))
    recorder.finish("completed", 0.3, telemetry(100))
    html, payload = report_payload(recorder)
    script = re.search(r"<script>(.*?)</script>", html, re.DOTALL)[1]
    wrapper = """
const fs=require('fs'),vm=require('vm'),input=JSON.parse(fs.readFileSync(0,'utf8'));
const elements={},context=new Proxy({}, {get:(obj,key)=>obj[key]??(()=>{})});
const document={getElementById:id=>elements[id]??=(id==='data'?{textContent:JSON.stringify(input.payload)}:{value:0,width:720,height:560,textContent:'',getContext:()=>context})};
vm.runInNewContext(input.script+`s.value=1;show();console.log(document.getElementById('decision').textContent);`,{document,console,setTimeout,clearTimeout});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", wrapper],
        input=json.dumps({"script": script, "payload": payload}),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    output = json.loads(result.stdout)
    assert output["type"] == "motor_sample"
    assert output["controls"] == {"steer": -0.4}
    assert output["policy_source"] == "fallback"
