# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "dash",
#   "plotly",
#   "numpy",
#   "onnxruntime",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "tqdm",
# ]
# ///
"""Plotly Dash interactive dashboard for the comma controls challenge."""

import importlib
import logging
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from plotly.subplots import make_subplots
from tqdm.contrib.concurrent import process_map

from controllers import BaseController
from tinyphysics import (
  CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
  TinyPhysicsModel, TinyPhysicsSimulator, get_available_controllers, run_rollout,
)

log = logging.getLogger("viz_dashboard")

# ── Cache ──────────────────────────────────────────────────────────────────
_cache: OrderedDict = OrderedDict()
MAX_CACHE = 50
MODEL_PATH = "./models/tinyphysics.onnx"
DATA_DIR = Path("./data")


def _cache_key(data_path, controller_type, pid_params=None):
  return (str(data_path), controller_type, tuple(pid_params) if pid_params else None)


def _run_sim(data_path, controller_type, pid_params=None):
  key = _cache_key(data_path, controller_type, pid_params)
  if key in _cache:
    _cache.move_to_end(key)
    log.info("Cache HIT for %s (controller=%s, pid_params=%s)", data_path, controller_type, pid_params)
    return _cache[key]

  log.info("Cache MISS — running simulation: data=%s, controller=%s, pid_params=%s",
           data_path, controller_type, pid_params)
  t0 = time.perf_counter()

  log.debug("Loading ONNX model from %s", MODEL_PATH)
  model = TinyPhysicsModel(MODEL_PATH, debug=False)

  if pid_params is not None:
    log.debug("Creating tuned PID controller with P=%.4f, I=%.4f, D=%.4f", *pid_params)
    controller = _make_pid_controller(*pid_params)
  else:
    log.debug("Importing controller module: controllers.%s", controller_type)
    controller = importlib.import_module(f"controllers.{controller_type}").Controller()

  log.debug("Creating simulator for %s", data_path)
  sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
  log.debug("Simulator created — data has %d rows, starting rollout", len(sim.data))
  cost = sim.rollout()
  elapsed = time.perf_counter() - t0

  result = {
    "cost": cost,
    "targets": sim.target_lataccel_history,
    "actuals": sim.current_lataccel_history,
    "actions": sim.action_history,
    "v_egos": [s.v_ego for s in sim.state_history],
  }

  _cache[key] = result
  if len(_cache) > MAX_CACHE:
    evicted = _cache.popitem(last=False)
    log.debug("Cache full (%d entries) — evicted oldest entry", MAX_CACHE)

  log.info("Simulation complete in %.2fs — lataccel_cost=%.4f, jerk_cost=%.4f, total_cost=%.4f",
           elapsed, cost['lataccel_cost'], cost['jerk_cost'], cost['total_cost'])
  return result


def _make_pid_controller(p, i, d):
  class TunedPID(BaseController):
    def __init__(self):
      self.p, self.i_gain, self.d = p, i, d
      self.error_integral = 0
      self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = target_lataccel - current_lataccel
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i_gain * self.error_integral + self.d * error_diff

  return TunedPID()


def _get_data_files():
  if DATA_DIR.exists():
    files = sorted([f.name for f in DATA_DIR.iterdir() if f.suffix == ".csv"])
    log.info("Found %d data files in %s", len(files), DATA_DIR)
    return files
  log.warning("Data directory %s does not exist", DATA_DIR)
  return []


# ── Layout helpers ─────────────────────────────────────────────────────────
CONTROLLERS = get_available_controllers()
DATA_FILES = _get_data_files()
log.info("Available controllers: %s", CONTROLLERS)
log.info("Available data files: %d", len(DATA_FILES))

CARD_STYLE = {
  "border": "1px solid #ddd", "borderRadius": "8px", "padding": "16px",
  "textAlign": "center", "flex": "1", "margin": "0 8px",
  "backgroundColor": "#fafafa",
}

app = dash.Dash(__name__, title="comma controls dashboard")

app.layout = html.Div([
  # ── Top bar ──
  html.Div([
    html.H2("comma controls challenge", style={"margin": "0 16px 0 0"}),
    html.Label("Controller:", style={"marginRight": "4px"}),
    dcc.Dropdown(
      id="controller-select", options=[{"label": c, "value": c} for c in CONTROLLERS],
      value="pid", clearable=False, style={"width": "140px", "display": "inline-block"},
    ),
    html.Label("Segment:", style={"margin": "0 4px 0 16px"}),
    dcc.Dropdown(
      id="data-select", options=[{"label": f, "value": f} for f in DATA_FILES],
      value=DATA_FILES[0] if DATA_FILES else None, clearable=False,
      style={"width": "200px", "display": "inline-block"},
    ),
    html.Button("Run", id="run-btn", n_clicks=0,
                style={"marginLeft": "16px", "padding": "6px 24px", "fontSize": "14px"}),
  ], style={"display": "flex", "alignItems": "center", "padding": "12px 20px",
            "borderBottom": "2px solid #eee", "flexWrap": "wrap", "gap": "4px"}),

  # ── PID tuning panel ──
  html.Div(id="pid-panel", children=[
    html.H4("PID Tuning", style={"margin": "0 0 8px"}),
    html.Div([
      html.Div([
        html.Label("P"), dcc.Slider(id="pid-p", min=0, max=1, step=0.005, value=0.195,
                                    marks={0: "0", 0.5: "0.5", 1: "1"}, tooltip={"placement": "bottom"}),
      ], style={"flex": "1", "margin": "0 8px"}),
      html.Div([
        html.Label("I"), dcc.Slider(id="pid-i", min=0, max=0.5, step=0.005, value=0.100,
                                    marks={0: "0", 0.25: "0.25", 0.5: "0.5"}, tooltip={"placement": "bottom"}),
      ], style={"flex": "1", "margin": "0 8px"}),
      html.Div([
        html.Label("D"), dcc.Slider(id="pid-d", min=-0.5, max=0.5, step=0.001, value=-0.053,
                                    marks={-0.5: "-0.5", 0: "0", 0.5: "0.5"}, tooltip={"placement": "bottom"}),
      ], style={"flex": "1", "margin": "0 8px"}),
    ], style={"display": "flex"}),
    dcc.Checklist(id="auto-update", options=[{"label": " Auto-update on slider change", "value": "on"}],
                  value=["on"], style={"marginTop": "4px"}),
  ], style={"padding": "12px 20px", "borderBottom": "1px solid #eee", "backgroundColor": "#f9f9ff"}),

  # ── Cost cards ──
  html.Div(id="cost-cards", children=[
    html.Div([html.H5("Lat Accel Cost"), html.H3(id="cost-lat", children="—")], style=CARD_STYLE),
    html.Div([html.H5("Jerk Cost"), html.H3(id="cost-jerk", children="—")], style=CARD_STYLE),
    html.Div([html.H5("Total Cost"), html.H3(id="cost-total", children="—")], style=CARD_STYLE),
  ], style={"display": "flex", "padding": "12px 20px"}),

  # ── Charts ──
  dcc.Graph(id="chart-lataccel", style={"height": "320px"}),
  html.Div([
    dcc.Graph(id="chart-steer", style={"flex": "1", "height": "280px"}),
    dcc.Graph(id="chart-jerk", style={"flex": "1", "height": "280px"}),
  ], style={"display": "flex"}),
  html.Div([
    dcc.Graph(id="chart-velocity", style={"flex": "1", "height": "280px"}),
    dcc.Graph(id="chart-error", style={"flex": "1", "height": "280px"}),
  ], style={"display": "flex"}),

  # ── Batch mode ──
  html.Details([
    html.Summary("Batch Mode", style={"fontSize": "16px", "fontWeight": "bold", "cursor": "pointer"}),
    html.Div([
      html.Label("Number of segments:"),
      dcc.Input(id="batch-n", type="number", value=10, min=1, max=500, style={"width": "80px", "margin": "0 8px"}),
      html.Button("Run Batch", id="batch-btn", n_clicks=0, style={"padding": "6px 16px"}),
      html.Div(id="batch-status", style={"marginTop": "8px", "color": "#888"}),
      dcc.Graph(id="chart-batch", style={"height": "300px"}),
    ], style={"padding": "8px 0"}),
  ], style={"padding": "12px 20px", "borderTop": "1px solid #eee"}),

  # ── Comparison mode ──
  html.Details([
    html.Summary("Comparison Mode", style={"fontSize": "16px", "fontWeight": "bold", "cursor": "pointer"}),
    html.Div([
      html.Label("Compare with:"),
      dcc.Dropdown(
        id="compare-controller", options=[{"label": c, "value": c} for c in CONTROLLERS],
        value=None, clearable=True, style={"width": "200px", "display": "inline-block", "margin": "0 8px"},
      ),
      html.Button("Compare", id="compare-btn", n_clicks=0, style={"padding": "6px 16px"}),
      dcc.Graph(id="chart-compare", style={"height": "360px"}),
    ], style={"padding": "8px 0"}),
  ], style={"padding": "12px 20px", "borderTop": "1px solid #eee"}),

  # ── Store ──
  dcc.Store(id="simulation-data"),
], style={"fontFamily": "system-ui, sans-serif", "maxWidth": "1400px", "margin": "0 auto"})


# ── Callbacks ──────────────────────────────────────────────────────────────

def _control_start_shape():
  return dict(type="line", x0=CONTROL_START_IDX, x1=CONTROL_START_IDX,
              y0=0, y1=1, yref="paper", line=dict(color="black", dash="dash", width=1))


def _cost_region_shape():
  return dict(type="rect", x0=CONTROL_START_IDX, x1=COST_END_IDX,
              y0=0, y1=1, yref="paper", fillcolor="rgba(52,152,219,0.06)", line=dict(width=0))


@callback(
  Output("pid-panel", "style"),
  Input("controller-select", "value"),
)
def toggle_pid_panel(ctrl):
  log.debug("toggle_pid_panel: controller=%s", ctrl)
  base = {"padding": "12px 20px", "borderBottom": "1px solid #eee", "backgroundColor": "#f9f9ff"}
  if ctrl != "pid":
    base["display"] = "none"
    log.debug("PID panel hidden (controller is not pid)")
  return base


@callback(
  Output("simulation-data", "data"),
  Input("run-btn", "n_clicks"),
  Input("pid-p", "value"), Input("pid-i", "value"), Input("pid-d", "value"),
  State("controller-select", "value"), State("data-select", "value"),
  State("auto-update", "value"),
  prevent_initial_call=True,
)
def run_simulation(n_clicks, p, i, d, controller, data_file, auto_update):
  triggered = ctx.triggered_id
  log.info("run_simulation callback triggered by '%s' — controller=%s, data=%s, P=%.4f, I=%.4f, D=%.4f",
           triggered, controller, data_file, p or 0, i or 0, d or 0)

  if triggered in ("pid-p", "pid-i", "pid-d") and "on" not in (auto_update or []):
    log.debug("Slider change ignored — auto-update is off")
    return dash.no_update

  if not data_file:
    log.warning("No data file selected — skipping simulation")
    return dash.no_update

  data_path = DATA_DIR / data_file
  pid_params = (p, i, d) if controller == "pid" else None
  log.info("Running sim: %s with %s (pid_params=%s)", data_path, controller, pid_params)
  result = _run_sim(data_path, controller, pid_params)

  log.info("Simulation result stored — %d steps, cost=%.4f",
           len(result["targets"]), result["cost"]["total_cost"])
  return {
    "cost": result["cost"],
    "targets": result["targets"],
    "actuals": result["actuals"],
    "actions": result["actions"],
    "v_egos": result["v_egos"],
    "controller": controller,
    "data_file": data_file,
    "pid_params": list(pid_params) if pid_params else None,
  }


@callback(
  Output("cost-lat", "children"), Output("cost-jerk", "children"), Output("cost-total", "children"),
  Input("simulation-data", "data"),
)
def update_costs(data):
  if not data:
    log.debug("update_costs: no data yet")
    return "—", "—", "—"
  c = data["cost"]
  log.debug("update_costs: lataccel=%.4f, jerk=%.4f, total=%.4f",
            c['lataccel_cost'], c['jerk_cost'], c['total_cost'])
  return f"{c['lataccel_cost']:.4f}", f"{c['jerk_cost']:.4f}", f"{c['total_cost']:.4f}"


@callback(Output("chart-lataccel", "figure"), Input("simulation-data", "data"))
def update_lataccel_chart(data):
  log.debug("update_lataccel_chart: data=%s", "present" if data else "none")
  fig = go.Figure()
  if data:
    fig.add_trace(go.Scatter(y=data["targets"], name="Target", line=dict(color="#27ae60")))
    fig.add_trace(go.Scatter(y=data["actuals"], name="Actual", line=dict(color="#c0392b")))
    fig.update_layout(shapes=[_control_start_shape(), _cost_region_shape()])
  fig.update_layout(title="Lateral Acceleration", xaxis_title="Step", yaxis_title="Lat Accel",
                    margin=dict(l=50, r=20, t=40, b=40), legend=dict(orientation="h", y=1.12))
  return fig


@callback(Output("chart-steer", "figure"), Input("simulation-data", "data"))
def update_steer_chart(data):
  log.debug("update_steer_chart: data=%s", "present" if data else "none")
  fig = go.Figure()
  if data:
    fig.add_trace(go.Scatter(y=data["actions"], name="Steer Cmd", line=dict(color="#2980b9")))
    fig.update_layout(shapes=[_control_start_shape()])
  fig.update_layout(title="Steering Commands", xaxis_title="Step", yaxis_title="Steer",
                    margin=dict(l=50, r=20, t=40, b=40))
  return fig


@callback(Output("chart-jerk", "figure"), Input("simulation-data", "data"))
def update_jerk_chart(data):
  log.debug("update_jerk_chart: data=%s", "present" if data else "none")
  fig = go.Figure()
  if data:
    actuals = np.array(data["actuals"])
    jerk = np.diff(actuals) / DEL_T
    fig.add_trace(go.Scatter(y=jerk, name="Jerk", line=dict(color="#f39c12")))
    fig.update_layout(shapes=[_control_start_shape()])
  fig.update_layout(title="Jerk (d(lataccel)/dt)", xaxis_title="Step", yaxis_title="Jerk",
                    margin=dict(l=50, r=20, t=40, b=40))
  return fig


@callback(Output("chart-velocity", "figure"), Input("simulation-data", "data"))
def update_velocity_chart(data):
  log.debug("update_velocity_chart: data=%s", "present" if data else "none")
  fig = go.Figure()
  if data:
    fig.add_trace(go.Scatter(y=data["v_egos"], name="v_ego", line=dict(color="#8e44ad")))
  fig.update_layout(title="Velocity", xaxis_title="Step", yaxis_title="m/s",
                    margin=dict(l=50, r=20, t=40, b=40))
  return fig


@callback(Output("chart-error", "figure"), Input("simulation-data", "data"))
def update_error_chart(data):
  log.debug("update_error_chart: data=%s", "present" if data else "none")
  fig = go.Figure()
  if data:
    err = np.array(data["targets"]) - np.array(data["actuals"])
    fig.add_trace(go.Scatter(y=err, name="Tracking Error", line=dict(color="#e74c3c"),
                             fill="tozeroy", fillcolor="rgba(231,76,60,0.15)"))
    fig.update_layout(shapes=[_control_start_shape(), _cost_region_shape()])
  fig.update_layout(title="Tracking Error (target - actual)", xaxis_title="Step", yaxis_title="Error",
                    margin=dict(l=50, r=20, t=40, b=40))
  return fig


# ── Batch mode ──

@callback(
  Output("chart-batch", "figure"), Output("batch-status", "children"),
  Input("batch-btn", "n_clicks"),
  State("batch-n", "value"), State("controller-select", "value"),
  State("pid-p", "value"), State("pid-i", "value"), State("pid-d", "value"),
  prevent_initial_call=True,
)
def run_batch(n_clicks, n_segs, controller, p, i, d):
  log.info("run_batch: n_segs=%s, controller=%s", n_segs, controller)
  if not n_segs or n_segs < 1:
    log.warning("Invalid segment count: %s", n_segs)
    return dash.no_update, "Invalid segment count"

  files = sorted(DATA_DIR.iterdir())[:n_segs]
  if not files:
    log.warning("No data files found in %s", DATA_DIR)
    return dash.no_update, "No data files found"

  log.info("Batch mode: running %d segments with controller=%s", len(files), controller)
  t0 = time.perf_counter()

  if controller == "pid":
    pid_params = (p, i, d)
    log.info("Batch PID params: P=%.4f, I=%.4f, D=%.4f — running sequentially", *pid_params)
    costs = []
    for idx, f in enumerate(files):
      log.debug("Batch segment %d/%d: %s", idx + 1, len(files), f.name)
      r = _run_sim(f, controller, pid_params)
      costs.append(r["cost"])
  else:
    log.info("Batch: using parallel process_map with 8 workers")
    rollout_partial = partial(run_rollout, controller_type=controller, model_path=MODEL_PATH, debug=False)
    results = process_map(rollout_partial, files, max_workers=8, chunksize=10)
    costs = [r[0] for r in results]

  elapsed = time.perf_counter() - t0
  lat_costs = [c["lataccel_cost"] for c in costs]
  jerk_costs = [c["jerk_cost"] for c in costs]
  total_costs = [c["total_cost"] for c in costs]

  log.info("Batch complete in %.2fs — %d segments, avg total_cost=%.4f, min=%.4f, max=%.4f",
           elapsed, len(files), np.mean(total_costs), np.min(total_costs), np.max(total_costs))

  fig = make_subplots(rows=1, cols=3, subplot_titles=["Lat Accel Cost", "Jerk Cost", "Total Cost"])
  fig.add_trace(go.Histogram(x=lat_costs, name="lataccel", marker_color="#27ae60", opacity=0.7), row=1, col=1)
  fig.add_trace(go.Histogram(x=jerk_costs, name="jerk", marker_color="#f39c12", opacity=0.7), row=1, col=2)
  fig.add_trace(go.Histogram(x=total_costs, name="total", marker_color="#c0392b", opacity=0.7), row=1, col=3)
  fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), showlegend=False)

  avg_total = np.mean(total_costs)
  status = f"Done — {len(files)} segments, avg total cost: {avg_total:.4f} ({elapsed:.1f}s)"
  return fig, status


# ── Comparison mode ──

@callback(
  Output("chart-compare", "figure"),
  Input("compare-btn", "n_clicks"),
  State("simulation-data", "data"), State("compare-controller", "value"),
  State("data-select", "value"),
  prevent_initial_call=True,
)
def run_comparison(n_clicks, sim_data, compare_ctrl, data_file):
  log.info("run_comparison: compare_ctrl=%s, data_file=%s, has_sim_data=%s",
           compare_ctrl, data_file, bool(sim_data))

  if not compare_ctrl or not data_file:
    log.warning("Comparison aborted — missing controller or data file")
    return go.Figure()

  data_path = DATA_DIR / data_file
  log.info("Running comparison simulation for controller=%s on %s", compare_ctrl, data_path)
  result_b = _run_sim(data_path, compare_ctrl)

  primary_ctrl = sim_data['controller'] if sim_data else '?'
  log.info("Comparison: %s (total=%.4f) vs %s (total=%.4f)",
           primary_ctrl, sim_data['cost']['total_cost'] if sim_data else 0,
           compare_ctrl, result_b['cost']['total_cost'])

  fig = go.Figure()
  fig.add_trace(go.Scatter(y=sim_data["targets"] if sim_data else [], name="Target",
                           line=dict(color="#27ae60", dash="dot")))
  if sim_data:
    fig.add_trace(go.Scatter(y=sim_data["actuals"],
                             name=f"{sim_data['controller']} (actual)",
                             line=dict(color="#c0392b")))
  fig.add_trace(go.Scatter(y=result_b["actuals"], name=f"{compare_ctrl} (actual)",
                           line=dict(color="#2980b9")))
  fig.update_layout(
    title=f"Comparison: {primary_ctrl} vs {compare_ctrl}",
    xaxis_title="Step", yaxis_title="Lat Accel",
    shapes=[_control_start_shape(), _cost_region_shape()],
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(orientation="h", y=1.12),
  )

  return fig


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Dash dashboard for comma controls challenge")
  parser.add_argument("--model_path", type=str, default=MODEL_PATH)
  parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
  parser.add_argument("--port", type=int, default=8050)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
  args = parser.parse_args()

  logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
  )

  MODEL_PATH = args.model_path
  DATA_DIR = Path(args.data_dir)
  DATA_FILES = _get_data_files()

  log.info("=== viz_dashboard starting ===")
  log.info("Args: model_path=%s, data_dir=%s, port=%d, debug=%s",
           args.model_path, args.data_dir, args.port, args.debug)
  log.info("Controllers: %s", CONTROLLERS)
  log.info("Data files: %d available", len(DATA_FILES))
  log.info("Starting Dash server on port %d …", args.port)

  app.run(debug=args.debug, port=args.port)
