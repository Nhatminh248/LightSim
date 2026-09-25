# LightSim Repository Summary & CTM Model Architecture

## 1. Overview
**LightSim** is a lightweight, pure-Python macro/mesoscopic traffic simulation framework designed for reinforcement learning (RL) research in traffic signal control. It is built on Daganzo's **Cell Transmission Model (CTM)** with vectorized NumPy computations.

---

## 2. Directory Summaries

### Top-Level Folders

* **[`lightsim/`](file:///Users/nhatleminh/thesis/LightSim/lightsim)**
  The core Python package containing the simulation engine, network models, RL environments, Decision Transformer modules, I/O serializers, metrics, and visualization server.

  * **[`lightsim/core/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/core)**: The simulation backend and mathematical dynamics. Contains the CTM flow model equations, the discrete-time simulation step loop, network compilation and representation, signal controllers (fixed-time, actuated, max-pressure, RL), demand injection/routing, and emergency vehicle (EV) tracking.
  * **[`lightsim/envs/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/envs)**: Gymnasium and PettingZoo RL environment wrappers. Implements single-agent (`SingleIntersectionEnv`) and multi-agent (`MultiAgentTrafficEnv`) interfaces, along with customizable observation builders, action spaces, and reward formulations (delay, pressure, queue, throughput, EV priority).
  * **[`lightsim/networks/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/networks)**: Network builders and generators. Provides synthetic topology generators (grid networks, arterial corridors), dictionary-based constructors, and OpenStreetMap (OSM) extractors using OSMnx/Overpass to build real-world road networks (e.g., Manhattan, Tokyo, Sioux Falls).
  * **[`lightsim/dt/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/dt)**: Offline Reinforcement Learning via Decision Transformer (DT). Includes PyTorch trajectory dataset collectors/loaders, causal transformer architecture for traffic control, autoregressive online controllers, and training routines.
  * **[`lightsim/benchmarks/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/benchmarks)**: Benchmark suites and evaluation scenarios. Contains standardized topologies (single intersection, 5-intersection arterial, 4x4 grid, OSM cities), speed profiling benchmarks, baseline evaluation scripts, and comparison tooling against SUMO.
  * **[`lightsim/io/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/io)**: Serialization and persistence. Handles exporting and loading networks, signal configurations, and demand profiles to and from JSON format.
  * **[`lightsim/viz/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/viz)**: Interactive web-based visualization. Houses a lightweight WebSocket/HTTP streaming server and frontend web assets (HTML/JS/Canvas) for live and replayed traffic animation and queue/density inspection.
  * **[`lightsim/utils/`](file:///Users/nhatleminh/thesis/LightSim/lightsim/utils)**: Measurement and validation utilities. Implements traffic performance metrics (average delay, queue lengths, throughput), travel time estimators, and network/demand consistency validators.

* **[`examples/`](file:///Users/nhatleminh/thesis/LightSim/examples)**
  Executable standalone examples demonstrating how to use the library: quickstart scripts, custom networks, JSON import/export, mesoscopic EV simulation, Max-Pressure control, multi-agent training, OSM pipelines, and record/replay workflows.

* **[`scripts/`](file:///Users/nhatleminh/thesis/LightSim/scripts)**
  Research and paper experiment scripts used for model validation, cross-validation against SUMO, Decision Transformer training/eval, multi-agent RL training, sample efficiency studies, figure generation (PaperBanana, Matplotlib), and demo generation.

* **[`tests/`](file:///Users/nhatleminh/thesis/LightSim/tests)**
  Comprehensive test suite using `pytest`. Covers CTM numerical properties, conservation of vehicles, flow models, signal transitions, RL environments, Decision Transformer components, EV tracking, edge cases, and visualization resilience.

* **[`docs/`](file:///Users/nhatleminh/thesis/LightSim/docs)**
  Documentation assets, primarily rendered GIF animations (e.g., London, Manhattan, San Francisco, Tokyo networks) and diagram figures for the documentation and README.

* **[`results/`](file:///Users/nhatleminh/thesis/LightSim/results)**
  Benchmark outputs, experimental run logs, and JSON summaries comparing LightSim against baselines (e.g., SUMO cross-validation, RL evaluation runs, fundamental diagram evaluations, speed benchmarks).

* **[`weights/`](file:///Users/nhatleminh/thesis/LightSim/weights)**
  Pre-trained RL model checkpoints (e.g., Stable-Baselines3 DQN and PPO weights for single-intersection and 4x4 grid environments) along with performance summary metrics.

---

## 3. Which File Handles the Equations of the CTM Model?

The core equations of the Cell Transmission Model are primarily implemented in:

### 📍 Primary File: [`lightsim/core/flow_model.py`](file:///Users/nhatleminh/thesis/LightSim/lightsim/core/flow_model.py)

In particular, the class **[`CTMFlowModel`](file:///Users/nhatleminh/thesis/LightSim/lightsim/core/flow_model.py#L61-L162)** implements the fundamental CTM equations based on Daganzo's triangular Fundamental Diagram (FD):

1. **Sending Flow (Demand Function)**:
   ```
   S_i(k_i) = min(v_f * k_i, Q) * lanes
   ```
   Implemented in `CTMFlowModel.compute_sending_flow()`:
   ```python
   return np.minimum(net.vf * density, net.Q) * net.lanes
   ```

2. **Receiving Flow (Supply Function)**:
   ```
   R_j(k_j) = min(Q, w * (k_jam - k_j)) * lanes
   ```
   Implemented in `CTMFlowModel.compute_receiving_flow()`:
   ```python
   return np.minimum(net.Q, net.w * (net.kj - density)) * net.lanes
   ```

3. **Intra-Link Cell-to-Cell Flow Transfer**:
   ```
   q_ij = min(S_i, R_j) * dt
   ```
   Implemented in `CTMFlowModel.compute_flow()`:
   ```python
   s_i = sending[src_idx]
   r_j = receiving[ds_idx]
   intra_flow[src_idx] = np.minimum(s_i, r_j) * dt
   ```

4. **Inter-Link Movement Flows & Node Dynamics**:
   In `CTMFlowModel.compute_flow()`:
   * Signal control masking: `effective_sending = mov_sending * signal_mask`
   * Start-up lost time ramp: `effective_sending *= capacity_factor`
   * Turn ratio allocation: `effective_sending *= net.mov_turn_ratio`
   * Saturation flow capping: `min(effective_sending, sat_rate)`
   * Merge conflict resolution (scaling when downstream receiving capacity is exceeded):
     `merge_scale = min(1.0, R_to / sum(effective_sending))`
   * Diverge conflict resolution:
     `diverge_scale = min(1.0, S_from / sum(effective_sending))`

### 📍 Secondary / Complementary File: [`lightsim/core/engine.py`](file:///Users/nhatleminh/thesis/LightSim/lightsim/core/engine.py)

While `flow_model.py` computes the flow exchange rates ($q_{ij}$), the **Conservation of Vehicles equation** (cell density update):
```
n_i(t + dt) = n_i(t) + q_{in}(t) - q_{out}(t) + demand_injected - sink_exited
k_i(t + dt) = n_i(t + dt) / (L_i * lanes_i)
```
is executed in the step loop of **[`SimulationEngine.step()`](file:///Users/nhatleminh/thesis/LightSim/lightsim/core/engine.py#L111-L180)**.
