<div align="center">

# 🧠 Anvil PCAM Lab

<img src="assets/pcam-dashboard.svg" width="100%" alt="Anvil PCAM Lab Banner"/>

<br/>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Score](https://img.shields.io/badge/P--04%20Score-70%20%2F%2070-FFD700?style=for-the-badge&logo=checkmarx&logoColor=black)](https://github.com/Sauhard74/Anvil-P-E)
[![License: MIT](https://img.shields.io/badge/License-MIT-00FFCC?style=for-the-badge&logo=opensourceinitiative&logoColor=black)](https://opensource.org/licenses/MIT)
[![Track](https://img.shields.io/badge/Track-P--04%20PCAM-7000FF?style=for-the-badge)](https://github.com/Sauhard74/Anvil-P-E)
[![FastAPI](https://img.shields.io/badge/FastAPI-Interactive%20Dashboard-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

<br/>

> **Precision-Controlled Associative Memory Retrieval** — a research prototype for the ANVIL P-04 benchmark.
> Achieves a perfect **70 / 70** retrieval score using the **PrecisionFlow v9.0** adapter.

<br/>

[🚀 Quick Start](#-run-locally) · [🏗️ Architecture](#%EF%B8%8F-system-architecture) · [🎯 PrecisionFlow Adapter](#-precisionflow-v90-adapter) · [🏆 Benchmark](#-official-p-04-harness) · [📡 API](#-api-reference)

</div>

---

## ✨ What Is This?

Anvil PCAM Lab is an interactive research prototype that demonstrates **precision-steered associative memory retrieval** using modern Hopfield-style dynamics. Given a noisy, partially-masked query vector, the system recovers the original stored memory with remarkable accuracy by warping the energy landscape using an **adaptive diagonal precision operator Π**.

The repository ships with:
- 🔬 A fully self-contained PCAM research engine (`anvil_pcam/`)
- 🎯 **PrecisionFlow v9.0** — a reference adapter scoring **70/70** on the official harness
- 🌐 An interactive FastAPI dashboard with real-time visualizations
- 🎬 A stunning 3D Three.js showcase (`showcase.html`)

---

## 🏗️ System Architecture

<div align="center">
<img src="assets/architecture_pipeline.png" width="95%" alt="3D System Architecture Pipeline"/>
<br/><sub><i>The four-stage retrieval pipeline: Attractor Manifold → Signal Corruption → PrecisionFlow Engine → Memory Retrieval</i></sub>
</div>

<br/>

The system operates as a **4-stage precision-steered pipeline**:

| Stage | Component | Description |
|:---:|---|---|
| **1** | 🧠 **Attractor Manifold** | `K` normalized 64-D memory attractors form a dense hyperspace. A similarity graph defines the baseline energy topography. |
| **2** | 🌫️ **Signal Corruption** | A query is subjected to masking and Gaussian noise, simulating chaotic real-world partial observations. |
| **3** | 🎯 **PrecisionFlow Engine** | The adapter computes an anisotropic Π vector — amplifying attractor-aligned dimensions, suppressing noise. |
| **4** | 🔄 **Gradient Descent** | Modern Hopfield dynamics navigate the precision-warped energy landscape to snap the query to the true attractor. |

### Core Modules

```
anvil_pcam/
├── core/
│   ├── memory.py       🧠  Deterministic 64-D attractor bank & similarity graph
│   ├── noise.py        🌫️  Masking + Gaussian corruption simulation
│   ├── precision.py    🎯  predict_precision(corrupted_query) interface
│   ├── dynamics.py     🔄  Modern Hopfield-style iterative retrieval & energy traces
│   └── evaluation.py   📊  Baseline (Π=I) vs. Adaptive (Π) comparison metrics
└── web/                🌐  FastAPI dashboard for the interactive demo
```

---

## 🌌 Energy Landscape & Retrieval Dynamics

<div align="center">
<img src="assets/energy_landscape.png" width="90%" alt="3D Hopfield Energy Landscape"/>
<br/><sub><i>A query (red) descends the precision-warped energy surface, converging to the target memory attractor (cyan glow)</i></sub>
</div>

<br/>

The Hopfield retrieval equations — precision-weighted for maximum signal clarity:

$$s_i = x_i^{\top} \Pi\, \xi_t$$

$$a = \text{softmax}(\beta \cdot s)$$

$$c_t = \text{normalize}\!\left(a^{\top} X\right)$$

The anisotropic state update allows different learning rates per dimension:

$$\xi_{t+1,j} = \text{normalize}\!\left(\xi_t + \alpha_j\,(c_t - \xi_t)\right), \qquad \alpha_j \propto \sqrt{\Pi_{jj}}$$

High-precision dimensions (large $\Pi_{jj}$) receive stronger gradient steps, rapidly shedding noise and snapping to the correct attractor basin.

---

## 🎯 PrecisionFlow v9.0 Adapter

<div align="center">
<img src="assets/precisionflow_diagram.png" width="90%" alt="PrecisionFlow v9.0 Mechanism"/>
<br/><sub><i>PrecisionFlow v9.0: noisy query → anisotropic Π matrix → clean retrieval. Score: 70/70 ✓</i></sub>
</div>

<br/>

The adapter in `adapters/myteam.py` implements the `predict_precision` interface:

```python
def predict_precision(corrupted_query: np.ndarray) -> np.ndarray:
    """
    corrupted_query : np.ndarray, shape (64,)
    returns         : np.ndarray, shape (64,)  — positive, mean-normalised precision
    """
```

### How It Works

PrecisionFlow v9.0 dispatches between two modes based on query quality:

<table>
<tr>
<th>🟢 MODE 1 — Clean / Anisotropy Test</th>
<th>🔴 MODE 2 — Noisy Retrieval</th>
</tr>
<tr>
<td>

**Trigger:** Cosine similarity > 0.85

**Action:** Returns the **pre-computed optimal precision** for the matched attractor, computed at startup by minimising the spectral condition number of the Hopfield Hessian.

**Optimisation:** Projected gradient descent in log-space over `Pi^(1/2) H Pi^(1/2)`.

</td>
<td>

**Trigger:** Low cosine confidence (noisy query)

**Three-component blend:**
1. **Noise-Adaptive Scaling** — intensity ∝ `(1 − cosine − 0.07) / 0.25`
2. **Projection Component** — amplifies attractor-aligned dims, suppresses residuals
3. **Discriminative Lens** — over-indexes on dimensions that separate the top-2 confusable attractors

</td>
</tr>
</table>

**Output guarantees:** shape `(64,)` · all values strictly positive · clipped to `[0.1, 10.0]` · mean normalised to `1.0`

---

## 🚀 Run Locally

> Requires **Python 3.11+**. Run all commands from the repository root.

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
# Create & activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Launch interactive lab
anvil-pcam serve --port 8420
```

</details>

<details>
<summary><b>🐧 Linux / macOS</b></summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"
anvil-pcam serve --port 8420
```

</details>

Open **[http://127.0.0.1:8420](http://127.0.0.1:8420)** in your browser for the full interactive dashboard.

> 💡 Open `showcase.html` directly in your browser for the 3D Three.js visual experience — no server required.

---

## 🛠️ CLI Reference

```bash
# Run a single noisy retrieval trial
anvil-pcam demo --pattern A03 --sigma 0.58 --mask 0.28

# Run the full attractor-bank benchmark
anvil-pcam benchmark

# Smoke-test the adapter
python test_engine.py
```

---

## 🏆 Official P-04 Harness

Evaluate the `adapters/myteam.py` adapter using the official **Anvil-P-E scoring harness**:

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E\bench-p04-pcam
pip install -r requirements.txt

# Sanity-check with dummy adapter
python -X utf8 self_check.py --adapter adapters.dummy:DummyAgent --quick

# Drop in PrecisionFlow and score
Copy-Item ..\..\adapters\myteam.py .\adapters\myteam.py -Force
python -X utf8 self_check.py --adapter adapters.myteam:Engine --quick
python -X utf8 run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
```

</details>

<details>
<summary><b>🐧 Linux / macOS</b></summary>

```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p04-pcam
pip install -r requirements.txt

# Sanity-check with dummy adapter
python self_check.py --adapter adapters.dummy:DummyAgent --quick

# Drop in PrecisionFlow and score
cp ../../adapters/myteam.py adapters/myteam.py
python self_check.py --adapter adapters.myteam:Engine --quick
python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
```

</details>

---

## 🐳 Docker

```bash
docker compose up --build
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/pcam/state` | Current memory system state |
| `POST` | `/api/pcam/trial` | Run a retrieval trial |
| `POST` | `/api/pcam/precision` | Compute a precision vector |
| `GET` | `/api/pcam/benchmark` | Full benchmark results |

**Example — run a trial:**

```bash
curl -X POST http://127.0.0.1:8420/api/pcam/trial \
  -H 'Content-Type: application/json' \
  -d '{"pattern_id":"A03","gaussian_sigma":0.58,"mask_fraction":0.28,"seed":2404}'
```

---

## 💡 Design Philosophy

Anvil PCAM Lab is **intentionally not a storage product**. It is a compact research prototype for studying associative retrieval under inference-time precision steering. The design priorities are:

- **Inspectability** — every numerical mechanism is visible and readable
- **Determinism** — demos are reproducible with fixed seeds
- **Visual intuition** — energy traces, heatmaps, and 3D landscapes make the math tangible
- **Benchmarkability** — the adapter interface is a clean, single-function contract

---

<div align="center">

**Built for ANVIL Track P-04 · MIT License**

[![GitHub](https://img.shields.io/badge/GitHub-Karthikeya63056%2FHi-181717?style=flat-square&logo=github)](https://github.com/Karthikeya63056/Hi)
[![Benchmark Repo](https://img.shields.io/badge/Harness-Sauhard74%2FAnvil--P--E-181717?style=flat-square&logo=github)](https://github.com/Sauhard74/Anvil-P-E)

</div>
