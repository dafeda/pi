# Agentic history matching: a simple model-discovery example

This experiment adapts ideas from [Murphy's (2026) Model Discovery Agent (MDA)](#relationship-to-murphy-2026) to a simple history-matching problem: inferring a hidden permeability field from synthetic pressure observations.

The LLM proposes geological **model structures**, while **ES-MDA** fits their permeability parameters. Residuals guide revisions, validation scores rank models, and protected final observations test the frozen model's pressure forecasts under predefined changes in injection and production rates.

The experiment:

1. Generates the hidden field and noisy pressure observations.
2. Fits and assesses an initial homogeneous model structure.
3. Lets the agent propose alternative model structures, then uses ES-MDA to fit each structure's permeability parameters to the conditioning observations.
4. Locks in the selected model and its fitted ensemble before the final test.
5. Grades against truth and final observations withheld during search.

**Operator-only:** this README describes the hidden truth. Do not give it to the hypothesis-proposing agent, called the *proposer* in the code.

## Quick start

Requires `uv`, Python 3.13, and credentials for the selected model provider. The default configuration uses an OpenCode pay-as-you-go API key with paid credits.

Set `OPENAI_API_KEY` to your OpenCode key and `OPENAI_BASE_URL` to `https://opencode.ai/inference/openai/v1` in this directory's `.env` (see [`.env.example`](.env.example)).

**Output:** the configured `output` path must not already exist. Choose a different path or delete existing output manually before running. Simultaneous runs must use different output directories.

From the repository root:

```bash
cd agentic-history-matching
uv sync --locked --no-build
uv run --env-file .env python -m controller configs/run.yaml
```

## Fixed hidden geology

![Top-down map of the 100 m × 100 m reservoir, with a 100 mD background, a full-height 1 mD band spanning x = 60–70 m, and four circle-marked wells (west, east, south, north) with diamond-marked pressure gauges at their centers. Three wells and gauges lie west of the band and one lies east.](assets/reservoir.svg)

Circles mark wells; diamonds at their centers mark colocated pressure gauges. Idealized reservoir pressure is measured.

- **Background:** 100 mD.
- **Vertical band:** a low-permeability strip at 1 mD, 10 m wide, centered at x = 65 m and spanning the full y-direction. “Vertical” refers to the 2D map, not depth.

**Wells:** Well locations stay fixed, but their roles change over time. Each well injects fluid, produces fluid, or is off according to the schedule. See [Wells and controls](#wells-and-controls) for details.

## Which measurements can the agent see?

The agent can see the fixed well locations and injection/production schedule, but cannot change them. The true permeability field and parameters are stored under `protected/` with the final observations and are never sent to the proposer.

| Measurements | Purpose | What the agent receives |
| --- | --- | --- |
| **Conditioning**, 0–32 hours | Fit each hypothesis's parameters to these 64 observations. | Raw observations, times/locations, predictions, and residuals. |
| **Validation**, 32–48 hours | Rank hypotheses using 32 observations excluded from parameter fitting. | Validation scores, not raw observations. These scores guide search. |
| **Protected final**, 48–64 hours | Grade the frozen selection using 32 further observations. | Nothing during search. |

The agent can use validation scores to revise its hypotheses, even though it cannot see the validation measurements. Only the protected final period remains independent of search. Truth-generation seeds and truth-recovery metrics must also stay out of proposer context.

## How the experiment works

The experiment separates **model-structure search** from **parameter fitting** through two loops:

- **Outer loop:** The LLM proposes geological model structures using observations and numerical feedback.
- **Inner loop:** The controller holds each structure fixed, fits its parameter ensemble with ES-MDA, and scores its forecasts.

A **hypothesis** is a proposed model structure. Each **candidate** pairs one structure with a parameter ensemble for numerical evaluation.

| Term | Meaning here |
| --- | --- |
| **Model structure**, `m` | The assumed layout: the same permeability everywhere (**homogeneous**), or one vertical or horizontal low-permeability strip (**band**). For a band, the structure also specifies its position and width. The pressure equation stays fixed. |
| **Model parameters**, `θ` | The background permeability and, for a band, its permeability relative to the background (the band/background ratio). ES-MDA adjusts these numbers to fit the pressure measurements. |

The code calls the three pattern types **families**: `homogeneous`, `vertical_band`, and `horizontal_band`.

For example, a background permeability of **100 mD** and a band/background ratio of **0.02** give a band permeability of **2 mD**.

The LLM cannot edit code, run shell commands, or control individual numerical updates.

The experiment runs sequentially: each numerical evaluation completes before the next LLM request. The synchronous CLI uses Pydantic AI's `run_sync()`; async tool callbacks keep numerical work inline rather than dispatching it to a background worker. Numerical, provider, or host failures abort the run instead of freezing a fallback decision or continuing with failed candidates. Final summaries are written only after normal completion; files already written before a failure may remain, but there is no recovery guarantee.

### A complete agent run

The controller, Python proposer, and evaluator below are components of `agentic-history-matching`. A Pydantic AI `Agent` owns conversation history, typed tool validation, dispatch, and the model-turn limit. Request hooks record campaign spending and transcripts, reject incomplete responses, and allow only one action per response. The **LLM proposer** chooses model structures. There is no custom model wrapper, context-size estimate, repeated-call-ID tracking, or duplicate-JSON-key check; argument validation uses Pydantic AI's typed tools.

```text
agentic-history-matching prepares hidden truth and observations
  → Controller reserves simulator budget, then fits and scores the initial homogeneous structure
  → Proposer host supplies agent-visible observations, completed results, and remaining budgets
  → LLM proposer chooses one action with its rationale:
      evaluate_hypothesis
        → Controller validates the proposal and reserves simulator budget
        → Controller samples a parameter prior ensemble once and runs inline:
            [generate fields → simulate conditioning history → ES-MDA parameter update] × 4
              → full-history forecast → conditioning/validation scores and residuals
        → Completed scores are returned to the next LLM turn
      finish_search (or host-enforced stop)
        → Controller ranks successful candidates by validation CRPS
          and freezes the selected structure/ensemble
  → Protected evaluator verifies the frozen selection, then reports final scores and truth recovery
```

Each successful candidate uses **250 simulator calls** by default: **50 realizations** are simulated in each of **four ES-MDA iterations**, then once more for the full-history forecast.

### What instructs the agent?

The LLM's instructions are `SYSTEM_PROMPT` in [`controller/proposer.py`](controller/proposer.py). Each turn also receives agent-visible information, evaluated hypotheses/results, and remaining budgets.

The agent uses **residuals**—simulated pressure minus measured pressure—to identify mismatches and propose a different permeability layout. The prompt asks it to explain these mismatches using the conditioning observations and known well controls. Validation scores also guide its proposals.

Residuals supplied to the agent use the ensemble mean pressure and are divided by the measurement-noise standard deviation:

```text
(mean simulated pressure - measured pressure) / measurement-noise standard deviation
```

Diagnostics are grouped by gauge, time, and pulse. Agent-visible pressure observations are changes from initial pressure in bar.

The agent has two tools:

- `evaluate_hypothesis`: validate, fit, and score one candidate before returning completed results.
- `finish_search`: freeze an `adequate`, `inadequate`, or `uncertain` outcome. An `adequate` claim must pass the numerical error check below; `uncertain` allows insufficient evidence or budget.

The **hypothesis grammar**, the allowed format for model-structure proposals, permits homogeneous permeability or **one** low-permeability band. The LLM chooses orientation, center, and width. For example:

```json
{
  "hypothesis": {
    "kind": "band",
    "orientation": "horizontal",
    "center_fraction": 0.4,
    "width_fraction": 0.2
  },
  "reason": "Test whether a horizontal band explains the directional pressure residuals."
}
```

A band runs across the map, is 10–50% wide by default, and must cover at least two rows or columns while leaving space on both sides.

The band's orientation, position, and width stay fixed while ES-MDA fits the permeability values. ES-MDA updates background permeability and **contrast**, the band/background permeability ratio. This lets the prior describe the band's relative permeability rather than an independent absolute value. Log and logit transforms enforce positive permeability and a less-permeable band without clipping; this is a modeling choice, not an ES-MDA requirement.

Priors, likelihood, observation error, solver, cutoffs, and scoring remain controller-owned and fixed. Diagonal bands, multiple bands, and new pattern types are unsupported.

Bands with the same orientation that cover the same grid cells reuse the existing evaluation and simulation reservation. Repeated model requests still consume the model-request budget.

The prompt discourages equivalent repeats but does not prescribe a vertical-then-horizontal sequence. YAML has no prompt or strategy setting.

This **adequacy gate** is a prototype noise-based predictive check, **not proof of geological truth** or a statistically calibrated adequacy test.

## What the simulator calculates

The simulator uses [FiPy](https://www.ctcms.nist.gov/fipy/) to solve pressure diffusion:

$$
S\frac{\partial p}{\partial t}
=\nabla\cdot\left(\frac{k(x;m,\theta)}{\mu}\nabla p\right)+q(x,t)
$$

| Term | Meaning |
| --- | --- |
| $p(x,t)$ | Fluid pressure (Pa) at spatial position $x$ (2D, in m) and time $t$ (s). |
| $k(x;m,\theta)$ | Permeability field (m²), determined by layout $m$ and parameters $\theta$. |
| $m$ | Layout: homogeneous or a band with specified orientation, position, and width. |
| $\theta$ | Permeability parameters: background permeability and, for a band, the band/background ratio. |
| $S$ | Storage coefficient: fluid volume stored per bulk volume per unit pressure increase; fixed at `1e-9` Pa⁻¹. |
| $\mu$ | Fluid dynamic viscosity; fixed at `1e-3` Pa·s. |
| $q(x,t)$ | Injection/production rate per bulk volume (s⁻¹); positive injects, negative produces. |
| $\partial p/\partial t$ | Pressure-change rate (Pa/s). |
| $\nabla p$ | Spatial pressure gradient (Pa/m). |
| $\nabla\cdot$ | Divergence, measuring a vector field's net outward flux per unit volume. |

The equation predicts pressure changes caused by flow, injection, and production. It combines fluid conservation with **Darcy's law**, $\mathbf{u}=-(k/\mu)\nabla p$, where $\mathbf{u}$ is volumetric flow per bulk cross-sectional area (m/s), directed from high to low pressure.

Storage, viscosity, and initial pressure are fixed; exterior boundaries are no-flow. Gravity and multiphase flow are excluded; this is a simplified pressure-inference example, not a realistic reservoir model.

### Wells and controls

Coordinates are fractions of domain length, in `(x, y)` order. Well and colocated gauge arrays both follow the order below:

| Well and gauge | Fractional coordinates |
| --- | --- |
| `west` | `(0.20, 0.25)` |
| `east` | `(0.80, 0.25)` |
| `south` | `(0.25, 0.20)` |
| `north` | `(0.25, 0.80)` |

Controls are balanced source/sink pairs. Rates below are multipliers of **`Q = 1e-4` m³/s**, in west/east/south/north well order. Positive rates inject; negative rates produce. Each row applies for eight one-hour steps.

The table below shows the well schedule for the 64-hour experiment, split into conditioning, validation, and protected final periods.

| Time (hours) | Steps | West / Q | East / Q | South / Q | North / Q | Use |
| --- | --- | --- | --- | --- | --- | --- |
| 0–8 | 1–8 | +1.0 | -1.0 | 0 | 0 | Conditioning |
| 8–16 | 9–16 | 0 | 0 | 0 | 0 | Conditioning |
| 16–24 | 17–24 | 0 | 0 | +1.0 | -1.0 | Conditioning |
| 24–32 | 25–32 | 0 | 0 | 0 | 0 | Conditioning |
| 32–40 | 33–40 | -0.6 | +0.6 | 0 | 0 | Validation |
| 40–48 | 41–48 | 0 | 0 | 0 | 0 | Validation |
| 48–56 | 49–56 | +0.4 | -0.4 | -0.8 | +0.8 | Protected final |
| 56–64 | 57–64 | 0 | 0 | 0 | 0 | Protected final |

## How parameter fitting works

For each proposed layout, ES-MDA keeps the geometry fixed and jointly updates these parameters:

| Parameter | Prior | Interpretation |
| --- | --- | --- |
| Background permeability, `k_bg` | `log(k_bg / mD) ~ Normal(log(100), 0.7)` | Positive permeability with prior median **100 mD**. |
| Band/background permeability ratio, `r`, when a band is present | `logit(r) ~ Normal(logit(0.02), 1.3)` | Ratio between zero and one, with prior median **0.02**. |

Band permeability is derived as `k_band = r × k_bg`; it is not fitted independently.

## Selection and evaluation

The controller ranks successful candidates by validation **CRPS**, *continuous ranked probability score*. These scores assess pressure predictions, not the probability that a model is true.

CRPS scores the predictive ensemble, including its spread, rather than just the mean prediction. For one observation:

```text
mean |prediction - observation|
  - 0.5 × average |prediction_i - prediction_j| over all realization pairs
```

Scores are averaged over observations; lower is better. Predictive ensembles include measurement error before scoring noisy observations.

## Relationship to Murphy (2026)

Murphy's MDA learns **mechanistic world models**: scientifically meaningful models that explain observations and predict responses to interventions. This experiment keeps the pressure-diffusion equation fixed and searches only the permeability representation. It adapts MDA's separation between LLM structure proposals and numerical parameter fitting, not the full algorithm (Section 3; Appendix A.3 and A.8). MDA names Murphy's discovery agent, not the ES-MDA inference algorithm.

- **Bounded hypothesis-space expansion:** in Murphy's **M-open** setting, the current hypothesis class can be insufficient and is expanded through new proposals. Here the evaluated candidate pool can grow from the homogeneous model to band structures, but the reviewed grammar never changes; the fixed truth lies within that grammar. This is a bounded analogue of M-open expansion, not open-ended mechanism discovery.
- **Parameter fitting and ranking models by validation scores:** ES-MDA fits the permeability values while holding each proposed layout fixed. Validation **CRPS** (continuous ranked probability score) ranks candidates; the controller does not compute marginal likelihoods, posterior model probabilities, or Bayesian model averaging.
- **Fixed experiment design:** well locations, gauges, and injection/production controls are prescribed. MDA's Value-of-Information experiment selection—choosing what to measure next for its expected benefit—is not implemented.
- **Independent final assessment:** validation scores guide proposals and therefore participate in search. Only protected final observations remain independent of search and assess the frozen selection without refitting.

Bayesian model weighting, adaptive experiment design, and proposals outside the grammar are possible future extensions, not implemented features.

## References

- Murphy, Kevin (2026). *Model Discovery Agent: LLM-Assisted Bayesian Experiment Design for Data-Efficient Discovery of Mechanistic World Models*. Section 3 and Appendix A.3/A.8. Reference key: `murphy2026`; source document: `murphy2026_model_discovery_agent.pdf`.
