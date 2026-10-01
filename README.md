# multimodal-agent-defer

An image + clinical-note anomaly-detection agent system with an uncertainty-based 'defer to human' rule.

**Research question:** Can an agent reading an image and a note know when to defer?

**Paper:** _arXiv link — TBD (add here as soon as the preprint is posted)_

## Goal

Clean up the multi-agent anomaly detection system, add an uncertainty-based defer rule, and measure how often it defers correctly versus wrongly. Ablation: single agent vs consensus (and entropy vs disagreement as the uncertainty signal).

## Data

Run your agents on a labeled set of (image, note) cases and save each agent's P(anomaly) to `data/agent_probs_{val,test}.csv` (columns `label, agent_0, agent_1, ...`). Use a held-out validation split to set the defer threshold. The synthetic generator is only for the smoke test and demo.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (one command)

```bash
make run        # ablation on your agent outputs -> results/
make smoke      # synthetic sanity check
make demo       # prints ACCEPT / DEFER decisions for example cases
```

`make smoke` runs the whole pipeline end-to-end on synthetic data in under a minute and checks that the code works. The numbers it produces are NOT results.

All experiments use fixed seeds (`seed: 42` in the configs, applied through `utils.set_seed`). After your final run, lock the environment with `pip freeze > requirements.lock.txt`.

## Repository layout

```
agent_defer/
  agents.py      agent interface (plug in your agents)
  consensus.py   single / consensus strategies + uncertainty measures
  defer.py       threshold tuning and defer statistics
  evaluate.py    ablation table + risk-coverage figure
  demo.py        CLI demo
configs/ results/ paper/
```

## Results

### Ablation (TBD after real runs)

| strategy | coverage | selective acc | defer precision | error recall | wrong-defer rate | missed-error rate |
|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

`defer precision` = deferred cases that the agent would have gotten wrong; `error recall` = wrong cases that were deferred; `wrong-defer rate` = unnecessary deferrals / all cases; `missed-error rate` = wrong and auto-accepted / all cases. The threshold is tuned on validation to cap selective error at `max_selective_error`.

## Main figure

![risk-coverage](results/risk_coverage.png)

## Limitations

- `agents.py` is an interface only; connect your system and produce the probability CSVs.
- LLM-verbalized probabilities are poorly calibrated; consider calibrating each agent on validation before consensus.
- Agents with correlated errors limit what consensus can detect.

## Citation

See `CITATION.cff`. A BibTeX entry will be added once the arXiv paper is out.
