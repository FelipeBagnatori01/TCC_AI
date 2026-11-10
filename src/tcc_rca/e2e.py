"""End-to-end smoke pipeline exercising seed, tracking, LSTM stub and the LLM client.

It validates the E0 infrastructure only; no OpenRCA data is involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from tcc_rca.config import Config
from tcc_rca.llm import LLMClient, Message
from tcc_rca.tracking import Run


def _tiny_lstm_metrics() -> dict[str, float]:
    """Seed-dependent computation using NumPy data and a PyTorch LSTM."""
    data = torch.from_numpy(np.random.rand(4, 8, 3).astype(np.float32))
    lstm = nn.LSTM(input_size=3, hidden_size=5, batch_first=True)
    with torch.no_grad():
        out, _ = lstm(data)
    return {
        "lstm_output_sum": float(out.sum().item()),
        "lstm_output_last_mean": float(out[:, -1, :].mean().item()),
    }


def run_e2e(config: Config) -> Path:
    """Run the smoke pipeline and return the run directory."""
    with Run(config) as run:
        run.log_metrics(_tiny_lstm_metrics())

        client = LLMClient(config.llm, call_log=run.llm_call_log)
        prompt = (Message(role="user", content="ping"),)
        client.complete(prompt)
        client.complete(prompt)  # identical call: must be served from cache, at zero cost
        # LLM usage depends on cache state (not on the seed), so it is kept apart from metrics.
        run.write_artifact("llm_usage.json", client.usage.as_dict())
        run.logger.info("llm usage: %s", client.usage.as_dict())
        return run.dir
