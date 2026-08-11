"""Numerical parity test: torch Boltz-2 AffinityModule vs the JAX/joltz port.

Loads the real affinity weights from the public ``boltz2_aff.ckpt`` checkpoint,
builds the torch reference module and its ``joltz.from_torch`` translation, feeds
both identical synthetic-but-valid inputs, and checks that the predicted affinity
value and binding logit agree to a tight tolerance across several seeds and
multiplicities.  Also checks the whole point of the port: that P(bind) is
differentiable w.r.t. its inputs.

Run:
    python tests/test_affinity_parity.py
    # or: pytest tests/test_affinity_parity.py
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import jax

import boltz.model.modules.affinity as boltz_affinity
import joltz

CKPT = Path(os.environ.get("BOLTZ2_AFF_CKPT", Path.home() / ".boltz" / "boltz2_aff.ckpt"))
TOL = 2e-3  # max abs error on the head outputs (float32 CPU vs jax)

# The 1.9 GB affinity checkpoint is not shipped; skip when it isn't cached.
pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason=f"affinity checkpoint not found at {CKPT}"
)


def _build_torch_affinity_module(member=1):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    token_s, token_z = hp["token_s"], hp["token_z"]

    args = dict(hp[f"affinity_model_args{member}"])
    # Disable GPU-only / training-only paths so the reference runs on CPU.
    args["pairformer_args"] = {
        **args["pairformer_args"],
        "use_trifast": False,
        "activation_checkpointing": False,
    }
    args["transformer_args"] = {
        **args["transformer_args"],
        "activation_checkpointing": False,
    }

    module = boltz_affinity.AffinityModule(token_s, token_z, **args)

    prefix = f"affinity_module{member}."
    sub_sd = {
        k[len(prefix):]: v
        for k, v in ck["state_dict"].items()
        if k.startswith(prefix)
    }
    missing, unexpected = module.load_state_dict(sub_sd, strict=False)
    # Strict correspondence: every checkpoint weight is consumed and every model
    # parameter is filled -- otherwise parity would compare shared random inits.
    assert not missing and not unexpected, (
        f"checkpoint did not map cleanly: missing={missing[:5]} unexpected={unexpected[:5]}"
    )
    module.eval()
    return module, token_s, token_z


def _synthetic_inputs(token_s, token_z, n_tokens=24, seed=0, multiplicity=1):
    g = torch.Generator().manual_seed(seed)
    B, N, M = 1, n_tokens, n_tokens  # one rep atom per token

    # s_inputs / x_pred carry the multiplicity dim (the module repeats z, the
    # token->atom map and the masks internally, but assumes s_inputs/x_pred are
    # already at multiplicity -- same contract as the torch caller).
    s_inputs = torch.randn(multiplicity, N, token_s, generator=g)
    z = torch.randn(B, N, N, token_z, generator=g) * 0.1
    x_pred = torch.randn(multiplicity, M, 3, generator=g) * 10.0

    token_to_rep_atom = torch.eye(M).unsqueeze(0)  # [1, N, M] identity map
    token_pad_mask = torch.ones(B, N)
    mol_type = torch.zeros(B, N, dtype=torch.long)  # 0 == protein/receptor
    affinity_token_mask = torch.zeros(B, N)
    affinity_token_mask[:, -N // 3:] = 1.0          # last third = ligand
    mol_type[:, -N // 3:] = 3                        # NONPOLYMER

    feats = {
        "token_to_rep_atom": token_to_rep_atom,
        "token_pad_mask": token_pad_mask,
        "mol_type": mol_type,
        "affinity_token_mask": affinity_token_mask,
    }
    return s_inputs, z, x_pred, feats


def _run_pair(module, jax_module, token_s, token_z, seed, multiplicity):
    s_inputs, z, x_pred, feats = _synthetic_inputs(
        token_s, token_z, seed=seed, multiplicity=multiplicity
    )
    with torch.no_grad():
        torch_out = module(s_inputs, z, x_pred, feats, multiplicity=multiplicity)

    jfeats = {k: joltz.from_torch(v) for k, v in feats.items()}
    with jax.default_matmul_precision("float32"):
        jax_out = jax_module(
            joltz.from_torch(s_inputs),
            joltz.from_torch(z),
            joltz.from_torch(x_pred),
            jfeats,
            multiplicity=multiplicity,
            deterministic=True,
        )

    for key in ("affinity_pred_value", "affinity_logits_binary"):
        t = torch_out[key].detach().numpy()
        j = np.asarray(jax_out[key])
        assert t.shape == j.shape, f"{key} shape {t.shape} != {j.shape}"
        err = np.abs(t - j).max()
        assert err < TOL, (
            f"[m{multiplicity} seed{seed}] {key} parity failed: {err:.3e} > {TOL}"
        )
    return err


def test_affinity_parity():
    for member in (1, 2):  # both ensemble members (8- and 4-block pairformers)
        module, token_s, token_z = _build_torch_affinity_module(member)
        jax_module = joltz.from_torch(module)
        worst = 0.0
        for multiplicity in (1, 2):        # exercise the repeat_interleave path
            for seed in (0, 1, 2):         # robustness across inputs/geometries
                worst = max(worst, _run_pair(
                    module, jax_module, token_s, token_z, seed, multiplicity))
        print(f"[affinity_module{member}] worst max_abs_err over "
              f"seeds/mult = {worst:.3e}")
    print("PARITY OK")


def test_affinity_differentiable():
    """The reason for the JAX port: P(bind) must be differentiable w.r.t. inputs."""
    module, token_s, token_z = _build_torch_affinity_module(1)
    s_inputs, z, x_pred, feats = _synthetic_inputs(token_s, token_z)
    jax_module = joltz.from_torch(module)
    jfeats = {k: joltz.from_torch(v) for k, v in feats.items()}
    zj, xj = joltz.from_torch(z), joltz.from_torch(x_pred)

    def loss(s):
        out = jax_module(s, zj, xj, jfeats, multiplicity=1, deterministic=True)
        return out["affinity_logits_binary"].sum()

    grad = np.asarray(jax.grad(loss)(joltz.from_torch(s_inputs)))
    assert np.isfinite(grad).all(), "non-finite gradient"
    assert np.abs(grad).max() > 0, "zero gradient (no signal)"
    print(f"grad(P(bind) logit) wrt s_inputs: finite, max|grad|={np.abs(grad).max():.3e}")
    print("DIFFERENTIABLE OK")


if __name__ == "__main__":
    test_affinity_parity()
    test_affinity_differentiable()
