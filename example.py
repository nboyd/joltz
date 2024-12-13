import shutil
import time
from dataclasses import asdict
from pathlib import Path

import boltz
import equinox as eqx
import jax
import numpy as np
import torch
from boltz.main import (
    BoltzDiffusionParams,
    BoltzInferenceDataModule,
    BoltzProcessedInput,
    BoltzWriter,
    Manifest,
    check_inputs,
    process_inputs,
)
from boltz.model.model import Boltz1
from jax import numpy as jnp
from optax.losses import softmax_cross_entropy

import joltz

predict_args = {
    "recycling_steps": 3,
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "write_confidence_summary": True,
    "write_full_pae": True,
    "write_full_pde": True,
}


# load model and convert to JAX

joltz = joltz.from_torch(
    Boltz1.load_from_checkpoint(
        Path("~/.boltz/boltz1_conf.ckpt").expanduser(),
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
    )
)


# use the standard boltz data loading machinery

out_dir = Path("test_prediction")
cache = Path("~/.boltz/").expanduser()
data = check_inputs(Path("test.fasta"), out_dir, override=True)
# Process inputs
ccd_path = cache / "ccd.pkl"
process_inputs(
    data=data,
    out_dir=out_dir,
    ccd_path=ccd_path,
    use_msa_server=True,
    msa_server_url="https://api.colabfold.com",
    msa_pairing_strategy="greedy",
)
# Load processed data
processed_dir = out_dir / "processed"
processed = BoltzProcessedInput(
    manifest=Manifest.load(processed_dir / "manifest.json"),
    targets_dir=processed_dir / "structures",
    msa_dir=processed_dir / "msa",
)

# Create data module
data_module = BoltzInferenceDataModule(
    manifest=processed.manifest,
    target_dir=processed.targets_dir,
    msa_dir=processed.msa_dir,
    num_workers=1,
)

# Load the features for the single example
features_dict = list(data_module.predict_dataloader())[0]

# convert features to numpy arrays
jax_features = {k: np.array(v) for k, v in features_dict.items() if k != "record"}

# Now we can finally make a prediction
start_time = time.time()
prediction = joltz(jax_features)
print(f"joltz(jax_features): {time.time() - start_time: 0.3f}s")

# We're also compatible with the JIT

jit_joltz = eqx.filter_jit(joltz)
start_time = time.time()
jit_joltz(jax_features)  # slow
print(f"jit_joltz(jax_features): {time.time() - start_time: 0.3f}s")

start_time = time.time()
jit_joltz(jax_features)  # fast (?)
print(f"jit_joltz(jax_features): {time.time() - start_time: 0.3f}s")

# We can also run the structure module

start_time = time.time()
structure_prediction = jit_joltz(
    jax_features, key=jax.random.key(0), sample_structure=True
)
print(
    f"jit_joltz(jax_features, sample_structure=True): {time.time() - start_time: 0.3f}s"
)
start_time = time.time()
structure_prediction = jit_joltz(
    jax_features, key=jax.random.key(0), sample_structure=True
)
print(
    f"jit_joltz(jax_features, sample_structure=True): {time.time() - start_time: 0.3f}s"
)


# If we want to save a .pdb or .cif to disk we can again hijack the boltz machinery

pred_writer = BoltzWriter(
    data_dir=processed.targets_dir,
    output_dir=out_dir / "predictions",
    output_format="pdb",
)
_pred_dict = {
    "exception": False,
    "coords": torch.tensor(
        np.array(structure_prediction["sample_atom_coords"])
    ).unsqueeze(0),
    "masks": features_dict["atom_pad_mask"].unsqueeze(0),
    "confidence_score": torch.ones(1),
}
pred_writer.write_on_batch_end(
    None,
    None,
    _pred_dict,
    None,
    {"record": features_dict["record"]},
    None,
    None,
)
# Not sure how pred_writer works so I'm just going to move the output .pdb so it doesn't get clobbered
shutil.copy(
    out_dir / "predictions/test/test_model_0.pdb",
    out_dir / "predictions/test/original.pdb",
)

# should write a structure to out_dir/predictions


# Let's do something more interesting: RSO (https://www.science.org/doi/10.1126/science.adq1741) to redesign this backbone.

target_distogram = jax.nn.softmax(jnp.array(prediction["pdistogram"]))


def loss_function(seq_relaxed, inputs, model, target, *, key):
    with jax.default_matmul_precision("float32"):
        # replace sequence and MSA with seq_relaxed
        inputs = inputs | {
            "res_type": 2 * seq_relaxed,
            "profile": 2 * seq_relaxed,
        }
        o = model(inputs)
        p_dist_logits = o["pdistogram"]
        return softmax_cross_entropy(p_dist_logits, target).mean()


j_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss_function))


def norm_seq_grad(g):
    eff_L = (jnp.square(g).sum(-1, keepdims=True) > 0).sum(-2, keepdims=True)
    gn = jnp.linalg.norm(g, axis=(-1, -2), keepdims=True)
    return g * jnp.sqrt(eff_L) / (gn + 1e-7)


seq_features = jnp.array(0.3 * np.random.randn(*jax_features["res_type"].shape)).astype(
    jnp.float32
)

for _iter in range(300):
    v, g = j_grad(
        seq_features, jax_features, joltz, target_distogram, key=jax.random.key(0)
    )
    seq_features = jax.tree.map(
        lambda a, b: a - 0.1 * b, seq_features, norm_seq_grad(g)
    )
    if _iter % 20 == 0:
        print(_iter, v)

# repredict with structure module

redesigned = jit_joltz(
    jax_features | {"res_type": 2 * seq_features, "profile": 2 * seq_features},
    key=jax.random.key(0),
    sample_structure=True,
)

# save redesigned structure using boltz machinery
_pred_dict = {
    "exception": False,
    "coords": torch.tensor(np.array(redesigned["sample_atom_coords"])).unsqueeze(0),
    "masks": features_dict["atom_pad_mask"].unsqueeze(0),
    "confidence_score": torch.ones(1),
}
pred_writer.write_on_batch_end(
    None,
    None,
    _pred_dict,
    None,
    {"record": features_dict["record"]},
    None,
    None,
)
shutil.copy(
    out_dir / "predictions/test/test_model_0.pdb",
    out_dir / "predictions/test/redesigned.pdb",
)
