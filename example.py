import shutil
import time
from dataclasses import asdict
from pathlib import Path

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

print("Loading torch model")
start_time = time.time()
torch_model = Boltz1.load_from_checkpoint(
    Path("~/.boltz/boltz1_conf.ckpt").expanduser(),
    strict=True,
    map_location="cpu",
    predict_args=predict_args,
    diffusion_process_args=asdict(BoltzDiffusionParams()),
    ema=False,
)
print(f"Boltz1.load_from_checkpoint: {time.time() - start_time: 0.3f}s")
print("Converting to JAX")
start_time = time.time()
joltz_model = joltz.from_torch(torch_model)
print(f"joltz.from_torch(torch_model): {time.time() - start_time: 0.3f}s")


# use the standard boltz data loading machinery

print("Loading data")

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
    num_workers=0,
)

# Load the features for the single example
features_dict = list(data_module.predict_dataloader())[0]

# convert features to numpy arrays
jax_features = {k: np.array(v) for k, v in features_dict.items() if k != "record"}

# We're also compatible with the JIT
jit_joltz = eqx.filter_jit(joltz_model)

# Now we can finally make a prediction
start_time = time.time()
prediction = jit_joltz(jax_features)
print(f"joltz(jax_features): {time.time() - start_time: 0.3f}s")


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
