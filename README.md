### Joltz
`joltz` is a straightforward translation of [boltz-1 (and boltz-2!)](https://github.com/jwohlwend/boltz) from pytorch to JAX, which is compatible with all the nice features of JAX (JIT/vmap/etc).

This is primarily used for protein design using hallucination see [boltz-binder-design](https://github.com/escalante-bio/boltz-binder-design).

For a bare-bones example of how to load and use the model see the [example script](example.py).

Work in progress, collaboration/feedback/PRs welcome!

Tested with boltz 2.0.3; will almost certainly break with more recent versions.

#### TODO:
- [ ] Chunking ?
- [ ] Replace dictionaries with `eqx.Module`s
- [ ] Tastefully sprinkle some `jax.lax.stop_grad`s in Boltz-2
- [ ] Finish boltz-2 confidence module
- [ ] Implement affinity module






