### Joltz
`joltz` is a straightforward translation of [boltz-1](https://github.com/jwohlwend/boltz) from pytorch to JAX, which is compatable with all the nice features of JAX (JIT/vmap/etc).

For an example of how to load and use the model see the [example script](example.py).

Work in progress, any feedback/PRs welcome!

#### TODO:
- [ ] Finish confidence module
- [ ] Chunking
- [ ] Model cache
- [ ] Dropout
- [ ] Fix structure module on CPU
- [ ] Refactor to remove explicit batching
- [ ] Replace dictionaries with `eqx.Module`s







