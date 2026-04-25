from caspo.rollout.sampler import HFRolloutSampler, RolloutBatch

__all__ = ["HFRolloutSampler", "RolloutBatch", "build_rollout_engine"]


_VALID_BACKENDS = ("hf", "vllm")


def build_rollout_engine(cfg, reward_fn, **kwargs):
    """Construct a rollout sampler/engine based on ``cfg.rollout_backend``.

    - ``hf`` (default): :class:`HFRolloutSampler`. Requires the policy model
      to be passed in via ``kwargs['model']`` and tokenizer via ``kwargs['tokenizer']``.
      Any additional kwargs are forwarded to ``HFRolloutSampler``.
    - ``vllm``: :class:`VLLMRolloutEngine`. Constructs its own model + tokenizer
      from ``cfg.model_name_or_path``. Subsequent calls to ``sync_weights_from_path``
      reload the weights from a trainer-saved checkpoint. ``model`` and
      ``tokenizer`` kwargs (if present) are stripped before forwarding so that
      callers can pass a single uniform kwargs dict regardless of backend.

    Returns an object with a ``sample(examples) -> RolloutBatch`` method.

    Raises:
        ValueError: if ``cfg.rollout_backend`` is not one of ``hf`` or ``vllm``.
        KeyError: if backend is ``hf`` and ``model`` or ``tokenizer`` is missing
            from ``kwargs``.
    """
    backend = getattr(cfg, "rollout_backend", "hf")
    if not isinstance(backend, str):
        raise ValueError(
            f"cfg.rollout_backend must be a string, got {type(backend).__name__}: {backend!r}; "
            f"must be one of {_VALID_BACKENDS}"
        )
    backend = backend.lower()

    if backend == "hf":
        missing = [k for k in ("model", "tokenizer") if k not in kwargs]
        if missing:
            raise KeyError(
                f"build_rollout_engine(backend='hf') requires kwargs {missing}; "
                f"the HF rollout sampler reuses the trainer's policy model and tokenizer"
            )
        model = kwargs.pop("model")
        tokenizer = kwargs.pop("tokenizer")
        return HFRolloutSampler(model, tokenizer, cfg, reward_fn, **kwargs)

    if backend == "vllm":
        from caspo.rollout.vllm_engine import VLLMRolloutEngine
        # vLLM constructs its own model/tokenizer from cfg.model_name_or_path,
        # so silently drop these if the caller passed them in.
        engine_kwargs = {
            k: v for k, v in kwargs.items() if k not in {"model", "tokenizer"}
        }
        return VLLMRolloutEngine(cfg, reward_fn, **engine_kwargs)

    raise ValueError(
        f"unknown rollout_backend {backend!r}; must be one of {_VALID_BACKENDS}"
    )
