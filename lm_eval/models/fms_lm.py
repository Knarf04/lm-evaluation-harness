import logging

import torch

import lm_eval.models.utils_hf
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


def _strip_compiled_prefix(sd):
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


@register_model("fms")
class FMSLMWrapper(HFLM):
    """
    lm-eval-harness model adapter for FMS LLaMA checkpoints via HF-adapted wrapper.

    Extends HFLM to leverage all its evaluation infrastructure (batching,
    caching, loglikelihood, generation) while loading models via FMS's
    sharded checkpoint system and HF-adapted wrappers.

    Currently only supports LLaMA architectures.

    Expected model_args:
      pretrained: str  (path to checkpoint directory readable by FileSystemReader)
      variant: str     (e.g. "llama_7b" — must be "<arch>_<variant>")
      tokenizer: str   (HF tokenizer name/path, e.g. "meta-llama/Llama-2-7b-hf")
      + all HFLM args (batch_size, max_length, dtype, device, trust_remote_code, etc.)

    Example usage:
      lm_eval --model fms \\
        --model_args pretrained=/path/to/checkpoint,variant=llama_7b,tokenizer=meta-llama/Llama-2-7b-hf \\
        --tasks hellaswag \\
        --batch_size 8
    """

    def __init__(
        self,
        pretrained: str,
        variant: str,
        **kwargs,
    ) -> None:
        parts = variant.split("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"variant must be '<arch>_<variant>' (e.g. 'llama_7b'), got '{variant}'"
            )
        self.variant = variant
        if "backend" in kwargs and kwargs["backend"] != "causal":
            raise ValueError(
                f"FMS only supports causal (decoder-only) models, got backend='{kwargs['backend']}'"
            )

        # Load FMS checkpoint on CPU BEFORE super().__init__() triggers
        # Accelerator, which initializes torch.distributed and changes how
        # torch.distributed._shard.checkpoint.load() behaves (each rank
        # would only load its shard instead of the full checkpoint).
        self._fms_model_cpu = self._load_fms_checkpoint(pretrained, variant)

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )

    @staticmethod
    def _load_fms_checkpoint(pretrained: str, variant: str):
        """Load FMS model + sharded checkpoint on CPU, before distributed init."""
        try:
            from fms import models
            from fms.models.llama import LLaMA, _llama_factory_factory
            from fms_fsdp.utils.config_utils import get_model_config
        except ModuleNotFoundError as exc:
            raise type(exc)(
                "attempted to use 'fms' LM type, but packages `fms` and "
                "`fms_fsdp` are not installed."
            ) from exc
        from torch.distributed._shard.checkpoint import FileSystemReader, load

        arch, var = variant.split("_", 1)
        if arch != "llama":
            raise NotImplementedError(
                f"Only llama architecture is supported (got arch={arch})"
            )

        config_data = get_model_config(variant)
        models.register_model(arch, var, _llama_factory_factory(config_data))

        fms_model = LLaMA(config_data)
        if pretrained.endswith('.pth'):
            ckpt = torch.load(pretrained, map_location="cpu")
            fms_model.load_state_dict(_strip_compiled_prefix(ckpt["model_state"]))
        else:
            state_dict = {"model_state": fms_model.state_dict()}
            load(state_dict=state_dict, storage_reader=FileSystemReader(pretrained))
            fms_model.load_state_dict(_strip_compiled_prefix(state_dict["model_state"]))

        return fms_model

    def _get_config(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        self._config = {}

    def _create_model(
        self,
        pretrained: str,
        dtype: str | torch.dtype | None = "auto",
        parallelize: bool | None = False,
        **kwargs,
    ) -> None:
        assert not parallelize, (
            "FMSLMWrapper does not support model parallelism (parallelize=True). "
            "Use data parallelism via `accelerate launch` instead."
        )
        from fms.models.hf.llama.modeling_llama_hf import (
            HFAdaptedLLaMAForCausalLM,
            HFAdaptedLLaMAConfig,
        )

        fms_model = self._fms_model_cpu

        # Resolve dtype — default to bfloat16 for FMS models
        _dtype = (
            lm_eval.models.utils_hf.get_dtype(dtype)
            if dtype is not None and dtype != "auto"
            else torch.bfloat16
        )
        fms_model.to(self._device, dtype=_dtype)

        torch.set_grad_enabled(False)
        fms_model.eval()

        # Convert to HF-adapted causal LM.
        # Must disable weight init: from_fms_model triggers PreTrainedModel.__init__
        # -> post_init() -> init_weights() which re-initializes all submodule weights.
        from transformers.modeling_utils import no_init_weights

        fms_hf_config = HFAdaptedLLaMAConfig.from_fms_config(fms_model.get_config())
        with no_init_weights():
            self._model = HFAdaptedLLaMAForCausalLM.from_fms_model(
                fms_model, **fms_hf_config.to_dict()
            )
        self._model.eval()

        self._config = self._model.config

        # Clean up reference
        del self._fms_model_cpu
