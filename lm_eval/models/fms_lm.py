import logging

import torch

import lm_eval.models.utils_hf
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


def _patch_triton_autotuner_compat():
    """Triton 3.1.0 excludes tl.constexpr params from fn.arg_names, but fla
    kernels use them as autotuner keys (e.g. STAGE, H, K, V, BT).

    Two patches:
      __init__: filter key list to only params in arg_names (fixes import crash)
      run:      catch IndexError from stale key_idx and fall back to first config
                (fixes runtime crash from arg-index mismatch)
    """
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return
    if getattr(Autotuner, '_fms_patched', False):
        return

    _orig_init = Autotuner.__init__
    _orig_run = Autotuner.run

    def _patched_init(self, fn, arg_names, configs, key, *args, **kwargs):
        key = [k for k in key if k in arg_names]
        return _orig_init(self, fn, arg_names, configs, key, *args, **kwargs)

    def _patched_run(self, *args, **kwargs):
        try:
            return _orig_run(self, *args, **kwargs)
        except IndexError:
            # key_idx out of range — constexpr/arg_names mismatch at runtime.
            # Fall back to first config (correct but may skip autotuning).
            config = self.configs[0]
            self.best_config = config
            if config.pre_hook is not None:
                config.pre_hook({
                    **dict(zip(self.arg_names, args)),
                    **kwargs,
                    **config.all_kwargs(),
                })
            ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
            self.nargs = None
            return ret

    Autotuner.__init__ = _patched_init
    Autotuner.run = _patched_run
    Autotuner._fms_patched = True


def _strip_compiled_prefix(sd):
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


@register_model("fms")
class FMSLMWrapper(HFLM):
    """
    lm-eval-harness model adapter for FMS checkpoints via HF-adapted wrappers.

    Extends HFLM to leverage all its evaluation infrastructure (batching,
    caching, loglikelihood, generation) while loading models via FMS's
    sharded checkpoint system and HF-adapted wrappers.

    Supported architectures: llama, gdn (Gated DeltaNet via fla).

    Expected model_args:
      pretrained: str  (path to checkpoint .pth file or distributed checkpoint dir)
      variant: str     (e.g. "llama_7b" or "gdn_100m" — must be "<arch>_<variant>")
      tokenizer: str   (HF tokenizer name/path, e.g. "meta-llama/Llama-2-7b-hf")
      + all HFLM args (batch_size, max_length, dtype, device, trust_remote_code, etc.)

    Example usage:
      lm_eval --model fms \\
        --model_args pretrained=/path/to/checkpoint,variant=llama_7b,tokenizer=meta-llama/Llama-2-7b-hf \\
        --tasks hellaswag \\
        --batch_size 8

    Data parallelism via accelerate launch is supported. Model parallelism is not.
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
        self._arch = parts[0]
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
        """Load FMS model + checkpoint on CPU, before distributed init.

        Supports llama (via FMS) and gdn (via fla) architectures.
        """
        try:
            from fms_fsdp.utils.config_utils import get_model_config
        except ModuleNotFoundError as exc:
            raise type(exc)(
                "attempted to use 'fms' LM type, but package `fms_fsdp` is not installed."
            ) from exc
        from torch.distributed._shard.checkpoint import FileSystemReader, load

        arch, var = variant.split("_", 1)
        config_data = get_model_config(variant)

        if arch == "llama":
            from fms import models
            from fms.models.llama import LLaMA, _llama_factory_factory

            models.register_model(arch, var, _llama_factory_factory(config_data))
            model = LLaMA(config_data)

        elif arch == "gdn":
            _patch_triton_autotuner_compat()
            from fla.models.gated_deltanet import (
                GatedDeltaNetForCausalLM,
                GatedDeltaNetConfig as FLAGDNConfig,
            )

            # fla uses vocab_size; fms config_utils returns src_vocab_size
            fla_config_data = dict(config_data)
            if "src_vocab_size" in fla_config_data:
                fla_config_data["vocab_size"] = fla_config_data.pop("src_vocab_size")
            fla_config = FLAGDNConfig(**fla_config_data)
            model = GatedDeltaNetForCausalLM(fla_config)

        else:
            raise NotImplementedError(
                f"Unsupported architecture '{arch}'. Supported: llama, gdn."
            )

        eval_logger.info(f"Reading state dict from {pretrained}")
        if pretrained.endswith('.pth'):
            ckpt = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(_strip_compiled_prefix(ckpt["model_state"]))
        else:
            state_dict = {"model_state": model.state_dict()}
            load(state_dict=state_dict, storage_reader=FileSystemReader(pretrained))
            model.load_state_dict(_strip_compiled_prefix(state_dict["model_state"]))

        return model

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

        model = self._fms_model_cpu

        # Resolve dtype — default to bfloat16 for FMS models
        _dtype = (
            lm_eval.models.utils_hf.get_dtype(dtype)
            if dtype is not None and dtype != "auto"
            else torch.bfloat16
        )
        model.to(self._device, dtype=_dtype)

        torch.set_grad_enabled(False)
        model.eval()

        # Convert to HF-adapted causal LM.
        # Must disable weight init: from_fms_model triggers PreTrainedModel.__init__
        # -> post_init() -> init_weights() which re-initializes all submodule weights.
        from transformers.modeling_utils import no_init_weights

        if self._arch == "llama":
            from fms.models.hf.llama.modeling_llama_hf import (
                HFAdaptedLLaMAForCausalLM,
                HFAdaptedLLaMAConfig,
            )
            fms_hf_config = HFAdaptedLLaMAConfig.from_fms_config(model.get_config())
            with no_init_weights():
                self._model = HFAdaptedLLaMAForCausalLM.from_fms_model(
                    model, **fms_hf_config.to_dict()
                )

        elif self._arch == "gdn":
            from fms.models.hf.gated_delta_net.modeling_gated_delta_net_hf import (
                HFAdaptedGDNForCausalLM,
            )
            from fms.models.hf.gated_delta_net.configuration_gated_delta_net_hf import (
                HFAdaptedGDNConfig,
            )
            fms_hf_config = HFAdaptedGDNConfig.from_dict(model.config.to_dict())
            with no_init_weights():
                self._model = HFAdaptedGDNForCausalLM._hf_model_from_fms(
                    model, fms_hf_config
                )

        self._model.eval()
        self._config = self._model.config

        # Clean up reference
        del self._fms_model_cpu
