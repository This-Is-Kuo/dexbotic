import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import ActionDenorm, AbsoluteAction
from dexbotic.exp.dm0_exp import DM0ActionConfig as _DM0ActionConfig
from dexbotic.exp.dm0_exp import DM0ComputeNormActionConfig as _DM0ComputeNormActionConfig
from dexbotic.exp.dm0_exp import DM0DataConfig as _DM0DataConfig
from dexbotic.exp.dm0_exp import DM0Exp as _DM0Exp
from dexbotic.exp.dm0_exp import DM0InferenceConfig as _DM0InferenceConfig
from dexbotic.exp.dm0_exp import DM0ModelConfig as _DM0ModelConfig
from dexbotic.exp.dm0_exp import DM0OptimizerConfig as _DM0OptimizerConfig
from dexbotic.exp.dm0_exp import DM0TokenizerConfig as _DM0TokenizerConfig
from dexbotic.exp.dm0_exp import DM0TrainerConfig as _DM0TrainerConfig
from dexbotic.exp.base_exp import safe_save_model_for_hf_trainer
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="train", choices=["train", "inference", "compute_norm_stats"])
    args, _ = parser.parse_known_args()
    return args


def _default_wandb_project() -> Optional[str]:
    value = os.getenv("DEXBOTIC_WANDB_PROJECT", "dm0_sft_post_data_01")
    if value.lower() in {"", "none", "null", "disabled"}:
        return None
    return value


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _eval_torch_dtype() -> torch.dtype:
    value = os.getenv("DEXBOTIC_EVAL_TORCH_DTYPE", "float32").strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if value not in mapping:
        raise ValueError(
            f"Unsupported DEXBOTIC_EVAL_TORCH_DTYPE={value}; "
            "expected float32, bfloat16, or float16."
        )
    return mapping[value]


def _stats_shape(stats, key: str) -> str:
    if not isinstance(stats, dict) or key not in stats:
        return "missing"
    item = stats[key]
    if isinstance(item, dict):
        for stat_key in ("min", "max", "mean", "std"):
            if stat_key in item:
                value = item[stat_key]
                shape = getattr(value, "shape", None)
                if shape is not None:
                    return str(tuple(shape))
                try:
                    return str((len(value),))
                except TypeError:
                    return "scalar"
    shape = getattr(item, "shape", None)
    if shape is not None:
        return str(tuple(shape))
    return type(item).__name__


def _latest_checkpoint_by_mtime(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def _resolve_resume_checkpoint(output_dir: str) -> str:
    checkpoint = os.getenv("DEXBOTIC_RESUME_CHECKPOINT", "latest").strip()
    if checkpoint.lower() in {"", "latest", "mtime", "auto"}:
        checkpoint = _latest_checkpoint_by_mtime(output_dir)
    elif not os.path.isabs(checkpoint):
        checkpoint = os.path.join(output_dir, checkpoint)

    if checkpoint is None or not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Resume checkpoint not found under output_dir={output_dir!r}")
    return checkpoint


def _allow_trusted_rng_state_load() -> None:
    original_load = torch.load
    if getattr(original_load, "_dexbotic_rng_retry_patched", False):
        return

    def load_with_rng_retry(*args, **kwargs):
        try:
            return original_load(*args, **kwargs)
        except pickle.UnpicklingError:
            path = str(args[0]) if args else ""
            if kwargs.get("weights_only") is True and os.path.basename(path).startswith("rng_state_"):
                retry_kwargs = dict(kwargs)
                retry_kwargs["weights_only"] = False
                logger.warning(f"Retrying trusted RNG state load with weights_only=False: {path}")
                return original_load(*args, **retry_kwargs)
            raise

    load_with_rng_retry._dexbotic_rng_retry_patched = True
    torch.load = load_with_rng_retry


@dataclass
class DM0OptimizerConfig(_DM0OptimizerConfig):
    base_lr: float = field(default_factory=lambda: float(os.getenv("DEXBOTIC_BASE_LR", "1e-5")))
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default_factory=lambda: int(os.getenv("DEXBOTIC_WARMUP_STEPS", "1000")))
    weight_decay: float = field(default=1e-10)


@dataclass
class DM0TrainerConfig(_DM0TrainerConfig):
    deepspeed: Optional[str] = field(
        default_factory=lambda: os.getenv("DEXBOTIC_DEEPSPEED_CONFIG", "./script/deepspeed/zero3.json")
    )
    wandb_project: Optional[str] = field(default_factory=_default_wandb_project)
    bf16: bool = field(default=True)
    num_train_steps: int = field(default_factory=lambda: int(os.getenv("DEXBOTIC_NUM_TRAIN_STEPS", "15000")))
    save_steps: int = field(default_factory=lambda: int(os.getenv("DEXBOTIC_SAVE_STEPS", "500")))
    save_total_limit: int = field(default=20)
    save_only_model: bool = field(default=False)
    per_device_train_batch_size: int = field(
        default_factory=lambda: int(os.getenv("DEXBOTIC_TRAIN_BATCH_SIZE", "2"))
    )
    gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(
        default_factory=lambda: int(os.getenv("DEXBOTIC_GRAD_ACCUM", "8"))
    )
    model_max_length: int = field(default=200)
    output_dir: str = field(
        default_factory=lambda: os.getenv(
            "DEXBOTIC_OUTPUT_DIR",
            (
                "/mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/"
                f"post_data_01_deltafix-{datetime.now().strftime('%m%d')}"
            ),
        )
    )
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(
        default_factory=lambda: {"min_lr": float(os.getenv("DEXBOTIC_MIN_LR", "1e-6"))}
    )
    logging_steps: int = field(default_factory=lambda: int(os.getenv("DEXBOTIC_LOGGING_STEPS", "10")))
    dataloader_num_workers: int = field(default_factory=lambda: int(os.getenv("DEXBOTIC_TRAIN_NUM_WORKERS", "4")))

    def __post_init__(self):
        super().__post_init__()
        run_name = os.getenv("DEXBOTIC_WANDB_RUN_NAME")
        if run_name:
            self.run_name = run_name


class DM0PostData01DeltaComputeNormActionConfig(_DM0ComputeNormActionConfig):
    def build_action_process_func(self) -> Pipeline:
        # post_data_01 converter already exports delta tcp actions plus next-step pinch targets
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=False),
                ToList(),
            ]
        )


@dataclass
class DM0ActionConfig(_DM0ActionConfig):
    statistic_mapping: str = field(default_factory=lambda: os.getenv("DEXBOTIC_NORM_STATS_PATH") or None)
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        external_norm_stats_path = os.getenv("DEXBOTIC_NORM_STATS_PATH")
        if external_norm_stats_path:
            print(f"[INFO] DEXBOTIC_NORM_STATS_PATH={external_norm_stats_path}")
            if not os.path.isfile(external_norm_stats_path):
                raise FileNotFoundError(
                    f"DEXBOTIC_NORM_STATS_PATH is set but file does not exist: {external_norm_stats_path}"
                )
            self.statistic_mapping = external_norm_stats_path
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        if statistic_mapping is None:
            raise RuntimeError("Loaded statistic_mapping is None; refusing to train without explicit norm_stats.")
        if external_norm_stats_path:
            print(f"[INFO] Loaded external norm_stats from {external_norm_stats_path}")
            print("[INFO] statistic_mapping is not None")
            print(f"[INFO] statistic_mapping keys = {sorted(statistic_mapping.keys())}")
            print("[INFO] norm_stats source = external")
            print(f"[INFO] action norm stats shape = {_stats_shape(statistic_mapping, 'action')}")
            print(f"[INFO] state norm stats shape = {_stats_shape(statistic_mapping, 'state')}")
        print("[INFO] raw state/action dim = 16/14")
        print("[INFO] configured padded state/action dim = 32/32")
        print(f"[INFO] chunk_size = {self.trajectory_length}")
        print("[INFO] transform order includes PadState -> PadAction -> AddTrajectory -> DeltaAction -> ActionNorm")
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=False),
                ActionNorm(statistic_mapping=statistic_mapping, use_quantiles=True),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )


@dataclass
class DM0DataConfig(_DM0DataConfig):
    dataset_name: str = field(default_factory=lambda: os.getenv("DEXBOTIC_DATASET_NAME", "post_data_01_default"))
    num_images: int = field(default=3)
    data_keys: list[str] = field(
        default_factory=lambda: ["input_ids", "labels", "action", "image", "state", "image_masks"]
    )
    aug_policy: str | list[str] = field(default_factory=lambda: ["dm0", "color_dm0", "color_dm0"])
    action_config: DM0ActionConfig = field(default_factory=DM0ActionConfig)


@dataclass
class DM0ModelConfig(_DM0ModelConfig):
    model_name_or_path: str = field(
        default_factory=lambda: os.getenv("DEXBOTIC_BASE_MODEL", "/dexbotic/checkpoints/DM0-base")
    )

    def build_model(self) -> DM0ForCausalLM:
        return DM0ForCausalLM.from_pretrained(self.model_name_or_path)


@dataclass
class DM0TokenizerConfig(_DM0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class DM0InferenceConfig(_DM0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "DEXBOTIC_INFERENCE_MODEL_PATH",
            "./user_checkpoints/dexbotic/custom_dm0/post_data_01_deltafix",
        )
    )
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [6, 13])
    action_dim: int = field(default=14)
    device_map: Optional[dict | str] = field(default_factory=lambda: os.getenv("DEXBOTIC_EVAL_DEVICE_MAP", "single"))
    cuda_device: Optional[int] = field(default=None)

    def _load_model(self) -> None:
        requested_device = os.getenv("DEXBOTIC_EVAL_DEVICE", "cuda")
        if requested_device:
            self.device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
        elif torch.cuda.is_available():
            if self.cuda_device is not None:
                self.device = torch.device(f"cuda:{self.cuda_device}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        eval_device_map = os.getenv("DEXBOTIC_EVAL_DEVICE_MAP", self.device_map or "single")
        if isinstance(eval_device_map, str):
            eval_device_map = eval_device_map.strip().lower()
        if eval_device_map in {"single", "none", "false", "0"}:
            self.device_map = None
            device_map_mode = "single"
        elif eval_device_map == "auto":
            self.device_map = "auto"
            device_map_mode = "auto"
        elif isinstance(self.device_map, dict):
            device_map_mode = "custom"
        else:
            raise ValueError(f"Unsupported DEXBOTIC_EVAL_DEVICE_MAP={eval_device_map}")

        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"[INFO] eval device = {self.device}")
        logger.info(f"[INFO] eval device_map mode = {device_map_mode}")
        if device_map_mode == "auto":
            logger.warning(
                "[WARN] DEXBOTIC_EVAL_DEVICE_MAP=auto may split DM0 across GPUs and can trigger "
                "cross-device tensor errors."
            )
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if len([part for part in visible_devices.split(",") if part.strip()]) > 1:
                logger.warning(
                    f"[WARN] CUDA_VISIBLE_DEVICES exposes multiple GPUs ({visible_devices}) while "
                    "DEXBOTIC_EVAL_DEVICE_MAP=auto is enabled."
                )

        torch_dtype = _eval_torch_dtype()
        logger.info(f"[INFO] eval torch_dtype = {torch_dtype}")
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
            logger.info(f"Using device_map: {self.device_map}")
        try:
            model = DM0ForCausalLM.from_pretrained(self.model_name_or_path, **load_kwargs)
            if torch_dtype == torch.float32:
                # DM0 initialization intentionally keeps a mixed BF16/FP32 model
                # when config.bf16 is enabled. Force a uniform FP32 inference
                # model so the action projections and denoising inputs agree.
                model = model.to(dtype=torch.float32)
                model.config.bf16 = False
                model.model.config.bf16 = False
            if self.device_map is None:
                model = model.to(self.device)
        except RuntimeError as exc:
            message = str(exc).lower()
            if self.device_map is None and ("out of memory" in message or "cuda oom" in message):
                raise RuntimeError(
                    "Single-GPU eval model loading ran out of CUDA memory. Lower "
                    "DEXBOTIC_OPENLOOP_BATCH_SIZE/DEXBOTIC_EVAL_BATCH_SIZE or choose a GPU with more memory; "
                    "not switching to device_map=auto automatically."
                ) from exc
            raise
        parameter_dtypes = sorted(
            {str(param.dtype) for param in model.parameters() if param.is_floating_point()}
        )
        logger.info(f"[INFO] model floating parameter dtypes = {parameter_dtypes}")
        if torch_dtype == torch.float32 and parameter_dtypes != ["torch.float32"]:
            raise RuntimeError(
                "FP32 inference requested, but model parameters still have mixed dtypes: "
                f"{parameter_dtypes}"
            )
        first_param_device = next(model.parameters()).device
        logger.info(f"[INFO] model first parameter device = {first_param_device}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = DM0Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(statistic_mapping=self.norm_stats, strict=False, use_quantiles=True),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False, use_quantiles=True),
                AbsoluteAction(),
            ]
        )

    def _get_response(
        self,
        text: str | list[str],
        images: list[str],
        states: Optional[str | list[str]] = None,
        batch_size: int = 1,
    ) -> list[list[float]]:
        t0 = time.monotonic()
        batch_size = int(batch_size)
        assert len(images) % batch_size == 0, (
            f"Number of images {len(images)} is not divisible by batch size {batch_size}"
        )
        num_images = len(images) // batch_size
        images = [
            images[i * num_images : (i + 1) * num_images] for i in range(batch_size)
        ]
        if isinstance(text, str):
            text = [text] * batch_size

        batch_images = [
            [Image.open(i).convert("RGB") for i in image_items]
            for image_items in images
        ]
        batch_images_tensor = [
            self.model.process_images(image_items).to(dtype=self.model.dtype)
            for image_items in batch_images
        ]

        if num_images != self.num_images:
            batch_images_tensor = [
                torch.cat(
                    [
                        image_tensor,
                        torch.zeros_like(image_tensor[0:1]).repeat(
                            self.num_images - num_images, 1, 1, 1
                        ),
                    ],
                    dim=0,
                )
                if len(image_tensor) < self.num_images
                else image_tensor[: self.num_images]
                for image_tensor in batch_images_tensor
            ]

        batch_image_masks = [
            torch.tensor(
                [True for _ in range(num_images)]
                + [False for _ in range(self.num_images - num_images)],
                device=image_tensor.device,
            )
            for image_tensor in batch_images_tensor
        ]
        batch_images_tensor = torch.stack(batch_images_tensor, dim=0)
        batch_image_masks = torch.stack(batch_image_masks, dim=0)

        self._save_image(images[0], text[0])

        batch_input_ids = np.array(
            [
                self.tokenization_func([{"from": "human", "value": p}])["input_ids"]
                for p in text
            ]
        )
        logger.info(
            f"prompt: {self.tokenization_func.tokenizer.decode(batch_input_ids[0])}"
        )
        batch_attention_mask = np.array(
            [np.array(ids != self.tokenizer.pad_token_id) for ids in batch_input_ids]
        )

        if states is not None:
            if isinstance(states, str):
                batch_states = np.array(json.loads(states))
                if batch_states.ndim == 1:
                    batch_states = batch_states[None]
                assert batch_states.shape[0] == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got length {len(batch_states)}."
                )
            elif isinstance(states, (list, tuple)) and all(
                isinstance(s, str) for s in states
            ):
                assert len(states) == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got {type(states)} with length {len(states)}."
                )
                batch_states = np.array([json.loads(s) for s in states])
        else:
            batch_states = np.zeros(
                (batch_size, self.model.model.config.action_dim),
                dtype=np.float32,
            )

        inference_args = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": batch_images_tensor,
            "image_masks": batch_image_masks,
            "state": batch_states,
            "meta_data": {
                "non_delta_mask": np.array(self.non_delta_mask),
            },
        }

        inputs = self.input_transform(inference_args)
        inputs["states"] = inputs["state"]

        def move_input_to_device(value):
            if not isinstance(value, torch.Tensor):
                return value
            value = value.to(self.device)
            if value.is_floating_point():
                value = value.to(dtype=self.model.dtype)
            return value

        inputs = {k: move_input_to_device(v) for k, v in inputs.items()}
        actions = self.model.inference_action(**inputs)

        def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
            value = value.detach()
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.cpu().numpy()

        outputs = {
            k: tensor_to_numpy(v) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = tensor_to_numpy(actions)
        outputs = self.output_transform(outputs)
        logger.info(f"Processing time: {time.monotonic() - t0}")
        response = outputs["action"][..., : self.action_dim].tolist()
        if batch_size == 1:
            response = response[0]
        return response


@dataclass
class DM0Exp(_DM0Exp):
    model_config: DM0ModelConfig = field(default_factory=DM0ModelConfig)
    optimizer_config: DM0OptimizerConfig = field(default_factory=DM0OptimizerConfig)
    trainer_config: DM0TrainerConfig = field(default_factory=DM0TrainerConfig)
    data_config: DM0DataConfig = field(default_factory=DM0DataConfig)
    tokenizer_config: DM0TokenizerConfig = field(default_factory=DM0TokenizerConfig)
    inference_config: DM0InferenceConfig = field(default_factory=DM0InferenceConfig)

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = DM0PostData01DeltaComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)

    def _log_trainable_parameters(self) -> None:
        total_params = 0
        trainable_params = 0
        trainable_by_root: dict[str, int] = {}
        sample_names: list[str] = []

        for name, param in self.model.named_parameters():
            count = param.numel()
            total_params += count
            if not param.requires_grad:
                continue
            trainable_params += count
            root = name.split(".", 1)[0]
            trainable_by_root[root] = trainable_by_root.get(root, 0) + count
            if len(sample_names) < 30:
                sample_names.append(name)

        ratio = trainable_params / total_params if total_params else 0.0
        logger.info(f"[INFO] total params = {total_params:,}")
        logger.info(f"[INFO] trainable params = {trainable_params:,}")
        logger.info(f"[INFO] trainable ratio = {ratio:.6%}")
        if ratio < 0.9:
            logger.warning(f"[WARNING] trainable ratio is below 90%; expected full fine-tune, got {ratio:.6%}")
        top_modules = sorted(trainable_by_root.items(), key=lambda item: item[1], reverse=True)
        logger.info(
            "[INFO] trainable module roots: "
            + ", ".join(f"{name}={count:,}" for name, count in top_modules[:20])
        )
        logger.info("[INFO] sample trainable parameter names: " + ", ".join(sample_names))

    def _apply_log_step_offset(self) -> None:
        step_offset = _env_int("DEXBOTIC_LOG_STEP_OFFSET", 0)
        if step_offset <= 0:
            return

        original_log = self.trainer.log
        original_save_checkpoint = self.trainer._save_checkpoint

        def log_with_offset(logs, *args, **kwargs):
            actual_global_step = self.trainer.state.global_step
            self.trainer.state.global_step = actual_global_step + step_offset
            try:
                return original_log(logs, *args, **kwargs)
            finally:
                self.trainer.state.global_step = actual_global_step

        def save_checkpoint_with_offset(*args, **kwargs):
            actual_global_step = self.trainer.state.global_step
            self.trainer.state.global_step = actual_global_step + step_offset
            try:
                return original_save_checkpoint(*args, **kwargs)
            finally:
                self.trainer.state.global_step = actual_global_step

        self.trainer.log = log_with_offset
        self.trainer._save_checkpoint = save_checkpoint_with_offset
        logger.info(f"Applying log/checkpoint step offset: {step_offset}")

    def _infer_log_step_offset(self, checkpoint: str) -> int:
        configured_offset = _env_int("DEXBOTIC_LOG_STEP_OFFSET", 0)
        if configured_offset > 0:
            return configured_offset

        state = self._read_checkpoint_state(checkpoint)
        if not state:
            return 0

        global_step = int(state.get("global_step") or 0)
        logged_steps = [
            int(entry["step"])
            for entry in state.get("log_history", [])
            if isinstance(entry, dict) and isinstance(entry.get("step"), int)
        ]
        if global_step <= 0 or not logged_steps:
            return 0

        return max(0, max(logged_steps) - global_step)

    def _read_checkpoint_state(self, checkpoint: str) -> dict:
        state_path = os.path.join(checkpoint, "trainer_state.json")
        if not os.path.exists(state_path):
            return {}

        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read trainer state from {state_path}: {exc}")
            return {}

    def _apply_target_train_steps(self, checkpoint: str, step_offset: int) -> None:
        target_steps = _env_int("DEXBOTIC_TARGET_TRAIN_STEPS", 0)
        if target_steps <= 0 or step_offset <= 0:
            return

        state = self._read_checkpoint_state(checkpoint)
        global_step = int(state.get("global_step") or 0)
        local_target_steps = max(global_step, target_steps - step_offset)
        self.trainer.args.max_steps = local_target_steps
        self.trainer_config.num_train_steps = local_target_steps

        logger.info(
            f"Using target train steps: total={target_steps}, offset={step_offset}, "
            f"trainer max_steps={local_target_steps}"
        )

    def train(self) -> None:
        if _env_flag("DEXBOTIC_RESUME_FROM_CHECKPOINT", False):
            resume_checkpoint = _resolve_resume_checkpoint(self.trainer_config.output_dir)
            self._initialize_train()
            self._log_trainable_parameters()
            inferred_offset = self._infer_log_step_offset(resume_checkpoint)
            if inferred_offset > 0:
                os.environ["DEXBOTIC_LOG_STEP_OFFSET"] = str(inferred_offset)
            self._apply_log_step_offset()
            self._apply_target_train_steps(resume_checkpoint, inferred_offset)
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            _allow_trusted_rng_state_load()
            self.trainer.train(resume_from_checkpoint=resume_checkpoint)
            self.trainer.save_state()
            self.model.config.use_cache = True
            self.model.model.llm.config.use_cache = True
            safe_save_model_for_hf_trainer(
                trainer=self.trainer,
                output_dir=self.trainer_config.output_dir,
            )
            logger.info(f"Training completed and model saved to {self.trainer_config.output_dir}")
            return

        self._initialize_train()
        self._log_trainable_parameters()
        self._apply_log_step_offset()
        logger.info(
            f"Starting fresh training; resume disabled by DEXBOTIC_RESUME_FROM_CHECKPOINT=0. "
            f"Output dir: {self.trainer_config.output_dir}"
        )
        self.trainer.train()
        self.trainer.save_state()
        self.model.config.use_cache = True
        self.model.model.llm.config.use_cache = True
        safe_save_model_for_hf_trainer(
            trainer=self.trainer,
            output_dir=self.trainer_config.output_dir,
        )
        logger.info(f"Training completed and model saved to {self.trainer_config.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    exp = DM0Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
