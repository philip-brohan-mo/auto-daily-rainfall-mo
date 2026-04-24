from dataclasses import dataclass, field
from pathlib import Path

# Short names that can be passed on the command line.
# Values are the full HuggingFace model IDs.
MODEL_PRESETS: dict[str, str] = {
    "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "granite": "ibm-granite/granite-vision-3.2-2b",
}


@dataclass
class ProjectPaths:
    data_dir: Path = Path("data")
    raw_images_dir: Path = Path("data/raw_images")
    annotations_dir: Path = Path("data/annotations")
    outputs_dir: Path = Path("outputs")
    models_dir: Path = Path("models")


@dataclass
class IngestConfig:
    images_dir: Path = Path("Daily_rainfall_sample/images")
    transcriptions_dir: Path = Path("Daily_rainfall_sample/transcriptions")
    output_dir: Path = Path("data/dataset")


@dataclass
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    device: str = "auto"


@dataclass
class TrainingConfig:
    output_dir: Path = Path("outputs/checkpoints")
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    eval_split: float = 0.1
    # LoRA hyper-parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # None → "all-linear" (PEFT selects all linear layers automatically)
    lora_target_modules: list[str] | None = None
    extra_args: dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
