"""One-time preparation; downloads are never part of benchmark timing."""
import whisper
from huggingface_hub import snapshot_download

# This private helper belongs to the pinned openai-whisper==20231117 package;
# it verifies the checkpoint SHA256 and does not load the model into memory.
whisper._download(whisper._MODELS["large-v3"], "/models/openai", False)
snapshot_download(
    "Systran/faster-whisper-large-v3",
    revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
    local_dir="/models/faster",
    allow_patterns=["model.bin", "config.json", "tokenizer.json",
                    "preprocessor_config.json", "vocabulary.json"],
)
