"""Al-Ansar Voice Cloning on Modal (serverless GPU, pay-per-second, scales to zero).
Deploy once with:  modal deploy backend/modal_app.py   (see colab/Deploy_Server.ipynb)"""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("alansar-voice")


def _prefetch():
    """Runs once while the image is built, so models are already on disk when the server wakes up."""
    from huggingface_hub import snapshot_download
    for repo in ["Systran/faster-whisper-large-v3", "facebook/nllb-200-distilled-1.3B", "k2-fsa/OmniVoice"]:
        snapshot_download(repo)
    try:
        import torch
        from omnivoice import OmniVoice
        OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
    except Exception as e:  # noqa
        print("prefetch warning:", e)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi", "python-multipart", "omnivoice", "faster-whisper", "transformers",
                 "sentencepiece", "soundfile", "librosa", "accelerate", "huggingface_hub")
    .run_function(_prefetch)
    .env({"ALANSAR_NO_AUTOSTART": "1", "ALANSAR_WORK": "/tmp/alansar"})
    .add_local_dir(HERE, remote_path="/app")
)


@app.cls(gpu="A10G", image=image, scaledown_window=120, max_containers=1, timeout=3600)
@modal.concurrent(max_inputs=20)
class Api:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/app")
        import pipeline
        import server
        pipeline.load_models()
        server.STATE["ready"] = True
        server.start_background(load_models=False)

    @modal.asgi_app()
    def web(self):
        import server
        return server.app
