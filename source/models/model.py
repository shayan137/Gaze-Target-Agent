"""Qwen3-VL wrapper: resolve model name, load once, greedy generate()."""
import os
import warnings


def resolve_model_path(qwen_model: str, cache_dir: str = None) -> str:

    if os.path.isdir(qwen_model):
        return qwen_model

    repo_id = qwen_model if "/" in qwen_model else f"Qwen/{qwen_model}"
    hf_cache = os.path.join(cache_dir, "huggingface") if cache_dir else None

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, cache_dir=hf_cache)


class QwenGazeVLM:
    """Thin inference wrapper around Qwen3-VL-Instruct."""

    def __init__(self, qwen_model: str, device: str = "cuda", cache_dir: str = None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        # cuDNN 9.x fails to init on some machines. This is a local
        # driver quirk, not a hardware problem, so it is opt-in.
        if os.environ.get("DISABLE_CUDNN", "0") == "1":
            torch.backends.cudnn.enabled = False

        model_path = resolve_model_path(qwen_model, cache_dir)

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()

    def generate(self, images, prompt: str, max_new_tokens: int = 32, with_scores: bool = False):

        if not isinstance(images, (list, tuple)):
            images = [images]

        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy = temperature 0
                    return_dict_in_generate=with_scores,
                    output_scores=with_scores,
                )

        prompt_len = inputs["input_ids"].shape[1]
        if with_scores:
            gen_ids = out.sequences[:, prompt_len:]
            scores = out.scores
        else:
            gen_ids = out[:, prompt_len:]
            scores = None

        text = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return text, scores, gen_ids
