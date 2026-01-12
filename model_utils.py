import torch
import transformers
from peft import PeftModel,  LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import is_sentence_transformer_model
from utils import get_device, get_tokenizer_dir


def load_model(
    model_dir,
    quantization_config = None,
):
    # from MUSE
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    return model


def load_tokenizer(
    tokenizer_dir,
    add_pad_token = True,
    use_fast = True
):
    # from MUSE
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast)
    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_dir,
    tokenizer_dir = None,
    add_pad_token = True,
    quantization_config = None,
    use_fast = True
):
    # from MUSE
    model = load_model(
        model_dir, quantization_config,
    )
    tokenizer = (load_tokenizer(tokenizer_dir, add_pad_token, use_fast=use_fast)
                 if tokenizer_dir is not None
                 else None)
    return model, tokenizer


def get_model_tokenizer(model_name, return_pipeline=False):
    pipeline = transformers.pipeline("text-generation", model=model_name,
                                       model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    if return_pipeline:
        return pipeline
    else:
        model, tokenizer = pipeline.model, pipeline.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer


def get_embedding_model(model_name, mean_pooling=False, layer=-1, model=None, tokenizer=None):
    if "pythia" in model_name:
        if "EleutherAI" not in model_name:
            model_name = f"EleutherAI/{model_name}"
        model, tokenizer = get_model_tokenizer(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = 512

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        def pythia_hidden_state(model, texts):
            max_length = tokenizer.model_max_length
            input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_length)

            if model.device.type == "cuda":
                input_ids = {k: v.cuda() for k, v in input_ids.items()}

            with torch.no_grad():
                outputs = model(**input_ids, output_hidden_states=True)
                if mean_pooling:
                    return outputs.hidden_states[layer].mean(dim=1).cpu()
                else:
                    return outputs.hidden_states[layer][:, -1, :].cpu()

        return model, pythia_hidden_state
    elif not is_sentence_transformer_model(model_name):
        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                tokenizer_dir=get_tokenizer_dir(model_name)
            )
        model.eval()

        def get_last_hidden_states(model, sentences, max_length=512, device=get_device()):
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            if mean_pooling:
                last_hidden_states = outputs.hidden_states[layer].mean(dim=1).cpu().float()
            else:
                last_hidden_states = outputs.hidden_states[layer][:, -1, :].cpu().float()
            return last_hidden_states
        return model, get_last_hidden_states
    else:
        model = SentenceTransformer(model_name)
        return model, lambda model, x: model.encode(x, show_progress_bar=False)

