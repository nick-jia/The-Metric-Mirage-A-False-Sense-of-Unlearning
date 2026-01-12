import numpy as np
import torch
import torch.nn.functional as F
import os


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device('cpu')
        except AttributeError:
            return torch.device('cpu')


def get_tokenizer_dir(model_dir):
    if "Llama-3" in model_dir.lower():
        tokenizer_dir = "meta-llama/Meta-Llama-3-8B"
    elif "Llama-2" in model_dir.lower():
        tokenizer_dir = "meta-llama/Llama-2-7b-hf"
    elif "phi-3" in model_dir.lower():
        tokenizer_dir = "microsoft/Phi-3-mini-4k-instruct"
    elif "Qwen2.5" in model_dir.lower():
        tokenizer_dir = "Qwen/Qwen2.5-7B"
    elif "gemma" in model_dir.lower():
        tokenizer_dir = "google/gemma-3-4b-it"
    elif "zephyr" in model_dir.lower():
        tokenizer_dir = "HuggingFaceH4/zephyr-7b-beta"
    else:
        tokenizer_dir = model_dir
        print("arg.tokenizer_dir not specified, set to model_dir by default")
    return tokenizer_dir


def load_embedding(embed_filename, data=None):
    if data is None:
        data = {'sentences': [], 'scores': {}, 'embeddings': {}}
    if os.path.exists(embed_filename) or os.path.exists(embed_filename.replace(".npy", "")):
        embedding_dict = load_large_dict(embed_filename)
        if len(data['sentences']) > 0:
            for i, sent in enumerate(data['sentences']):
                try:
                    data['embeddings'][i] = torch.from_numpy(embedding_dict["embeddings"][embedding_dict["body"].index(sent)])
                except TypeError:
                    continue
                except (IndexError, ValueError):
                    print(len(embedding_dict["embeddings"]), i, len(embedding_dict["body"]))
                    break
        else:
            data['sentences'] = embedding_dict["body"]
            data['embeddings'] = embedding_dict["embeddings"]
    elif "_percent" in embed_filename:
        data["need_to_load_parent_embed"] = True
    return data


def load_large_dict(load_path):
    if load_path[-4:] == ".npy":
        load_path = load_path.replace(".npy", "")
    try:
        metadata = np.load(os.path.join(load_path, "metadata.npy"), allow_pickle=True).item()
    except FileNotFoundError:
        return np.load(load_path + ".npy", allow_pickle=True)
    num_chunks = metadata["num_chunks"]

    # Initialize lists for reassembly
    sentences = []
    embeddings_list = []

    # Load chunks and reassemble
    for i in range(num_chunks):
        chunk_sentences = np.load(os.path.join(load_path, f"sentences_chunk_{i}.npy"), allow_pickle=True)
        chunk_embeddings = np.load(os.path.join(load_path, f"embeddings_chunk_{i}.npy"), allow_pickle=True)

        sentences.extend(chunk_sentences)
        embeddings_list.append(chunk_embeddings)

    # Concatenate embeddings into a single numpy array
    embeddings = np.concatenate(embeddings_list, axis=0)

    return {"body": sentences, "embeddings": embeddings}


def save_large_dict(save_path, data, chunk_size=10000):
    if save_path[-4:] == ".npy":
        save_path = save_path.replace(".npy", "")

    sentences = data["body"]
    embeddings = data["embeddings"]

    os.makedirs(save_path, exist_ok=True)
    num_chunks = (len(sentences) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        chunk_sentences = sentences[i * chunk_size:(i + 1) * chunk_size]
        chunk_embeddings = embeddings[i * chunk_size:(i + 1) * chunk_size]

        # Save each chunk separately
        np.save(os.path.join(save_path, f"sentences_chunk_{i}.npy"), chunk_sentences)
        np.save(os.path.join(save_path, f"embeddings_chunk_{i}.npy"), chunk_embeddings)

    # Save metadata
    metadata = {"num_chunks": num_chunks, "chunk_size": chunk_size}
    np.save(os.path.join(save_path, "metadata.npy"), metadata)


def save_progress(data, embed_filename):

    if embed_filename is not None:
        if len(data['embeddings']) > 0:
            if not isinstance(data['embeddings'], np.ndarray):
                try:
                    embeddings = torch.stack([data["embeddings"][i] for i in range(len(data["embeddings"]))]).numpy()
                except Exception as e:
                    embeddings = np.stack([data["embeddings"][i] for i in range(len(data["embeddings"]))])
            else:
                try:
                    embeddings = np.stack(data["embeddings"])
                except ValueError:
                    embeddings = data["embeddings"]
            embeddings_dict = {"body": data['sentences'], 'embeddings': embeddings}
            save_large_dict(embed_filename, embeddings_dict)


def compute_batched_perplexity(model, encodings):
    """Compute batched perplexity for transition scoring"""
    with torch.no_grad():
        outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = encodings['input_ids'][..., 1:].contiguous()
        shift_attention_mask = encodings['attention_mask'][..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)

        valid_tokens = shift_attention_mask.float()
        sample_losses = (loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)
        return - sample_losses


def find_last_true_index(satisfied_ind, embed_satisfied_ind):
    """Find indices where satisfied_ind is True - essential for embedding filtering"""
    # Find indices where satisfied_ind is True
    satisfied_true = np.where(satisfied_ind)[0]

    # If no True in satisfied_ind, return None
    if len(satisfied_true) == 0:
        return None

    # Find indices where embed_satisfied_ind is True
    embed_true = np.where(embed_satisfied_ind)[0]

    # If no True in embed_satisfied_ind, return None
    if len(embed_true) == 0:
        return None

    # Check if any True in embed_satisfied_ind is at or before last True in satisfied_ind
    if embed_true[0] <= satisfied_true[-1]:
        satisfied_ind[:embed_true[0]] = False
        return satisfied_ind
    return None
