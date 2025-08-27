import argparse
import torch
import transformers
import os
import json
import numpy as np
import random
import warnings
import functools
import copy
import hashlib
import gcg
from data_utils import obtain_sentence_dataset
from utils import get_tokenizer_dir, load_embedding, find_last_true_index, compute_batched_perplexity
from model_utils import load_model_and_tokenizer, load_model, get_embedding_model
from method_utils import SNNLCrossEntropy, prompt_rephrase_sentence, generate_sentence


def compute_ppl_change(original_model, finetune_model, tokenizer, unlearn_data, retain_data=None,
                       tokenize_mode="", save_name=None, overwrite=False, batch_size=4, sent_dataset=False,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = {}
    if not sent_dataset:
        dataset["unlearn"] = obtain_sentence_dataset(unlearn_data, mode=tokenize_mode)
        if retain_data is not None:
            dataset["retain"] = obtain_sentence_dataset(retain_data, mode=tokenize_mode)
    else:
        dataset["unlearn"] = unlearn_data
        if retain_data is not None:
            dataset["retain"] = retain_data

    changes = {}
    for key, sents in dataset.items():
        changes[f"{key}_data_finetune_change"] = []
        changes[f"{key}_data_unlearn_change"] = []
        changes[f"{key}_finetune_ll"] = []
        changes[f"{key}_original_ll"] = []
        input_batches = []
        for i in range(0, len(sents), batch_size):
            batch_sentences = sents[i:i + batch_size]
            input_batches.append(tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to(device))
        for inputs in input_batches:
            changes[f"{key}_finetune_ll"].append(compute_batched_perplexity(finetune_model, inputs).cpu().numpy())
        changes[f"{key}_finetune_ll"] = np.concatenate(changes[f"{key}_finetune_ll"])

        for inputs in input_batches:
            changes[f"{key}_original_ll"].append(compute_batched_perplexity(original_model, inputs).cpu().numpy())
        changes[f"{key}_original_ll"] = np.concatenate(changes[f"{key}_original_ll"])

        changes[f"{key}_data_finetune_change"] = (changes[f"{key}_finetune_ll"] - changes[f"{key}_original_ll"]).tolist()
        changes[f"{key}_finetune_ll"] = changes[f"{key}_finetune_ll"].tolist()
        changes[f"{key}_original_ll"] = changes[f"{key}_original_ll"].tolist()

    avg_unlearn_finetune_change = np.mean(changes[f"unlearn_data_finetune_change"])
    if not sent_dataset:
        print(f"Unlearn data: finetune change: {avg_unlearn_finetune_change:.4f}")
    changes["unlearn"] = dataset["unlearn"]

    if retain_data is not None:
        avg_retain_finetune_change = np.mean(changes[f"retain_data_finetune_change"])
        avg_retain_unlearn_change = np.mean(changes[f"retain_data_unlearn_change"])
        print(f"Retain data: finetune change: {avg_retain_finetune_change:.4f}, unlearn change: {avg_retain_unlearn_change:.4f}")
        changes["retain"] = dataset["retain"]

    if save_name is not None:
        if overwrite and os.path.exists(save_name):
            os.remove(save_name)
        with open(save_name, "w") as f:
            json.dump(changes, f)

    return changes


def find_k_nearest_neighbors(embeddings1, embeddings2, k, dist="l2"):
    """
    Find k nearest neighbors for embeddings2 in embeddings1 and return distances.
    Uses PyTorch with GPU acceleration.
    """
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert embeddings to PyTorch tensors and move to GPU
    embed1_tensor = torch.tensor([embeddings1[i] for i in range(len(embeddings1))], dtype=torch.float32).to(device)
    embed2_tensor = torch.tensor([embeddings2[i] for i in range(len(embeddings2))], dtype=torch.float32).to(device)

    # Compute pairwise Euclidean distances
    # Shape: (len(embed2), len(embed1))
    snnl_loss = SNNLCrossEntropy()
    if dist == "cos":
        distances = snnl_loss.pairwise_cos_distance(embed2_tensor, embed1_tensor)
    elif dist == "l2":
        distances = torch.cdist(embed2_tensor, embed1_tensor, p=2)
        # distances = torch.sqrt(snnl_loss.pairwise_euclid_distance(embed2_tensor, embed1_tensor))
    else:
        raise NotImplementedError(dist)

    # Get indices and distances of k nearest neighbors
    # torch.topk returns values and indices; we want the smallest distances
    nearest_distances, nearest_indices = torch.topk(distances, k, dim=1, largest=False, sorted=True)

    # Convert back to CPU and NumPy if needed for compatibility with downstream code
    nearest_indices = nearest_indices.cpu().numpy()
    nearest_distances = nearest_distances.cpu().numpy()

    return nearest_indices, nearest_distances


def find_knn_embeddings(sentences, embed_model, encode_fn, embeddings1, k=100, batch_size=32, dist="l2"):
    embeddings2 = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        try:
            batch_embeddings = encode_fn(embed_model, batch_sentences)
            embeddings2.extend(
                batch_embeddings.cpu().numpy() if torch.is_tensor(batch_embeddings) else batch_embeddings)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            half_batch = len(batch_sentences) // 2
            batch_embeddings = torch.concat([
                encode_fn(embed_model, batch_sentences[:half_batch]),
                encode_fn(embed_model, batch_sentences[half_batch:])
            ])
            embeddings2.extend(
                batch_embeddings.cpu().numpy() if torch.is_tensor(batch_embeddings) else batch_embeddings)
    nearest_indices, nearest_distances = find_k_nearest_neighbors(embeddings1, embeddings2, k, dist=dist)
    return embeddings2, nearest_distances, nearest_indices


def gcg_rephrase_sentence(original_model, finetune_model, tokenizer, sentence, config, negative_embeddings=None, positive_embeddings=None):
    sent = sentence + f"{'. ' if sentence[-1] != '.' else ''}"
    for itr in range(config["gcg_freq"]):
        temp_res = gcg.prep_text_for_embedding_loss(sent, original_model, tokenizer, config["device"],
                                                    num_free_tokens=config["num_free_tokens"],
                                                    adv_suffix=config['adv_suffix'], adv_prefix=config['adv_prefix'])
        input_ids, input_embeddings, free_token_slice, target_slice, loss_slice, prompt_slice, input_slice = temp_res
        loss_fn = functools.partial(gcg.embedding_loss_fn, negative_embeddings=negative_embeddings,
                                    positive_embeddings=positive_embeddings, config=config,
                                    original_embedding=input_embeddings, input_is_embed=True)
        eval_fn = functools.partial(gcg.embedding_loss_fn, negative_embeddings=negative_embeddings,
                                    positive_embeddings=positive_embeddings, config=config,
                                    original_embedding=input_embeddings, input_is_embed=False)
        res = gcg.optimize_gcg(original_model, input_ids, free_token_slice, target_slice, loss_slice, loss_fn, num_steps=10, eval_fn=eval_fn)

        if config["adv_prefix"] or config['adv_suffix'] or itr == config["gcg_freq"] - 1:
            input_ids = res["input_ids"][prompt_slice]
            template_applied = True
        else:
            input_ids = res["input_ids"][input_slice]
            template_applied = False

        sent = tokenizer.decode(input_ids)

    for _ in range(config["sample_freq"]):
        sent = prompt_rephrase_sentence(
            original_model, finetune_model, tokenizer, sent, config, template_applied=template_applied
        )
        template_applied = False
    return sent


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default="continue", help="write or continue")
    parser.add_argument('--original-model', type=str)
    parser.add_argument('--finetune-model', type=str)
    parser.add_argument('--embed-model', type=str, default=None)
    parser.add_argument('--tokenizer-dir', type=str, default=None)
    parser.add_argument("--w-po", type=float, default=0.7, help="weight for preference optimization between original and finetune")
    parser.add_argument("--w-llo", type=float, default=0.3, help="Weight for likelihood of original")
    parser.add_argument("--w-cos", type=float, default=0, help="should be negative to maximize embedding distance")
    parser.add_argument("--w-l2", type=float, default=-1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument('--ll-percentile', type=float, default=25, help="Percentile threshold for log-likelihood change")
    parser.add_argument('--ll-inc-percentile', type=float, default=25, help="Percentile threshold for log-likelihood change")
    parser.add_argument('--embed-ratio', type=float, default=1.5)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--embed-path', type=str, default=None)
    parser.add_argument('--mean-pooling', action="store_true")
    parser.add_argument('--early-layer', action="store_true")
    parser.add_argument('--data', type=str, default="wmdp_bio")
    parser.add_argument('--retain-data', type=str, default="wikitext_train")
    parser.add_argument('--tokenize', type=str, default='sentence_improved')
    parser.add_argument('--num-samples', type=int, default=1e8)
    parser.add_argument('--max-sents', type=int, default=-1)
    parser.add_argument('--num-needed', type=int, default=100, help="Number of sentences needed in modified_dataset")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--adv-prefix', action="store_true")
    parser.add_argument('--adv-suffix', action="store_true")
    parser.add_argument('--gcg-freq', type=int, default=1)
    parser.add_argument('--sample-freq', type=int, default=1)
    parser.add_argument('--num-free-tokens', type=int, default=10)
    parser.add_argument('--only-embed-increase', action="store_true")
    arg = parser.parse_args()
    
    arg_str = str(sorted(vars(arg).items()))
    arg_hash = hashlib.md5(arg_str.encode()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if arg.tokenizer_dir is None:
        arg.tokenizer_dir = get_tokenizer_dir(arg.original_model)

    if (arg.adv_suffix or arg.adv_prefix) and arg.gcg_freq > 1:
        raise ValueError("gcg-freq cannot be greater than 1 when adv_suffix or adv_prefix is true")

    # Setup checkpoint directory
    ckpt_prefix = "res"
    os.makedirs(ckpt_prefix, exist_ok=True)

    # Load models
    original_model, tokenizer = load_model_and_tokenizer(arg.original_model, tokenizer_dir=arg.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    finetune_model = load_model(arg.finetune_model)

    # Load dataset
    sentences = obtain_sentence_dataset(arg.data, mode=arg.tokenize, num_samples=arg.num_samples)
    
    # Load embeddings
    neg_embedding_dict = neg_embeddings = embed_model = encode_fn = None
    if arg.embed_path is None:
        arg.embed_path = f"{arg.original_model.split('/')[-1]}{'_meanpool' if arg.mean_pooling else ''}_{arg.data}_{arg.tokenize}_embeddings"
    neg_embed_filename = f"{ckpt_prefix}/{arg.embed_path}"
    neg_data = load_embedding(neg_embed_filename)
    neg_embedding_dict = neg_data["embeddings"]
    
    if len(neg_embedding_dict) == 0:
        print(f"No negative embedding found in {neg_embed_filename}")
        exit(1)
        
    neg_embeddings = torch.tensor([neg_embedding_dict[i] for i in range(len(neg_embedding_dict))], dtype=torch.float32).to(device)
    
    embed_model, encode_fn = get_embedding_model(arg.original_model, mean_pooling=arg.mean_pooling,
                                                             layer=-20 if arg.early_layer else -1,
                                                             model=original_model, tokenizer=tokenizer)
    
    if len(sentences) > arg.max_sents and arg.max_sents > 0:
        sentences = random.sample(sentences, arg.max_sents)

    # Setup configuration
    config = {
        "max_length": 50,
        "min_length": 20,
        "w_po": arg.w_po,
        "w_llo": arg.w_llo,
        "w_cos": arg.w_cos,
        "w_l2": arg.w_l2,
        "temperature": arg.temperature,
        "top_p": 0.9,
        "top_k": 50,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "embedding_batch_size": 32,
        "embedding_model": "original",
        "gcg_freq": arg.gcg_freq,
        "sample_freq": arg.sample_freq,
        'adv_suffix': arg.adv_suffix,
        "adv_prefix": arg.adv_prefix,
        "num_free_tokens": arg.num_free_tokens,
    }

    if arg.w_l2 != 0:
        dist = "l2"
    else:
        dist = "cos"

    # Setup thresholds for filtering
    thresholds = [None, None]
    if arg.ll_percentile > 0 or arg.ll_inc_percentile > 0:
        ll_metrics_file = os.path.join(ckpt_prefix, f"{arg.original_model.split('/')[-1]}_{arg.data}_{arg.retain_data}_ll_changes.json")
        if not os.path.exists(ll_metrics_file):
            print(f"Computing log-likelihood and its change... (will be saved to {ll_metrics_file})")
            compute_ppl_change(original_model, finetune_model, tokenizer, arg.data, None, arg.tokenize, save_name=ll_metrics_file)
            print(f"Computed and saved log-likelihood to {ll_metrics_file}")

        with open(ll_metrics_file, 'r') as f:
            ll_changes = json.load(f)
        if arg.ll_percentile > 0:
            ll_dist = ll_changes["unlearn_original_ll"]
            thresholds[0] = np.percentile(ll_dist, arg.ll_percentile)
        if arg.ll_inc_percentile > 0:
            unlearn_changes = ll_changes['unlearn_data_finetune_change']
            thresholds[1] = np.percentile(unlearn_changes, arg.ll_inc_percentile)

    random.shuffle(sentences)

    # Initialize output
    modified_dataset = []
    json_filename = f"{ckpt_prefix}/D_f_prime_{arg.data}_{arg_hash}.json"
    
    # Check for existing results to resume
    start_index = 0
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            modified_dataset = json.load(f)
        if len(modified_dataset) > 0:
            start_index = sentences.index(modified_dataset[-1][0]) + 1
            print(f"Resuming from index {start_index} based on {json_filename}")

    print(f"Starting surrogate dataset generation...")
    print(f"Target: {arg.num_needed} sentences")
    print(f"Strategy: {arg.strategy}")
    print(f"Iterations: {arg.iter}")
    
    # Main generation loop
    for i in range(start_index, len(sentences)):
        original_sent = sentences[i]
        sent = copy.copy(original_sent)
        sents = [sent]
        
        # Generate continuation if using continue strategy
        if "continue" in arg.strategy:
            temperature = config["temperature"]
            config["temperature"] = 0.2
            continue_sent = generate_sentence(original_model, finetune_model, tokenizer, sent, config)
            config["temperature"] = temperature
            new_generation = continue_sent[len(sent):]
            sents.append(new_generation)
            sent = new_generation
        
        # Apply iterative rephrasing
        for j in range(arg.iter):
            sent = gcg_rephrase_sentence(original_model, finetune_model, tokenizer, sents[-1], config, neg_embeddings)
            sents.append(copy.copy(sent))
        
        final_sent = sents[-1]
        appended = True
        
        # Apply filtering if thresholds are set
        if thresholds[0] is None and thresholds[1] is None:
            modified_dataset.append([original_sent, final_sent])
            final_idx = -1
        else:
            # Compute perplexity changes for filtering
            ppl_change_res = compute_ppl_change(original_model, finetune_model, tokenizer, sents[1:], sent_dataset=True)
            # Check likelihood threshold (G3)
            if thresholds[0] is not None:
                sur_original_lls = np.array(ppl_change_res["unlearn_original_ll"])
                satisfied_ind0 = sur_original_lls >= thresholds[0]
            else:
                satisfied_ind0 = True
            
            # Check preference threshold (G1)
            if thresholds[1] is not None:
                sur_finetune_changes = np.array(ppl_change_res[f"unlearn_data_finetune_change"])
                satisfied_ind1 = sur_finetune_changes >= thresholds[1]
            else:
                satisfied_ind1 = True
            
            satisfied_ind = satisfied_ind0 & satisfied_ind1
            
            if np.sum(satisfied_ind) == 0:
                appended = False
                continue
            
            # Select sentences for embedding analysis
            if arg.only_embed_increase:
                interest_sents = sents
            else:
                interest_sents = [sents[0]] + list(np.array(sents[1:])[satisfied_ind])
            
            # Compute embedding distances (G2)
            embeddings2, nearest_distances, nearest_indices = find_knn_embeddings(interest_sents, embed_model, encode_fn, neg_embedding_dict, dist=dist)
            avg_neighbor_distances = np.mean(nearest_distances, axis=1)
            embed_threshold = avg_neighbor_distances[0] * arg.embed_ratio
            embed_satisfied_ind = avg_neighbor_distances > embed_threshold
            
            res_sents = list(np.array(interest_sents)[embed_satisfied_ind])
            
            if arg.only_embed_increase:
                embed_satisfied_ind = embed_satisfied_ind[1:]
                found_indices = find_last_true_index(satisfied_ind, embed_satisfied_ind)
                if found_indices is None:
                    found_idx = None
                else:
                    found_idx = int(np.argmax(avg_neighbor_distances[1:] * found_indices))
                    assert found_indices[found_idx]
                
                if found_idx is None:
                    sent_found = False
                else:
                    sent_found = True
                    res_sents = [sents[found_idx + 1]]
            else:
                sent_found = len(res_sents) > 0

            if sent_found:
                modified_dataset.append([original_sent, res_sents[-1]])
                final_idx = sents.index(res_sents[-1])
                if arg.verbose:
                    print(f"******Sentence {i} APPENDED******")
                    print(f"{final_idx}th sentence appended: " + res_sents[-1] + "\n")
            else:
                appended = False
                continue

        if appended:
            print(f"\n{i} ({len(modified_dataset)} / {arg.num_needed})")

            # Save progress
            with open(json_filename, 'w') as f:
                json.dump(modified_dataset, f)

            # Check if target number reached
            if arg.num_needed > 0 and len(modified_dataset) >= arg.num_needed:
                print(f"Reached desired number of sentences ({len(modified_dataset)}/{arg.num_needed}). Stopping.")
                print(f"Surrogate dataset saved to {json_filename}")
                break

    print(f"Surrogate dataset generation complete!")
    print(f"Generated {len(modified_dataset)} sentences")
    print(f"Results saved to: {json_filename}")
