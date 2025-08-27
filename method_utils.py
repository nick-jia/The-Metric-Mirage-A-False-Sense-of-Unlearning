import torch
import torch.nn.functional as F


class SNNLCrossEntropy:
    def __init__(self, temperature=100.0, factor=-10.0, cos_distance=True):
        self.temperature = temperature
        self.factor = factor
        self.cos_distance = cos_distance
        self.eps = 1e-5

    def pairwise_cos_distance(self, A, B):
        A_norm = F.normalize(A, p=2, dim=-1)
        B_norm = F.normalize(B, p=2, dim=-1)
        return 1 - torch.matmul(A_norm, B_norm.T)

    def pairwise_euclid_distance(self, A, B):
        A2 = torch.sum(A**2, dim=-1, keepdim=True)
        B2 = torch.sum(B**2, dim=-1, keepdim=True).T
        AB = torch.matmul(A, B.T)
        return A2 + B2 - 2 * AB

    def compute_fits(self, A, B):
        if self.cos_distance:
            dist = self.pairwise_cos_distance(A, B)
        else:
            dist = self.pairwise_euclid_distance(A, B)
        return torch.exp(-dist / self.temperature)

    def snnl(self, x, y):
        fits = self.compute_fits(x, x)
        mask = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        fits = fits.masked_fill(mask, 0)
        same_labels = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        masked_fits = fits * same_labels
        sum_masked = masked_fits.sum(dim=-1)
        return -torch.log(self.eps + sum_masked).mean()


def get_token_scores(model, input_ids, ll=True, embed=False):
    """Calculate log probabilities for next token predictions."""
    with torch.no_grad():
        if embed:
            outputs = model(input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            embedding = torch.mean(last_hidden, dim=1).float()
        else:
            outputs = model(input_ids)
        if ll:
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
        else:
            log_probs = 0
    if embed:
        return log_probs, embedding
    else:
        return log_probs


def compute_embedding_distances(current_embedding, negative_embeddings, positive_embeddings, config,
                                original_embedding=None):
    snnl_loss = SNNLCrossEntropy()
    batch_size = min(config["embedding_batch_size"], negative_embeddings.shape[0])
    ref_idx = torch.randperm(negative_embeddings.shape[0])[:batch_size]
    if original_embedding is not None:
        ref_batch = torch.cat([original_embedding, negative_embeddings[ref_idx]], dim=0).to(config["device"])
    else:
        ref_batch = negative_embeddings[ref_idx].to(config["device"])

    cos_dist, l2_dist = 0, 0
    if config["w_cos"] != 0:
        cos_dist = snnl_loss.pairwise_cos_distance(current_embedding, ref_batch).mean()
    if config["w_l2"] != 0:
        l2_dist = torch.log(snnl_loss.pairwise_euclid_distance(current_embedding, ref_batch).mean())

    return config["w_cos"] * cos_dist + config["w_l2"] * l2_dist


def combine_objectives(original_model, finetune_model, input_ids, config):
    original_ll, perference = 0, 0

    if config["w_llo"] != 0 or config["w_po"] != 0:
        original_ll = get_token_scores(original_model, input_ids)

    if config["w_po"] != 0:
        finetune_ll = get_token_scores(finetune_model, input_ids)
        perference = config["w_po"] * finetune_ll
    return config["w_llo"] * original_ll + perference


def sample_next_token(scores, config):
    """Sample next token using top-p with temperature and top-k fallback."""
    temperature = config["temperature"]
    top_p = config["top_p"]
    top_k = config["top_k"]

    # Top-p (nucleus) sampling
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_scores / temperature, dim=-1), dim=-1)
    sorted_indices_to_keep = cumulative_probs <= top_p
    sorted_indices_to_keep[..., 0] = True  # Ensure at least one token

    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_keep.to(mask.dtype)
    )
    filtered_scores = scores.masked_fill(~mask, float("-inf"))

    # Fallback to top-k if no tokens remain
    if torch.all(filtered_scores == float("-inf")):
        top_k_scores, top_k_indices = torch.topk(scores, min(top_k, scores.size(-1)))
        probs = torch.softmax(top_k_scores, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices.gather(1, next_token_idx)
    else:
        probs = torch.softmax(filtered_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def generate_sentence(original_model, finetune_model, tokenizer, prompt, config):
    if isinstance(prompt, str):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config["device"])
    else:
        input_ids = prompt.to(config["device"])
    max_length = config["max_length"]
    min_length = config["min_length"]

    new_length = 0
    while new_length < min_length:
        generated_ids = input_ids.clone()
        new_length = 0
        for _ in range(max_length):
            final_scores = combine_objectives(original_model, finetune_model, generated_ids, config)
            next_token = sample_next_token(final_scores, config)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            new_length += 1
            if next_token.item() == tokenizer.eos_token_id:
                break
    res_ids = generated_ids
    return tokenizer.decode(res_ids[0], skip_special_tokens=True)


def prompt_rephrase_sentence(original_model, finetune_model, tokenizer, sentence, config, template_applied=False):
    if not template_applied:
        prompt = "Rephrase the following: " + sentence + f"{'. ' if sentence[-1] != '.' else ' '}Rephrased version:"
    else:
        prompt = sentence
    generated_sentence = generate_sentence(original_model, finetune_model, tokenizer, prompt, config)
    return generated_sentence[len(prompt):]
