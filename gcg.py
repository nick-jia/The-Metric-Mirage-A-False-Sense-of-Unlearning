# modified based on https://github.com/locuslab/acr-memorization/blob/main/prompt_optimization/gcg.py
import torch
import torch.nn.functional as F
from method_utils import compute_embedding_distances


def embedding_loss_fn(model, inputs, target_labels, loss_slice, negative_embeddings, positive_embeddings,
                      config, original_embedding, input_is_embed=True):
    """Compute embedding-based loss for GCG optimization - essential for core algorithm"""
    if input_is_embed:
        outputs = model(inputs_embeds=inputs, output_hidden_states=True)
    else:
        outputs = model(input_ids=inputs, output_hidden_states=True)

    # Get embeddings for embedding score
    last_hidden = outputs.hidden_states[-1][:, loss_slice]
    current_embedding = torch.mean(last_hidden, dim=1).float()

    # Compute embedding score
    embedding_score = compute_embedding_distances(
        current_embedding,
        negative_embeddings,
        positive_embeddings,
        config,
        original_embedding=original_embedding
    )

    # Fluency penalty based on logit distribution (no specific target)
    logits = outputs.logits[:, :-1, :]  # Predict next tokens
    probs = F.softmax(logits, dim=-1)

    if config['fluency_penalty'] > 0:
        # Measure entropy or max probability to encourage fluency
        entropy = - torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        fluency_penalty = config['fluency_penalty'] * entropy
        total_loss = embedding_score + fluency_penalty
    else:
        total_loss = embedding_score

    return total_loss


def prep_text_for_embedding_loss(input_str, model, tokenizer, device, num_free_tokens=10, system_prompt="", chat_template=["Rephrase the following: ", " Rephrased version:"], adv_suffix=False, adv_prefix=False):
    """Prepare text tokens for embedding-based loss optimization - essential for core algorithm"""
    input_tokens = tokenizer.encode(input_str, return_tensors="pt", add_special_tokens=False).to(device=device)
    with torch.no_grad():
        input_embeddings = model(input_ids=input_tokens, output_hidden_states=True).hidden_states[-1].mean(dim=1).float()
    
    chat_template_tokens = (
        tokenizer.encode(chat_template[0], return_tensors="pt", add_special_tokens=False).to(device=device),
        tokenizer.encode(chat_template[1], return_tensors="pt", add_special_tokens=False).to(device=device))
    free_tokens = torch.randint(0, tokenizer.vocab_size, (1, num_free_tokens)).to(device=device)

    # For embedding-based loss, we might not need target tokens at the end
    # Just concatenate system prompt, input, free tokens, and closing chat template
    if adv_suffix:
        input_ids = torch.cat((chat_template_tokens[0], input_tokens, chat_template_tokens[1], free_tokens, input_tokens), dim=1).squeeze().long()
        free_token_slice = slice(input_ids.size(-1) - input_tokens.size(-1) - num_free_tokens, input_ids.size(-1) - input_tokens.size(-1))
        input_slice = None
    elif adv_prefix:
        input_ids = torch.cat((free_tokens, chat_template_tokens[0], input_tokens, chat_template_tokens[1], input_tokens), dim=1).squeeze().long()
        free_token_slice = slice(0, num_free_tokens)
        input_slice = None
    else:
        input_ids = torch.cat((chat_template_tokens[0], free_tokens, input_tokens, chat_template_tokens[1], input_tokens), dim=1).squeeze().long()
        free_token_slice = slice(chat_template_tokens[0].size(-1), free_tokens.size(-1) + chat_template_tokens[0].size(-1))
        input_slice = slice(chat_template_tokens[0].size(-1), input_ids.size(-1) - input_tokens.size(-1) - chat_template_tokens[1].size(-1))

    target_slice = slice(input_ids.size(-1) - input_tokens.size(-1), input_ids.size(-1))
    loss_slice = slice(input_ids.size(-1) - input_tokens.size(-1) - 1, input_ids.size(-1) - 1)
    prompt_slice = slice(0, input_ids.size(-1) - input_tokens.size(-1))

    return input_ids, input_embeddings, free_token_slice, target_slice, loss_slice, prompt_slice, input_slice


def optimize_gcg(model, input_ids, free_token_slice, target_slice, loss_slice, loss_fn,
                 num_steps, topk=250, batch_size=50, mini_batch_size=50, eval_fn=None):
    """Optimize text using GCG - essential for core algorithm"""
    # Get embedding matrix more efficiently with error handling
    try:
        embedding_matrix = model.get_input_embeddings().weight
    except (NotImplementedError, AttributeError):
        try:
            embedding_matrix = model.transformer.wte.weight
        except (AttributeError, KeyError):
            embedding_matrix = model.model.embed_tokens.weight  # For some newer model architectures
    device = input_ids.device
    vocab_size = embedding_matrix.size(0)

    # Correctly handle free_token_slice as a slice object
    free_token_len = free_token_slice.stop - free_token_slice.start

    if eval_fn is None:
        # Use embedding loss function as default eval function
        eval_fn = lambda model, candidate_ids, target_labels, loss_slice: loss_fn(
            model, candidate_ids, target_labels, loss_slice
        )

    best_loss = float('inf')
    best_input = input_ids.clone()

    # Pre-allocate tensors that will be reused
    candidates_input_ids = torch.empty((batch_size, input_ids.size(0)),
                                       dtype=input_ids.dtype,
                                       device=device)
    loss_tensor = torch.zeros(batch_size, device=device)

    # Extract target labels once instead of in each iteration
    target_labels = input_ids[target_slice].squeeze()

    for i in range(num_steps):
        # Compute gradients for the free tokens
        inputs_one_hot = F.one_hot(input_ids, vocab_size).to(embedding_matrix.dtype).unsqueeze(0)
        inputs_one_hot.requires_grad_(True)
        inputs_embeds = torch.matmul(inputs_one_hot, embedding_matrix)

        # Compute loss
        loss = loss_fn(model, inputs_embeds, target_labels, loss_slice)

        # Get gradient for the free token slice
        grad = torch.autograd.grad(loss, inputs_one_hot)[0][0, free_token_slice]

        with torch.no_grad():
            # Get topk most promising tokens based on gradient
            top_values, top_indices = torch.topk(-grad, topk, dim=1)

            # Fill candidates with the current input_ids
            candidates_input_ids.copy_(input_ids.expand(batch_size, -1))

            # Select random positions to modify within the free token slice
            new_token_loc = torch.randint(0, free_token_len, (batch_size,), device=device)

            # Select random tokens from the top-k options for each position
            new_token_vals = top_indices[new_token_loc, torch.randint(0, topk, (batch_size,), device=device)]

            # Apply modifications to the free tokens
            for b in range(batch_size):
                absolute_pos = free_token_slice.start + new_token_loc[b]
                candidates_input_ids[b, absolute_pos] = new_token_vals[b]

            # Process in mini-batches to avoid OOM errors
            try:
                for mini_batch in range(0, batch_size, mini_batch_size):
                    end_idx = min(mini_batch + mini_batch_size, batch_size)

                    loss_tensor[mini_batch:end_idx] = eval_fn(
                        model,
                        candidates_input_ids[mini_batch:end_idx],
                        target_labels,
                        loss_slice
                    )
            except:
                mini_batch_size = mini_batch_size // 2
                for mini_batch in range(0, batch_size, mini_batch_size):
                    end_idx = min(mini_batch + mini_batch_size, batch_size)

                    loss_tensor[mini_batch:end_idx] = eval_fn(
                        model,
                        candidates_input_ids[mini_batch:end_idx],
                        target_labels,
                        loss_slice
                    )

            # Get best candidate
            best_candidate = torch.argmin(loss_tensor)
            candidate_loss = loss_tensor[best_candidate].item()
            input_ids = candidates_input_ids[best_candidate].clone()

            # Update best result if this is better
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_input = input_ids.clone()

    # Return best result found
    return {"input_ids": best_input, "inputs_embeds": model.get_input_embeddings()(best_input).unsqueeze(0)}
