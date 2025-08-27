import argparse
import os
import torch
import numpy as np
import random
import warnings
import data_utils
import model_utils
import utils


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--tokenizer-dir', type=str, default=None)
    parser.add_argument('--ckpt-interval', type=int, default=128)
    parser.add_argument('--embed-model-name', type=str, default=None)
    parser.add_argument('--max-sents', type=int, default=-1)
    parser.add_argument('--num-samples', type=int, default=1e8)
    parser.add_argument('--data', type=str)
    parser.add_argument('--tokenize', type=str, default='sentence_improved')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--mean-pooling', action="store_true")
    parser.add_argument('--early-layer', action="store_true")
    arg = parser.parse_args()

    if arg.tokenizer_dir is None:
        arg.tokenizer_dir = utils.get_tokenizer_dir(arg.model_dir)
    if arg.embed_model_name is None:
        arg.embed_model_name = arg.model_dir

    ckpt_prefix = "res"
    os.makedirs(ckpt_prefix, exist_ok=True)

    # Generate filenames
    embed_model_name = arg.embed_model_name.split('/')[-1]
    if arg.mean_pooling:
        embed_model_name += "_meanpool"
    if arg.early_layer:
        embed_model_name += "_earlylayer"
    data_name = arg.data.split('/')[-1]
    metrics_str = '_'.join(arg.metrics)

    embed_filename = f"{ckpt_prefix}/{embed_model_name}_{data_name}_{arg.tokenize}_embeddings.npy"

    # Load dataset
    sentences = data_utils.obtain_sentence_dataset(arg.data, mode=arg.tokenize, num_samples=arg.num_samples)
    if len(sentences) > arg.max_sents and arg.max_sents > 0:
        sentences = random.sample(sentences, arg.max_sents)

    # Initialize data structure
    data = utils.load_embedding(embed_filename) or {
        'sentences': sentences,
        'embeddings': np.array([None] * len(sentences), dtype=object),
    }

    total_sentences = len(sentences)
    missing_indices = [i for i, emb in enumerate(data['embeddings']) if emb is None]
    if len(missing_indices) > 0:
        if arg.early_layer:
            embed_model, encode_fn = model_utils.get_embedding_model(arg.embed_model_name,
                                                                     mean_pooling=arg.mean_pooling,
                                                                     layer=-20)
        else:
            embed_model, encode_fn = model_utils.get_embedding_model(arg.embed_model_name,
                                                                     mean_pooling=arg.mean_pooling)
        for i in range(0, len(missing_indices), arg.batch_size):
            batch_indices = missing_indices[i:i + arg.batch_size]
            batch_sentences = [data['sentences'][idx] for idx in batch_indices]

            try:
                batch_embeddings = encode_fn(embed_model, batch_sentences)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                half_batch = len(batch_sentences) // 2
                batch_embeddings = torch.concat([
                    encode_fn(embed_model, batch_sentences[:half_batch]),
                    encode_fn(embed_model, batch_sentences[half_batch:])
                ])

            for j, emb in enumerate(batch_embeddings):
                data['embeddings'][batch_indices[j]] = emb

            if (i + arg.batch_size) % arg.ckpt_interval == 0 or (i + arg.batch_size) >= total_sentences:
                utils.save_progress(data, embed_filename)

    utils.save_progress(data, embed_filename)
    print(f"Processed {total_sentences} sentences")
    if data.get('embeddings') is not None and len(data['embeddings']) > 0:
        first_embedding_shape = np.array(data['embeddings'][0]).shape
    else:
        first_embedding_shape = 'N/A'
    print(f"Embedding shape: {len(data['embeddings']) if data.get('embeddings') is not None else 0} x {first_embedding_shape}")
