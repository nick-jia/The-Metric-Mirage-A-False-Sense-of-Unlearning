import numpy as np
import re
import nltk
import os
import json
from collections import deque
from datasets import load_dataset


def raw_text_to_sentences(text, mode="sentence_naive", window_size=50, stride=20):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)
    if mode == "sentence_naive":
        if isinstance(text, list):
            sentences = [sent for sent in text if len(sent) > 10]
        else:
            sentences = [sent for sent in nltk.sent_tokenize(text) if len(sent) > 10]
        word_counts = []
        for sent in sentences:
            word_counts.append(len(nltk.word_tokenize(sent)))
        print(f"average {np.average(word_counts)} +/- {np.std(word_counts)} words "
              f"([{np.min(word_counts)}, {np.max(word_counts)}])")
    elif mode == "sentence_improved":
        sentences = []
        if isinstance(text, list):
            temp_sentences = text
        else:
            temp_sentences = nltk.sent_tokenize(text)
        word_counts = []
        processed_sentences = []
        for sent in temp_sentences:
            word_count = len(nltk.word_tokenize(sent))
            if word_count > window_size:
                breakdowns = re.split(r';\s*', sent.strip())
                for breakdown_sent in breakdowns:
                    word_count = len(nltk.word_tokenize(breakdown_sent))
                    if word_count > window_size * 2:
                        hard_breakdowns = split_sentence_by_words(breakdown_sent, window_size)
                        for hard_breakdown in hard_breakdowns:
                            word_count = len(nltk.word_tokenize(hard_breakdown))
                            if word_count > 0:
                                processed_sentences.append(hard_breakdown)
                                word_counts.append(word_count)
                    elif word_count > 0:
                        processed_sentences.append(breakdown_sent)
                        word_counts.append(word_count)
            else:
                if word_count > 0:
                    processed_sentences.append(sent)
                    word_counts.append(word_count)
        print(f"max words in sentences before sliding window: {np.max(word_counts)}")
        ideal_sent_per_window = 2
        if stride >= window_size:
            ideal_sent_per_window = 1

        i = 0
        current_combination = deque()
        current_word_count = deque()
        final_word_counts = []
        total_word_count = sum(word_counts)
        running_word_count = 0
        while i < len(processed_sentences):
            current_combination.append(processed_sentences[i])
            current_word_count.append(word_counts[i])
            running_word_count += word_counts[i]
            total_word_count -= word_counts[i]
            i += 1
            if running_word_count + total_word_count < window_size:
                # do not add sentence if the remaning #tokens is smaller than window size
                break
            if running_word_count >= window_size and len(current_combination) >= ideal_sent_per_window:
                sentences.append(' '.join(current_combination))
                final_word_counts.append(running_word_count)
                remove_word_counts = 0
                while current_combination and remove_word_counts < stride:
                    remove_word_counts += current_word_count[0]
                    running_word_count -= current_word_count.popleft()
                    current_combination.popleft()
        assert len(final_word_counts) == len(sentences)
        print(f"average {np.average(final_word_counts)} +/- {np.std(final_word_counts)} words "
              f"([{np.min(final_word_counts)}, {np.max(final_word_counts)}])")
    elif mode == "words":
        if isinstance(text, list):
            text = ' '.join(text)
        words = nltk.word_tokenize(text)
        sentences = []
        for i in range(0, len(words) - window_size + 1, stride):
            if i + window_size > len(words):
                break
            sentence = words[i:i + window_size]
            sentences.append(' '.join(sentence))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return sentences


def split_sentence_by_words(sentence, max_words):
    """Split a sentence into chunks with maximum word count"""
    words = nltk.word_tokenize(sentence)
    chunks = []
    current_chunk = []
    current_count = 0
    
    for word in words:
        if current_count + 1 <= max_words:
            current_chunk.append(word)
            current_count += 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_count = 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def read_dataset(dataset_name, ret_type="sent", num_samples=1e10, split="train", min_words=0, mode="sentence_naive", percent=None):
    assert ret_type == "sent" or ret_type == "doc"

    wmdp_datasets = {
        "wmdp_bio": ("data/wmdp/bio-forget-corpus.jsonl", True),  # True: use 'text' field
        "wmdp_bio_retain": ("data/wmdp/bio-retain-corpus.jsonl", False),
        "wmdp_cyber": ("data/wmdp/cyber-forget-corpus.jsonl", False),  # False: use raw line
        "wmdp_cyber_retain": ("data/wmdp/cyber-retain-corpus.jsonl", False)
    }
    if dataset_name in wmdp_datasets:
        file_path, use_text_field = wmdp_datasets[dataset_name]
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    if use_text_field:
                        raw_text = json.loads(line.strip())['text']
                    else:
                        raw_text = line.strip()
                    if len(raw_text) > min_words:
                        data.append(raw_text)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file_path}: {line[:50]}...")
        data = " ".join(data)
        if ret_type == "sent":
            dataset = raw_text_to_sentences(data, mode=mode, window_size=50, stride=50)
            if percent is not None:
                sample_size = max(1, round(len(dataset) * percent / 100))
                dataset = dataset[:sample_size]
                print(f"Dataset: {dataset_name} ({percent}%)| {len(dataset)} sentences")
            else:
                print(f"Dataset: {dataset_name}| {len(dataset)} sentences")
        else:
            dataset = " ".join(data)
            print(f"Dataset: {dataset_name}")
        return dataset

    if dataset_name in ["wikitext_train", "wikitext_test"]:
        split = "train" if dataset_name == "wikitext_train" else "test"
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        data = [item['text'] for item in dataset if len(item['text']) > min_words]
        if ret_type == "sent":
            dataset = raw_text_to_sentences(data, mode=mode)
            if percent is not None:
                sample_size = max(1, round(len(dataset) * percent / 100))
                dataset = dataset[:sample_size]
                print(f"Dataset: {dataset_name} ({percent}%)| {len(dataset)} sentences")
            else:
                print(f"Dataset: {dataset_name} | {len(dataset)} sentences")
        else:
            dataset = " ".join(data)
            print(f"Dataset: {dataset_name}")
        return dataset

    if dataset_name == "toxic":
        book = open(f"data/civil_comments_toxic.txt", 'rb').read().decode(encoding='utf-8')
        if ret_type == "sent":
            dataset = raw_text_to_sentences(book, mode=mode)
            if percent is not None:
                sample_size = max(1, round(len(dataset) * percent / 100))
                dataset = dataset[:sample_size]
                print(f"Dataset: Google's civil comments dataset (toxic) ({percent}%)| {len(dataset)} sentences")
            else:
                print(f"Dataset: Google's civil comments dataset (toxic)| {len(dataset)} sentences")
        elif ret_type == "doc":
            dataset = re.sub(r'\s+', ' ', book)
            print(f"Dataset: Google's civil comments dataset (toxic)")
    elif dataset_name == "nontoxic":
        book = open(f"data/civil_comments_benign.txt", 'rb').read().decode(encoding='utf-8')
        if ret_type == "sent":
            dataset = raw_text_to_sentences(book, mode=mode)
            if percent is not None:
                sample_size = max(1, round(len(dataset) * percent / 100))
                dataset = dataset[:sample_size]
                print(f"Dataset: Google's civil comments dataset (nontoxic) ({percent}%)| {len(dataset)} sentences")
            else:
                print(f"Dataset: Google's civil comments dataset (nontoxic)| {len(dataset)} sentences")
        elif ret_type == "doc":
            dataset = re.sub(r'\s+', ' ', book)
            print(f"Dataset: Google's civil comments dataset (nontoxic)")
    else:
        print("Not using pre-stored datasets, so the dataset is expected to be the path to a txt file")
        book = open(dataset_name, 'rb').read()
        data_name = dataset_name.split("/")[-1]
        data_name = data_name.split(".")[0]
        if ret_type == "sent":
            dataset = raw_text_to_sentences(book, mode=mode)
            print(f"Dataset: {data_name}| {len(dataset)} sentences")
        elif ret_type == "doc":
            dataset = re.sub(r'\s+', ' ', book)
            print(f"Dataset: {data_name}")
    return dataset


def obtain_sentence_dataset(dataset_name, save_path=None, return_save_path=False, num_samples=1e10,
                            mode="sentence_naive", ret_type='sent', verbose=True):
    print(f"loading dataset {dataset_name}")
    all_datasets = [] if ret_type == "sent" else ""

    if save_path is None:
        if isinstance(dataset_name, list):
            data_name = "&".join(subfile.split("/")[-1] for subfile in dataset_name)
        else:
            data_name = dataset_name
        if not os.path.exists(data_name):
            save_path = f"data/dataset_{mode}/{data_name}_{mode}"
        else:
            print("dataset itself is a file and save_path is not set, so will not save another file")

    need_process = True
    if save_path is not None:
        if ret_type == "sent":
            save_path = f"{save_path}.json"
            if os.path.exists(save_path):
                need_process = False
                with open(save_path, "r") as infile:
                    all_datasets = json.load(infile)["text"]
        else:
            save_path = f"{save_path}.txt"
            if os.path.exists(save_path):
                need_process = False
                with open(save_path, "r") as infile:
                    all_datasets = infile.read()


    percents = []
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    new_dataset_name = []
    for subset_name in dataset_name:
        if "percent" in subset_name and "D_f_prime" not in subset_name:
            subset_name, percent = subset_name.split("_percent")
            new_dataset_name.append(subset_name)
            percents.append(float(percent))
            if not ("wmdp" in subset_name or "toxic" in subset_name or "wikitext_train" in subset_name):
                raise NotImplementedError(f"percent is currently only implemented for wmdp, toxic and wikitext_train")
        else:
            new_dataset_name.append(subset_name)
            percents.append(None)
    dataset_name = new_dataset_name

    if verbose:
        print(f"Datasets: {dataset_name}", flush=True)
    assert len(dataset_name) == len(percents)
    if need_process:
        assert len(dataset_name) == len(percents)
        for name, perc in zip(dataset_name, percents):
            sent_dataset = read_dataset(name, ret_type=ret_type, num_samples=num_samples, mode=mode, percent=perc)
            if ret_type == "sent":
                all_datasets.extend(sent_dataset)
            else:
                all_datasets += sent_dataset

        if save_path is not None:
            if ret_type == "sent":
                books = {"text": all_datasets}
                with open(save_path, "w") as outfile:
                    json.dump(books, outfile)
            else:
                with open(save_path, "w") as outfile:
                    outfile.write(all_datasets)

    return all_datasets


def load_wmdp_bio_dataset(split, num_samples=1e10):
    """Load wmdp_bio dataset - essential for core algorithm"""
    # Simplified wmdp_bio loading
    dataset_path = f"data/wmdp_bio_{split}.txt"
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    else:
        # Return sample text if file doesn't exist
        return "Sample biomedical text for testing purposes."


def load_wikitext_dataset(split, num_samples=1e10):
    """Load wikitext dataset - essential for core algorithm"""
    # Simplified wikitext loading
    dataset_path = f"data/wikitext_{split}.txt"
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    else:
        # Return sample text if file doesn't exist
        return "Sample wikitext for testing purposes."
