# Example Surrogate Datasets

This folder contains surrogate datasets created using 3DS, for the combination of 2 datasets (Civil Comments, and WMDP-bio) and 3 LLMs (Meta-Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-Beta), corresponding to Figures 10-15 in the manuscript.


> [!CAUTION]
> The Civil Comments dataset is used to simulate unlearning toxic content from LLMs; therefore, the unlearning dataset is the subset of the data with a toxicity score >= 0.5. The surrogate datasets created by 3DS contain knowledge derived from this unlearning dataset, meaning they may also contain toxic and NSFW content.
