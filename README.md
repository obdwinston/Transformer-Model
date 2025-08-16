Decoder-only generative model implemented from scratch based on GPT-2/GPT-Neo architecture, featuring Transformer blocks with masked multi-head self-attention (local and global) for autoregressive text generation.

The configured 41M model was pre-trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), which comprises short, synthetically-generated (using GPT-3.5 and GPT-4) children’s stories. The [TinyStories paper by Microsoft Research](https://arxiv.org/abs/2305.07759) showed that very Small Language Models (SLMs) can learn to generate coherent, diverse texts, and even demonstrate limited reasoning capabilities.

Besides reducing the number of transformer blocks, the small parameter counts reported (≤33M parameters) can also be attributed to a reduced vocabulary size, from the original ~50K tokens for the default GPT-2/GPT-Neo tokeniser, to just the top 10K tokens based on frequency (this critical detail was relegated to a footnote). This drastic reduction is enabled by the simpler corpus, resulting in significantly smaller embedding and output layers, and therefore a much smaller model (~60% reduction in total parameters, see table below).

| Component            | Full Vocabulary<br>(50,257 tokens) |            | Reduced Vocabulary<br>(8,000 tokens) |            |
| -------------------- | ---------------------------------- | ---------- | ------------------------------------ | ---------- |
|                      | Parameters                         | Percentage | Parameters                           | Percentage |
| Embedding Layers     | 38,990,592                         | 36.8%      | 6,537,984                            | 15.9%      |
| Transformer Blocks   | 28,342,272                         | 26.8%      | 28,342,272                           | 69.1%      |
| Output Layers        | 38,598,912                         | 36.4%      | 6,146,304                            | 15.0%      |
| **Total Parameters** | **105,931,776**                    |            | **41,026,560**                       |            |

Despite replicating the paper's 33M [model architecture](https://huggingface.co/roneneldan/TinyStories-33M/blob/main/config.json) and further reducing the vocabulary size from 10K to 8K tokens, the current model still has roughly 41M parameters. Notably, the top 10K and 8K tokens cover 99.9% and 99.8% of all words in the TinyStories dataset respectively.

The training cross-entropy loss for the current 41M model was ~1.4 after 1 epoch, which is higher than the ~1.0 (assumed to also be cross-entropy loss) reported for the paper's 33M model after 2,500 training steps. Validation losses for both models were ~1.5 and ~1.2 respectively. The only apparent differences from the paper's setup are the reduced vocabulary size (8K versus 10K tokens), the smaller effective batch size (16 batch size × 5 gradient accumulation steps = 80 versus 1,280 based on the [model card](https://huggingface.co/roneneldan/TinyStories-33M)), and the total number of training steps/epochs (what constitutes a training step was not defined). Despite the ~40% higher training losses, the current 41M model was still able to generate coherent, logical sentences.

Examples of the model-generated texts are given below, with the prompt highlighted.

> <mark>Once upon a time in a land far, far away, there lived a</mark> little girl named Lily. Lily loved to play outside in the sun and feel the warm breeze on her face. One day, Lily's mommy told her they were going to visit her grandma. Lily was so excited! When they arrived at her grandma's house, Lily saw a big, scary dog. She was afraid and started to cry. Her mommy hugged her and said, "Don't worry, Lily. The dog is just playing. He won't hurt you."

> <mark>There was once a beautiful princess</mark> who lived in a big castle. One day, the princess decided to go for a walk in the forest. As she was walking, she saw a little bird with a broken wing. The princess felt sad for the bird and wanted to help. She gently picked up the bird and brought it home. The princess took care of the bird and made sure it was safe. She gave it food and water and talked to it.

> <mark>There once was a boy who lived in a small house.</mark> He was very curious and wanted to explore the world around him. One day, he decided to go outside and see what he could find.
> He walked around the garden, looking at all the plants and trees. He was so excited to explore. Suddenly, he heard a loud noise and saw a big, scary monster! The boy was so scared, he ran away as fast as he could.

<img width="1367" height="1892" alt="transformer-model" src="https://github.com/user-attachments/assets/6820bb61-6151-4f7f-9f36-084d197f43cf" />
