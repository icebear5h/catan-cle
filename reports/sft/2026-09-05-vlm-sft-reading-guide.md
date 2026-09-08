# A practical reading curriculum for VLM supervised fine-tuning

Research checked September 5, 2026. Prepared for someone building a structured visual reader and wanting a broader understanding of SFT—not just another proposed fix for one run.

This guide contains **24 readings: 22 practical blogs, tutorials or documentation chapters, plus two optional papers**. Dated material is predominantly from 2024–2026; one older paper supplies the precise language for gradient conflict. Living documentation is labeled by access date, not presented as newly published research.

The goal is to understand what each part of a training system does, what its measurements mean, and which experiments can distinguish competing explanations. Nothing here establishes that our present bottleneck is LoRA rank, token geometry, vision adaptation or data mix. No training recipe or live run was changed for this research.

Reading times are editorial estimates for the suggested sections, excluding running notebooks. “Learn” summarizes the source; “Boundary” marks limits and our caution about applying it. Public pages and relevant sections were inspected; linked training notebooks were not executed.

## Start here: six readings, roughly 2–2¾ hours

Read in this order; you do not need to finish the whole list before it becomes useful.

| Order | Reading | Question to answer afterward |
| --- | --- | --- |
| 1 | [01 — nanoVLM](#01-nanovlm) | What do the encoder, connector and decoder each receive and produce? |
| 2 | [06 — End-to-end VLM SFT](#06-vlm-trl-walkthrough) | How does an image/question/answer become a training batch and a saved model? |
| 3 | [09 — Dataset design](#09-datasets-guide) | Does each example unambiguously demonstrate the desired behavior? |
| 4 | [18 — Gradient accumulation](#18-gradient-accumulation) | How are short answers and long readouts weighted by the actual loss? |
| 5 | [14 — LoRA Without Regret](#14-lora-without-regret) | Which limitations involve rank, and which involve optimization or placement? |
| 6 | [19 — Evaluation](#19-evaluation) | What can the reported metric establish—and what does it leave untested? |

For the closest practical VLM retention example, read [21 — SmolVLM](#21-smolvlm-retention) next. For the newest architecture alternative, read [05 — MolmoPoint](#05-molmopoint). For a recent comparison of adaptation methods, read [16 — Beyond LoRA](#16-beyond-lora).

## The curriculum

Vocabulary tokens are discrete symbols; visual tokens are usually continuous image-region embeddings. A connector translates visual features for the decoder, sometimes compressing them. The exact modules vary by model. [nanoVLM](https://huggingface.co/blog/nanovlm)

## A. Architecture: what the image and text pathways actually do

<a id="01-nanovlm"></a>

### 01. [nanoVLM: The simplest repository to train your VLM in pure PyTorch](https://huggingface.co/blog/nanovlm)

Aritra Roy Gosthipaty, Luis (lusxvr), Andres Marafioti, Sergio Paniego, Merve Noyan, Pedro Cuenca and Vaibhav Srivastav · May 21, 2025\
Engineering tutorial · Beginner–intermediate · 20–25 min

Read: The architecture explanation, model initialization, and the two-learning-rate optimizer setup. Follow the linked module files if you want code.

Learn: The encoder turns pixels into features; pixel shuffle and a projection compress and translate those features; the decoder consumes the resulting embeddings alongside text. Newly initialized connectors and pretrained backbones need not share an optimizer group.

Boundary: An educational VQA model, not Qwen’s exact implementation. Lower backbone learning rates are a design choice, not a guarantee of retention.


<a id="02-smaller-smolvlm"></a>

### 02. [SmolVLM Grows Smaller – Introducing the 256M & 500M Models!](https://huggingface.co/blog/smolervlm)

Andres Marafioti, Miquel Farré and Merve Noyan · January 23, 2025\
Model-team blog · Intermediate · 10–15 min

Read: The overview and the changes relative to the earlier 2B model, especially encoder selection and tokenization.

Learn: Image resolution, visual compression and text tokenization are different knobs. The team reports that replacing multi-token crop-position strings with special tokens improved stability and downstream results even though training loss looked worse.

Boundary: These are compact-model results. They do not establish a good initialization distribution for our atlas rows, nor make low row cosine a correctness certificate. Discrete vocabulary tokens and continuous image embeddings are different objects.


<a id="03-idefics2"></a>

### 03. [Introducing Idefics2: A Powerful 8B Vision-Language Model for the community](https://huggingface.co/blog/idefics2)

Leo Tronchon, Hugo Laurençon and Victor Sanh · April 15, 2024\
Model-team blog · Beginner–intermediate · 10–15 min

Read: The architecture comparison, training-data description and improvements over Idefics1.

Learn: Native aspect ratios, image splitting and feature pooling solve different problems. Its split/non-split comparison illustrates that spending more visual tokens can help document tasks without improving every benchmark. Cauldron provides an example of standardizing multiple instruction datasets.

Boundary: Its Perceiver pooling is not Qwen’s merger. A task-dependent comparison does not establish a universally optimal image size or visual-token count.


<a id="04-paligemma2"></a>

### 04. [Welcome PaliGemma 2 – New vision language models by Google](https://huggingface.co/blog/paligemma2)

Merve Noyan, Andreas P. Steiner, Pedro Cuenca and Aritra Roy Gosthipaty · December 5, 2024\
Technical release/tutorial · Beginner–intermediate · 15–20 min

Read: The introduction, capabilities and fine-tuning sections; follow the technical report for controlled size/resolution comparisons.

Learn: A pretrained transfer checkpoint is not the same as a task-finetuned model. Model size and image resolution offer distinct compute tradeoffs. Caption length and factual correctness also require different measurements.

Boundary: Success after separate task-specific fine-tunes is not evidence that one shared checkpoint retains all tasks simultaneously. The demonstration uses part of a validation split for tuning; maintain a genuinely untouched test split in your own work.


<a id="05-molmopoint"></a>

### 05. [MolmoPoint: Better pointing architecture for vision-language models](https://allenai.org/blog/molmopoint)

Ai2 · March 18, 2026\
Model-team technical blog · Intermediate · 12–15 min

Read: The introduction, visual-evidence selection mechanism and image/video pointing results.

Learn: An alternative to emitting coordinate text: select a coarse patch, refine within it and predict local coordinates. The system also models stopping and point ordering. This is a useful case study in changing an output representation rather than only adding examples.

Boundary: The grounding tokens invoke specialized architecture; adding ordinary special tokens to Qwen would not reproduce it. Pointing is not attribute reading, and neither establishes whole-board reconstruction or retention.


## B. SFT fundamentals: what your training loop is optimizing

<a id="06-vlm-trl-walkthrough"></a>

### 06. [How to Fine-Tune Multimodal Models or VLMs with Hugging Face TRL](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl)

Philipp Schmid · September 30, 2024\
End-to-end tutorial · Beginner–intermediate · 20–30 min

Read: Defining a use case and baseline, preparing examples, the collator/trainer, and loading the adapter for inference.

Learn: The clearest practical bridge from an image-plus-text example to tokenization, labels, optimization, saving and generated answers. Read it as a complete workflow rather than a collection of hyperparameters.

Boundary: It uses older Qwen2-VL/TRL APIs and product descriptions with text metadata. Its collator masks padding/image tokens, not every user/system token. Its adapter-only saving matches its training scope; it would not automatically preserve separately trained vision weights.


<a id="07-trl-sft-trainer"></a>

### 07. [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

Hugging Face TRL maintainers · Living documentation; no fixed publication date; accessed September 5, 2026\
Official reference · Intermediate · 20–25 min

Read: The SFT objective, logged metrics, completion/assistant-only loss and VLM-specific training guidance. Skip the exhaustive API listing initially.

Learn: Loss depends on which labels remain unmasked. Token accuracy under gold-prefix conditioning is different from free-running generation. Assistant-only masking relies on compatible templates; image-token truncation is a separate VLM hazard.

Boundary: Current documentation may not match an installed trainer version. Verify the actual processed batch, labels and reduction. This guide does not imply our current code has a masking bug.


<a id="08-chat-templates"></a>

### 08. [Chat templates](https://huggingface.co/docs/transformers/chat_templating)

Hugging Face Transformers maintainers · Living documentation; no fixed publication date; accessed September 5, 2026\
Official reference · Beginner–intermediate · 10–15 min

Read: Message serialization, generation prompts, special tokens and formatting examples for training.

Learn: A conversation object must become the exact token sequence expected by the model. Role markers, turn endings and generation prefixes are part of that contract. Formatting messages correctly does not, by itself, establish the desired loss mask.

Boundary: Do not transplant one model’s control tokens into another. For VLMs, also inspect the processor’s image placeholders and image/text alignment. A visually plausible rendered prompt can still encode incorrectly.


## C. Dataset design: coverage, ambiguity and effective supervision

<a id="09-datasets-guide"></a>

### 09. [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)

Unsloth documentation team · Living documentation; no fixed publication date; accessed September 5, 2026\
Practical guide · Beginner · 15–20 min

Read: Task definition, instruction/conversation formatting, synthetic-data quality, combining datasets and vision examples. Skip product-UI instructions.

Learn: Start with the behavior and answer format you want, then build examples that actually demonstrate it. Normalize multiple sources to a consistent schema and inspect generated labels rather than trusting synthetic volume.

Boundary: Dataset-size recommendations are starting heuristics. Combining sources does not guarantee adequate coverage, useful per-task gradients or retention. A previous mixed run is evidence to audit its exact supervision—not a reason to advertise mixing as an untried cure.


<a id="10-molmo-data"></a>

### 10. [Molmo](https://allenai.org/blog/molmo)

Ai2 · September 25, 2024\
Model-team data/architecture blog · Intermediate · 20–25 min

Read: The PixMo dataset descriptions, evaluation methodology and architecture summary.

Learn: Dense descriptions, pointing and synthetic documents provide different forms of supervision. Pointing examples include all matching objects and absent-object cases: a useful precedent for replacing ambiguous singular retrieval questions with complete-set or explicitly disambiguated targets.

Boundary: The models inherit pretrained language and vision backbones. The relatively small new dataset is not their total lifetime training data. Human preference and benchmark scores are not exact structured-state validation.


<a id="11-multimodal-pipeline"></a>

### 11. [Efficient MultiModal Data Pipeline](https://huggingface.co/blog/mmdp)

Aritra Roy Gosthipaty, Luis (lusxvr), Andres Marafioti, Sergio Paniego and Pedro Cuenca · July 8, 2025\
Engineering tutorial · Intermediate · 20–30 min

Read: Stages 1–3 for loading/padding intuition, then the packing and multimodal knapsack sections.

Learn: Useful throughput depends on loading, padding and packing, not just nominal batch size. Multimodal batches must budget image work as well as text tokens. Inspect the length distribution before changing batching.

Boundary: Packing efficiency is not task balance. Filtering long examples can change the curriculum. Check that a real implementation preserves image alignment and intended attention boundaries; diagrams alone do not prove those properties.


## D. Optimization and LoRA: capacity is only one variable

<a id="12-lora-dora"></a>

### 12. [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)

Sebastian Raschka · February 18, 2024\
Code-first explainer · Beginner–intermediate · 25–40 min

Read: The LoRA recap and implementation, applying it to linear layers, then the magnitude/direction decomposition for DoRA.

Learn: Understand adapter matrix shapes, rank, an initially zero adapter contribution and why merged versus additive execution can be equivalent. This makes configuration fields much less mysterious.

Boundary: The teaching implementation scales directly by alpha; standard PEFT LoRA uses alpha divided by rank. Numerical alpha values are therefore not interchangeable. Variant-performance claims come from the linked paper, not a new board-reading experiment.


<a id="13-peft-lora"></a>

### 13. [LoRA](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora)

Hugging Face PEFT maintainers · Version 0.20.0 documentation; no fixed publication date; accessed September 5, 2026\
Official reference · Intermediate · 15–25 min

Read: Rank stabilization, target-module selection, fine-grained ranks/scaling, trainable token indices and modules to save.

Learn: Rank, scaling, adapter placement and extra trainable parameters are separate choices. Learn how to identify what is trained and what the saved artifact contains, including newly trained vocabulary rows.

Boundary: API support is not experimental evidence that a technique solves forgetting. If visual weights were independently updated, turning off a language adapter does not restore the old visual model. Check your installed PEFT version.


<a id="14-lora-without-regret"></a>

### 14. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)

John Schulman and colleagues, Thinking Machines Lab · September 29, 2025\
Original experimental blog · Intermediate · 25–35 min

Read: Rank, batch-size effects, adapter placement and parameterization invariances; inspect the learning-rate sweep setup.

Learn: Their SFT results distinguish rank-dependent limits from a large-batch disadvantage that higher rank does not remove. Attention-only adaptation also underperforms broader placement in matched comparisons. A fair rank experiment must control more than parameter count.

Boundary: These are text instruction/reasoning experiments, not generative VLM SFT. Rank-one success in their math-RL setting does not establish adequate capacity for our board reader. Their numerical learning-rate guidance is empirical.


<a id="15-unsloth-hyperparameters"></a>

### 15. [LoRA fine-tuning Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

Unsloth documentation team; Eyera acknowledged · Living documentation; no fixed publication date; accessed September 5, 2026\
Practical parameter guide · Beginner–intermediate · 15–20 min

Read: Rank/scaling, target modules, LoRA versus QLoRA, and checking whether weights actually changed.

Learn: A useful vocabulary and checklist for translating experiments into configuration. Compare its recommendations with the controlled experiments in reading 14 rather than treating one set of defaults as law.

Boundary: The page’s suggestion that loss below 0.2 likely indicates overfitting is not a general diagnostic. Loss depends on masking, task entropy, tokenization and reduction; examine held-out behavior. Rank/alpha recommendations are also heuristics.


<a id="16-beyond-lora"></a>

### 16. [Beyond LoRA: Can you beat the most popular fine-tuning technique?](https://huggingface.co/blog/peft-beyond-lora)

Benjamin Bossan, Sayak Paul, Marian Tietz and Kashif Rasul · June 18, 2026\
Framework-team experimental blog · Intermediate · 12–18 min

Read: The problems with comparing paper results, the PEFT benchmarking approach and the findings.

Learn: Compare accuracy, retention, memory, runtime and deployment compatibility—not just a winning benchmark score. The post illustrates why tuned baselines and matched experimental conditions matter when evaluating alternatives to vanilla LoRA.

Boundary: Its experiments include text-math SFT and diffusion image generation. That is not an image-to-text VLM comparison. Treat its method-ranking advice as a result in those settings, not a universal replacement recommendation.


<a id="17-mixed-precision"></a>

### 17. [Introducing Mixed Precision Training in Opacus](https://pytorch.org/blog/introducing-mixed-precision-training-in-opacus/)

Iden Kalemaj and Huanyu Zhang · August 12, 2025\
PyTorch engineering blog · Intermediate · 10–15 min

Read: The explanation and diagrams contrasting low-precision and mixed-precision training, then the implementation example.

Learn: Parameter storage, forward/backward arithmetic and optimizer updates are distinct precision decisions. BF16 computation does not necessarily mean BF16 master weights; understanding that distinction is essential when diagnosing very small updates.

Boundary: The implementation concerns privacy-preserving Opacus training. Do not transfer its DP-SGD performance results to AdamW or Qwen. Use it for the numerical model, not as a recommendation to adopt differential privacy.


<a id="18-gradient-accumulation"></a>

### 18. [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)

Hugging Face Transformers team: Lysandre, Arthur Zucker, Zachary Mueller, Yih-Dar Shieh, Benjamin Bossan and Pedro Cuenca · October 16, 2024\
Engineering investigation · Intermediate · 10–15 min

Read: The bug’s origin and the loss-reduction fix; trace the numerator and denominator in the code.

Learn: For token-mean causal-language-model loss, accumulated microbatches should contribute summed loss divided by the total supervised-token count. Averaging microbatch means can change weighting when valid-token counts differ. This matters when mixing one-word answers and long readouts.

Boundary: The historical bug is not evidence that our installed trainer is broken. Token counts help audit weighting but do not completely determine gradient contributions. Correct normalization also does not guarantee bitwise equivalence across batching strategies.


## E. Evaluation: distinguish learning signals from usable behavior

<a id="19-evaluation"></a>

### 19. [Let’s talk about LLM evaluation](https://huggingface.co/blog/clefourrier/llm-evaluation)

Clémentine Fourrier · May 23, 2024\
Evaluation-methodology blog · Beginner–intermediate · 20–25 min

Read: The different evaluation methods and why we evaluate: ranking, capability measurement and non-regression.

Learn: A dataset, prompt and scoring rule jointly define what a result means. Choice ranking, generated responses, human judgments and model judges answer different questions. Evaluation needs to match the intended use rather than a convenient leaderboard.

Boundary: Primarily text-LLM evaluation. Applying its principles to image controls and board reconstruction is our methodological extension, not an experiment reported in the post. Historical benchmark examples are not current model rankings.


<a id="20-eval-reproducibility"></a>

### 20. [Troubleshooting reproducibility](https://github.com/huggingface/evaluation-guidebook/blob/main/contents/troubleshooting/troubleshooting-reproducibility.md)

Clémentine Fourrier and Hugging Face Evaluation Guidebook contributors · Undated archived chapter; accessed September 5, 2026\
Practical guidebook chapter · Intermediate · 10–15 min

Read: Differences across codebases, prompts/templates, generation parameters and model-loading configuration.

Learn: An apparent regression can come from a different prompt, stopping rule, scorer or loaded artifact. Even identically named exact-match metrics may differ in how answers are selected or normalized.

Boundary: This links the readable archived chapter, not an inspected copy of the newer interactive edition. Apply the debugging principles while checking current library behavior. It is not permission to relax the scorer until an incorrect prediction passes.


## F. Joint learning, retention and alternatives to a single-cause story

<a id="21-smolvlm-retention"></a>

### 21. [SmolVLM - small yet mighty Vision Language Model](https://huggingface.co/blog/smolvlm)

Andres Marafioti, Merve Noyan, Miquel Farré, Elie Bakouch and Pedro Cuenca · November 26, 2024\
Model-team training report · Intermediate · 15–20 min

Read: Architecture and training details, especially context extension and checkpoint selection.

Learn: A concrete VLM example of uneven capability trajectories: DocVQA could deteriorate while most benchmarks improved. The team also changed math sampling after observing a decline during context extension. Read this for the measurements that motivated recipe changes.

Boundary: Their weighted checkpoint score is application-specific. It is not a universal replacement for per-behavior retention floors, and their data proportions are not a proven fix for our mixture.


<a id="22-forgetting-explainer"></a>

### 22. [LLM Research Insights: Instruction Masking and New LoRA Finetuning Experiments](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction)

Sebastian Raschka · June 2, 2024\
Research explainer · Intermediate · 10–20 min

Read: The sections on learning versus forgetting and higher-rank adaptation; use the instruction-masking section as a companion to reading 7.

Learn: An accessible introduction to separating acquisition of new capabilities from preservation of old ones. It also distinguishes instruction tuning from adaptation that demands substantial new domain knowledge.

Boundary: This is a secondary explanation of linked studies. Use reading 23 for consequential empirical claims; some proposed mechanisms in the post are explicitly speculative. The studies do not establish our model’s best achievable joint performance.


<a id="23-lora-learns-less"></a>

### 23. [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)

Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz and colleagues · May 15, 2024; final arXiv revision September 20, 2024\
Optional primary paper · Advanced · 35–50 min

Read: Acquisition and retention results, then update-rank and hyperparameter analyses (§§4.1–4.3, 4.6–4.7).

Learn: In the studied settings, full fine-tuning often acquires more while retaining less; higher-rank instruction tuning can narrow acquisition gaps. More adaptation capacity and better retention are therefore not interchangeable objectives.

Boundary: Llama-2 text code/math, with both instruction tuning and much larger continued-pretraining workloads. This supports testing rank as one hypothesis, not declaring rank 8 the cause of a VLM regression.


<a id="24-gradient-surgery"></a>

### 24. [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman and Chelsea Finn · January 19, 2020; NeurIPS 2020 — older foundation, intentionally included\
Optional primary paper · Advanced · 20–30 min

Read: Conflicting gradients, the projection algorithm and its theoretical conditions (§§2.2–2.4).

Learn: The paper defines conflict using negative gradient cosine and studies its interaction with magnitude imbalance and curvature. Orthogonal gradients are not the same as conflicting ones. PCGrad modifies conflicting components rather than treating every difference between tasks as harmful.

Boundary: Experiments concern multitask supervised learning and RL, not contemporary generative VLM SFT. The local theory is not a global no-forgetting guarantee or proof that gradient surgery is the next fix.

## Claims these readings should make us more careful about

### “A falling loss means the visual skill is improving”

Not necessarily. SmolVLM’s smaller-model report describes a tokenization change that improved downstream performance while making training loss look worse. More generally, comparing losses after changing tokenization or supervision masks may compare different objectives. This does not make loss useless; it makes held-out behavior essential. [Smaller SmolVLM](https://huggingface.co/blog/smolervlm), [TRL objective and metrics](https://huggingface.co/docs/trl/sft_trainer)

### “Orthogonal gradients explain catastrophic interference”

That is not the definition of gradient conflict in PCGrad: it uses a negative cosine. For a small plain-SGD step on task B, the first-order change to task A’s loss is approximately `−η g_A · g_B`. Zero dot product means no first-order change under those assumptions—not proof of interference, compatibility over an entire run, or sufficient capacity. [Gradient Surgery](https://arxiv.org/abs/2001.06782)

For AdamW, a more relevant local diagnostic is `g_A · Δθ_actual`, because the update includes optimizer state, coordinate-wise scaling and decay. This is our mathematical application of the optimizer equations, not a reported diagnosis of this project. Measure by parameter group and check the observed loss change as well. [AdamW algorithm](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)

### “We should always freeze—or always unfreeze—the vision tower”

There is direct counterevidence to both blanket rules. Prismatic reported degradation, especially in localization, when fine-tuning the visual backbone. Cambrian reported benefits from unfreezing under a different recipe and data setup. Compare their conditions rather than voting between conclusions. [Prismatic, §4.1/Figure 5](https://arxiv.org/html/2402.07865v2), [Cambrian-1, §3.3/Figure 5 in the linked HTML edition](https://arxiv.org/html/2406.16860)

These are VLM training studies, but neither isolates what failed in our checkpoint. The relevant experiment would control the starting model, data, schedule and evaluation while changing trainable scope.

### “Unsloth says we need 75% reasoning”

The recommendation appears in Unsloth’s Qwen3.5 fine-tuning guide. The inspected passage provides no accompanying threshold experiment and does not specify whether the fraction counts rows or tokens. Treat it as model-guide advice about reasoning retention, not a universal rehearsal ratio or evidence that structured board labels need long rationales. [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

### “We already mixed tasks, so the remaining gap proves a capacity ceiling”

An unsuccessful mixture is important evidence, but it does not by itself identify a fundamental ceiling. Loss reduction, sampling, answer lengths, optimization and adaptation placement can all change the result. Conversely, suggesting another arbitrary mixture without auditing the first does not constitute a new mechanism. [Loss normalization](https://huggingface.co/blog/gradient_accumulation), [Controlled LoRA experiments](https://thinkingmachines.ai/blog/lora/)

For our project, this is a research question: did the mixed run supply adequate, correctly weighted supervision and fail across reasonable controlled settings, or have we only observed one trajectory? Nothing in these readings warrants promising that replay alone will fix it.

## Before the next run: a checklist to answer from code and held-out data

These are our proposed audit questions, synthesized from the readings—not a new training launch plan or experimentally established recipe.

1. **Define the target.** Is the job attribute reading, text-to-location retrieval, complete reconstruction, or decision making? Define ambiguous and absent-object answers explicitly. See [09](#09-datasets-guide) and [10](#10-molmo-data).
2. **Decode one real batch.** Inspect image alignment, role tokens, answer labels, padding and end-of-turn supervision. Confirm that the model receives the evidence the task requires. See [06–08](#06-vlm-trl-walkthrough).
3. **Audit effective supervision.** Count rows, supervised answer tokens, occupied/empty items and relevant coverage intersections. Inspect loss normalization and explicit weights; do not equate a row quota with gradient share. See [18](#18-gradient-accumulation).
4. **State the visual budget.** Record preprocessing, crop policy, resolution and visual-token count. Check whether fine details survive the actual processor, not just the original screenshot. See [01–03](#01-nanovlm).
5. **Separate the trainable components.** List vision, connector/merger, language adapters, input token rows and output rows with their own optimizer groups. Record adapter targets, rank and scaling. See [12–14](#12-lora-dora).
6. **Verify updates and persistence.** Distinguish gradients from parameter changes; inspect update norms and dtypes. Reload the exact saved bundle and verify it contains every trained component. See [13](#13-peft-lora) and [17](#17-mixed-precision).
7. **Use a fixed behavior panel.** Separate free-generated answers from candidate-scoring diagnostics. Report occupied-piece accuracy, nearby-empty false positives, terrain/port retention, malformed answers and complete-board exactness independently. See [19–21](#19-evaluation).
8. **Make visual dependence testable.** Use controlled image removal/shuffling and target-versus-control perturbations, plus held-out layouts and rendering conditions. Check labeling and artifact confounds before interpreting a drop. This is our board-reader application of [19](#19-evaluation), not a prescribed control suite from that article.
9. **Measure forgetting consistently.** On a fixed higher-is-better behavior metric, report `max(score at checkpoints up to now) − current score`, plus change from the stage-start baseline. Treat a best-of-many noisy estimate cautiously; keep final testing separate from checkpoint selection. See [20–23](#20-eval-reproducibility).
10. **Choose a discriminating comparison and a stop rule.** Specify what would support an optimization, capacity or representation explanation. Match compute, data and relevant hyperparameters; select for new-skill gains subject to explicit retention floors. Keep the text-board baseline as a legitimate system alternative and compare end-to-end accuracy, latency and cost—not novelty. See [14](#14-lora-without-regret), [16](#16-beyond-lora) and [23](#23-lora-learns-less).

## How far this research goes

This is a curated curriculum, not an exhaustive bibliography or a causal verdict on our runs. Original model/framework blogs provide the practical core; the two numbered papers make learning-versus-retention and gradient-conflict claims easier to verify. Supporting primary links above resolve specific disagreements without adding another mandatory paper stack.

Several useful pages are living documentation. Unsloth’s official Markdown representations were used where the browser reader could not extract the page. The evaluation chapter is explicitly the readable archived version; the migrated interactive edition was not substantively accessible during this review.

The remaining uncertainty is substantive: none of these sources demonstrates that our board reader has reached its best achievable joint performance, that atlas-row cosine caused forgetting, or that a particular rank, initialization or rehearsal fraction will solve it. Those claims still need project-specific controlled evidence.
