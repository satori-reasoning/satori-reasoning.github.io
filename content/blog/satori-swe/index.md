---
title: "Satori-SWE: Evolutionary Test-Time Scaling for Sample-Efficient Software Engineering"
date: 2025-05-25T12:00:00+08:00
weight: 1
# aliases: ["/first"]
# tags: ["Research"]
# draft: true
# comments: false
# description: "Desc Text."
# disable_share: false
# hide_meta: false
# hide_summary: false # to hide summary in list
# hide_footer: false
math: true
# search_hidden: false # to hide from search page
show_reading_time: true
show_bread_crumbs: true
show_post_nav_links: false # the prev/next after the content
show_code_copy_buttons: true
show_word_count: true
# use_hugo_toc: true
# show_toc: true
# toc_open: true # default expand all
# cover:
#     image: "path"
#     # can also paste direct link from external site
#     # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
#     alt: "<alt text>"
#     caption: "<text>"
#     relative: true # To use relative path for cover image, used in hugo Page-bundles
#     responsive_images: true
#     hidden: false
# header:
#   background: "" # background css value
#   background_image: ""
#   gradient: false
#   blur: false
---
{{< button href="" label="Paper" external=true >}}
{{< button href="https://github.com/satori-reasoning/Satori-SWE" label="Github" external=true >}}
{{< button href="https://huggingface.co/Satori-reasoning" label="Huggingface" external=true >}}


<style>
.code-box {
    max-height: 350px;
    overflow-y: auto;
    padding: 10px;
    font-family: 'Menlo', Consolas, Monaco, 'Courier New', monospace;
    font-size: 18px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    border-radius: 5px; 
}

.issue-box {
    max-height: 350px;
    overflow-y: auto;
    padding: 10px;
    background-color: #f8f8f8; 
    font-family: 'Menlo', Consolas, Monaco, 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    border-radius: 5px; 
}

.correct_patch {
    background-color:rgb(207, 248, 206); 
}

.code-box b {
    font-weight: 600 !important; /* Force stronger boldness */
    color: #111111; /* Change text color to something more vibrant */
}

.code-box b, .code-box strong {
    font-weight: bold !important;
}

</style>




## **Introduction**

LLMs perform well on coding benchmarks like LiveCodeBench but struggle with real-world software engineering (SWE) tasks (<a href="https://openreview.net/pdf?id=VTF8yNQM66">Jimenez et al. 2024</a>). Even large models like Claude reach only around 60\% accuracy on SWE-bench, despite using carefully engineered prompting pipelines (<a href="https://arxiv.org/pdf/2407.01489">Xia et al. 2024</a>). Smaller models (under 100B parameters) perform significantly worse, typically scoring below 10\% in zero-shot settings and plateauing around 30\% after supervised fine-tuning (SFT) (<a href="https://arxiv.org/pdf/2412.21139">Pan et al. 2024</a>, <a href="https://arxiv.org/pdf/2501.05040">Xie et al. 2025</a>) on GitHub issue datasets. Improving the performance of these small models remains a key challenge for practical deployment, where repeatedly querying large models is often too costly or inefficient.

Existing approaches primarily rely on supervised fine-tuning (SFT) with high-quality data, which is expensive to curate at scale. An alternative is test-time scaling: generating multiple outputs, scoring them using a verifier, and selecting the best one. Although effective, this strategy often requires excessive sampling and costly scoring. This work aims to explore a new research direction: sample-efficient test-time scaling methods that can identify correct solutions with fewer samples. We propose **Evolutionary Test-Time Scaling (EvoScale)** that treats generation as an evolutionary process. By iteratively refining outputs via selection and mutation, EvoScale shifts the output distribution toward higher-scoring regions. Our approach results in **Satori-SWE-32B**, a 32B model trained on open-source model and data. Key features of Satori-SWE include:

- A new perspective of formulating **test-time scaling as an evolutionary process**, improving sample efficiency for software engineering tasks.
- A novel **RL training approach that enables self-evolution**, eliminating the need for external reward models or verifiers at inference time.
- Satori-SWE-32B with EvoScale achieves performance **comparable to models exceeding 100B parameters**, while requiring only a small number of samples.




## **Our Approach**
### **1. Preliminaries**
<div align="center">
  <img src="/img/swe-satori/overview.png" alt="overview" style="width:70%">
</div>

We study the problem of using LMs to resolve real-world GitHub issues, where each issue consists of a textual description and a corresponding code repository. This work follows pipeline-based methods that decompose the task into retrieval and editing. Retrieval refers to identifying the files or functions relevant to the issue, while editing involves generating the code changes needed to resolve it.

Formally, given an issue description $x$, the goal is to produce a code edit (i.e., patch) $y$ that fixes the bug or implements the requested change. A retrieval model selects a subset code context $C(x) \subseteq \mathcal{C}$ from the full codebase $\mathcal{C}$, and an editing model $\pi$ generates the patch $y = \pi(x, C(x))$ that modifies the code context $C(x)$. While retrieval has reached around 70\% accuracy in prior work, editing remains the main bottleneck. This work focuses on improving editing performance in the pipeline-based setting, using off-the-shelf localization methods in experiments. 



### **Why is test-time scaling sample-inefficient in SWE task?**
Test-time scaling improves model performance during inference without training. For SWE task, correct solutions exist but are rarely sampled. As for hard issues, the model’s output distribution is not concentrated around high-scoring regions. The figure below shows reward score distribution of outputs from a SFT model, with high-scoring outputs concentrated in the long tail. Given a sample budget $N$, typical test-time scaling methods in SWE (<a href="https://arxiv.org/pdf/2412.21139">Pan et al. 2024</a>, <a href="https://arxiv.org/pdf/2502.18449">Wei et al. 2025</a>) draw $N$ outputs (patches) { $\{y\_i\}$ }$\_{i=1}^{N}$ from a frozen editor model $\pi$, score them with a score function $R$ (e.g., reward model or unit tests), and selects the best one $\arg\max_{y_i} R(x, y_i)$. While high-scoring outputs near the mode could be sampled easily, the challenge of test-time scaling is to identify high-scoring outputs from the tail of $\pi(\cdot \mid x, C(x))$. **However, doing so typically requires a large sample size $N$, making the process sample-inefficient.**
<div align="center">
  <img src="/img/swe-satori/intro_kde.png" alt="intro_kde" style="width:40%">
</div>



### **2. Formulation: Patch Generation as Evolution**
<div align="center">
  <img src="/img/swe-satori/inference.png" alt="inference">
</div>
Our goal is to enable sample-efficient test-time scaling, achieving stronger performance with fewer samples. We propose <b>Evolutionary Test-Time Scaling (EvoScale), which iteratively refines generation by using earlier outputs to guide subsequent sampling.</b> We recast patch generation for a GitHub issue as an evolutionary process and use a LLM as a <i>mutation operator</i> (<a href="https://arxiv.org/pdf/2305.00593">Shen et al. 2023</a>), leveraging its ability to produce syntactically and semantically valid patches. The objective is to explore the patch space with a small number of samples, identify high-scoring patches, and iteratively refine the generated patches. At each iteration $t$, the LM generates a batch of patches $\mathcal{Y}^{t+1} = \{y^{t+1}_1, \dots, y^{t+1}_M\}$ conditioned on a set of prior patches $\mathcal{E}^t$: $\mathcal{Y}^{t+1} \sim \pi(\cdot \mid x, C(x), \mathcal{E}^t)$. We refer to $\mathcal{E}^t$ as <i>conditioning examples</i> consisting of patches generated at iteration $t$. Following the selection step in evolutionary algorithms, $\mathcal{E}^t$ could be selected as the top-$K$ patches ranked by a scoring function $R$ (i.e., fitness function in evolutionary algorithms). Note that we find that our model after training can self-evolve without this selector.

### **Can a language model naturally perform mutation?**
Ideally, the mutation operator should generate patches that improve scores. However, we find that models trained with classical SFT—conditioned only on the issue and code context—struggle to refine existing patches. To this end, we propose a **two-stage SFT** to overcome this limitation.

<div align="center">
  <img src="/img/swe-satori/algorithm.png" alt="algorithm">
</div>

### **3. Small-scale Mutation SFT**
We introduce a two-stage supervised fine-tuning (SFT) process: classical SFT followed by mutation SFT. The classical SFT model is first trained and then used to generate conditioning examples for training the mutation SFT model.

1. **Classical SFT:** 
We fine-tune a base model on inputs consisting of the issue description $x$ and code context $C(x)$, with targets that include a chain-of-thought (CoT) trace and the ground-truth patch, jointly denoted as $y^*_{\text{SFT}}$. The training objective is:
<div class="code-box">
$$
\begin{align}
    \max_{\pi_{\text{SFT}}} \; \mathbb{E}_{x \sim \mathcal{D},\, y^*_{\text{SFT}} \sim \mu(\cdot \mid x, C(x))} \left[ \log \pi_{\text{SFT}}(y^*_{\text{SFT}} \mid x, C(x)) \right]. \nonumber
\end{align}
$$
</div>
We refer to the resulting model $\pi_{\text{SFT}}$ as the classical SFT model.

2.  **Mutation SFT:** 
We fine-tune a second model, initialized from the same base model, using inputs $x$, $C(x)$, and a set of conditioning examples $\mathcal{E}$ consisting of patches sampled from the classical SFT model $\pi\_{\text{SFT}}$. The target $y^*_{\text{M-SFT}}$ includes a CoT trace generated by the teacher model $\mu$ conditioned on $\mathcal{E}$, along with the ground-truth patch. The training objective is:
<div class="code-box">
$$
\begin{align}
    \max_{\pi_{\text{M-SFT}}} \; \mathbb{E}_{\substack{x \sim \mathcal{D},\, \mathcal{E} \sim \pi_{\text{SFT}}(\cdot \mid x, C(x)), y^*_{\text{M-SFT}} \sim \mu(\cdot \mid x, C(x), \mathcal{E})}} \left[ \log \pi_{\text{M-SFT}}(y^*_{\text{M-SFT}} \mid x, C(x), \mathcal{E}) \right]. \nonumber
\end{align}
$$
</div>
We refer to the resulting model $\pi_{\text{M-SFT}}$ as the mutation SFT model.

### **Can SFT Model after the two-stage training learns to self-evolve?**
Self-evolution requires the model to improve low-scoring patches on its own, without relying on reward models to select high-scoring examples. If so, we could eliminate the selection step (Line 3 in Algorithm), reducing scoring costs and sample usage. However, we find that **SFT alone cannot enable self-evolution**. We then introduce a reinforcement learning approach that trains the model to self-evolve without scoring or filtering.

### **4. Large‑Scale RL for Self‑Evolution**
To *self-evolve*, the model must generate patches that maximize a scoring function $R$, given conditioning examples $\mathcal{E}$ from previous patches. This setup naturally aligns with reinforcement learning (RL), where a policy $\pi$ is optimized to maximize expected rewards (i.e., scores) over time. Since our goal is to maximize the reward at the final iteration $T$, a naïve RL objective is:
<div class="code-box">
$$
\begin{align}
\max_{\pi} \mathbb{E}_{y^t \sim \pi(\cdot | x, C(x), \mathcal{E}^{t-1})}\Bigl[\sum_{t=0}^T r_t \Bigr], \quad \text{where} \quad r_t =
\begin{cases}
R(x, y^t), & t = T \\
0, & \text{otherwise}
\end{cases}\nonumber
\end{align} 
$$
</div>
This objective focuses solely on maximizing the final reward. However, it presents two key challenges:

- **Rewards are sparse**, with feedback only at iteration $T$, making learning inefficient (<a href="https://arxiv.org/pdf/2502.02508">Shen et al. 2025</a>, <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/f96af360d2a1b1585c3e3a5b82ba4ef7-Paper-Conference.pdf">Lee et al. 2024</a>).
- Generating full $T$-step trajectories is **computationally expensive** (<a href="https://openreview.net/pdf?id=4FWAwZtd2n">Snell et al. 2025</a>).


We address the sparse reward challenge using ***potential-based reward shaping*** (<a href="https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf">Ng et al. 1999</a>), where the potential function is defined as $\Phi(y) = R(x, y)$. The potential reward at step-$t$ is:
<div class="code-box">
$$
\begin{align}
r_t = \Phi(y^t) - \Phi(y^{t-1}) = R(x, y^t) - R(x, y^{t-1}). \nonumber
\end{align}
$$
</div>
Unlike the naïve formulation, this provides non-zero potential rewards at every step, mitigating the sparse reward challenge. The cumulative potential reward forms a telescoping sum: $\sum_{t=1}^T r_t = R(x, y^T) - R(x, y^0)$. Since $y^0$ is fixed, maximizing this sum is equivalent to maximizing the final score.

In practice, we fine-tune the mutation SFT model $\pi_{\text{M-SFT}}$ to maximize the expected potential rewards in score between a newly generated patch $y$ and a previous patch $y^\prime$ drawn from the conditioning examples $\mathcal{E}$:
<div class="code-box">
$$
\begin{align}
\max_{\pi_{\text{RL}}}  \mathbb{E}_{\substack{y \sim \pi_{\text{RL}}(\cdot \mid x, C(x), \mathcal{E}), y' \sim \mathcal{E}}} \bigl[R(x, y) - R(x, y') - \lambda F(y)\bigr]. \nonumber
\end{align} 
$$
</div>
This objective encourages the model to generate patches that consistently improve upon previous ones. To ensure the outputs follow the required syntax, we incorporate a formatting penalty term $F$ (<i>string matching</i> and <i>syntax checking</i>) into the reward function.

## **Benchmarking Performance on SWE‑Bench‑Verified**
We use the *Qwen2.5-Coder-32B-Instruct model* as our base model for training Satori-SWE-32B. Through the two-stage SFT training and RL training, **Satori-SWE-32B outperforms all small-scale models** under greedy decoding, while achieving comparable performance with current SOTA SWE-RL with smaller model scale (32B v.s. 70B),  much fewer training data (30K v.s. million-scale) and test-time scaling samples (50 v.s. 500).


<!-- | Model                     | Params   | Samples | Acc. (%) |
| ------------------------- | -------- | ------- | -------- |
| GPT‑4o (Agentless)        | –        | 1       | 38.8     |
| Claude 3.5 (Agentless)    | –        | 1       | 50.8     |
| DeepSeek-V3 (Agentless)    | –        | -       | 42.0     |
| SWE-Fixer                 | 72 B     | 1       | 30.2     |
| SWE-Gym-32B            | 32 B     | 1      | 20.6     |
| SWE-Gym-32B            | 32 B     | 16      | 32.0     |
| Llama‑3 SWE‑RL            | 70 B     | 80      | 37.0     |
| Llama‑3 SWE‑RL            | 70 B     | 500     | 41.0     |
| **Satori‑SWE‑32B** | **32 B** | **1**   | **35.8** |
| **Satori‑SWE‑32B**        | **32 B**     | **10**      |  **37.0**     |
| **Satori‑SWE‑32B**       | **32 B**     | **50**  | **41.6** | -->

<table>
  <thead>
    <tr>
      <th style="width: 30%; text-align: center;">Model</th>
      <th style="width: 15%; text-align: center;">Params</th>
      <th style="width: 15%; text-align: center;">Best@N</th>
      <th style="width: 20%; text-align: center;">Accuracy (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align: center;">GPT‑4o (Agentless)</td><td style="text-align: center;">–</td><td style="text-align: center;">1</td><td style="text-align: center;">38.8</td></tr>
    <tr><td style="text-align: center;">Claude 3.5 (Agentless)</td><td style="text-align: center;">–</td><td style="text-align: center;">1</td><td style="text-align: center;">50.8</td></tr>
    <tr><td style="text-align: center;">DeepSeek‑V3 (Agentless)</td><td style="text-align: center;">–</td><td style="text-align: center;">–</td><td style="text-align: center;">42.0</td></tr>
    <tr><td style="text-align: center;">SWE‑Fixer</td><td style="text-align: center;">72 B</td><td style="text-align: center;">1</td><td style="text-align: center;">30.2</td></tr>
    <tr><td style="text-align: center;">SWE‑Gym‑32B</td><td style="text-align: center;">32 B</td><td style="text-align: center;">1</td><td style="text-align: center;">20.6</td></tr>
    <tr><td style="text-align: center;">SWE‑Gym‑32B</td><td style="text-align: center;">32 B</td><td style="text-align: center;">16</td><td style="text-align: center;">32.0</td></tr>
    <tr><td style="text-align: center;">Llama‑3 SWE‑RL</td><td style="text-align: center;">70 B</td><td style="text-align: center;">80</td><td style="text-align: center;">37.0</td></tr>
    <tr><td style="text-align: center;">Llama‑3 SWE‑RL</td><td style="text-align: center;">70 B</td><td style="text-align: center;">500</td><td style="text-align: center;">41.0</td></tr>
    <tr><td style="text-align: center;"><b>Satori‑SWE‑32B</b></td><td style="text-align: center;"><b>32 B</b></td><td style="text-align: center;"><b>1</b></td><td style="text-align: center;"><b>35.8</b></td></tr>
    <tr><td style="text-align: center;"><b>Satori‑SWE‑32B</b></td><td style="text-align: center;"><b>32 B</b></td><td style="text-align: center;"><b>10</b></td><td style="text-align: center;"><b>38.9</b></td></tr>
    <tr><td style="text-align: center;"><b>Satori‑SWE‑32B</b></td><td style="text-align: center;"><b>32 B</b></td><td style="text-align: center;"><b>25</b></td><td style="text-align: center;"><b>40.2</b></td></tr>
    <tr><td style="text-align: center;"><b>Satori‑SWE‑32B</b></td><td style="text-align: center;"><b>32 B</b></td><td style="text-align: center;"><b>50</b></td><td style="text-align: center;"><b>41.6</b></td></tr>
  </tbody>
</table>



## **Anaysis of SWE-Satori**
We further present a comprehensive analysis of the proposed EvoScale approach. To simplify our analysis, we use ground-truth localization (retrieval) and focus on the code editing part.
### **Can LLMs Iteratively Evolve without Mutation SFT Training?**
First, we investigate whether the mutation SFT is necessary for LLMs to learn how to iteratively improve their generations. Specifically, we fine-tune base LLMs using either classical SFT (without conditional generation) or mutation SFT. As shown in Figure, models trained with classical SFT fail to naturally improve their outputs when conditioned on previous samples. In contrast, mutation SFT enables the model to iteratively improve under the guidance of a reward model. The performance of the mutation SFT model at later iterations can surpass the classical SFT model by scaling up the samples (e.g., Best@40). <b>Moreover, this iterative refinement capability can be learned effectively even with a small number of training data.</b>
<div align="center">
  <img src="/img/swe-satori/sft.png" alt="sft" style="width:60%">
</div>

### **RL Enables Self-evolve Capability.**
While mutation SFT model demonstrates evolutionary behavior when guided by a reward model, we further examine whether it can self-evolve without such guidance. Specifically, instead of selecting the top-$K$ candidates to ensure generation quality, we allow the model to generate $M=K=5$ random samples for the next iteration of conditional generation. However, the SFT model fails to learn self-evolution without reward model selection. Interestingly, RL training significantly improves the SFT model in two key aspects. First, **RL substantially boosts the model's greedy performance**, surpassing even the Best@$N$ performance of 30 randomly generated samples from the SFT model. Second, we observe that the **RL-trained model exhibits strong self-evolution capability**: even when conditioned on its random outputs, the model can self-refine and improve performance across iterations without reward model guidance.
<div align="center">
  <img src="/img/swe-satori/ablation_rl_evolution.png" alt="ablation_rl_evolution" style="width:60%">
</div>

### **Do our SFT and RL Models Monotonically Improve Reward Scores over Iterations?**
We further analyze the evolutionary behavior of the SFT and RL models by measuring the average reward score of the patch samples generated at each iteration. As shown in the Figure, although the SFT model learns to iteratively improve reward scores, it relies on the reward model to select high-quality conditioning examples to achieve significant improvements. In contrast, **the RL model trained with potential-based reward, naturally learns to self-evolve without any external guidance.** Its reward scores improve monotonically across iterations.
<div align="center">
  <img src="/img/swe-satori/rm_score.png" alt="rm_score" style="width:60%">
</div>


### **Evolutionary Test-time Scaling v.s. Other Test-time Scaling Methods.**
Next, we further compare evolutionary test-time scaling with other test-time scaling methods. Starting from the RL model, we first randomly sample $N={5, 10, 15, 20, 25, 50}$ patch samples and let the reward model and unit tests select the best sample among the subsets. Also starting from the RL model, we let the model perform self-evolution with $K=5$ samples per iteration, up to four iterations ($20$ samples in total). The test-time scaling results presented in the Figure demonstrate **both efficiency and effectiveness of evolutionary test-time scaling.**
<div align="center">
  <img src="/img/swe-satori/scaling_methods.png" alt="scaling_methods" style="width:60%">
</div>


## **Demo Examples**
The core idea of EvoScale is to use a mix of correct and incorrect patches as context to generate a
correct patch. In this part, we present an example below to illustrate the capability.

### Issue Description

<div class="issue-box" markdown="1">
Contains.as_set returns Contains:

```python
>>> Contains(x, Reals).as_set()
Contains(x, Reals)
```

This is **wrong** because `Contains` is not a set (it's a boolean).  
It causes failures elsewhere because `Contains` lacks `as_relational`
(since it isn’t a set). For example  
(from <https://github.com/sympy/sympy/pull/14965#discussion_r205281989>):

```pytb
>>> Piecewise((6, Contains(x, Reals)), (7, True))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "./sympy/functions/elementary/piecewise.py", line 136, in __new__
    r = cls.eval(*newargs)
  File "./sympy/functions/elementary/piecewise.py", line 185, in eval
    c = c.as_set().as_relational(x)
AttributeError: 'Contains' object has no attribute 'as_relational'
```

</div>

### Conditional Patches
#### **Conditional patch 1 ✅**
```diff
diff --git a/sympy/sets/contains.py b/sympy/sets/contains.py
index 0b1e933..3fd780b 100644
--- a/sympy/sets/contains.py
+++ b/sympy/sets/contains.py
@@ -45,4 +45,6 @@ class Contains(BooleanFunction):
     def as_set(self):
-        raise NotImplementedError()
+        # The set being contained in is the only set we can return
+        # from a Contains object in any meaningful way
+        return self.args[1]
```

#### **Conditional patch 2 ✅**
```diff
diff --git a/sympy/sets/contains.py b/sympy/sets/contains.py
index 0b1e933..4c089aa 100644
--- a/sympy/sets/contains.py
+++ b/sympy/sets/contains.py
@@ -45,4 +45,5 @@ class Contains(BooleanFunction):
     def as_set(self):
-        raise NotImplementedError()
+        # The set that is being tested for contains
+        return self.args[1]
```

#### **Conditional patch 3 ✅**
```diff
diff --git a/sympy/sets/contains.py b/sympy/sets/contains.py
index 0b1e933..985c88e 100644
--- a/sympy/sets/contains.py
+++ b/sympy/sets/contains.py
@@ -45,4 +45,6 @@ class Contains(BooleanFunction):
     def as_set(self):
-        raise NotImplementedError()
+        # The set being contained in is the set representation
+        # of Contains objects
+        return self.args[1]
```

#### **Conditional patch 4 ✅**
```diff
diff --git a/sympy/sets/contains.py b/sympy/sets/contains.py
index 0b1e933..e732791 100644
--- a/sympy/sets/contains.py
+++ b/sympy/sets/contains.py
@@ -45,4 +45,6 @@ class Contains(BooleanFunction):
     def as_set(self):
-        raise NotImplementedError()
+        # Contains(x, S) means "x is in S", so as_set returns S
+        return self.args[1]
```

#### **Conditional patch 5 ❌**
```diff
diff --git a/sympy/sets/contains.py b/sympy/sets/contains.py
index 0b1e933..4980364 100644
--- a/sympy/sets/contains.py
+++ b/sympy/sets/contains.py
@@ -45,4 +45,22 @@ class Contains(BooleanFunction):
     def as_set(self):
+        # try to evaluate Contains(x, S)
+        try:
+            res = self._eval_evalf()
+            if res == S.true:
+                return S.UniversalSet
+            elif res == S.false:
+                return S.EmptySet
+        except (TypeError, AttributeError):
+            pass
+        # fallback
     raise NotImplementedError()
```

### Evolve Response
#### **Thinking Process**
<div class="issue-box">
1. Peer Patch Review:

- Patch 1: This patch correctly identifies that the set being contained in (self.args[1]) is the most meaningful set representation of a Contains object. The implementation is simple and correct.

- Patch 2: Similar to Patch 1 but with slightly different wording in the comment. The functionality is correct but the comment could be more precise about why this is the right approach.

- Patch 3: This patch adds a note about potential invalid sets, which isn't strictly necessary since the issue is about the basic behavior. The core implementation is correct but the comment is overly cautious.

- Patch 4: This patch provides a clear explanation of why returning the set (self.args[1]) is the correct behavior - because Contains(x, S) means "x is in S", so the set representation is just S. This is a correct and well-explained solution.

- Patch 5: This patch takes a different approach by trying to evaluate the Contains expression and return either the universal or empty set based on the result. While creative, this doesn't match the simple and correct solution of just returning the set being contained in. The approach is overly complex for this simple issue.

2. Issue Analysis:
   - The issue occurs when Contains.as_set() returns the Contains object itself instead of a proper set.
   - This causes problems because other code expects a set with as_relational() method.
   - The correct behavior should be to return the set that x is being tested for containment in (self.args[1]), since that's the only meaningful set representation of a Contains object.

3. Solution Approach:
   - The simplest and most correct solution is to modify as_set() to return self.args[1], which is the set being tested for containment.
   - This matches the mathematical interpretation that Contains(x, S) means "x is in S", so the set representation is just S.
   - No other evaluation or special cases are needed since the set being contained in is the only meaningful set representation of a Contains object.

4. Code Localization:
   - The issue is in sympy/sets/contains.py in the Contains class.
   - The problematic method is as_set() which currently raises NotImplementedError.
   - The fix is to modify this method to return self.args[1].
</div>

#### **Evolved Patch** ✅
```diff
diff --git a/sympy/sets/contains.py b/sympy/sets/contains.py
index 0b1e933..e732791 100644
--- a/sympy/sets/contains.py
+++ b/sympy/sets/contains.py
@@ -45,4 +45,6 @@ class Contains(BooleanFunction):
     def as_set(self):
-        raise NotImplementedError()
+        # Contains(x, S) means "x is in S", so as_set returns S
+        return self.args[1]
```

------

## **Commitment to Open-Source Community**

We plan to provide the following resources:

- 📜 **Paper** with detailed implementation details in Appendix.
- 🧩 [SFT data with CoT, Mutation SFT data with CoT](https://huggingface.co/datasets/Satori-reasoning/Satori-SWE-two-stage-SFT-data), and [RL data](https://huggingface.co/datasets/Satori-reasoning/SWE-Satori-RL-data).
- ⚙️ Code for EvoScale Inference Piepline & RL training framework.
- 🏋️‍♂️ Model Checkpoints:
  - Code Editing Model (SFT): `Satori‑SWE‑SFT-32B`
  - Code Editing Model (RL): `Satori‑SWE‑RL-32B`
  - Code Editing Reward Model: `Satori‑SWE‑RM-32B`
  - Retrieval Model: `Satori‑SWE‑Retrieval-32B`
  - Retrieval Reward Model: `Satori‑SWE‑Retrieval-RM-32B`

Stay tuned on our [GitHub](https://github.com/satori-reasoning/Satori-SWE).

------

## **Satori Team Members**
<span>$&#8224;$</span>: Project lead
### **Core Contributors**
- [Guangtao Zeng, SUTD](https://chaoscodes.github.io/)
- [Maohao Shen, MIT](https://maohaos2.github.io/Maohao/)
- [Zhang-Wei Hong<span>$^&#8224;$</span>, MIT](https://williamd4112.github.io/)
### **Contributors**
- Delin Chen, UMass Amherst
- Zhenting Qi, Harvard
- Wei Lu, SUTD
- Gregory W. Wornell, MIT
- Subhro Das, MIT-IBM Watson AI Lab
- David Cox, MIT-IBM Watson AI Lab
- Chuang Gan<span>$^&#8224;$</span>, UMass, MIT-IBM Watson AI Lab

------
## **Contact Information**

For questions, please:
- Raise an issue in our GitHub repository
- Contact us at: satori2025@outlook.com

------

<!-- ## **Citation**

```bibtex
@misc{zeng2025satoriswe,
  title   = {Satori-SWE: Evolutionary Test-Time Scaling for Sample-Efficient Software Engineering with Reinforcement Learning},
  author  = {Guangtao Zeng and Maohao Shen and Delin Chen and Zhenting Qi and Zhenfang Chen and Subhro Das and Dan Gutfreund and David Cox and Gregory Wornell and Wei Lu and Zhang-Wei Hong and Chuang Gan},
  year    = {2025},
  eprint  = {},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url     = {}
}
``` -->
