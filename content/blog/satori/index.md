---
title: "Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search"
date: 2025-02-01T12:00:00+08:00
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
{{< button href="https://github.com/satori-reasoning/Satori" label="Github" external=true >}}
{{< button href="https://huggingface.co/Satori-reasoning" label="Huggingface" external=true >}}


<style>
.code-box {
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

.code-box b {
    font-weight: 600 !important; /* Force stronger boldness */
    color: #111111; /* Change text color to something more vibrant */
}

.code-box b, .code-box strong {
    font-weight: bold !important;
}

</style>


## **Introduction**
Since the release of OpenAI&#39;s o1, significant efforts have been made within the research community to enhance open-source LLMs with advanced reasoning capabilities. This includes various approaches, including distillation using a strong teacher model, MCTS, and reward model guided search. This work aims to explore a new research direction: enabling LLMs with autoregressive search capabilities, i.e., a single LLM performs an extended reasoning process with self-reflection and self-exploration of new strategies. To achieve this, we develop a LLM post-training paradigm with several key concepts and ideas inspired by classical reinforcement learning (RL) communities. Our approach results in Satori, a 7B LLM trained on open-source model (Qwen-2.5-Math-7B) and open-source data (OpenMathInstruct-2 and NuminaMath). Key features of Satori include:
  - Capable of self-reflection and self-exploration without external guidance.
  - Achieve state-of-the-art reasoning performance mainly through self-improvement (RL).
  - Exhibit transferability of reasoning capabilities on unseen domains beyond math.

## **Our Approach**
We formulate LLM reasoning as a sequential decision-making problem, where reasoning is a process of constructing and refining an answer step by step. Specifically, the LLM (agent&#39;s policy) starts with an input context (initial state), generates a reasoning step (action), and updates the context (next state). The LLM repeats this process until it reaches a final answer, and receives a reward that evaluates whether the final answer matches the ground truth. With this formulation, we could train the LLM to reason using RL, aiming to generate a sequence of reasoning steps that maximize the expected reward.

### **Chain-of-Action-Thought reasoning (COAT)**
The key challenge of achieving autoregressive search is enabling the LLM to determine when to reflect, continue, or explore alternative solutions without external intervention. 
To enable this, we introduce several special meta-action tokens that guide the LLM&#39;s reasoning process,
  - **Continue Reasoning** (<|continue|>): encourages the LLM to build upon its current reasoning trajectory by generating the next intermediate step. 
  - **Reflect** (<|reflect|>): prompts the model to pause and verify the correctness of prior reasoning steps.
  - **Explore Alternative Solution** (<|explore|>): signals the model to identify critical flaws in its reasoning and explore a new solution.

We refer to this formulation as Chain-of-Action-Thought (COAT) reasoning. Each COAT reasoning step is a sequence of tokens, starting with one of the meta-action tokens.

### **Overview of Training Framework**
A standard LLM is not able to perform COAT reasoning, and achieving this will encounter two key challenges:
  - **Unawareness of meta-action tokens**: Without training, the LLM doesn&#39;t recognize that encountering special meta-action tokens may require reflection or proposing alternative solutions.
  - **Long horizon and sparse rewards**: Reasoning requires long-term decision-making with rewards only at the end, the LLM must take many correct reasoning steps before receiving rewards, and failures force it to restart from the initial state. This makes learning difficult because rewards are scarce, yet rewards are essential for driving RL progress.

To address these two challenges, we propose a two-stage training framework\*: 
  1) A small-scale format tuning (FT) stage that helps the base LLM to internalize the COAT reasoning format.
  2) A large-scale self-improvement stage that utilizes reinforcement learning with "Restart and Explore" (RAE) techniques.
![schematic](/img/schematic.png)

<p style="font-size: 14px;"> <b>*Note:</b> <i>our concurrent work <a href="https://arxiv.org/abs/2501.12948">DeepSeek-R1</a> adopts a similar training framework of small-scale cold-start SFT followed by large-scale RL training. Although both works coincide in this high-level idea, our work differs from R1 in key methodologies, including the data synthesis framework and RL algorithms.</i></p>

### **Format Tuning Through Imitation Learning** 
This stage aims to fine-tune a pre-trained base LLM to imitate a few demonstrated reasoning trajectories with COAT reasoning format. To synthesize such COAT trajectories that incorporate trials
and errors, we propose a multi-agent data synthesis framework that leverages three LLMs:
  - **Generator**: Given an input problem, a generator generates multiple reasoning paths for a given input problem using classical CoT techniques.
  - **Critic**: A critic evaluates the correctness of the reasoning paths generated by the generator, providing feedback to refine the reasoning and address suboptimal steps.
  - **Reward Model**: A reward model assigns scores to the refined reasoning paths and selects the most effective path as the final demonstration trajectory.

These three models collaborate to construct high-quality demonstration trajectories. We observe that a small number (10K) of demonstration trajectories is sufficient for base LLM to follow the COAT reasoning format.

### **Self-improvement via Reinforcement Learning**
Through format tuning, the LLM has adopted the COAT reasoning style but struggles to generalize to unseen problems. The RL stage aims to incentivize the actual capabilities of leveraging self-reflection to improve reasoning. We start with the format-tuned LLM and further optimize it using the classical PPO algorithm with two additional key strategies,
 - **Restart and Explore (RAE)**: Inspired by [Go-Explore](https://arxiv.org/abs/1901.10995), we train the LLM policy to reason not only from the problem statement but also from intermediate steps sampled from past trajectories, both correct and incorrect. We also add exploration bonuses to encourage deeper reflection, further increasing opportunities for the policy to arrive at correct answers.
 - **Iterative Self-improvement**: The policy might converge to a local sub-optimum and cannot further improve. Inspired by [Kickstarting](https://arxiv.org/abs/1803.03835), after each round of RL training, we distill the knowledge of the current teacher policy into the student model (base LLM) through supervised fine-tuning. Starting from the newly fine-tuned LLM, we then perform another round of RL training.

## **Benchmarking Performance**
Satori is evaluated on both in-domain reasoning benchmarks (math reasoning) and out-of-domain benchmarks (general reasoning tasks). All results are reported as the zero-shot pass@1 accuracy with greedy sampling.

### **Evaluation Tasks**
- **Mathmatics Reasoning Benchmarks**: [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH500](https://arxiv.org/abs/2305.20050), AMC2023, AIME2024, and [OlympiadBench](https://github.com/OpenBMB/OlympiadBench). Except for GSM8K, all other datasets feature competition-level problems.
- **General Domain Reasoning Benchmarks**: 
  - Logical Reasoning: [FOLIO](https://arxiv.org/abs/2209.00840), [BoardgameQA (BGQA)](https://arxiv.org/abs/2306.07934).
  - Code Reasoning: [CRUXEval](https://arxiv.org/abs/2401.03065).
  - Commonsense Reasoning: [StrategyQA (STGQA)](https://arxiv.org/abs/2101.02235).
  - Tabular Reasoning: [TableBench](https://arxiv.org/abs/2408.09174).
  - Domain-specific Reasoning: [MMLUPro](https://arxiv.org/abs/2406.01574) STEM subsets (STEM), including physics, chemistry, computer science, engineering, biology, and economics.

### **Math Reasoning Benchmarks**
Satori-Qwen-7B achieves SOTA performance and outperforms Qwen-2.5-Math-7B-Instruct which uses the same base model (Qwen-2.5-Math-7B). After round 2 training, Satori-Qwen-7B (Round 2)
demonstrates even stronger performance on hard tasks.

| Scale | Model | GSM8K | MATH500 | OlymBench | AMC2023 | AIME2024 | AVG. |
|-------|------------|-------------|------------|------------|------------|------------|------------|
| **Large** | Llama-3.1-70B-Instruct    | 94.1 | 68.0 | 29.4 | 42.5 | 13.3 | 49.5 |
|           | OpenMath2-Llama3.1-70B    | 94.1 | 71.8 | 30.1 | 45.0 | 13.3 | 50.9 |
|           | QwQ-32B-Preview           | 95.5 | 90.6 | 61.2 | 77.5 | 50.0 | 75.0 |
| **Small** | Llama-3.1-8b-Instruct     | 84.4 | 51.9 | 15.1 | 22.5 | 3.3  | 35.4 |
|           | OpenMath2-Llama3.1-8B     | 90.5 | 67.8 | 28.9 | 37.5 | 6.7  | 46.3 |
|           | NuminaMath-7B-CoT         | 78.9 | 54.6 | 15.9 | 20.0 | 10.0 | 35.9 |
|           | Qwen-2.5-7B-Instruct      | 91.6 | 75.5 | 35.5 | 52.5 | 6.7  | 52.4 |
|           | Qwen-2.5-Math-7B-Instruct |95.2  | 83.6 | 41.6 | 62.5 | 16.7 | 59.9 |
|           | **Satori-Qwen-7B**            | 93.2 | 85.6 | 46.6 | 67.5 | 20.0 | **62.6** |
|           | **Satori-Qwen-7B (Round 2)**  | 93.9 | 83.6 | 48.5 | 72.5 | 23.3 | **64.4** |

### **General Domain Reasoning Benchmarks**
Trained **only on math** datasets, Satori-Qwen-7B exhibits strong transferability across diverse out-of-domain reasoning benchmarks and outperforms Qwen-2.5-Math-7B-Instruct by a large margin. 
Moreover, despite not being trained in other domains, Satori-Qwen-7B achieves performance comparable to or exceeding other small-scale general instruct models.

| Scale  | Model | FOLIO | BGQA | CRUXEval | StrategyQA | TableBench | STEM | Avg. |
|-------|------------|-------------|------------|------------|------------|------------|------------|------------|
| **Large** | Llama-3.1-70B-Instruct    | 65.0 | 58.3 | 59.6 | 88.8 | 34.2 | 61.7 | 61.3 |
|           | OpenMath2-Llama3.1-70B    | 68.5 | 68.7 | 35.1 | 95.6 | 46.8 | 15.1 | 55.0 |
|           | QwQ-32B-Preview           | 84.2 | 71.1 | 65.2 | 88.2 | 51.5 | 71.3 | 71.9 |
| **Small** | Llama-3.1-8b-Instruct     | 63.5 | 50.3 | 38.5 | 92.2 | 32.4 | 43.4 | 53.4 |
|           | OpenMath2-Llama3.1-8B     | 57.1 | 49.0 | 11.1 | 84.4 | 34.2 | 10.9 | 41.1 |
|           | NuminaMath-7B-CoT         | 53.2 | 44.6 | 28.0 | 77.8 | 29.1 | 11.3 | 40.7 |
|           | Qwen-2.5-7B-Instruct      | 72.4 | 53.0 | 58.1 | 91.3 | 43.2 | 57.1 | **62.5** |
|           | Qwen-2.5-Math-7B-Instruct | 68.9 | 51.3 | 28.0 | 85.3 | 36.2 | 45.2 | 52.5 |
|           | **Satori-Qwen-7B**            | 71.4 | 61.8 | 42.5 | 86.3 | 43.4 | 56.7 | 60.4 |
|           | **Satori-Qwen-7B (Round 2)**  | 72.9 | 58.5 | 41.1 | 90.4 | 44.6 | 57.4 | **60.8** |




## **Demo Examples**

### **Mathematical Reasoning (AIME2024)**

#### **Problem Statement:**
<div class="code-box">
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
</div>

#### **Satori&#39;s Response:**
<div class="code-box">
First, we determine the time Aya spends walking when she walks at a speed of $ s $ kilometers per hour. The total time for the walk including the time spent in the coffee shop is 4 hours. Therefore, the time spent walking is:

$$ 4 - \frac{t}{60} \text{ hours} $$

Since the distance of the walk is 9 kilometers, we can write the equation:

$$ s \left( 4 - \frac{t}{60} \right) = 9 $$

Next, we determine the time Aya spends walking when she walks at a speed of $ s + 2 $ kilometers per hour. The total time for the walk including the time spent in the coffee shop is 2 hours and 24 minutes, which is equivalent to 2.4 hours. Therefore, the time spent walking is:

$$ 2.4 - \frac{t}{60} \text{ hours} $$

Since the distance of the walk is 9 kilometers, we can write the equation:

$$ (s + 2) \left( 2.4 - \frac{t}{60} \right) = 9 $$
We now have a system of two equations:

1. $$ s \left( 4 - \frac{t}{60} \right) = 9 $$
2. $$ (s + 2) \left( 2.4 - \frac{t}{60} \right) = 9 $$

Let&#39;s solve these equations step by step. First, expand both equations:

1. $$ 4s - \frac{st}{60} = 9 $$

2. $$ 2.4s + 4.8 - \frac{st}{60} - \frac{2t}{60} = 9 $$
Subtract the first equation from the second equation to eliminate $ \frac{st}{60} $:

$$ (2.4s + 4.8 - \frac{st}{60} - \frac{2t}{60}) - (4s - \frac{st}{60}) = 9 - 9 $$
$$ 2.4s + 4.8 - 4s - \frac{2t}{60} = 0 $$
$$ -1.6s + 4.8 - \frac{2t}{60} = 0 $$
$$ -1.6s + 4.8 = \frac{2t}{60} $$
$$ -96s + 288 = 2t $$$$ t = 144 - 48s $$
Substitute $ t = 144 - 48s $ back into the first equation:

$$ s \left( 4 - \frac{144 - 48s}{60} \right) = 9 $$
$$ s \left( 4 - \frac{144}{60} + \frac{48s}{60} \right) = 9 $$
$$ s \left( 4 - 2.4 + \frac{4s}{5} \right) = 9 $$
$$ s \left( 1.6 + \frac{4s}{5} \right) = 9 $$
$$ 1.6s + \frac{4s^2}{5} = 9 $$
$$ 8s + \frac{4s^2}{5} = 45 $$
$$ 40s + 4s^2 = 225 $$
$$ 4s^2 + 40s - 225 = 0 $$
Solve the quadratic equation using the quadratic formula $ s = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $, where $ a = 4 $, $ b = 40 $, and $ c = -225 $:

$$
s = \frac{-40 \pm \sqrt{40^2 - 4 \cdot 4 \cdot (-225)}}{2 \cdot 4} $$
$$ s = \frac{-40 \pm \sqrt{1600 + 3600}}{8} $$
$$ s = \frac{-40 \pm \sqrt{5200}}{8} $$
$$ s = \frac{-40 \pm 20\sqrt{13}}{8} $$
$$ s = \frac{-10 \pm 5\sqrt{13}}{2} $$

Since $ s $ must be positive, we take the positive root:$ s = \frac{-10 + 5\sqrt{13}}{2} $
Now, we need to find the time Aya spends walking when she walks at a speed of $ s + \frac{1}{2} $ kilometers per hour. The time spent walking is:

$$ \frac{9}{s + \frac{1}{2}} $$

Substitute $ s = \frac{-10 + 5\sqrt{13}}{2} $:

$$ s + \frac{1}{2} = \frac{-10 + 5\sqrt{13}}{2} + \frac{1}{2} = \frac{-9 + 5\sqrt{13}}{2} $$

The time spent walking is:

$$ \frac{9}{\frac{-9 + 5\sqrt{13}}{2}} = \frac{18}{-9 + 5\sqrt{13}} $$Rationalize the denominator:
$$
\begin{align*}
&\frac{18}{-9 + 5\sqrt{13}} \cdot \frac{-9 - 5\sqrt{13}}{-9 - 5\sqrt{13}} \\
&= \frac{18(-9 - 5\sqrt{13})}{81 - 325}  \\
&= \frac{18(-9 - 5\sqrt{13})}{-244}  \\
&= \frac{9(9 + 5\sqrt{13})}{122} = \frac{81 + 45\sqrt{13}}{122}
\end{align*}
$$
The total time for the walk including the time spent in the coffee shop is:

$$\frac{81 + 45\sqrt{13}}{122} + \frac{144 - 48 \cdot \frac{-10 + 5\sqrt{13}}{2}}{60}$$

After simplifying, we find that the total time is 3 hours and 12 minutes. Therefore, the number of minutes the walk takes her, including the $ t $ minutes spent in the coffee shop, is:$$ \boxed{192} $$

<span style="color: #B8860B;">
Let&#39;s reassess my previous conclusion. The approach to solving the problem is overly complicated and involves unnecessary steps, such as solving a system of equations with time spent in the coffee shop included.
</span>
<span style="color: blue;">
Let&#39;s switch to a new solution. First, calculate the walking times for the given speeds without considering the time spent in the coffee shop, and then use the difference in total times to find the time spent walking at the new speed.
</span>

Let $ t_w $ be the time Aya spends walking when she walks at $ s $ kilometers per hour. We know:

$$ s \cdot t_w = 9 $$ 
$$  t_w = \frac{9}{s} $$
When she walks at $ s + 2 $ kilometers per hour, the time spent walking is:

$$  \frac{9}{s+2} $$ 
The difference in walking times is:

$$  \frac{9}{s} - \frac{9}{s+2} = 4 - 2.4 = 1.6 $$ 
Simplify the left side:

$$  \frac{9(s+2) - 9s}{s(s+2)} = \frac{18}{s(s+2)} = 1.6 $$
$$ s(s+2) = \frac{18}{1.6} = 11.25 $$ 
$$  s^2 + 2s - 11.25 = 0 $$ 

Solve the quadratic equation using the quadratic formula $  s = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $:

$$  s = \frac{-2 \pm \sqrt{4 + 45}}{2} = \frac{-2 \pm 7}{2} $$ 

Since $ s $ must be positive, we have:

$$  s = \frac{5}{2} = 2.5 $$ 
Now, find the time spent walking at $ s + \frac{1}{2} = 3 $ kilometers per hour:

$$  \text{Time spent walking} = \frac{9}{3} = 3 \text{ hours} $$ 
Convert 3 hours to minutes:$$  3 \text{ hours} = 180 \text{ minutes} $$ 

Since $ t $ minutes are spent in the coffee shop, the total time is:$$  180 + t $$ 

From the original equations, we know:

$$  2.5 \left( 4 - \frac{t}{60} \right) = 9 $$
$$  10 - \frac{2.5t}{60} = 9 $$
$$  \frac{2.5t}{60} = 1 $$ $$  2.5t = 60 $$ 
$$  t = 24 $$ Therefore, the total time is:

$$  180 + 24 = 204 $$ 

The number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop, is:

$$  \boxed{204} $$ 

</div>


### **Mathematical Reasoning (MATH500)**

#### **Problem Statement:**
<div class="code-box">
Given $\|\mathbf{v}\| = 5$ and $\|\mathbf{w}\| = 8,$ find the largest possible value of $\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\|.$
</div>

#### **Satori&#39;s Response:**
<div class="code-box">
The formula for the projection of vector $\mathbf{v}$ onto vector $\mathbf{w}$ is given by:
$$
\operatorname{proj}_{\mathbf{w}} \mathbf{v} = \frac{\mathbf{v} \cdot \mathbf{w}}{\mathbf{w} \cdot \mathbf{w}} \mathbf{w}
$$

The magnitude of the projection $\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\|$ is:

$$
\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\| = \left\| \frac{\mathbf{v} \cdot \mathbf{w}}{\mathbf{w} \cdot \mathbf{w}} \mathbf{w} \right\| = \left| \frac{\mathbf{v} \cdot \mathbf{w}}{\mathbf{w} \cdot \mathbf{w}} \right| \|\mathbf{w}\|
$$

Since $\|\mathbf{w}\| = 8$, we have $\mathbf{w} \cdot \mathbf{w} = 8^2 = 64$. Therefore, the expression for the magnitude of the projection becomes:

$$
\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\| = \left| \frac{\mathbf{v} \cdot \mathbf{w}}{64} \right| \cdot 8 = \frac{|\mathbf{v} \cdot \mathbf{w}|}{8}
$$

The dot product $\mathbf{v} \cdot \mathbf{w}$ can be expressed in terms of the magnitudes of $\mathbf{v}$ and $\mathbf{w}$ and the cosine of the angle $\theta$ between them:
$$
\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \|\mathbf{w}\| \cos \theta
$$

Given $\|\mathbf{v}\| = 5$ and $\|\mathbf{w}\| = 8$, we have:

$$
\mathbf{v} \cdot \mathbf{w} = 5 \cdot 8 \cdot \cos \theta = 40 \cos \theta
$$

Therefore, the magnitude of the projection is:



$$
\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\| = \frac{|40 \cos \theta|}{8} = 5 |\cos \theta|
$$

The maximum value of $|\cos \theta|$ is 1, which occurs when $\theta = 0^\circ$ (i.e., when $\mathbf{v}$ and $\mathbf{w}$ are in the same direction). Thus, the largest possible value of $\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\|$ is:



$$
5 \cdot 1 = 5
$$



Therefore, the largest possible value of $\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\|$ is $\boxed{5}$.

<span style="color: #B8860B;">
Wait. The explanation is correct but the step could be more concise.</span>
<span style="color: blue;">
Let&#39;s recompute the magnitude of the projection directly using the formula $\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\| = \frac{|\mathbf{v} \cdot \mathbf{w}|}{\|\mathbf{w}\|}$.
</span>

We know that $\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \|\mathbf{w}\| \cos \theta$, so:



$$
\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\| = \frac{|\|\mathbf{v}\| \|\mathbf{w}\| \cos \theta|}{\|\mathbf{w}\|} = \|\mathbf{v}\| |\cos \theta|
$$

Given $\|\mathbf{v}\| = 5$, we have:



$$
\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\| = 5 |\cos \theta|
$$

The maximum value of $|\cos \theta|$ is 1, so the maximum value of $\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\|$ is:


$$
5 \cdot 1 = 5
$$


Therefore, the largest possible value of $\|\operatorname{proj}_{\mathbf{w}} \mathbf{v}\|$ is $\boxed{5}$.
</div>

### **Logical Reasoning (BoardgameQA)**

#### **Problem Statement:**
<div class="code-box">

Solve the following problem. Your final answer should be &#34;True&#34;, &#34;False&#34;, or &#34;Uncertain&#34;.


A few players are playing a boardgame. The current state of the game is as follows. The liger has 1 friend that is easy going and 3 friends that are not. The liger is a physiotherapist. The seal has a football with a radius of 22 inches, and is a teacher assistant. The seal hides the cards that she has from the bee but does not build a power plant near the green fields of the husky. 

And the rules of the game are as follows. Rule1: If the liger works in healthcare, then the liger does not manage to convince the pelikan. Rule2: The seal will bring an oil tank for the pelikan if it (the seal) has a football that fits in a 52.3 x 43.6 x 36.1 inches box. Rule3: Regarding the seal, if it works in education, then we can conclude that it brings an oil tank for the pelikan. Rule4: If you see that something hides the cards that she has from the bee but does not build a power plant near the green fields of the husky, what can you certainly conclude? You can conclude that it does not bring an oil tank for the pelikan. Rule5: For the pelikan, if you have two pieces of evidence (1) the liger manages to convince the pelikan and (2) the seal brings an oil tank for the pelikan, then you can add &#34;pelikan will never bring an oil tank for the dragon&#34; to your conclusions. Rule6: The liger will manage to persuade the pelikan if it (the liger) has fewer than five friends.


Rule2 is preferred over Rule4. Rule3 is preferred over Rule4. Rule6 is preferred over Rule1. 

Based on the game state and the rules and preferences, does the pelikan bring an oil tank for the dragon?

</div>

#### **Satori&#39;s Response:**
<div class="code-box">
First, we analyze the given information and rules:

1. The liger has 1 friend that is easy going and 3 friends that are not.

2. The liger is a physiotherapist.

3. The seal has a football with a radius of 22 inches.

4. The seal hides the cards that she has from the bee but does not build a power plant near the green fields of the husky.

5. Rule1: If the liger works in healthcare, then the liger does not manage to convince the pelikan.

6. Rule2: The seal will bring an oil tank for the pelikan if it (the seal) has a football that fits in a 52.3 x 43.6 x 36.1 inches box.

7. Rule3: Regarding the seal, if it works in education, then we can conclude that it brings an oil tank for the pelikan.

8. Rule4: If you see that something hides the cards that she has from the bee but does not build a power plant near the green fields of the husky, what can you certainly conclude? You can conclude that it does not bring an oil tank for the pelikan.


9. Rule5: For the pelikan, if you have two pieces of evidence (1) the liger manages to convince the pelikan and (2) the seal brings an oil tank for the pelikan, then you can add &#34;pelikan will never bring an oil tank for the dragon&#34; to your conclusions.

10. Rule6: The liger will manage to persuade the pelikan if it (the liger) has fewer than five friends.

<span style="color: #B8860B;">
Wait. The mistake is that the solution starts analyzing the given information and rules without directly addressing the question of whether the pelikan brings an oil tank for the dragon.</span>
<span style="color: blue;">
Let&#39;s re-evaluate the problem step-by-step to determine if the pelikan brings an oil tank for the dragon.
</span>

1. **Liger&#39;s Friends and Occupation:**
   - The liger has 1 friend that is easy going and 3 friends that are not.
   
   - The liger is a physiotherapist.
   
   - According to Rule6: The liger will manage to persuade the pelikan if it (the liger) has fewer than five friends.
   
   - Since the liger has 4 friends (1 easy going + 3 not), the liger will manage to convince the pelikan.

2. **Seal&#39;s Football and Box:**
   - The seal has a football with a radius of 22 inches.
   
   - The dimensions of the box are 52.3 x 43.6 x 36.1 inches.
   
   - The volume of the football is given by $V = \frac{4}{3} \pi r^3 $.
     $$ V = \frac{4}{3} \pi (22)^3 \approx \frac{4}{3} \pi (10648) \approx 44922.67 \text{ cubic inches} $$
     
   - The volume of the box is:
     $$ V_{\text{box}} = 52.3 \times 43.6 \times 36.1 \approx 83257.69 \text{ cubic inches} $$
     
   - Since the volume of the football is less than the volume of the box, the seal&#39;s football fits in the box.
   
   - According to Rule2: The seal will bring an oil tank for the pelikan if it (the seal) has a football that fits in a 52.3 x 43.6 x 36.1 inches box.
   
   - Therefore, the seal will bring an oil tank for the pelikan.

3. **Conclusion Based on Rules:**
   - From Rule4: If you see that something hides the cards that she has from the bee but does not build a power plant near the green fields of the husky, what can you certainly conclude? You can conclude that it does not bring an oil tank for the pelikan.
   
     - The seal hides the cards that she has from the bee but does not build a power plant near the green fields of the husky.
     
     - Therefore, the seal does not bring an oil tank for the pelikan.
     
   - This creates a contradiction with Rule2, which states that the seal will bring an oil tank for the pelikan.
   
   - Given the rules and the information, the most logical conclusion is that the seal will bring an oil tank for the pelikan, despite the contradiction.

4. **Final Conclusion:**

   - Based on the analysis of the rules and the given information, the seal will bring an oil tank for the pelikan.

Therefore, the final answer is: $\boxed{\text{True}}$.

<span style="color: #B8860B;">
Wait. The mistake is that the solution starts analyzing the given information and rules without directly addressing the specific question about the pelikan bringing an oil tank for the dragon.</span>
<span style="color: blue;">
Let&#39;s simplify the approach and focus on the direct question. We need to determine if the pelikan brings an oil tank for the dragon based on the given rules and the information about the liger and the seal.
</span>

Based on the given information:

- The liger has 4 friends (1 easy going and 3 not), so the liger will manage to convince the pelikan (Rule6).

- The seal has a football with a radius of 22 inches, which fits in a 52.3 x 43.6 x 36.1 inches box, so the seal will bring an oil tank for the pelikan (Rule2).

- The seal hides the cards that she has from the bee but does not build a power plant near the green fields of the husky, so according to Rule4, the seal does not bring an oil tank for the pelikan.

There is a contradiction between Rule2 and Rule4. However, since Rule4 states a condition that leads to a specific conclusion if both conditions are met, we should consider the direct implications of each rule.

Given the contradiction, we need to re-evaluate the logical flow. The key is to recognize that Rule4 provides a clear condition that negates the oil tank bringing behavior if both conditions are met. Since the seal meets the condition of hiding cards and not building a power plant, Rule4 takes precedence over Rule2.

Therefore, the pelikan does not bring an oil tank for the dragon.

The final answer is: $\boxed{\text{False}}$.
</div>

### **Code Reasoning (CRUXEval)**

#### **Problem Statement:**
<div class="code-box">
You are given the following function:

```python
def f(text, sep, maxsplit):
    splitted = text.rsplit(sep, maxsplit)
    length = len(splitted)
    new_splitted = splitted[:length // 2]
    new_splitted.reverse()
    new_splitted += splitted[length // 2:]
    return sep.join(new_splitted)
```

You are also given the following input (the order of the input is the same as the order of the input variables in the function):


&#34;ertubwi&#34;, &#34;p&#34;, 5


With the given function and input, what would be the output?
</div>

#### **Satori&#39;s Response:**
<div class="code-box">
First, we need to understand the function $f$ and how it processes the given input. The function $f$ takes three arguments: a string <i>text</i>, a separator <i>sep</i>, and a maximum number of splits <i>maxsplit</i>. It splits the string <i>text</i> into parts using the separator <i>sep</i> up to <i>maxsplit</i> times, reverses the first half of the resulting list of parts, and then concatenates the reversed first half with the second half using the separator <i>sep</i>.

<span style="color: #B8860B;">
Wait. Let&#39;s clarify this. The function correctly processes the input by splitting the string, reversing the first half, and concatenating the parts, which aligns with the function&#39;s intended behavior.
</span>

Let&#39;s break down the function step by step with the given input: &#34;ertubwi&#34;, &#34;p&#34;, 5.

1. **Split the string:** The function uses *text.rsplit(sep, maxsplit)*. Here, <i> text = &#34;ertubwi&#34; </i>, sep = &#34;p&#34;, and *maxsplit = 5*. Since &#34;ertubwi&#34; does not contain the separator &#34;p&#34;, <i>text.rsplit(sep, maxsplit)</i> will split the string into a list containing the entire string and four empty strings (because <i>maxsplit</i> is 5, but there are no occurrences of &#34;p&#34;):

   <i>
   splitted = [&#34;ertubwi&#34;, &#34;&#34;, &#34;&#34;, &#34;&#34;, &#34;&#34;]
   </i>

2. **Determine the length:** The length of <i>splitted</i> is 5.

3. **Split the list into two halves:** Since the length is 5, the first half will be the first 2 elements (index 0 and 1), and the second half will be the remaining elements (index 2, 3, and 4):
   <i>
   new_splitted = splitted[:5 // 2] = [&#34;ertubwi&#34;, &#34;&#34;]
   </i>

   <i>
   new_splitted += splitted[5 // 2:] = [&#34;ertubwi&#34;, &#34;&#34;, &#34;&#34;, &#34;&#34;, &#34;&#34;]
   </i>

4. **Reverse the first half:** Reverse the elements in *new_splitted[:2]*:
   <i>
   new_splitted[:2] = [&#34;&#34;, &#34;ertubwi&#34;]
   </i>

5. **Concatenate the parts:** Now, concatenate the reversed first half with the second half using the separator &#34;p&#34;:
   <i>
   new_splitted = [&#34;&#34;, &#34;ertubwi&#34;, &#34;&#34;, &#34;&#34;, &#34;&#34;]
   </i>

   <i>
   result = &#34;p&#34;.join(new_splitted) = &#34;p&#34; + &#34;&#34; + &#34;ertubwi&#34; + &#34;p&#34; + &#34;&#34; + &#34;p&#34; + &#34;&#34; + &#34;p&#34; + &#34;&#34; = &#34;p ertubwi p p p p&#34;
   </i>


However, since the second half should only include the remaining elements after the first half, we need to re-evaluate the correct approach. Let&#39;s consider the correct approach:


1. **Split the string:** Since &#34;ertubwi&#34; does not contain &#34;p&#34;, we need to consider the correct approach to split and reverse.

Given the function&#39;s behavior and the input, the correct approach is to split the string into two equal parts and reverse the first part.

1. **Split the string:** &#34;ertubwi&#34; into &#34;ertu&#34; and &#34;bwi&#34;.

2. **Reverse the first part:** &#34;ertu&#34; reversed is &#34;uert&#34;.

3. **Concatenate: &#34;uert&#34; and &#34;bwi&#34; with separator &#34;p&#34;:** &#34;uertpbwi&#34;.

Therefore, the final answer is: $\boxed{\text{uertpbwi}}$.

<span style="color: #B8860B;">
Wait. The mistake is in the incorrect handling of the split and the subsequent concatenation, leading to an incorrect final string.</span>
<span style="color: blue;">
Let&#39;s switch to a new solution. We should correctly split the string &#34;ertubwi&#34; into parts using the separator &#34;p&#34;, which in this case does not occur, so we should consider the string as a whole and apply the function&#39;s logic directly.
</span>

Let&#39;s re-evaluate the function step-by-step with the given input: &#34;ertubwi&#34;, &#34;p&#34;, 5.

1. **Split the string:** Since &#34;ertubwi&#34; does not contain the separator &#34;p&#34;, the function will split it into a list with one element: <i>[&#34;ertubwi&#34;]</i>.

2. **Determine the length:** The length of the list is 1.

3. **Split the list into two halves:** Since the length is odd, the first half will have 0 elements, and the second half will have 1 element.

4. **Reverse the first half:** The first half is empty, so it remains empty.

5. **Concatenate the parts:** Concatenate the reversed first half with the second half using the separator &#34;p&#34;: <i>&#34;&#34; + &#34;ertubwi&#34; = &#34;ertubwi&#34;</i>.

Therefore, the final answer is: $\boxed{\text{ertubwi}}$.
</div>


### **Commonsense Reasoning (StrategyQA)**

#### **Problem Statement:**
<div class="code-box">
You are given the following facts:

1. Christopher Reeve suffered a serious spinal cord injury that left him a quadriplegic. 

2. Christopher Reeve required a portable ventilator after his injury.

3. At one point, Christopher Reeve&#39;s treatment was costing $400,000 yearly.

Based on the facts above, answer the following question. Your final answer should be either &#34;True&#34; or &#34;False&#34;.

If he were poor, would Christopher Reeve have lived?
</div>

#### **Satori&#39;s Response:**
<div class="code-box">
The question asks whether Christopher Reeve would have lived if he were poor.

To answer this, we need to consider the impact of his spinal cord injury and the cost of his treatment.

Christopher Reeve&#39;s spinal cord injury left him a quadriplegic, which means he was paralyzed from the neck down. This made it difficult for him to move and perform many daily activities.

The fact that he required a portable ventilator indicates that he was dependent on mechanical assistance for breathing.

The cost of his treatment was \$400,000 yearly, which is a significant amount of money.

If Christopher Reeve were poor, he would likely not have been able to afford this level of medical care.

Without the necessary medical treatment, it is highly unlikely that he would have been able to live a normal life.

<span style="color: #B8860B;">
Wait. The question is not about the impact of his condition or the cost of treatment on his ability to live, but rather if he would have lived given the circumstances.</span>
<span style="color: blue;">
Let&#39;s switch to a new solution. Focus on the fact that the question asks if he would have lived if he were poor, and consider whether the cost of treatment alone would have been a determining factor in his survival.
</span>

Let&#39;s re-evaluate the question more carefully. The question asks if Christopher Reeve would have lived if he were poor.

The key factors to consider are:

1. The severity of his spinal cord injury.

2. The cost of his treatment.

3. The impact of poverty on his ability to access and afford medical care.

Christopher Reeve&#39;s spinal cord injury was severe, and he required specialized medical equipment and ongoing treatment. The cost of \$400,000 yearly was substantial and likely beyond the means of many people, especially those who are poor.

If Christopher Reeve were poor, he would likely not have been able to afford this level of medical care. The cost of his treatment would have been a significant financial burden.

Given the severity of his condition and the high cost of treatment, it is reasonable to conclude that he would not have been able to live a normal life if he were poor.

Therefore, the final answer is: $\boxed{\text{False}}$.
</div>

### **Tabular Reasoning (TableBench)**

#### **Problem Statement:**
<div class="code-box">
You are given a table with the following data:
  <div class="code-box">
    <table style="width: 80%; font-size: 12px; margin: 0 auto;">
        <thead>
            <tr>
                <th style="font-size: 12px;"><strong>Peak</strong></th>
                <th style="font-size: 12px;"><strong>Country</strong></th>
                <th style="font-size: 12px;"><strong>Elevation (m)</strong></th>
                <th style="font-size: 12px;"><strong>Prominence (m)</strong></th>
                <th style="font-size: 12px;"><strong>Col (m)</strong></th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Mount Stanley</td>
                <td>Democratic Republic of the Congo / Uganda</td>
                <td>5109</td>
                <td>3951</td>
                <td>1158</td>
            </tr>
            <tr>
                <td>Mount Karisimbi</td>
                <td>Rwanda / Democratic Republic of the Congo</td>
                <td>4507</td>
                <td>3312</td>
                <td>1195</td>
            </tr>
            <tr>
                <td>Kinyeti</td>
                <td>South Sudan</td>
                <td>3187</td>
                <td>2120</td>
                <td>1067</td>
            </tr>
            <tr>
                <td>Emogadong</td>
                <td>South Sudan</td>
                <td>2623</td>
                <td>1730</td>
                <td>893</td>
            </tr>
            <tr>
                <td>Kabobo</td>
                <td>Democratic Republic of the Congo</td>
                <td>2725</td>
                <td>1604</td>
                <td>1121</td>
            </tr>
            <tr>
                <td>Mont Mohi</td>
                <td>Democratic Republic of the Congo</td>
                <td>3480</td>
                <td>1592</td>
                <td>1888</td>
            </tr>
            <tr>
                <td>Wuhevi</td>
                <td>Democratic Republic of the Congo</td>
                <td>3095</td>
                <td>1570</td>
                <td>1525</td>
            </tr>
            <tr>
                <td>Mount Muhabura</td>
                <td>Rwanda / Uganda</td>
                <td>4127</td>
                <td>1530</td>
                <td>2597</td>
            </tr>
        </tbody>
    </table>
    <p style="text-align: center;"><strong>Table of peaks and their details.</strong></p>
  </div>
Based on the table, answer the following question. If your answer is extracted from the table, make sure that the answer is exactly the same as the corresponding content in the table.

What is the average prominence of mountain peaks in the Democratic Republic of the Congo that have an elevation of at least 3000 meters?
</div>

#### **Satori&#39;s Response:**
<div class="code-box">
First, we identify the mountain peaks in the Democratic Republic of the Congo that have an elevation of at least 3000 meters. From the table, these peaks are:

- Mount Stanley (5109 meters)

- Mount Karisimbi (4507 meters)

- Kabobo (2725 meters)

- Mont Mohi (3480 meters)

- Wuhevi (3095 meters)

Next, we check the prominences of these peaks:

- Mount Stanley: 3951 meters

- Mount Karisimbi: 3312 meters

- Kabobo: 1604 meters

- Mont Mohi: 1592 meters

- Wuhevi: 1570 meters

We sum the prominences of these peaks:

$$ 3951 + 3312 + 1604 + 1592 + 1570 $$

Let's calculate the sum step by step:

$$ 3951 + 3312 = 7263, 7263 + 1604 = 8867,  8867 + 1592 = 10459, 10459 + 1570 = 12029 $$

There are 5 peaks, so we divide the total sum of the prominences by 5 to find the average: $ \frac{12029}{5} = 2405.8 $


Therefore, the average prominence of mountain peaks in the Democratic Republic of the Congo that have an elevation of at least 3000 meters is $\boxed{2405.8}$.

<span style="color: #B8860B;">
Wait. The mistake is including Kabobo, which has an elevation of 2725 meters, which is less than 3000 meters.</span>
<span style="color: blue;">
Let&#39;s switch to a new solution. Only consider the peaks with elevations of at least 3000 meters, which are Mount Stanley, Mount Karisimbi, Mont Mohi, and Wuhevi.
</span>

Identify the mountain peaks in the Democratic Republic of the Congo with an elevation of at least 3000 meters:

- Mount Stanley (5109 meters)

- Mount Karisimbi (4507 meters)

- Mont Mohi (3480 meters)

- Wuhevi (3095 meters)

Check the prominences of these peaks:

- Mount Stanley: 3951 meters

- Mount Karisimbi: 3312 meters

- Mont Mohi: 1592 meters

- Wuhevi: 1570 meters

Sum the prominences of these peaks:

$$ 3951 + 3312 + 1592 + 1570 $$

Calculate the sum step by step:

$$ 3951 + 3312 = 7263, 7263 + 1592 = 8855,  8855 + 1570 = 10425 $$

There are 4 peaks, so divide the total sum of the prominences by 4 to find the average:
$ \frac{10425}{4} = 2606.25 $

Therefore, the average prominence of mountain peaks in the Democratic Republic of the Congo that have an elevation of at least 3000 meters is $\boxed{2606.25}$.
</div>

### **Domain-specific Reasoning (MMLUPro - Economics)**

#### **Problem Statement:**
<div class="code-box">
What does the Harrod-Domar model, taken with the Keynesian theory of savings, imply for the growth rates of poor countries?

(A) Rich and poor countries will experience the same growth rates regardless of savings rates

(B) Poor countries will have fluctuating growth rates independent of savings

(C) The level of technology determines growth rates, not savings

(D) The level of income has no impact on savings

(E) Savings have no correlation with growth rates in any country

(F) Rich countries will have low growth rates

(G) Poor countries will have high growth rates

(H) High levels of savings will always lead to high growth rates

(I) Poor countries will have low growth rates

(J) Poor countries can achieve high growth rates regardless of investment levels
</div>

#### **Satori&#39;s Response:**
<div class="code-box">
The Harrod-Domar model is an economic model that relates the growth rate of an economy to its savings rate and its capital-output ratio. The model suggests that the growth rate of an economy is determined by the ratio of savings to capital, and that this ratio is influenced by the capital-output ratio.


<span style="color: #B8860B;">
Wait. Let&#39;s clarify this. The Harrod-Domar model indeed relates the growth rate to the savings rate and the capital-output ratio, which is a correct starting point for analyzing the growth rates of poor countries.
</span>

The Harrod-Domar model states that the growth rate $ g $ of an economy is given by:

$$ g = \frac{s}{k} $$

where $ s $ is the savings rate and $ k $ is the capital-output ratio.


<span style="color: #B8860B;">
Wait. The explanation of the Harrod-Domar model is correct but the step does not directly address the implications for poor countries in the context of the Keynesian theory of savings.</span>
<span style="color: blue;">
Let&#39;s switch to a new solution. Start by explaining how the Harrod-Domar model, when combined with the Keynesian theory of savings, suggests that higher savings rates in poor countries can lead to higher growth rates due to increased investment.
</span>


The Harrod-Domar model suggests that the growth rate of an economy is determined by the ratio of savings to capital. According to the Keynesian theory of savings, countries with higher savings rates will have more funds available for investment.

In poor countries, where the capital stock is often lower relative to their population, a higher savings rate can lead to a higher capital stock over time. This increase in capital stock can then support higher levels of output and, consequently, higher growth rates.

<span style="color: #B8860B;">
Wait. The explanation does not directly address the options provided in the question, which are specific to the growth rates of poor countries.</span>
<span style="color: blue;">
Let&#39;s switch to a new solution. Analyze each option to determine which one best fits the implications of the Harrod-Domar model and the Keynesian theory of savings for poor countries.
</span>

Let&#39;s re-evaluate the options given:

(A) Rich and poor countries will experience the same growth rates regardless of savings rates.
  - This is not true because the Harrod-Domar model suggests that growth rates are determined by the savings rate and the capital-output ratio, which can differ between rich and poor countries.

(B) Poor countries will have fluctuating growth rates independent of savings.
  - This is not supported by the Harrod-Domar model, which suggests that growth rates are directly related to savings rates.

(C) The level of technology determines growth rates, not savings.
  - This is not true according to the Harrod-Domar model, which explicitly states that growth rates are determined by the savings rate and the capital-output ratio.

(D) The level of income has no impact on savings.
  - This is not true according to the Keynesian theory of savings, which suggests that higher income can lead to higher savings.

(E) Savings have no correlation with growth rates in any country.
  - This is not true according to the Harrod-Domar model, which shows a direct relationship between savings and growth.

(F) Rich countries will have low growth rates.
  - This is not necessarily true. Rich countries may have higher growth rates if they have high savings rates and a low capital-output ratio.

(G) Poor countries will have high growth rates.
  - This is not necessarily true. Poor countries may have lower growth rates if they have low savings rates and a high capital-output ratio.

(H) High levels of savings will always lead to high growth rates.
  - This is not true because the Harrod-Domar model also considers the capital-output ratio. High savings rates alone are not sufficient for high growth rates.

(I) Poor countries will have low growth rates.
  - This is a possible implication, but it depends on the specific savings and capital-output ratios of poor countries.

(J) Poor countries can achieve high growth rates regardless of investment levels.
  - This is not true according to the Harrod-Domar model, which suggests that high growth rates require high savings rates and a low capital-output ratio.

Given the analysis, the most accurate statement based on the Harrod-Domar model and the Keynesian theory of savings is that poor countries can achieve high growth rates if they have high savings rates and a low capital-output ratio.

Therefore, the final answer is: $\boxed{I}$.
</div>


## **Commitment to Open-Source Community**
We aim to contribute to open-source communities by supporting the development of new research methods that can enhance LLM reasoning capabilities. We plan to provide the following resources:
- Technique details (our research paper)
- Synthetic data for format tuning
- Reinforcement learning (RL) training data 
- Code for multi-agent data synthesis
- Code for RL training

## **Satori Team Members**
### **Core Contributors**
- [Maohao Shen, MIT](https://maohaos2.github.io/Maohao/)
- [Guangtao Zeng, SUTD](https://chaoscodes.github.io/)
- [Zhenting Qi, Harvard](https://zhentingqi.github.io/)
### **Contributors**
<span>$&#8224;$</span>: Project lead
- Zhang-Wei Hong, MIT
- Zhenfang Chen, MIT-IBM Watson AI Lab
- Wei Lu, SUTD
- Gregory W. Wornell, MIT
- Subhro Das, MIT-IBM Watson AI Lab
- David Cox, MIT-IBM Watson AI Lab
- Chuang Gan<span>$^&#8224;$</span>, UMass, MIT-IBM Watson AI Lab

## **Contact Information**
For questions, please:
- Raise an issue in our GitHub repository
- Contact us at:
  - [satori2025@outlook.com](mailto:satori2025@outlook.com)


## **Citation**
(TODO)
```
@article{TBD,
  title={Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search},
  author={TBD},
  journal={TBD},
  year={2025}
}
```
