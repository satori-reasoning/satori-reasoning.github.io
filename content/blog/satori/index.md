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
{{< button href="" label="Code" external=true >}}
{{< button href="" label="Data" external=true >}}
{{< button href="" label="Model" external=true >}}

<style>
.code-box {
    max-height: 300px;
    overflow-y: auto;
    padding: 10px;
    background-color: #f8f8f8; /* 代码背景色 */
    font-family: Consolas, Monaco, 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    border-radius: 5px; /* 圆角 */
}

</style>

## **Introduction**
Since the release of OpenAI's o1, significant efforts have been made within the research community to enhance open-source LLMs with advanced reasoning capabilities. This includes various approaches, including distillation using a strong teacher model, MCTS, and reward model guided search. This work aims to explore a new research direction: enabling LLMs with autoregressive search capabilities, i.e., a single LLM performs an extended reasoning process with self-reflection and self-exploration of new strategies. To achieve this, we develop a LLM post-training paradigm with several key concepts and ideas inspired by classical reinforcement learning (RL) communities. Our approach results in Satori, a 7B LLM trained on open-source model (Qwen-2.5-Math-7B) and open-source data (OpenMathInstruct-2 and NuminaMath). Key features of Satori include:
  - Capable of self-reflection and self-exploration without external guidance.
  - Achieve state-of-the-art reasoning performance mainly through self-improvement (RL).
  - Exhibit transferability of reasoning capabilities on unseen domains beyond math.

## **Our Approach**
We formulate LLM reasoning as a sequential decision-making problem, where reasoning is a process of constructing and refining an answer step by step. Specifically, the LLM (agent's policy) starts with an input context (initial state), generates a reasoning step (action), and updates the context (next state). The LLM repeats this process until it reaches a final answer, and receives a reward that evaluates whether the final answer matches the ground truth. With this formulation, we could train the LLM to reason using RL, aiming to generate a sequence of reasoning steps that maximize the expected reward.

### **Chain-of-Action-Thought reasoning (COAT)**
The key challenge of achieving autoregressive search is enabling the LLM to determine when to reflect, continue, or explore alternative solutions without external intervention. 
To enable this, we introduce several special meta-action tokens that guide the LLM's reasoning process,
  - **Continue Reasoning** (<|continue|>): encourages the LLM to build upon its current reasoning trajectory by generating the next intermediate step. 
  - **Reflect** (<|reflect|>): prompts the model to pause and verify the correctness of prior reasoning steps.
  - **Explore Alternative Solution** (<|explore|>): signals the model to identify critical flaws in its reasoning and explore a new solution.

We refer to this formulation as Chain-of-Action-Thought (COAT) reasoning. Each COAT reasoning step is a sequence of tokens, starting with one of the meta-action tokens.

### **Overview of Training Framework**
A standard LLM is not able to perform COAT reasoning, and achieving this will encounter two key challenges:
  - **Unawareness of meta-action tokens**: Without training, the LLM doesn't recognize that encountering special meta-action tokens may require reflection or proposing alternative solutions.
  - **Long horizon and sparse rewards**: Reasoning requires long-term decision-making with rewards only at the end, the LLM must take many correct reasoning steps before receiving rewards, and failures force it to restart from the initial state. This makes learning difficult because rewards are scarce, yet rewards are essential for driving RL progress.

To address these two challenges, we propose a two-stage training framework\*: 
  1) A small-scale format tuning (FT) stage that helps the base LLM to internalize the COAT reasoning format.
  2) A large-scale self-improvement stage that utilizes reinforcement learning with "Restart and Explore" (RAE) techniques.
![schematic](/img/schematic.png)

\***Note**: *our concurrent work [DeepSeek-R1](https://arxiv.org/abs/2501.12948) adopts a similar training framework of small-scale cold-start SFT followed by large-scale RL training. Although both works coincide in this high-level idea, our work differs from R1 in key methodologies, including the data synthesis framework and RL algorithms.*
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

Mathematical Reasoning (from AIME2024)

Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.

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

Let's solve these equations step by step. First, expand both equations:

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

$$ \[ \frac{81 + 45\sqrt{13}}{122} + \frac{144 - 48 \cdot \frac{-10 + 5\sqrt{13}}{2}}{60} \] $$

After simplifying, we find that the total time is 3 hours and 12 minutes. Therefore, the number of minutes the walk takes her, including the $ t $ minutes spent in the coffee shop, is:$$ \boxed{192} $$

<p><span style="color: #B8860B;">
Let's reassess my previous conclusion. The approach to solving the problem is overly complicated and involves unnecessary steps, such as solving a system of equations with time spent in the coffee shop included.
</span></p>
<p><span style="color: blue;">
Let's switch to a new solution. First, calculate the walking times for the given speeds without considering the time spent in the coffee shop, and then use the difference in total times to find the time spent walking at the new speed.
</span></p>

Let $ t_w $ be the time Aya spends walking when she walks at $ s \) kilometers per hour. We know:

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

Solve the quadratic equation using the quadratic formula $$  s = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$ :

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

The number of minutes the walk takes her, including the \( t \) minutes spent in the coffee shop, is:

$$  \boxed{204} $$ 

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
- [Maohao Shen](https://maohaos2.github.io/Maohao/), MIT
- [Guangtao Zeng](https://chaoscodes.github.io/), SUTD
- [Zhenting Qi](https://zhentingqi.github.io/), Harvard
### **Contributors**
   \*: Project lead
- Zhang-Wei Hong, MIT
- Zhenfang Chen, MIT-IBM Watson AI Lab
- Wei Lu, SUTD
- Gregory W. Wornell, MIT
- Subhro Das, MIT-IBM Watson AI Lab
- David Cox, MIT-IBM Watson AI Lab
- Chuang Gan\*, UMass, MIT-IBM Watson AI Lab

## **Contact Information**
For questions or collaboration, please:
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
