# Reasoning-Tree Construction Prompts

This file documents the LLM prompts used to **construct and verify reasoning
trees** (Section IV-A of the paper). Placeholders such as `{root_question}` are
populated with dataset-specific content at runtime. The reasoning-tree
construction code lives in a separate private environment; these prompts are
provided for reproducibility.

The pipeline consists of two phases: (1) **top-down decomposition** (Prompt 1–4)
and (2) **bottom-up rationale update** (Prompt 5), followed by (3) **automated
verification** (Prompt 6).

---

## Prompt 1: Decomposition (Top-down)

In the top-down decomposition phase, the LLM decomposes a node into multiple
child subproblems. The root node is always decomposed by default; for non-root
nodes, decomposition proceeds only if the decomposition score exceeds a
predefined threshold.

```
Please further decompose the sub-question in [Sub-question]. You can refer to
the examples of similar questions and their decomposed sub-questions in
[Logic Heuristics].

[Logic Heuristics:]
{logic_heuristics}

[Root question:]
{root_question}

[Sub-question:]
{sub_question}

Instruction:
1. Output the decomposed sub-questions as a string in list format:
   ["sub-question1", "sub-question2", "sub-question3"].
2. If the sub-question can be decomposed, output the sub-question list;
   otherwise, output an empty list: [].
3. Output only a list without any additional text, and keep the sub-questions
   as concise as possible.
```

## Prompt 2: Generating Rationale

After decomposing a node into child subproblems, the LLM generates a logical
rationale (step-by-step reasoning) for each subproblem, conditioned on the
root question as context.

```
Please answer the sub-question based on the root question provided as context
and provide a reasoning process.

[Root question:]
{root_question}

[Sub-question:]
{sub_question}

Instruction: Output only a reasoning process without any additional text, and
keep the rationale as concise as possible.
```

## Prompt 3: Generating Accuracy Scores

The LLM scores each subproblem's rationale on a 1–5 scale for correctness and
logical soundness. These accuracy scores guide quality control during
subsequent decomposition steps.

```
Rate the overall correctness and logical soundness of the provided rationale
for the sub-question. Use a scale of 1-5, where 1 indicates completely
incorrect/illogical reasoning and 5 indicates flawless reasoning.

[Root question:]
{root_question}

[Sub-question:]
{sub_question}

[Rationale for sub-question:]
{rationale}

Instruction: Output only an integer rating without any additional text or
format.
```

## Prompt 4: Generating Decomposition Scores

The LLM rates whether a given subproblem requires further decomposition on a
1–5 scale. If the score exceeds a predefined threshold, the subproblem is
further decomposed; otherwise, decomposition stops at that node.

```
Rate the necessity of further decomposing the given sub-question based on the
rationale for the sub-question. Use a scale of 1-5, where 1 indicates that the
problem is straightforward and requires no decomposition, and 5 indicates that
the problem is complex and must be broken down into smaller parts.

[Root question:]
{root_question}

[Sub-question:]
{sub_question}

[Rationale for sub-question:]
{rationale}

Instruction: Output only an integer rating without any additional text or
format.
```

## Prompt 5: Updating Logical Rationales (Bottom-up)

In the bottom-up update phase, after generating a rationale for the n-th node,
the LLM sequentially compares it against all preceding nodes and updates their
rationales to maintain cross-node consistency.

```
Update the rationale for question b based on the rationale for question a,
given that both question a and question b are sub-questions of root question q.

[Root question q:]
{root_question}

[Question a:]
{question_a}

[The rationale for question a:]
{rationale_a}

[Question b:]
{question_b}

[The rationale for question b:]
{rationale_b}

Instruction: Output only the updated rationale without any additional text,
and keep the rationale as concise as possible.
```

## Prompt 6: Reasoning-Tree Verification

As an automated quality assurance step, an LLM verifier examines every
generated reasoning tree for structural integrity and logical coherence before
it is used for downstream training. Only trees that pass this review are
retained.

```
You are a quality assurance reviewer. Examine the following reasoning tree
generated for a math problem. Verify that: (1) the tree structure is logically
coherent, (2) each child node is a genuine subproblem of its parent, and
(3) rationales are mathematically correct. Output "PASS" if the tree meets all
criteria, or "FAIL" with a brief explanation otherwise.

Problem:
{item_stem}

Reasoning Tree:
{reasoning_tree_json}
```
