# Neuro-Symbolic AI - Card Suit Game with Semantic Loss

**Neuro-Symbolic AI Assignment 2026/2027**

This university project investigates the integration of symbolic knowledge into neural network training through **Semantic Loss**.

The objective is to develop and compare two models capable of recognizing the suits of two playing cards and predicting the outcome of a game directly from their images.

The central research question is whether incorporating logical constraints into the neural learning objective can improve prediction consistency while preserving classification performance.

---

## 1. Problem Definition

The task consists of predicting the game outcome given two input card images:

- `I_x`: image of the first card.
- `I_y`: image of the second card.

The game involves three suits governed by the following cyclic dominance rules:

| Suit | Dominates |
|:---|:---|
| Hearts | Clubs |
| Clubs | Spades |
| Spades | Hearts |

Two cards with identical suits result in a **Draw**. Otherwise, the player holding the dominant suit wins.

Card ranks are completely ignored.

### Dataset

The project uses the [Cards Image Dataset - Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) from Kaggle.

Only the following classes are retained:

- Hearts
- Clubs
- Spades

Diamonds and Jokers are excluded.

---

## 2. Neural Architecture

The network receives two card images simultaneously and produces **nine output logits**, organized into three groups:

| Output group | Classes | Purpose |
|:---|:---|:---|
| First card | Hearts, Clubs, Spades | First suit classification |
| Second card | Hearts, Clubs, Spades | Second suit classification |
| Game outcome | Player 1, Draw, Player 2 | Outcome prediction |

The architecture performs three related classification tasks within a shared multi-output learning framework.

For supervised classification, raw logits are used with standard multi-class Cross Entropy. For logical regularization, sigmoid transformations provide nine independent Bernoulli probabilities associated with the corresponding Boolean propositions.

### Label Encoding

| Variable | Class | Integer label |
|:---|:---|:---:|
| Suit | Hearts | 0 |
| Suit | Clubs | 1 |
| Suit | Spades | 2 |
| Outcome | Player 1 wins (1) | 0 |
| Outcome | Draw (x) | 1 |
| Outcome | Player 2 wins (2) | 2 |

The outcome encoding follows the assignment convention `{1, x, 2}`.

---

## 3. Neuro-Symbolic Learning: Two Approaches

The central experimental contribution is the comparison between two models sharing the same neural architecture, training examples, labels, and data partitions.

The only difference is the loss function used during optimization.

### A. Neural Baseline - Purely Supervised Learning

The baseline represents a conventional data-driven approach.

It learns the three classification tasks exclusively through supervised Cross Entropy:

```text
L_neural = L_x + L_y + L_o
```

Where:

- `L_x`: Cross Entropy for the first card.
- `L_y`: Cross Entropy for the second card.
- `L_o`: Cross Entropy for the game outcome.

No explicit logical constraints are included in the training objective.

### B. Semantic Loss Model - Neuro-Symbolic Learning

The second approach extends supervised learning by integrating the formal rules of the game directly into the optimization process.

Two categories of propositional constraints are defined.

**1. Exactly-One Constraint (alpha_EO)**

Each card must have exactly one active suit, and the game outcome must also correspond to exactly one class.

This constraint is decomposed into three Semantic Loss terms:

- `EO_x`: Exactly-One constraint for the first card.
- `EO_y`: Exactly-One constraint for the second card.
- `EO_o`: Exactly-One constraint for the game outcome.

**2. Game Rules Constraint (alpha_Rules)**

The logical rules formalize the relationship between the two card suits and the resulting game outcome.

For example, if the first card is Hearts and the second card is Clubs, Player 1 must win.

The corresponding Semantic Loss term is `SL_xy`.

### Complete Optimization Objective

The neuro-symbolic model is trained using the following objective:

```text
L_total = omega_1 * (L_x + L_y + L_o)
        + omega_2 * (EO_x + EO_y + EO_o)
        + omega_3 * SL_xy
```

The three hyperparameters control the contribution of each component:

| Hyperparameter | Contribution |
|:---|:---|
| `omega_1` | Supervised Cross Entropy |
| `omega_2` | Exactly-One Semantic Loss |
| `omega_3` | Game-rule Semantic Loss |

This formulation allows symbolic knowledge to influence the network parameters through differentiable logical regularization and backpropagation.

---

## 4. Semantic Loss Implementation

Semantic Loss measures the negative logarithm of the probability that a Boolean formula is satisfied.

For a logical constraint `alpha`, its general formulation is:

```text
SL(alpha) = -log(P(alpha))
```

The probability is computed by summing the probabilities of all Boolean assignments satisfying the corresponding logical formula.

### Brute-Force Enumeration

Rather than relying on an external logical inference engine, the project uses exhaustive enumeration of Boolean assignments.

Since the model has nine Boolean propositions, the complete assignment space contains:

```text
2^9 = 512 possible Boolean assignments
```

For each assignment, the implementation evaluates whether the corresponding logical constraints are satisfied and computes the associated probability.

This approach is computationally feasible for the problem size and is explicitly permitted by the assignment as an alternative to compiling the constraints into Sentential Decision Diagrams (SDDs).

**Implementation choice:** Semantic Loss is computed directly within the PyTorch training pipeline. No ASP or DeepProbLog inference engine is used.

---

## 5. Data Preparation and Organization

The dataset is processed through a custom PyTorch Dataset named `CardDuelDataset`, which loads pairs of card images and their associated labels.

### Image Preprocessing

- Image resizing: `128 x 128` pixels.
- Color representation: RGB.
- Tensor conversion: `torchvision.transforms.ToTensor()`.
- Normalization: standard ImageNet channel mean and standard deviation.

The DataLoaders use a batch size of `32`.

### CSV Manifests

To guarantee a controlled comparison, card pairs and their labels are stored in fixed CSV manifests.

The `manifests/` directory contains:

```text
manifests/
|-- train_100.csv
|-- train_50.csv
|-- train_20.csv
|-- val.csv
|-- test.csv
```

These manifests define the samples used in each experimental partition.

**Important:** Both the Neural Baseline and the Semantic Loss Model must use the same manifests. This ensures that the comparison is performed on identical training, validation, and test examples.

The full training set (100%) is used for the main experiments. The 50% and 20% configurations are retained as optional data-efficiency investigations.

---

## 6. Experimental Protocol

The experimental evaluation is organized into four main components.

### 6.1 Training

Both models are trained using the same data partitions and neural architecture:

- Neural Baseline: supervised Cross Entropy.
- Semantic Loss Model: supervised learning combined with logical regularization.

### 6.2 Ablation Study

The full neuro-symbolic model is compared against the purely neural baseline to investigate the contribution of the logical constraints.

The purpose is to evaluate whether Semantic Loss improves logical consistency and how this affects predictive performance.

### 6.3 Hyperparameter Tuning

Different configurations of `omega_1`, `omega_2`, and `omega_3` are investigated.

The validation set is used to identify suitable loss-weight configurations before final evaluation.

### 6.4 Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Description |
|:---|:---|
| First card accuracy | Correct suit predictions for the first image |
| Second card accuracy | Correct suit predictions for the second image |
| Outcome accuracy | Correct game-outcome predictions |
| Exactly-One satisfaction | Compliance with the three Exactly-One constraints |
| Rule consistency | Compliance with the logical game rules |
| Training time | Computational cost of model optimization |

The **test set is reserved for the final comparison**, avoiding its use during hyperparameter selection.

---

## 7. Research Objectives and Expected Results

The central objective is to investigate the effectiveness of Semantic Loss as a mechanism for integrating symbolic knowledge into neural learning.

The experiments aim to answer three main questions:

1. Does logical regularization improve the consistency of neural predictions?
2. What is the impact of Semantic Loss on classification accuracy?
3. What computational cost is introduced by logical regularization?

The project does not assume that the neuro-symbolic approach will necessarily outperform the baseline on every metric.

Instead, the experimental comparison will determine the actual benefits and limitations of incorporating symbolic constraints into the learning process.

---

## 8. Theoretical Component and Deliverables

The theoretical component consists of an **8-12 slide presentation** covering:

- The theoretical foundations of Semantic Loss.
- Formalization of the Exactly-One and game-rule constraints.
- Brute-force enumeration and probability computation.
- The influence of logical regularization on neural training.
- Experimental results, comparison, and conclusions.

The project deliverables include:

- Complete source code and Jupyter notebook.
- Experimental configurations and results.
- Theoretical presentation.
- Instructions required to reproduce the experiments.

---

## 9. Technologies

| Technology | Role |
|:---|:---|
| Python | Main programming language |
| PyTorch | Neural network implementation and training |
| Torchvision | Image preprocessing |
| Pandas | Dataset and CSV manifest management |
| Pillow (PIL) | Image loading and processing |
| Semantic Loss | Differentiable logical regularization |

---

**Neuro-Symbolic AI - Card Suit Game with Semantic Loss**

*Academic Year 2026/2027*
