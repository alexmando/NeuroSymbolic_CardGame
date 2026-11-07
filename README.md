NeuroSymbolic AI - Card Suit Game 🃏

University project for the Neuro-Symbolic AI course.
The primary objective of this project is to develop and compare two different AI systems capable of predicting the outcome of a card game by observing only the images of the two played cards.

📜 Game Rules

The game relies exclusively on card suits, entirely ignoring numerical ranks and face cards. It follows a circular logic:

Hearts beats Clubs

Clubs beats Spades

Spades beats Hearts

Two cards of the same suit result in a Draw

Note: The Diamonds suit and Jokers are strictly excluded from this game.

🏗️ Project Architecture & Comparison

To evaluate the benefits of neuro-symbolic integration, we will implement and compare two distinct approaches:

Approach A (DeepProbLog): A pure neuro-symbolic system. A Convolutional Neural Network (CNN) will handle the perception task (assigning a probability distribution over the suits of a single card). The ProbLog logic engine will then deduce the final outcome by applying explicit symbolic game rules to those probabilities.

Approach B (Neural Baseline): A purely sub-symbolic system (e.g., a Siamese Network). The neural network will process both card images simultaneously and attempt to predict the final outcome directly (Player 1 Wins, Player 2 Wins, or Draw), relying solely on data-driven learning without any explicit logical rules.

📂 Dataset & Transformation Strategy ("The Manifest")

The base dataset used is the Cards Image Dataset-Classification from Kaggle.

Data Preparation:
The raw dataset has been heavily filtered to meet the project requirements. We removed all Diamonds and Jokers, and we aggregated the images solely by suit (ignoring ranks).

The Manifest Strategy:
To ensure a scientifically valid comparison—especially for the required Data Efficiency experiments (evaluating the models on 20%, 50%, and 100% of the training data)—we cannot rely on random sampling at runtime.
Instead, the dataset has been transformed into fixed pairs of cards with pre-calculated outcomes. These pairs are saved as offline CSV files (Manifests) located in the manifests/ folder:

train_100.csv (Full training set)

train_50.csv (50% of the training set)

train_20.csv (20% of the training set)

test.csv (Fixed test set)

⚠️ Technical Guidelines for the Team

To guarantee consistency across all experiments, please adhere to the following rules during development:

Strict Dataloader Rule: Whether you are building the Neural Baseline or DeepProbLog, you MUST load the images into your PyTorch Dataset by reading exclusively from the aforementioned CSV files. Do not generate random pairs on the fly. This guarantees both models are trained and evaluated on the exact same card pairings.

Reproducibility: Always set the random seeds at the beginning of your scripts (torch.manual_seed(42), random.seed(42), np.random.seed(42)) to ensure reproducible weight initialization.

Evaluation Metrics: The models will be compared based on Accuracy, Macro F1-Score, and Training Time/Computational Efficiency across all three training subsets (20%, 50%, 100%).
