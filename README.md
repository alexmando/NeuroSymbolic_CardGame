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
