# Movie Gross Predictor

Movies blend art and analytics—while creativity drives storytelling, financial success often hinges on quantifiable patterns. This project explores whether machine learning can bridge that gap by predicting box office revenue from features like genre trends, audience engagement, and critical reception.

## What It Does

A deep learning model that predicts a movie’s log-transformed gross revenue using 464 engineered features:
- Genre combinations (encoded into 20+ binary columns via `MultiLabelBinarizer`)
- Log-transformed votes and gross to handle skew
- Runtime, Metascores, and IMDb ratings  
Achieves **87.3% accuracy** (5-fold cross-validation) and generalizes across diverse budgets and genres.

## How I Built It

### Data Engineering
- Merged multiple CSVs, resolving column conflicts (e.g., `Certificate_x` vs. `Certificate_y`).  
- Extracted gross revenue from unstructured text with regex.  
- Applied log transforms and one-hot encoded categorical variables.

### Model Architecture
- Constructed a 3-layer feedforward neural network (256 → 128 → 64 neurons) in PyTorch.  
- Added 20% dropout and batch normalization to reduce overfitting.  
- Trained with RMSprop (learning rate = 0.0005).

### Training & Validation
- Split data 80/20 (train/validation).  
- Ran 500 epochs using MSE loss.  
- Validated via 5-fold cross-validation to ensure model robustness.

## Challenges

- **Data Preprocessing:** Standardizing inconsistent column names and handling missing values.  
- **Feature Extraction:** Parsing unstructured “Info” text to isolate revenue figures.  
- **Overfitting:** Initial models memorized training data; dropout and batch norm were critical.  
- **Abandoned Experiment:** Director/cast fame scoring via ChatGPT API proved too time-consuming.

## Accomplishments

- Engineered a robust feature set of 464 predictors from messy, real-world data.  
- Achieved 87.3% predictive accuracy on log-gross revenue.  
- Implemented cross-validation to confirm generalizability.  

## What I Learned

- **Data Quality > Model Complexity:** Feature engineering (log transforms, genre encoding) drove more gains than layer tweaks.  
- **Regularization Matters:** Dropout and batch normalization transformed an overfit model into a reliable predictor.  
- **Optimizer Choice:** RMSprop outperformed both Adam and SGD on this dataset.  
- **Feature Pruning:** The director/cast fame experiment taught me to focus on actionable data.

## What’s Next

- **API Integration:** Automate director/cast fame scoring via IMDb or ChatGPT APIs.  
- **Real-Time Metrics:** Incorporate live social media buzz and trailer views.  
- **Web App Deployment:** Package into a user-friendly interface for filmmakers and investors.  
- **Transformer Models:** Experiment with attention mechanisms (e.g., LSTMs) to capture temporal release trends.

## Built With

- [pandas](https://pandas.pydata.org/)  
- [Python](https://www.python.org/)  
- [PyTorch](https://pytorch.org/)  
- [scikit-learn](https://scikit-learn.org/)  
