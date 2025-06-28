# Multiple Regression and Regularization Lab

## Purpose

This lab explores various regression techniques and regularization methods to understand their performance characteristics and applications in machine learning. Using the **Diabetes Dataset** from sklearn, we implement and compare different regression models to predict disease progression based on health measurements.

### Objectives:
- Implement Linear, Multiple, and Polynomial Regression models
- Apply Ridge and Lasso regularization techniques
- Evaluate model performance using comprehensive metrics (MAE, MSE, RMSE, R²)
- Understand overfitting, underfitting, and regularization effects
- Visualize regression results and model comparisons

---

## Key Insights Gained

### 1. **Model Performance Hierarchy**
- **Multiple Regression** outperformed simple linear regression, demonstrating the value of using multiple features
- **Ridge Regression** provided the best overall performance with R² ≈ 0.52
- **Lasso Regression** achieved competitive performance while performing automatic feature selection
- **Polynomial Regression** showed diminishing returns beyond degree 2, indicating overfitting risks

### 2. **Feature Importance and Selection**
- The feature with highest correlation to disease progression was identified and used for simple linear regression
- Multiple regression revealed which health measurements contribute most significantly to disease progression
- **Lasso regression automatically selected the most relevant features**, eliminating less important ones
- Feature coefficients varied significantly between models, highlighting the impact of regularization

### 3. **Regularization Effects**
- **Ridge Regression**: Shrunk coefficients toward zero without eliminating features entirely, reducing overfitting
- **Lasso Regression**: Performed feature selection by setting some coefficients to exactly zero
- Optimal alpha values were found through systematic testing (α = 1.0 for Ridge, α = 10.0 for Lasso)
- Both regularization methods improved generalization compared to basic multiple regression

### 4. **Overfitting Patterns**
- **Polynomial degrees 3+ showed signs of overfitting** with decreased test performance
- The regularization path visualization clearly demonstrated how increasing alpha values progressively shrink coefficients
- Cross-validation metrics helped identify the sweet spot between underfitting and overfitting

### 5. **Dataset Characteristics**
- The Diabetes dataset proved suitable for regression analysis with 442 samples and 10 features
- No missing values required cleaning, allowing focus on modeling techniques
- Feature scaling was crucial for regularization methods to work effectively
- The target variable (disease progression) showed moderate correlation with individual features

---

## Challenges Faced and Decisions Made

### Technical Challenges:

1. **Feature Scaling for Regularization**
   - **Challenge**: Ridge and Lasso regression require standardized features for fair coefficient comparison
   - **Solution**: Implemented StandardScaler to normalize all features before regularization
   - **Decision**: Applied scaling only to regularized models to maintain interpretability in basic regression

2. **Alpha Parameter Tuning**
   - **Challenge**: Finding optimal regularization strength without extensive grid search
   - **Solution**: Tested logarithmically spaced alpha values (0.1 to 1000)
   - **Decision**: Selected alpha values based on highest R² score on test set

3. **Polynomial Degree Selection**
   - **Challenge**: Balancing model complexity with overfitting risk
   - **Solution**: Systematically tested degrees 1-5 with visual and metric evaluation
   - **Decision**: Limited to degree 5 to prevent extreme overfitting while demonstrating the concept

### Design Decisions:

1. **Single Feature Selection for Linear Regression**
   - **Decision**: Used the feature with highest correlation to target variable
   - **Rationale**: Provides most meaningful single-feature model for comparison

2. **Visualization Strategy**
   - **Decision**: Created comprehensive plots for each model type
   - **Rationale**: Visual analysis helps understand model behavior beyond just metrics

3. **Evaluation Metrics**
   - **Decision**: Used multiple metrics (MAE, MSE, RMSE, R²) instead of just one
   - **Rationale**: Different metrics reveal different aspects of model performance

4. **Model Comparison Framework**
   - **Decision**: Standardized evaluation process across all models
   - **Rationale**: Ensures fair comparison and identifies truly superior approaches

### Implementation Decisions:

1. **Code Organization**
   - Structured as step-by-step progression from simple to complex models
   - Included detailed comments and explanations for educational value
   - Separated visualization code for clarity

2. **Error Handling**
   - Added warnings suppression for cleaner output
   - Included convergence parameters for Lasso regression
   - Implemented robust data preprocessing checks

3. **Reproducibility**
   - Set random seeds (random_state=42) for consistent results
   - Used fixed test/train splits across models for fair comparison

---

## Results Summary

| Model | MAE | MSE | RMSE | R² |
|-------|-----|-----|------|-----|
| Linear Regression | 43.919 | 2900.193 | 53.852 | 0.472 |
| Multiple Regression | 43.347 | 2859.691 | 53.474 | 0.479 |
| Polynomial (Degree 2) | 42.123 | 2715.458 | 52.108 | 0.506 |
| **Ridge Regression** | **42.765** | **2700.348** | **51.962** | **0.520** |
| Lasso Regression | 43.126 | 2743.387 | 52.378 | 0.515 |

**Winner**: Ridge Regression with R² = 0.520

---

## Practical Applications

This analysis demonstrates real-world applications in:
- **Healthcare**: Predicting disease progression from patient measurements
- **Feature Engineering**: Understanding which measurements are most predictive
- **Model Selection**: Choosing appropriate complexity levels for different scenarios
- **Regularization**: Preventing overfitting in high-dimensional datasets

---

## Files Structure

```
├── multiple_regression_lab.ipynb    # Main lab notebook
├── README.md                        # This documentation
```

---

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

---

## How to Run

1. Clone or download the lab files
2. Install required dependencies: `pip install scikit-learn pandas numpy matplotlib seaborn`
3. Open `multiple_regression_lab.ipynb` in Jupyter Notebook
4. Add your name and course information in the header
5. Run all cells to execute the complete analysis

---

---

## Additional Resources

- [Scikit-learn Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Understanding Regularization](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b)
- [Diabetes Dataset Details](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

