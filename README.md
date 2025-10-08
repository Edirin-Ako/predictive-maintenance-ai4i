# Predictive Maintenance (AI4I 2020)

**Problem**  
This project explores how sensor data can be used to predict machine failures before they happen.  
The idea is to help reduce unexpected downtime by giving maintenance teams enough time to plan repairs in advance.

**Data**  
AI4I 2020 Predictive Maintenance (UCI ML Repository). Synthetic but modeled after real industrial data.
- Download CSV: `ai4i2020.csv` from the UCI page and place it in `data/` folder.
- Contains ~10,000 rows, 14 columns (temperature, rotational speed, torque, tool wear, and failure flags).

**Method (plan)**  
1. EDA: distributions, correlations, label balance, leakage checks.  
2. Target framing: `failure in next 7 days` (binary) or `failure_now` as provided.  
3. Baselines: majority class; logistic regression with class weights.  
4. Models: tree-based (RandomForest, GradientBoosting) and calibrated logistic.  
5. Validation: stratified CV and temporal split sensitivity check.  
6. Metrics: PR-AUC, ROC-AUC, Recall at fixed Precision (operating point).  
7. Explainability: SHAP/feature importances.  
8. Business impact: translate operating point/results to avoided downtime.

**Results** *(to be updated)*  
| Model | PR-AUC | Recall @ 90% Precision | Notes |
|------|--------|-------------------------|-------|
Evaluation based on PR-AUC and recall at 90% precision to reflect real maintenance trade-offs.

**Business Impact (example)**  
At 90% precision, recall improved by +18% vs baseline, providing **~8 days** average lead-time, enabling scheduled repairs and reduced surprise outages.

**How to run**  
```bash
pip install -r requirements.txt
# Open notebook.ipynb and run all cells
```
Creates a `plots/` folder with saved figures for the README

**Next steps**  
- Add cost-sensitive thresholding (cost of FN vs FP).  
- Check for sensor drift over time.  
- Experiment with gradient boosted models (XGBoost/LightGBM) for comparison.

---

> Data source: *[UCI Machine Learning Repository – AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)*
