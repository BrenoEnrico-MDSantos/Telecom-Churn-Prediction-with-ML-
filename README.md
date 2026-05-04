# Telecom Churn Prediction with ML
Churner prediction using binary classifiers in python, alongside PowerBI for report and A/B testing.

## :ringed_planet: To use K-Medoids
- The library ```scikit-learn-extra``` is available only in **python <= 3.11**. Use a custom venv as kernel for projects, and install libraries with ```%p``` (due to compatibility, ```%pip install "numpy<2"```)
- Check all versions installed at CMD with ```py -0``` (asterisk marks the current)
    - ```dir``` for checking created venvs
- Install pip and upgrade if needed: ```python -m ensurepip --upgrade```
- Kernel creation: ```pip install ipyernel```, then ```python -m ipykernel install --user --name=venv_kmedoids --display-name "Python 3.11 (KMedoids)"```
    - For checking all kernels, ```jupyter kernelspec list```, and to uninstall: ```jupyter kernelspec uninstall venv_kmedoids```
- Activate the venv ```venv_kmedoids\Scripts\activate```
- Open JupyterLab: ```jupyter lab``` or [URL](http://localhost:8888/lab?token=04f81664309fa076b10351753c52ec47dd26265eca202945)
- Create the project with new kernel _Python 3.11 (KMedoids)_. Remember to use ```%pip install```

Uplift posting: [here](https://cmr.berkeley.edu/2025/11/to-treat-or-not-to-treat-five-lessons-learned-from-using-uplift-modeling-to-optimize-marketing-campaigns/)

# Steps
1. Extract -> Load -> Transform:
    - Understand columns for each tool, and create an average for direct monetary reading; 
    - Outlier detection and dealing with Tukey method;
    - Standardize numerical data with StandardScaler for FAMD;
2. Customer Clustering and Profiling:
    - FAMD coordinates with high explainability components fed on KMedoids model;
    - Cluster distribution and means/modes display;
    - Survivability curves with Kaplan-Meier Fitter;
    - Laplace Smoothed Odds Ratio + Lift for cluster prominent traits;
    - Labeling and expert domain strategy with AI;
        -  'State' table with top drivers
3. Churn Prediction for Recently Joined:
    - Weight of Evidence calculations for event of churn;
    - COX PH Hazard Ratios with relevant Information Value columns;
    - XGBoost + Optuna with unscaled numericals and WOEEncoded categoricals;
        - SHAP values for feature understanding;
    - 'Joined' status customers churn profiling with optimal threshold;
        - Custom function for individual cases with SHAP waterfall
4. Uplift Analysis on customers at risk:
    - A/B groups (simulated intervention and response across clusters);
    - The predicted churn labeling becomes the target for WOEEncoder;
    -  Calculation of Uplift and profiling to assess feasibility of interventions;
5. PBI Dashboards for churn situation, clusters, states and campaign predictions.   
