# Telecom Churn Prediction with ML
Churner prediction using binary classifiers in python, alongside PowerBI for report and A/B testing.

## Steps

1. Data loading and cleaning; initial exploration with Pandas and Numpy
2. Kprototypes clustering dropping most of service-related features;
- ```.fillna()``` Median will be inputted for NaN in numericals, and mode for categoricals (if Yes/No columns, NaN can be inputted No);
- use of StandardScaler for numericals;
- custom gamma for weighted categoricals;
- Elbow plot for optimal k;
- Apply **Gower Matrix** to correctly process mixed data distance (not Euclidian nor Hamming). With the "null dealt" original df, and ordinal columns turned into numbers, pass this non-standardized df to gower.gower_matrix(). This df cannot contain numerical outliers.

! Consider using HCA: dendogram for no. k choice:

```import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# 1. Converter a matriz de Gower para o formato 'condensed'
# O scipy exige que a matriz quadrada seja convertida em um vetor (forma condensada)
condensed_dist = squareform(dist_matrix, checks=False)

# 2. Gerar o Linkage (a estrutura da árvore)
# 'average' é o método que discutimos para melhor Silhouette
Z = linkage(condensed_dist, method='average')

# 3. Plotar o Dendrograma
plt.figure(figsize=(12, 7))
plt.title('Dendrograma de Propensão ao Consumo (Gower + Average Linkage)')
plt.xlabel('Índice dos Clientes (ou tamanho do grupo)')
plt.ylabel('Distância de Gower')

dendrogram(
    Z,
    truncate_mode='lastp',  # Mostra apenas os últimos 'p' clusters para não poluir
    p=12,                   # Ver apenas os últimos 12 agrupamentos
    leaf_rotation=45.,
    leaf_font_size=10.,
    show_contracted=True    # Mostra marcas nos grupos contraídos
)

# Linha de corte sugerida (ajuste conforme o gráfico gerado)
# plt.axhline(y=0.15, color='r', linestyle='--') 

plt.show()
```

Set the HC model with ```AgglomerativeClustering``` and average ```linkage``` with gower_dist,then ```.fit_predict(gower_dist)``` and get the metrics:

```score = silhouette_score(dist_matrix, clusters, metric='precomputed')
dbi = davies_bouldin_score(df.select_dtypes(include=[np.number]), clusters)```

- Scatterplot with Factorial Analysis for Mixed Data (FAMD); 
- Spider/Radar chart for profiling.

3. Kaplan-Meier Curve for main categorical features survivability over months of tenure
4. Feature correlation and COX PH for spotting columns of highest HR
5. Auto ranking and choice of classification model with pipelines and GridSearchCV tuning; recall and f1 as main metrics for retention strat
6. Apply best model to recently joined customers and divise retention and winback measures
7. PowerBI viz with slicers; report of losses to churn, distribution across categoricals and highest Hazard Ratio assessed; prediction panel with costs of strategies and imminent losses (emphasize emergency for top clusters)

Classification Models:
1. Tuned XGBoost
2. Reg cutoff BRFC
3. Vanilla EasyEnsembleClassifier
4. ADASYN Tuned reg cutoff RFC
5. Vanilla RFC
