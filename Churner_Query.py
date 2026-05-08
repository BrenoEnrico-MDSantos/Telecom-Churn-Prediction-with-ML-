import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklift.models import SoloModel
from sklearn.model_selection import train_test_split
from category_encoders import WOEEncoder

# 1. SETUP E DISTRIBUIÇÃO EQUILIBRADA
np.random.seed(42)
n = 10000

data = {
    'Cluster': np.random.choice(['Premium', 'Budget', 'Standard', 'Retention_Risk'], n),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
    'Payment': np.random.choice(['Auto', 'Manual'], n),
    'Churn_Score_Base': np.random.uniform(20, 80, n),
    'Tenure_Months': np.random.randint(1, 72, n)
}
df = pd.DataFrame(data)

# Garante probabilidade 50/50 de tratamento em todos os clusters (RCT perfeito)
df['treatment'] = np.random.binomial(1, 0.5, n)

# 2. LÓGICA DE RESPOSTA ESCALÁVEL (Matriz de Coeficientes)
# Criamos um "Vetor de Persuabilidade" individual
# Cada variável contribui para o quanto o cliente é sensível ao tratamento (Uplift)
X_dummy = pd.get_dummies(df[['Cluster', 'Contract', 'Payment']], drop_first=True)

# Coeficientes aleatórios para simular a "verdade oculta" do mercado
# Alguns positivos (sensíveis), outros negativos (irritáveis/Sleeping Dogs)
coefs = np.random.uniform(-0.5, 0.5, X_dummy.shape[1])
sensibilidade_base = X_dummy.dot(coefs)

# O efeito do tratamento (uplift) é uma função da sensibilidade + ruído individual
df['individual_uplift'] = (sensibilidade_base * 20) + np.random.normal(0, 5, n)

# Churn Final: Probabilístico (Logit-like)
# Se recebeu tratamento, reduzimos o score base pelo uplift individual
df['score_final'] = df['Churn_Score_Base'] - (df['treatment'] * df['individual_uplift'])

# Transforma score em probabilidade e então em label binário
prob = 1 / (1 + np.exp(-(df['score_final'] - 50) / 10))
df['Churn_Label'] = np.random.binomial(1, prob)

# 3. MODELAGEM (SoloLearner)
X = df[['Cluster', 'Contract', 'Payment', 'Churn_Score_Base', 'Tenure_Months']]
y = df['Churn_Label']
treat = df['treatment']

X_train, X_test, y_train, y_test, tr_train, tr_test = train_test_split(X, y, treat, test_size=0.3)

encoder = WOEEncoder(cols=['Cluster', 'Contract', 'Payment'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

sm = SoloModel(XGBClassifier(n_estimators=100, learning_rate=0.1))
sm.fit(X_train_encoded, y_train, tr_train)

# 4. AVALIAÇÃO POR DECIL
uplift_scores = sm.predict_tau(X_test_encoded)
eval_df = pd.DataFrame({'uplift': uplift_scores, 'target': y_test, 'tr': tr_test})
eval_df['decile'] = pd.qcut(eval_df['uplift'], 10, labels=False)

report = eval_df.groupby('decile').apply(
    lambda x: x[x['tr'] == 0]['target'].mean() - x[x['tr'] == 1]['target'].mean()
).reset_index(name='Real_Uplift').sort_values('decile', ascending=False)

print(report)




import pandas as pd
import numpy as np
from sklift.models import SoloModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from category_encoders import WOEEncoder

# 1. Definimos o Encoder focado no CHURN (target real/profiling)
encoder = WOEEncoder(cols=['Contract', 'PaymentMethod', 'Cluster'])

# 2. Fit apenas nos dados de treino para evitar leakage
X_train_encoded = encoder.fit_transform(X_train, y_train) # y_train = Churn_Label

# 3. O SoloModel recebe os dados com WoE + a flag de tratamento
sm.fit(X_train_encoded, y_train, treatment_train)

import pandas as pd
import numpy as np
from category_encoders import WOEEncoder
from sklift.models import SoloModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 1. Simulação do seu Dataset (df_joined)
np.random.seed(42)
n = 1000
data = {
    'Cluster': np.random.choice(['C1', 'C2', 'C3', 'C4', 'C5'], n),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], n),
    'Churn_Score': np.random.uniform(0, 100, n), # Seu score original
    'treatment': np.random.choice([0, 1], n)     # Simulação da oferta A/B
}
df = pd.DataFrame(data)

# Criando o Churn_Label (1 se Score > cutoff de 70, simulando seu processo)
df['Churn_Label'] = (df['Churn_Score'] > 70).astype(int)

# 2. Divisão de Treino e Teste
X = df.drop(['Churn_Label'], axis=1)
y = df['Churn_Label']
treat = df['treatment']

X_train, X_test, y_train, y_test, tr_train, tr_test = train_test_split(
    X, y, treat, test_size=0.3, random_state=42
)

# 3. Mapeamento WoE (Target é o Churn_Label)
# Nota: tr_train não entra no WoE, apenas as features categóricas e o target Y
cat_cols = ['Cluster', 'Contract', 'PaymentMethod']
encoder = WOEEncoder(cols=cat_cols)
X_train_encoded = encoder.fit_transform(X_train.drop('treatment', axis=1), y_train)
X_test_encoded = encoder.transform(X_test.drop('treatment', axis=1))

# 4. Implementação do S-Learner (Uplift)
# Usamos o XGBoost para aprender a interação entre WoE, Scores e Tratamento
sm = SoloModel(XGBClassifier(random_state=42))
sm.fit(X_train_encoded, y_train, tr_train)

# 5. Predição do Uplift Score
# uplift = P(Churn|Controle) - P(Churn|Tratamento)
# Score positivo = A oferta REDUZIU a probabilidade de churn.
uplift_scores = sm.predict_tau(X_test_encoded)

# 6. Profiling e Segmentação (Mapeamento de Tipos de Clientes)
results = X_test.copy()
results['uplift_score'] = uplift_scores
results['churn_label'] = y_test.values

def segment_customers(row):
    # Definindo limiares (podem ser ajustados conforme a distribuição)
    high_uplift = 0.1  # Sensível à oferta
    low_uplift = -0.1  # Reação negativa à oferta
    
    if row['uplift_score'] > high_uplift:
        return 'Persuadable'      # O alvo de ouro
    elif row['uplift_score'] < low_uplift:
        return 'Sleeping Dog'     # Não toque neles!
    else:
        if row['churn_label'] == 0:
            return 'Sure Thing'    # Vai ficar de qualquer jeito
        else:
            return 'Lost Cause'    # Vai sair de qualquer jeito

results['Segment'] = results.apply(segment_customers, axis=1)

# Exibição dos Resultados
print("Distribuição dos Segmentos de Uplift:")
print(results['Segment'].value_counts(normalize=True) * 100)

print("\nUplift Médio por Cluster (WoE-based):")
print(results.groupby('Cluster')['uplift_score'].mean().sort_values(ascending=False))
)


import pandas as pd
import shap
import matplotlib.pyplot as plt

def cockpit_atendimento(customer_id, df_flagged, df_states, X_data, explainer):
    # 1. Recuperar dados de predição do cliente
    try:
        cliente_info = df_flagged.loc[customer_id]
        prob_churn = cliente_info['churn_probability']
    except KeyError:
        return "Cliente não encontrado na base de risco."

    # 2. Recuperar dados contextuais do Estado
    estado_cliente = X_data.loc[customer_id, 'State_Original_Name'] # Ajuste para o nome da sua coluna
    info_estado = df_states[df_states['State'] == estado_cliente].iloc[0]

    # 3. Gerar Explicação SHAP (Local)
    # Calculamos o SHAP apenas para este cliente específico
    shap_values_custom = explainer(X_data.loc[[customer_id]])

    # --- OUTPUT PARA O CRM ---
    print(f"=== PERFIL DE RETENÇÃO: CLIENTE {customer_id} ===")
    print(f"Probabilidade de Churn: {prob_churn:.2%}")
    print(f"Estado de Origem: {estado_cliente} (Churn Médio do Estado: {info_estado['Churn_Rate']:.2%})")
    print(f"\nCONTEXTO REGIONAL ({estado_cliente}):")
    print(f"Principais Causas no Estado: {info_estado['Top_Churn_Drivers']}")
    print(f"Fatores de Retenção no Estado: {info_estado['Top_Retention_Drivers']}")
    
    print(f"\n--- POR QUE ESTE CLIENTE ESPECÍFICO VAI SAIR? (SHAP) ---")
    # O waterfall ou bar plot local é melhor para atendimento individual que o swarmplot
    shap.plots.bar(shap_values_custom[0]) 

# Exemplo de uso:
# cockpit_atendimento('ID_12345', df_flagged, report_df, X_test, explainer)

import pandas as pd

def cockpit_atendimento_v2(customer_id, df_flagged, df_states, X_data, explainer_values, feature_names):
    """
    Versão textual para CRM com tradução de SHAP para argumentos de negócio.
    """
    try:
        # 1. Localizar o índice do cliente nos shap_values
        idx = X_data.index.get_loc(customer_id)
        cliente_shap = explainer_values[idx]
        
        # 2. Informações de Probabilidade e Estado
        prob = df_flagged.loc[customer_id, 'churn_probability']
        estado_cliente = X_data.loc[customer_id, 'State'] # Nome da coluna de estado original
        info_estado = df_states[df_states['State'] == estado_cliente].iloc[0]

        # 3. Converter SHAP em texto (Top 3 drivers individuais)
        # Criamos um dict de Feature: Impacto
        impactos = dict(zip(feature_names, cliente_shap.values))
        # Ordenamos pelo valor absoluto do impacto (o que mais pesou na decisão)
        top_drivers = sorted(impactos.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        print(f"--- RELATÓRIO DE RETENÇÃO PARA O ATENDENTE ---")
        print(f"CLIENTE: {customer_id} | RISCO: {prob:.1%}")
        print(f"CONTEXTO: Estado {estado_cliente} (Churn da Região: {info_estado['Churn_Rate']:.1%})")
        print("-" * 50)
        
        print("MOTIVOS INDIVIDUAIS (POR QUE ELE QUER SAIR?):")
        for feature, impact in top_drivers:
            direcao = "AUMENTA o risco" if impact > 0 else "REDUZ o risco"
            valor_real = X_data.loc[customer_id, feature]
            print(f" - {feature} ({valor_real}): {direcao}")

        # 4. RECOMENDAÇÃO AUTOMÁTICA (A última granularidade)
        driver_principal = top_drivers[0][0]
        print("-" * 50)
        print("AÇÃO RECOMENDADA:")
        if "Charges" in driver_principal or "Price" in driver_principal:
            print(">> Oferecer downgrade de plano ou desconto temporário (Driver: Preço).")
        elif "Tenure" in driver_principal:
            print(">> Oferecer bônus de fidelidade (Driver: Tempo de Casa baixo).")
        elif "Contract" in driver_principal:
            print(">> Tentar migração para plano anual com benefícios (Driver: Contrato).")
        else:
            print(">> Sondagem geral: foco em qualidade e estabilidade do serviço.")

    except Exception as e:
        print(f"Erro ao processar ID {customer_id}: {e}")

# Para rodar, você precisará dos shap_values calculados:
# shap_values = explainer(X_test)
# cockpit_atendimento_v2('ID_CLIENTE', df_flagged, report_df, X_test, shap_values, X_test.columns)
