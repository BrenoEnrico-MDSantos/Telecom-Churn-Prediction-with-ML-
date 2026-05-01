import pandas as pd
import numpy as np

# Supondo que 'df' é seu dataframe original e 'woe_df' contém os valores de WoE por variável
def get_state_drivers(df, target_col, feature_cols):
    state_report = []
    
    for state in df['State'].unique():
        # Filtra os dados apenas para o estado atual
        subset = df[df['State'] == state]
        churn_rate = subset[target_col].mean()
        
        # Calcula a importância/correlação local das outras variáveis para este estado
        # Aqui usamos correlação simples, mas você pode usar o valor absoluto do WoE médio
        correlations = subset[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        
        # Top Drivers de Churn (Correlação positiva com churn)
        churn_drivers = correlations[correlations > 0].sort_values(ascending=False).index[:3].tolist()
        
        # Top Drivers de Retenção (Correlação negativa com churn / Protetores)
        retention_drivers = correlations[correlations < 0].sort_values(ascending=True).index[:3].tolist()
        
        state_report.append({
            'State': state,
            'Churn_Rate': churn_rate,
            'Top_Churn_Drivers': ", ".join(churn_drivers),
            'Top_Retention_Drivers': ", ".join(retention_drivers)
        })
    
    return pd.DataFrame(state_report)

# Lista de colunas de features (excluindo State e o Alvo)
features = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Type_WoE']
report_df = get_state_drivers(df, 'Churn', features)

# Ordenar pelos estados com maior churn para ação prioritária
report_df = report_df.sort_values(by='Churn_Rate', ascending=False)
print(report_df)

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
