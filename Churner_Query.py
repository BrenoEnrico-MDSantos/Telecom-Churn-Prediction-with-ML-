import numpy as np
import pandas as pd

def robust_cluster_profiling(df, cluster_col='Clusters', alpha=1.0, min_freq=0.1):
    """
    Calcula métricas de perfilamento robustas para evitar contradições em datasets pequenos.
    
    Parâmetros:
    - alpha: Constante de suavização de Laplace (1.0 é o padrão).
    - min_freq: Limiar mínimo (10%) para ignorar ruído estatisticamente irrelevante no cluster.
    """
    results = []
    features = [col for col in df.columns if col != cluster_col]
    global_total = len(df)

    for feature in features:
        global_counts = df[feature].value_counts()
        
        for cluster in df[cluster_col].unique():
            cluster_df = df[df[cluster_col] == cluster]
            n = len(cluster_df)
            cluster_counts = cluster_df[feature].value_counts()

            for val in global_counts.index:
                x = cluster_counts.get(val, 0)
                freq = x / n
                
                # Tweak 1: Filtro de Frequência Mínima
                if freq < min_freq:
                    continue

                # Tweak 2: Suavização de Laplace (Probabilidades ajustadas)
                p_cluster_smooth = (x + alpha) / (n + alpha * 2)
                
                x_out = global_counts[val] - x
                n_out = global_total - n
                p_out_smooth = (x_out + alpha) / (n_out + alpha * 2)

                # Tweak 3: Log-Odds Suavizado
                # Valores positivos indicam "Likes", negativos indicam "Dislikes"
                log_odds = np.log(p_cluster_smooth / (1 - p_cluster_smooth)) - \
                           np.log(p_out_smooth / (1 - p_out_smooth))

                # Tweak 4: Lift (Frequência no cluster vs Global)
                lift = freq / (global_counts[val] / global_total)

                results.append({
                    'Cluster': cluster,
                    'Feature': f"{feature}_{val}",
                    'Frequency': round(freq, 3),
                    'Log_Odds_Smooth': round(log_odds, 3),
                    'Lift': round(lift, 3)
                })

    return pd.DataFrame(results).sort_values(['Cluster', 'Log_Odds_Smooth'], ascending=[True, False])

# Aplicação
profile_df = robust_cluster_profiling(df)

# Para ver o "Top" de cada cluster sem as contradições:
print(profile_df[profile_df['Log_Odds_Smooth'] > 0])


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
