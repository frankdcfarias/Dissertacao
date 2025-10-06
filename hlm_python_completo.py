# ==============================================================================
# ANÁLISE HLM EM PYTHON - RADAR DA TRANSPARÊNCIA E IPS BRASIL
# ==============================================================================
#
# Este script implementa (HLM) em Python para analisar a relação
#  entre Radar da Transparência e IPS Brasil (2023-2024)
# 
# Estrutura: Tempo (Nível 1) → Municípios (Nível 2) → Estados (Nível 3)
#
# Autor: Frank Farias
# Data: Setembro 2025
# Dissertação de Mestrado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. PREPARAÇÃO DOS DADOS
# ==============================================================================

def preparar_dados_hlm():
    """Prepara dados longitudinais para análise HLM."""
    
    print("=== PREPARAÇÃO DOS DADOS PARA HLM ===\n")
    
    # Carregar dados de ambos os anos
    df_2023 = pd.read_excel('ips2023.xlsx')
    df_2024 = pd.read_excel('ips2024.xlsx')
    
    # Mapear colunas
    colunas_2023 = {
        'codigo_ibge': 'Código IBGE',
        'municipio': 'Município',
        'estado': 'Estado',
        'ips_brasil': 'IPS Brasil 2023',
        'radar_transparencia': 'Radar da transparência 2023'
    }
    
    colunas_2024 = {
        'codigo_ibge': 'Código IBGE',
        'municipio': 'Município',
        'estado': 'UF',
        'ips_brasil': 'IPS Brasil 2024',
        'radar_transparencia': 'Radar da transparência 2024'
    }
    
    # Preparar dados 2023
    dados_2023 = df_2023[list(colunas_2023.values())].copy()
    dados_2023.columns = list(colunas_2023.keys())
    dados_2023['ano'] = 2023
    dados_2023['tempo'] = 0  # Referência
    
    # Preparar dados 2024
    dados_2024 = df_2024[list(colunas_2024.values())].copy()
    dados_2024.columns = list(colunas_2024.keys())
    dados_2024['ano'] = 2024
    dados_2024['tempo'] = 1  # Um ano depois
    
    # Combinar dados
    dados_longitudinais = pd.concat([dados_2023, dados_2024], ignore_index=True)
    dados_longitudinais = dados_longitudinais.dropna()
    
    # Manter apenas municípios com ambos os anos
    municipios_completos = (dados_longitudinais.groupby('codigo_ibge')['ano']
                           .count() == 2)
    municipios_manter = municipios_completos[municipios_completos].index
    
    dados_final = dados_longitudinais[
        dados_longitudinais['codigo_ibge'].isin(municipios_manter)
    ].copy()
    
    # Criar variáveis adicionais
    dados_final['tempo_centrado'] = dados_final['tempo'] - 0.5  # -0.5, 0.5
    
    # Criar componentes between e within
    medias_municipio = dados_final.groupby('codigo_ibge').agg({
        'radar_transparencia': 'mean',
        'ips_brasil': 'mean'
    }).rename(columns={'radar_transparencia': 'radar_between',
                      'ips_brasil': 'ips_between'})
    
    dados_final = dados_final.merge(medias_municipio, left_on='codigo_ibge', right_index=True)
    dados_final['radar_within'] = dados_final['radar_transparencia'] - dados_final['radar_between']
    dados_final['ips_within'] = dados_final['ips_brasil'] - dados_final['ips_between']
    
    print(f"Dataset final: {dados_final.shape}")
    print(f"Municípios: {dados_final['codigo_ibge'].nunique()}")
    print(f"Estados: {dados_final['estado'].nunique()}")
    
    return dados_final

# ==============================================================================
# 2. ANÁLISE EXPLORATÓRIA
# ==============================================================================

def analise_exploratoria(dados):
    """Realiza análise exploratória dos dados longitudinais."""
    
    print("\n=== ANÁLISE EXPLORATÓRIA ===\n")
    
    # Estatísticas descritivas por ano
    stats_ano = dados.groupby('ano').agg({
        'ips_brasil': ['count', 'mean', 'std', 'min', 'max'],
        'radar_transparencia': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print("Estatísticas por ano:")
    print(stats_ano)
    
    # Correlações por ano
    print("\nCorrelações por ano:")
    for ano in [2023, 2024]:
        dados_ano = dados[dados['ano'] == ano]
        corr_pearson = dados_ano['radar_transparencia'].corr(dados_ano['ips_brasil'])
        corr_spearman = dados_ano['radar_transparencia'].corr(dados_ano['ips_brasil'], method='spearman')
        print(f"{ano}: Pearson = {corr_pearson:.4f}, Spearman = {corr_spearman:.4f}")
    
    # Mudanças entre anos
    dados_wide = dados.pivot(index=['codigo_ibge', 'municipio', 'estado'], 
                             columns='ano', 
                             values=['ips_brasil', 'radar_transparencia']).reset_index()
    
    dados_wide.columns = ['codigo_ibge', 'municipio', 'estado', 'ips_2023', 'ips_2024', 'radar_2023', 'radar_2024']
    dados_wide['delta_ips'] = dados_wide['ips_2024'] - dados_wide['ips_2023']
    dados_wide['delta_radar'] = dados_wide['radar_2024'] - dados_wide['radar_2023']
    
    print(f"\nMudanças 2023→2024:")
    print(f"Delta IPS: μ = {dados_wide['delta_ips'].mean():.3f}, σ = {dados_wide['delta_ips'].std():.3f}")
    print(f"Delta Radar: μ = {dados_wide['delta_radar'].mean():.3f}, σ = {dados_wide['delta_radar'].std():.3f}")
    
    # Correlação entre mudanças
    corr_mudancas = dados_wide['delta_radar'].corr(dados_wide['delta_ips'])
    print(f"Correlação entre mudanças: {corr_mudancas:.4f}")
    
    return dados_wide

# ==============================================================================
# 3. IMPLEMENTAÇÃO DE MODELOS HLM USANDO STATSMODELS
# ==============================================================================

def modelo_hlm_sequencial(dados):
    """Implementa sequência de modelos HLM usando statsmodels."""
    
    print("\n=== MODELOS HIERÁRQUICOS SEQUENCIAIS ===\n")
    
    # Dicionário para armazenar modelos
    modelos = {}
    resultados = {}
    
    # MODELO 1: Modelo Nulo (apenas intercepto aleatório por município)
    print("1. Modelo Nulo...")
    try:
        modelo1 = smf.mixedlm("ips_brasil ~ 1", data=dados, 
                             groups=dados["codigo_ibge"]).fit(reml=False)
        modelos['nulo'] = modelo1
        resultados['nulo'] = {
            'aic': modelo1.aic,
            'bic': modelo1.bic,
            'loglik': modelo1.llf
        }
        print(f"   AIC: {modelo1.aic:.2f}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    # MODELO 2: Crescimento Linear (tempo)
    print("2. Crescimento Linear...")
    try:
        modelo2 = smf.mixedlm("ips_brasil ~ tempo", data=dados,
                             groups=dados["codigo_ibge"]).fit(reml=False)
        modelos['linear'] = modelo2
        resultados['linear'] = {
            'aic': modelo2.aic,
            'bic': modelo2.bic,
            'loglik': modelo2.llf
        }
        print(f"   AIC: {modelo2.aic:.2f}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    # MODELO 3: Slopes Aleatórios (tempo varia por município)
    print("3. Slopes Aleatórios...")
    try:
        # statsmodels não suporta slopes aleatórios facilmente
        # Implementação aproximada usando interações
        modelo3 = smf.mixedlm("ips_brasil ~ tempo", data=dados,
                             groups=dados["codigo_ibge"],
                             re_formula="tempo").fit(reml=False)
        modelos['slopes'] = modelo3
        resultados['slopes'] = {
            'aic': modelo3.aic,
            'bic': modelo3.bic,
            'loglik': modelo3.llf
        }
        print(f"   AIC: {modelo3.aic:.2f}")
    except Exception as e:
        print(f"   Usando aproximação com interação município×tempo")
        # Alternativa: modelo com efeitos fixos por município
        dados['municipio_tempo'] = dados['codigo_ibge'].astype(str) + '_' + dados['tempo'].astype(str)
        modelo3 = smf.ols("ips_brasil ~ tempo + C(codigo_ibge)", data=dados).fit()
        modelos['slopes'] = modelo3
        resultados['slopes'] = {
            'aic': modelo3.aic,
            'bic': modelo3.bic,
            'loglik': modelo3.llf
        }
        print(f"   AIC: {modelo3.aic:.2f}")
    
    # MODELO 4: Efeito do Radar (between-within)
    print("4. Efeito do Radar...")
    try:
        modelo4 = smf.mixedlm("ips_brasil ~ tempo + radar_between + radar_within", 
                             data=dados,
                             groups=dados["codigo_ibge"]).fit(reml=False)
        modelos['radar'] = modelo4
        resultados['radar'] = {
            'aic': modelo4.aic,
            'bic': modelo4.bic,
            'loglik': modelo4.llf
        }
        print(f"   AIC: {modelo4.aic:.2f}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    # MODELO 5: Três níveis (estados)
    print("5. Três Níveis...")
    try:
        # Aproximação: efeitos fixos por estado
        modelo5 = smf.mixedlm("ips_brasil ~ tempo + radar_between + radar_within + C(estado)", 
                             data=dados,
                             groups=dados["codigo_ibge"]).fit(reml=False)
        modelos['tres_niveis'] = modelo5
        resultados['tres_niveis'] = {
            'aic': modelo5.aic,
            'bic': modelo5.bic,
            'loglik': modelo5.llf
        }
        print(f"   AIC: {modelo5.aic:.2f}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    # MODELO 6: Interação tempo×radar
    print("6. Interação Tempo×Radar...")
    try:
        modelo6 = smf.mixedlm("ips_brasil ~ tempo * radar_between + tempo * radar_within + C(estado)", 
                             data=dados,
                             groups=dados["codigo_ibge"]).fit(reml=False)
        modelos['interacao'] = modelo6
        resultados['interacao'] = {
            'aic': modelo6.aic,
            'bic': modelo6.bic,
            'loglik': modelo6.llf
        }
        print(f"   AIC: {modelo6.aic:.2f}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    return modelos, resultados

# ==============================================================================
# 4. ANÁLISE ALTERNATIVA USANDO SKLEARN E ANÁLISE MANUAL
# ==============================================================================

def analise_hlm_manual(dados):
    """Implementa análise HLM manual usando decomposição between-within."""
    
    print("\n=== ANÁLISE HLM MANUAL (SKLEARN) ===\n")
    
    resultados_hlm = {}
    
    # 1. ANÁLISE BETWEEN-PERSON (médias por município)
    print("1. Análise Between-Person...")
    medias_municipio = dados.groupby('codigo_ibge').agg({
        'ips_brasil': 'mean',
        'radar_transparencia': 'mean',
        'estado': 'first'
    }).reset_index()
    
    X_between = medias_municipio[['radar_transparencia']].values
    y_between = medias_municipio['ips_brasil'].values
    
    modelo_between = LinearRegression().fit(X_between, y_between)
    y_pred_between = modelo_between.predict(X_between)
    
    r2_between = r2_score(y_between, y_pred_between)
    rmse_between = np.sqrt(mean_squared_error(y_between, y_pred_between))
    
    print(f"   Coeficiente: {modelo_between.coef_[0]:.6f}")
    print(f"   Intercepto: {modelo_between.intercept_:.3f}")
    print(f"   R²: {r2_between:.4f}")
    print(f"   RMSE: {rmse_between:.3f}")
    
    resultados_hlm['between'] = {
        'coeficiente': modelo_between.coef_[0],
        'intercepto': modelo_between.intercept_,
        'r2': r2_between,
        'rmse': rmse_between
    }
    
    # 2. ANÁLISE WITHIN-PERSON (desvios das médias)
    print("\n2. Análise Within-Person...")
    X_within = dados[['radar_within']].values
    y_within = dados['ips_within'].values
    
    modelo_within = LinearRegression().fit(X_within, y_within)
    y_pred_within = modelo_within.predict(X_within)
    
    r2_within = r2_score(y_within, y_pred_within)
    rmse_within = np.sqrt(mean_squared_error(y_within, y_pred_within))
    
    print(f"   Coeficiente: {modelo_within.coef_[0]:.6f}")
    print(f"   Intercepto: {modelo_within.intercept_:.3f}")
    print(f"   R²: {r2_within:.4f}")
    print(f"   RMSE: {rmse_within:.3f}")
    
    resultados_hlm['within'] = {
        'coeficiente': modelo_within.coef_[0],
        'intercepto': modelo_within.intercept_,
        'r2': r2_within,
        'rmse': rmse_within
    }
    
    # 3. MODELO COMBINADO (between + within + tempo)
    print("\n3. Modelo Combinado...")
    X_combinado = dados[['tempo', 'radar_between', 'radar_within']].values
    y_combinado = dados['ips_brasil'].values
    
    modelo_combinado = LinearRegression().fit(X_combinado, y_combinado)
    y_pred_combinado = modelo_combinado.predict(X_combinado)
    
    r2_combinado = r2_score(y_combinado, y_pred_combinado)
    rmse_combinado = np.sqrt(mean_squared_error(y_combinado, y_pred_combinado))
    
    print(f"   Coef. Tempo: {modelo_combinado.coef_[0]:.6f}")
    print(f"   Coef. Between: {modelo_combinado.coef_[1]:.6f}")
    print(f"   Coef. Within: {modelo_combinado.coef_[2]:.6f}")
    print(f"   Intercepto: {modelo_combinado.intercept_:.3f}")
    print(f"   R²: {r2_combinado:.4f}")
    print(f"   RMSE: {rmse_combinado:.3f}")
    
    resultados_hlm['combinado'] = {
        'coef_tempo': modelo_combinado.coef_[0],
        'coef_between': modelo_combinado.coef_[1],
        'coef_within': modelo_combinado.coef_[2],
        'intercepto': modelo_combinado.intercept_,
        'r2': r2_combinado,
        'rmse': rmse_combinado
    }
    
    # 4. ANÁLISE DE MUDANÇAS (first-difference)
    print("\n4. Análise de Mudanças...")
    dados_wide = dados.pivot(index=['codigo_ibge', 'municipio', 'estado'], 
                             columns='ano', 
                             values=['ips_brasil', 'radar_transparencia']).reset_index()
    
    dados_wide.columns = ['codigo_ibge', 'municipio', 'estado', 'ips_2023', 'ips_2024', 'radar_2023', 'radar_2024']
    dados_wide['delta_ips'] = dados_wide['ips_2024'] - dados_wide['ips_2023']
    dados_wide['delta_radar'] = dados_wide['radar_2024'] - dados_wide['radar_2023']
    
    # Remover NaN
    mask = ~(np.isnan(dados_wide['delta_ips']) | np.isnan(dados_wide['delta_radar']))
    delta_ips = dados_wide.loc[mask, 'delta_ips'].values
    delta_radar = dados_wide.loc[mask, 'delta_radar'].values
    
    if len(delta_ips) > 0:
        modelo_mudancas = LinearRegression().fit(delta_radar.reshape(-1, 1), delta_ips)
        y_pred_mudancas = modelo_mudancas.predict(delta_radar.reshape(-1, 1))
        
        r2_mudancas = r2_score(delta_ips, y_pred_mudancas)
        rmse_mudancas = np.sqrt(mean_squared_error(delta_ips, y_pred_mudancas))
        
        print(f"   Coeficiente: {modelo_mudancas.coef_[0]:.6f}")
        print(f"   Intercepto: {modelo_mudancas.intercept_:.3f}")
        print(f"   R²: {r2_mudancas:.4f}")
        print(f"   RMSE: {rmse_mudancas:.3f}")
        
        resultados_hlm['mudancas'] = {
            'coeficiente': modelo_mudancas.coef_[0],
            'intercepto': modelo_mudancas.intercept_,
            'r2': r2_mudancas,
            'rmse': rmse_mudancas
        }
    
    return resultados_hlm, dados_wide

# ==============================================================================
# 5. CÁLCULO DE ICCs (INTRACLASS CORRELATION)
# ==============================================================================

def calcular_iccs(dados):
    """Calcula correlações intraclasse (ICCs) para variáveis."""
    
    print("\n=== CORRELAÇÕES INTRACLASSE (ICCs) ===\n")
    
    iccs = {}
    
    for variavel in ['ips_brasil', 'radar_transparencia']:
        # Converter para wide format
        dados_wide = dados.pivot(index='codigo_ibge', columns='ano', values=variavel)
        dados_wide = dados_wide.dropna()
        
        if dados_wide.shape[1] == 2:  # Temos ambos os anos
            # Variância total
            valores_all = dados_wide.values.flatten()
            var_total = np.var(valores_all, ddof=1)
            
            # Variância between (entre municípios)
            medias_municipio = dados_wide.mean(axis=1)
            var_between = np.var(medias_municipio, ddof=1) * 2  # * n_tempos
            
            # Variância within (temporal)
            var_within = var_total - var_between
            
            # ICC
            icc = var_between / var_total
            
            print(f"{variavel.upper()}:")
            print(f"   Variância Total: {var_total:.4f}")
            print(f"   Variância Between: {var_between:.4f}")
            print(f"   Variância Within: {var_within:.4f}")
            print(f"   ICC: {icc:.4f}")
            print()
            
            iccs[variavel] = {
                'var_total': var_total,
                'var_between': var_between,
                'var_within': var_within,
                'icc': icc
            }
    
    return iccs

# ==============================================================================
# 6. GERAÇÃO DE GRÁFICOS
# ==============================================================================

def gerar_graficos_hlm(dados, resultados_hlm, dados_wide):
    """Gera gráficos para análise HLM."""
    
    print("\n=== GERANDO GRÁFICOS HLM ===\n")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Trajetórias por estado
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    estados = sorted(dados['estado'].unique())
    cores = sns.color_palette("husl", len(estados))
    
    for i, estado in enumerate(estados):
        dados_estado = dados[dados['estado'] == estado]
        
        # Amostra de municípios para visualização
        municipios_amostra = dados_estado['codigo_ibge'].unique()[:15]
        
        for municipio in municipios_amostra:
            dados_mun = dados_estado[dados_estado['codigo_ibge'] == municipio]
            if len(dados_mun) == 2:
                anos = dados_mun['ano'].values
                ips = dados_mun['ips_brasil'].values
                axes[i].plot(anos, ips, alpha=0.3, color=cores[i], linewidth=1)
        
        # Média estadual
        media_estado = dados_estado.groupby('ano')['ips_brasil'].mean()
        axes[i].plot([2023, 2024], media_estado.values, 
                    color=cores[i], linewidth=3, marker='o', markersize=8)
        
        axes[i].set_title(f'{estado} (n={len(dados_estado)//2})', fontweight='bold')
        axes[i].set_ylabel('IPS Brasil')
        axes[i].set_xlabel('Ano')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(35, 75)
    
    # Remover subplot extra
    if len(estados) < 8:
        axes[7].remove()
    
    plt.suptitle('Trajetórias HLM: IPS Brasil por Estado (2023-2024)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('trajetorias_hlm_python.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Between vs Within Effects
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Between-person
    medias_municipio = dados.groupby('codigo_ibge').agg({
        'ips_brasil': 'mean',
        'radar_transparencia': 'mean',
        'estado': 'first'
    }).reset_index()
    
    for i, estado in enumerate(estados):
        dados_est = medias_municipio[medias_municipio['estado'] == estado]
        ax1.scatter(dados_est['radar_transparencia'], dados_est['ips_brasil'], 
                   c=cores[i], alpha=0.7, s=60, label=estado)
    
    # Linha de regressão between
    coef_between = resultados_hlm['between']['coeficiente']
    intercept_between = resultados_hlm['between']['intercepto']
    r2_between = resultados_hlm['between']['r2']
    
    x_line = np.linspace(medias_municipio['radar_transparencia'].min(),
                        medias_municipio['radar_transparencia'].max(), 100)
    y_line = intercept_between + coef_between * x_line
    ax1.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Radar da Transparência (Média)')
    ax1.set_ylabel('IPS Brasil (Média)')
    ax1.set_title(f'Between-Person\nβ = {coef_between:.6f}, R² = {r2_between:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Within-person
    for i, estado in enumerate(estados):
        dados_est = dados[dados['estado'] == estado]
        ax2.scatter(dados_est['radar_within'], dados_est['ips_within'], 
                   c=cores[i], alpha=0.7, s=60, label=estado)
    
    # Linha de regressão within
    coef_within = resultados_hlm['within']['coeficiente']
    intercept_within = resultados_hlm['within']['intercepto']
    r2_within = resultados_hlm['within']['r2']
    
    x_line_w = np.linspace(dados['radar_within'].min(),
                          dados['radar_within'].max(), 100)
    y_line_w = intercept_within + coef_within * x_line_w
    ax2.plot(x_line_w, y_line_w, 'k--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Radar da Transparência (Centrado)')
    ax2.set_ylabel('IPS Brasil (Centrado)')
    ax2.set_title(f'Within-Person\nβ = {coef_within:.6f}, R² = {r2_within:.4f}')
    ax2.grid(True, alpha=0.3)
    
    # Mudanças
    if 'mudancas' in resultados_hlm:
        ax3.scatter(dados_wide['delta_radar'], dados_wide['delta_ips'], alpha=0.6, s=50)
        
        coef_mudancas = resultados_hlm['mudancas']['coeficiente']
        intercept_mudancas = resultados_hlm['mudancas']['intercepto']
        r2_mudancas = resultados_hlm['mudancas']['r2']
        
        x_delta = np.linspace(dados_wide['delta_radar'].min(),
                             dados_wide['delta_radar'].max(), 100)
        y_delta = intercept_mudancas + coef_mudancas * x_delta
        ax3.plot(x_delta, y_delta, 'r--', linewidth=2)
        
        ax3.set_xlabel('Δ Radar da Transparência')
        ax3.set_ylabel('Δ IPS Brasil')
        ax3.set_title(f'Mudanças (2023→2024)\nβ = {coef_mudancas:.6f}, R² = {r2_mudancas:.4f}')
        ax3.grid(True, alpha=0.3)
    
    # Médias por estado ao longo do tempo
    medias_estado_tempo = dados.groupby(['estado', 'ano']).agg({
        'ips_brasil': 'mean'
    }).reset_index()
    
    for i, estado in enumerate(estados):
        dados_est = medias_estado_tempo[medias_estado_tempo['estado'] == estado]
        ax4.plot([2023, 2024], dados_est['ips_brasil'].values, 
                'o-', color=cores[i], linewidth=2, markersize=8, label=estado)
    
    ax4.set_xlabel('Ano')
    ax4.set_ylabel('IPS Brasil (Média Estadual)')
    ax4.set_title('Evolução por Estado')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Análise HLM: Between-Within Effects', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('between_within_hlm_python.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# 7. EQUAÇÕES DE REGRESSÃO
# ==============================================================================

def formular_equacoes(resultados_hlm, iccs):
    """Formula as equações de regressão descobertas."""
    
    print("\n=== EQUAÇÕES DE REGRESSÃO DESCOBERTAS ===\n")
    
    equacoes = {}
    
    # 1. Modelo Between-Person
    if 'between' in resultados_hlm:
        beta0 = resultados_hlm['between']['intercepto']
        beta1 = resultados_hlm['between']['coeficiente']
        r2 = resultados_hlm['between']['r2']
        
        print("1. MODELO BETWEEN-PERSON (Diferenças entre municípios):")
        print(f"   IPS_médio = {beta0:.3f} + {beta1:.6f} × Radar_médio")
        print(f"   R² = {r2:.4f}")
        print(f"   Interpretação: Municípios com 1 ponto a mais na transparência")
        print(f"   média têm IPS {beta1:.4f} pontos {'maior' if beta1 > 0 else 'menor'}")
        print()
        
        equacoes['between'] = {
            'formula': f"IPS_médio = {beta0:.3f} + {beta1:.6f} × Radar_médio",
            'r2': r2,
            'interpretacao': f"Efeito between-person: {beta1:.6f}"
        }
    
    # 2. Modelo Within-Person  
    if 'within' in resultados_hlm:
        beta0 = resultados_hlm['within']['intercepto']
        beta1 = resultados_hlm['within']['coeficiente']
        r2 = resultados_hlm['within']['r2']
        
        print("2. MODELO WITHIN-PERSON (Mudanças temporais):")
        print(f"   IPS_centrado = {beta0:.3f} + {beta1:.6f} × Radar_centrado")
        print(f"   R² = {r2:.4f}")
        print(f"   Interpretação: Quando um município aumenta 1 ponto na")
        print(f"   transparência, seu IPS muda {beta1:.4f} pontos")
        print()
        
        equacoes['within'] = {
            'formula': f"IPS_centrado = {beta0:.3f} + {beta1:.6f} × Radar_centrado",
            'r2': r2,
            'interpretacao': f"Efeito within-person: {beta1:.6f}"
        }
    
    # 3. Modelo Combinado HLM
    if 'combinado' in resultados_hlm:
        beta0 = resultados_hlm['combinado']['intercepto']
        beta1 = resultados_hlm['combinado']['coef_tempo']
        beta2 = resultados_hlm['combinado']['coef_between']
        beta3 = resultados_hlm['combinado']['coef_within']
        r2 = resultados_hlm['combinado']['r2']
        
        print("3. MODELO HLM COMBINADO (Multinível completo):")
        print(f"   IPS_ij = {beta0:.3f} + {beta1:.4f} × Tempo + {beta2:.6f} × Radar_between + {beta3:.6f} × Radar_within")
        print(f"   R² = {r2:.4f}")
        print(f"   onde:")
        print(f"     i = observação temporal (2023, 2024)")
        print(f"     j = município")
        print(f"     Tempo = 0 (2023), 1 (2024)")
        print(f"     Radar_between = média municipal da transparência")
        print(f"     Radar_within = desvio da média municipal")
        print()
        
        equacoes['combinado'] = {
            'formula': f"IPS_ij = {beta0:.3f} + {beta1:.4f}×Tempo + {beta2:.6f}×Radar_between + {beta3:.6f}×Radar_within",
            'r2': r2,
            'interpretacao': f"Modelo completo com efeitos temporais e espaciais"
        }
    
    # 4. Modelo de Mudanças
    if 'mudancas' in resultados_hlm:
        beta0 = resultados_hlm['mudancas']['intercepto']
        beta1 = resultados_hlm['mudancas']['coeficiente']
        r2 = resultados_hlm['mudancas']['r2']
        
        print("4. MODELO DE MUDANÇAS (First-difference):")
        print(f"   ΔIPS = {beta0:.3f} + {beta1:.6f} × ΔRadar")
        print(f"   R² = {r2:.4f}")
        print(f"   Interpretação: Para cada aumento de 1 ponto na transparência,")
        print(f"   o IPS muda {beta1:.4f} pontos entre 2023 e 2024")
        print()
        
        equacoes['mudancas'] = {
            'formula': f"ΔIPS = {beta0:.3f} + {beta1:.6f} × ΔRadar",
            'r2': r2,
            'interpretacao': f"Efeito de mudanças: {beta1:.6f}"
        }
    
    # 5. Componentes de Variância
    print("5. DECOMPOSIÇÃO DA VARIÂNCIA (ICCs):")
    for variavel, dados_icc in iccs.items():
        icc = dados_icc['icc']
        print(f"   {variavel.upper()}:")
        print(f"     ICC = {icc:.4f}")
        print(f"     {icc*100:.1f}% da variância é entre municípios")
        print(f"     {(1-icc)*100:.1f}% da variância é temporal (within)")
        print()
        
        equacoes[f'icc_{variavel}'] = {
            'formula': f"ICC = {icc:.4f}",
            'interpretacao': f"{icc*100:.1f}% da variância entre municípios"
        }
    
    return equacoes

# ==============================================================================
# 8. FUNÇÃO PRINCIPAL
# ==============================================================================

def main():
    """Executa análise HLM completa."""
    
    print("="*70)
    print("ANÁLISE HLM EM PYTHON - RADAR DA TRANSPARÊNCIA E IPS BRASIL")
    print("="*70)
    
    # 1. Preparar dados
    dados = preparar_dados_hlm()
    
    # 2. Análise exploratória
    dados_wide = analise_exploratoria(dados)
    
    # 3. Calcular ICCs
    iccs = calcular_iccs(dados)
    
    # 4. Modelos HLM usando statsmodels
    modelos_sm, resultados_sm = modelo_hlm_sequencial(dados)
    
    # 5. Análise manual complementar
    resultados_hlm, dados_wide = analise_hlm_manual(dados)
    
    # 6. Gerar gráficos
    gerar_graficos_hlm(dados, resultados_hlm, dados_wide)
    
    # 7. Formular equações
    equacoes = formular_equacoes(resultados_hlm, iccs)
    
    # 8. Salvar resultados
    print("\n=== SALVANDO RESULTADOS ===\n")
    
    # Salvar dados
    dados.to_csv('dados_hlm_python.csv', index=False)
    dados_wide.to_csv('dados_wide_python.csv', index=False)
    
    # Salvar resultados em formato JSON-like
    import json
    
    with open('resultados_hlm_python.json', 'w') as f:
        # Converter numpy para tipos serializáveis
        resultados_serial = {}
        for key, value in resultados_hlm.items():
            if isinstance(value, dict):
                resultados_serial[key] = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                        for k, v in value.items()}
            else:
                resultados_serial[key] = float(value) if isinstance(value, (np.float64, np.float32)) else value
        
        json.dump(resultados_serial, f, indent=2)
    
    with open('equacoes_hlm.json', 'w') as f:
        json.dump(equacoes, f, indent=2)
    
    with open('iccs_resultados.json', 'w') as f:
        iccs_serial = {}
        for key, value in iccs.items():
            iccs_serial[key] = {k: float(v) for k, v in value.items()}
        json.dump(iccs_serial, f, indent=2)
    
    print("Arquivos salvos:")
    print("• dados_hlm_python.csv")
    print("• dados_wide_python.csv") 
    print("• resultados_hlm_python.json")
    print("• equacoes_hlm.json")
    print("• iccs_resultados.json")
    print("• trajetorias_hlm_python.png")
    print("• between_within_hlm_python.png")
    
    print("\n" + "="*70)
    print("ANÁLISE HLM CONCLUÍDA!")
    print("="*70)
    
    return dados, resultados_hlm, equacoes, iccs

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    dados, resultados, equacoes, iccs = main()
