"""
Redundância e características textuais do WikiText-2.

Redundância:
  - parágrafos e frases duplicados (exatos) dentro do treino;
  - parágrafos da validação e do teste que também aparecem no treino (vazamento);
  - parágrafos semanticamente semelhantes, por similaridade de embeddings.

Características textuais:
  - comprimento (palavras por artigo, palavras por frase);
  - frequência de palavras (palavras mais comuns, faixas de frequência);
  - complexidade lexical (MTLD, tamanho médio das palavras e das frases).

Uso:
    python redundancia.py

Precisa do separar_artigos.py e dos arquivos wiki2_*.txt na mesma pasta.
Requer: pip install sentence-transformers scikit-learn pandas matplotlib numpy
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from separar_artigos import PARTICOES, separar_artigos, nivel_do_titulo

# ---------------------------------------------------------------- configuração
USAR_EMBEDDINGS = True    # False pula a parte semântica (mais rápido)
MODELO = "all-MiniLM-L6-v2"
LIMIAR_QUASE_DUPLICADO = 0.95   # cosseno a partir do qual dois parágrafos são "quase iguais"
LIMIAR_SEMELHANTE = 0.85        # cosseno a partir do qual são "muito semelhantes"
MIN_PALAVRAS = 5                # ignora parágrafos e frases muito curtos na redundância
CORES = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}
NOMES = {"train": "Treino", "validation": "Validação", "test": "Teste"}

RE_FIM_FRASE = re.compile(r"\s\.(?:\s|$)")
RE_MARCACAO = re.compile(r" @(.)@ ")


# ================================================================ dados
def carregar():
    """Para cada partição: lista de artigos com parágrafos e frases já limpos."""
    dados = {}
    for p in PARTICOES:
        artigos = []
        for art in separar_artigos(f"wiki2_{p}.txt"):
            paragrafos = []
            for linha in art["linhas"]:
                if not linha.strip() or nivel_do_titulo(linha)[0] is not None:
                    continue
                paragrafos.append(RE_MARCACAO.sub(r"\1", linha).strip())
            frases = [f.strip() for par in paragrafos
                      for f in RE_FIM_FRASE.split(par) if f.strip()]
            artigos.append({"titulo": art["titulo"], "paragrafos": paragrafos, "frases": frases})
        dados[p] = artigos
    return dados


def palavras(texto):
    """Palavras em minúsculas, sem pontuação."""
    return [w.lower() for w in texto.split() if any(c.isalnum() for c in w)]


def normalizar(texto):
    """Forma canônica para comparar duplicatas exatas."""
    return " ".join(palavras(texto))


# ================================================================ MTLD
def _fatores(tokens, limiar):
    fatores, tipos, n = 0.0, set(), 0
    for t in tokens:
        tipos.add(t)
        n += 1
        if len(tipos) / n <= limiar:
            fatores += 1
            tipos, n = set(), 0
    if n > 0:   # fator parcial do trecho final
        fatores += (1 - len(tipos) / n) / (1 - limiar)
    return fatores


def mtld(tokens, limiar=0.72):
    """MTLD (McCarthy e Jarvis, 2010): média das passadas direta e inversa."""
    if len(tokens) < 10:
        return np.nan
    ida = len(tokens) / max(_fatores(tokens, limiar), 1e-9)
    volta = len(tokens) / max(_fatores(tokens[::-1], limiar), 1e-9)
    return (ida + volta) / 2


# ================================================================ redundância
def redundancia_exata(dados):
    """Duplicatas exatas de parágrafos e frases."""
    print("=" * 70)
    print("REDUNDÂNCIA EXATA")
    resultados = {}
    conjuntos = {}

    for nivel in ["paragrafos", "frases"]:
        for p in PARTICOES:
            itens = [normalizar(x) for a in dados[p] for x in a[nivel]]
            itens = [x for x in itens if len(x.split()) >= MIN_PALAVRAS]
            contagem = Counter(itens)
            repetidos = sum(c - 1 for c in contagem.values() if c > 1)
            resultados[(nivel, p, "interno")] = 100 * repetidos / len(itens)
            conjuntos[(nivel, p)] = contagem

            if nivel == "paragrafos" and p == "train":
                print("  Parágrafos mais repetidos no treino:")
                for texto, c in contagem.most_common(5):
                    if c > 1:
                        print(f"    {c:>3}x  {texto[:90]}")

        # Vazamento: itens da validação/teste que também existem no treino.
        treino = conjuntos[(nivel, "train")]
        for p in ["validation", "test"]:
            itens = conjuntos[(nivel, p)]
            em_treino = sum(c for t, c in itens.items() if t in treino)
            resultados[(nivel, p, "no treino")] = 100 * em_treino / sum(itens.values())

    tabela = pd.DataFrame({
        "Parágrafos duplicados (%)": [resultados[("paragrafos", p, "interno")] for p in PARTICOES],
        "Frases duplicadas (%)": [resultados[("frases", p, "interno")] for p in PARTICOES],
        "Parágrafos também no treino (%)": [np.nan] + [resultados[("paragrafos", p, "no treino")]
                                                       for p in ["validation", "test"]],
        "Frases também no treino (%)": [np.nan] + [resultados[("frases", p, "no treino")]
                                                   for p in ["validation", "test"]],
    }, index=[NOMES[p] for p in PARTICOES])
    print("\n" + tabela.round(2).to_string(na_rep="–"))
    return tabela


def redundancia_semantica(dados):
    """Para cada parágrafo, a similaridade com o parágrafo mais parecido."""
    from sentence_transformers import SentenceTransformer

    print("\n" + "=" * 70)
    print("REDUNDÂNCIA SEMÂNTICA (embeddings)")
    modelo = SentenceTransformer(MODELO)
    textos, vetores = {}, {}
    for p in PARTICOES:
        textos[p] = [x for a in dados[p] for x in a["paragrafos"] if len(x.split()) >= MIN_PALAVRAS]
        print(f"  gerando embeddings de {len(textos[p]):,} parágrafos ({p})...")
        vetores[p] = modelo.encode(textos[p], normalize_embeddings=True, batch_size=64,
                                   show_progress_bar=False).astype(np.float32)

    def vizinho_mais_proximo(A, B, mesmo_conjunto):
        maximos, indices = np.empty(len(A)), np.empty(len(A), dtype=int)
        for ini in range(0, len(A), 1024):
            S = A[ini:ini + 1024] @ B.T
            if mesmo_conjunto:   # não comparar o parágrafo com ele mesmo
                S[np.arange(len(S)), np.arange(ini, ini + len(S))] = -1
            indices[ini:ini + len(S)] = S.argmax(axis=1)
            maximos[ini:ini + len(S)] = S.max(axis=1)
        return maximos, indices

    sims = {}
    sims["Treino × treino"], idx = vizinho_mais_proximo(vetores["train"], vetores["train"], True)
    for p in ["validation", "test"]:
        sims[f"{NOMES[p]} × treino"], _ = vizinho_mais_proximo(vetores[p], vetores["train"], False)

    linhas = []
    for nome, s in sims.items():
        linhas.append({"Comparação": nome,
                       "Similaridade mediana": np.median(s),
                       f"Quase duplicados ≥{LIMIAR_QUASE_DUPLICADO} (%)": 100 * (s >= LIMIAR_QUASE_DUPLICADO).mean(),
                       f"Muito semelhantes ≥{LIMIAR_SEMELHANTE} (%)": 100 * (s >= LIMIAR_SEMELHANTE).mean()})
    tabela = pd.DataFrame(linhas).set_index("Comparação")
    print("\n" + tabela.round(3).to_string())

    s = sims["Treino × treino"]
    print("\n  Exemplos de pares muito semelhantes no treino (não idênticos):")
    candidatos = [i for i in np.argsort(-s)
                  if normalizar(textos["train"][i]) != normalizar(textos["train"][idx[i]])][:3]
    for i in candidatos:
        print(f"    cosseno {s[i]:.2f}")
        print(f"      a) {textos['train'][i][:100]}")
        print(f"      b) {textos['train'][idx[i]][:100]}")
    return tabela, sims


# ================================================================ características
def caracteristicas(dados):
    """Uma linha por artigo com comprimento e complexidade lexical."""
    linhas, comprimentos_frases = [], {}
    for p in PARTICOES:
        comprimentos_frases[p] = []
        for a in dados[p]:
            tokens = [w for par in a["paragrafos"] for w in palavras(par)]
            if not tokens:
                continue
            tam_frases = [len(palavras(f)) for f in a["frases"] if palavras(f)]
            comprimentos_frases[p].extend(tam_frases)
            linhas.append({
                "particao": p, "titulo": a["titulo"],
                "palavras": len(tokens),
                "frases": len(tam_frases),
                "palavras_por_frase": np.mean(tam_frases),
                "letras_por_palavra": np.mean([len(w) for w in tokens]),
                "ttr": len(set(tokens)) / len(tokens),
                "mtld": mtld(tokens),
            })
    df = pd.DataFrame(linhas)

    print("\n" + "=" * 70)
    print("CARACTERÍSTICAS TEXTUAIS (média por artigo)")
    resumo = df.groupby("particao", sort=False)[
        ["palavras", "frases", "palavras_por_frase", "letras_por_palavra", "ttr", "mtld"]].mean()
    resumo.index = [NOMES[p] for p in resumo.index]
    print(resumo.round(2).to_string())
    return df, comprimentos_frases


def frequencias(dados):
    """Contagem de palavras do treino e faixas de frequência."""
    contagem = Counter(w for a in dados["train"] for par in a["paragrafos"] for w in palavras(par))
    total = sum(contagem.values())

    faixas = [("1 vez", 1, 1), ("2–10", 2, 10), ("11–100", 11, 100),
              ("101–1.000", 101, 1000), ("> 1.000", 1001, np.inf)]
    tabela = pd.DataFrame([{
        "Faixa de frequência": nome,
        "% do vocabulário": 100 * sum(1 for c in contagem.values() if lo <= c <= hi) / len(contagem),
        "% dos tokens": 100 * sum(c for c in contagem.values() if lo <= c <= hi) / total,
    } for nome, lo, hi in faixas]).set_index("Faixa de frequência")

    print("\n" + "=" * 70)
    print(f"FREQUÊNCIA DE PALAVRAS (treino): {total:,} tokens, {len(contagem):,} palavras distintas")
    print(tabela.round(2).to_string())
    return contagem, tabela


# ================================================================ gráficos
def grafico_redundancia(tab_exata, sims):
    n = 2 if sims else 1
    fig, eixos = plt.subplots(1, n, figsize=(7 * n, 4.8))
    eixos = np.atleast_1d(eixos)

    ax = eixos[0]
    colunas = list(tab_exata.columns)
    x = np.arange(len(colunas))
    for i, p in enumerate(PARTICOES):
        valores = tab_exata.loc[NOMES[p]].values
        ok = ~np.isnan(valores)   # o treino não tem as colunas de vazamento
        barras = ax.bar((x + (i - 1) * 0.27)[ok], valores[ok], 0.27, color=CORES[p], label=NOMES[p])
        ax.bar_label(barras, fmt="%.1f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" (%)", "").replace(" também", "\ntambém")
                        .replace(" duplic", "\nduplic") for c in colunas], fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Duplicatas exatas e vazamento para o treino")
    ax.legend()

    if sims:
        ax = eixos[1]
        cores = [CORES["train"], CORES["validation"], CORES["test"]]
        for (nome, s), cor in zip(sims.items(), cores):
            ax.hist(s, bins=60, range=(0, 1), density=True, alpha=0.5, color=cor, label=nome)
        ax.axvline(LIMIAR_SEMELHANTE, color="gray", linestyle="--", label=f"muito semelhante ({LIMIAR_SEMELHANTE})")
        ax.axvline(LIMIAR_QUASE_DUPLICADO, color="black", linestyle="--", label=f"quase duplicado ({LIMIAR_QUASE_DUPLICADO})")
        ax.set_xlabel("similaridade com o parágrafo mais parecido do treino (cosseno)")
        ax.set_ylabel("densidade")
        ax.set_title("Redundância semântica entre parágrafos")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("fig_redundancia.png", dpi=150)
    plt.close(fig)


def grafico_comprimento(df, comprimentos_frases):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    bins = np.logspace(np.log10(max(df["palavras"].min(), 1)), np.log10(df["palavras"].max()), 30)
    for p in PARTICOES:
        ax1.hist(df.loc[df["particao"] == p, "palavras"], bins=bins, density=True,
                 alpha=0.5, color=CORES[p], label=NOMES[p])
        ax2.hist(comprimentos_frases[p], bins=60, range=(0, 80), density=True,
                 alpha=0.5, color=CORES[p], label=NOMES[p])
    ax1.set_xscale("log")
    ax1.set_xlabel("palavras por artigo (escala log)")
    ax1.set_ylabel("densidade")
    ax1.set_title("Comprimento dos artigos")
    ax1.legend()
    ax2.set_xlabel("palavras por frase")
    ax2.set_ylabel("densidade")
    ax2.set_title("Comprimento das frases")
    ax2.legend()
    fig.tight_layout()
    fig.savefig("fig_comprimento.png", dpi=150)
    plt.close(fig)


def grafico_frequencia(contagem, tab_faixas):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

    conteudo = [(w, c) for w, c in contagem.most_common(300)
                if w not in ENGLISH_STOP_WORDS and w.isalpha()][:20]
    total = sum(contagem.values())
    ax1.barh([w for w, _ in conteudo][::-1], [100 * c / total for _, c in conteudo][::-1],
             color=CORES["train"])
    ax1.set_xlabel("% dos tokens do treino")
    ax1.set_title("20 palavras de conteúdo mais frequentes (sem stopwords)")

    x = np.arange(len(tab_faixas))
    b1 = ax2.bar(x - 0.2, tab_faixas["% do vocabulário"], 0.4, label="% do vocabulário", color="#8172B2")
    b2 = ax2.bar(x + 0.2, tab_faixas["% dos tokens"], 0.4, label="% dos tokens", color="#CCB974")
    ax2.bar_label(b1, fmt="%.1f", fontsize=8)
    ax2.bar_label(b2, fmt="%.1f", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tab_faixas.index)
    ax2.set_xlabel("quantas vezes a palavra aparece no treino")
    ax2.set_ylabel("%")
    ax2.set_title("Poucas palavras cobrem quase todo o texto")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("fig_frequencia.png", dpi=150)
    plt.close(fig)


def grafico_complexidade(df):
    medidas = [("mtld", "MTLD (diversidade lexical)"),
               ("palavras_por_frase", "Palavras por frase (média do artigo)"),
               ("letras_por_palavra", "Letras por palavra (média do artigo)")]
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, titulo) in zip(eixos, medidas):
        dados = [df.loc[df["particao"] == p, col].dropna() for p in PARTICOES]
        caixa = ax.boxplot(dados, patch_artist=True, widths=0.6)
        for patch, p in zip(caixa["boxes"], PARTICOES):
            patch.set_facecolor(CORES[p])
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([NOMES[p] for p in PARTICOES])
        ax.set_title(titulo)
    fig.suptitle("Complexidade lexical por artigo")
    fig.tight_layout()
    fig.savefig("fig_complexidade.png", dpi=150)
    plt.close(fig)


# ================================================================ principal
def main():
    dados = carregar()

    tab_exata = redundancia_exata(dados)
    tab_sem, sims = (redundancia_semantica(dados) if USAR_EMBEDDINGS else (None, None))

    df, comprimentos_frases = caracteristicas(dados)
    contagem, tab_faixas = frequencias(dados)

    grafico_redundancia(tab_exata, sims)
    grafico_comprimento(df, comprimentos_frases)
    grafico_frequencia(contagem, tab_faixas)
    grafico_complexidade(df)

    df.to_csv("caracteristicas_artigos.csv", index=False)
    print("\nGráficos salvos: fig_redundancia.png, fig_comprimento.png, "
          "fig_frequencia.png, fig_complexidade.png")
    print("Tabela por artigo salva em caracteristicas_artigos.csv")


if __name__ == "__main__":
    main()