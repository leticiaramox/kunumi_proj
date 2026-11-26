"""
Diversidade temática do WikiText-2: número de tópicos, distribuição dos assuntos
e entropia temática, usando LDA (Latent Dirichlet Allocation).

Uso:
    python tematica.py

Precisa do separar_artigos.py e dos arquivos wiki2_*.txt na mesma pasta.
Requer: pip install scikit-learn pandas matplotlib numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from separar_artigos import PARTICOES, separar_artigos, nivel_do_titulo

# ---------------------------------------------------------------- configuração
VALORES_K = [5, 8, 10, 15, 20, 25, 30]   # números de tópicos testados
K_FIXO = None        # ponha um número (ex.: 15) para pular a busca e usar esse K
N_PALAVRAS_TOPO = 10 # palavras exibidas por tópico
SEMENTE = 42


# ---------------------------------------------------------------- dados
def documentos(particao):
    """Um documento por artigo: texto em minúsculas, só palavras, sem títulos."""
    titulos, textos = [], []
    for art in separar_artigos(f"wiki2_{particao}.txt"):
        palavras = []
        for linha in art["linhas"]:
            if not linha.strip() or nivel_do_titulo(linha)[0] is not None:
                continue
            palavras.extend(p.lower() for p in linha.split() if p.isalpha())
        titulos.append(art["titulo"])
        textos.append(" ".join(palavras))
    return titulos, textos


# ---------------------------------------------------------------- coerência
def coerencia_npmi(modelo, X_binaria, n=N_PALAVRAS_TOPO):
    """Coerência NPMI média dos tópicos (de -1 a 1; maior = tópicos mais coerentes).

    Para cada par de palavras do topo de um tópico, mede o quanto elas aparecem
    juntas nos mesmos artigos além do que se esperaria por acaso (PMI
    normalizado). Tópicos coerentes reúnem palavras que de fato andam juntas.
    """
    n_docs = X_binaria.shape[0]
    df = np.asarray(X_binaria.sum(axis=0)).ravel() / n_docs
    notas = []
    for pesos in modelo.components_:
        topo = pesos.argsort()[::-1][:n]
        conjunta = (X_binaria[:, topo].T @ X_binaria[:, topo]).toarray() / n_docs
        pares = []
        for i in range(1, n):
            for j in range(i):
                p_ij = conjunta[i, j]
                if p_ij == 0:
                    pares.append(-1.0)            # nunca aparecem juntas
                elif p_ij == 1:
                    pares.append(1.0)
                else:
                    pmi = np.log(p_ij / (df[topo[i]] * df[topo[j]]))
                    pares.append(pmi / -np.log(p_ij))
        notas.append(np.mean(pares))
    return float(np.mean(notas))


def treinar_lda(X, k):
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch",
                                    max_iter=30, random_state=SEMENTE)
    lda.fit(X)
    return lda


# ---------------------------------------------------------------- entropia
def entropia(p, k):
    """Entropia de Shannon (em bits) e a versão normalizada por log2(K).

    Normalizada: 0 = um único tema, 1 = todos os K temas com o mesmo peso.
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    h = float(-(p * np.log2(p)).sum())
    return h, h / np.log2(k)


# ---------------------------------------------------------------- análise
def main():
    dados = {p: documentos(p) for p in PARTICOES}

    # O vocabulário e o LDA são aprendidos só no treino; validação e teste são
    # projetados nos mesmos tópicos, para que as partições sejam comparáveis.
    vetorizador = CountVectorizer(stop_words="english", min_df=5, max_df=0.5)
    X = {"train": vetorizador.fit_transform(dados["train"][1])}
    for p in ["validation", "test"]:
        X[p] = vetorizador.transform(dados[p][1])
    vocabulario = vetorizador.get_feature_names_out()
    X_binaria = (X["train"] > 0).astype(int)
    print(f"Vocabulário do LDA: {len(vocabulario):,} palavras "
          f"(sem stopwords, presentes em 5+ artigos e em no máx. 50% deles)\n")

    # ------------------------------------------------ 1. Número de tópicos
    print("=" * 70)
    print("1. Número de tópicos")
    if K_FIXO:
        k = K_FIXO
        lda = treinar_lda(X["train"], k)
        print(f"  K fixado em {k}")
    else:
        busca = []
        for kk in VALORES_K:
            modelo = treinar_lda(X["train"], kk)
            busca.append({"K": kk,
                          "coerencia": coerencia_npmi(modelo, X_binaria),
                          "perplexidade_val": modelo.perplexity(X["validation"]),
                          "modelo": modelo})
            print(f"  K = {kk:>2}  coerência NPMI = {busca[-1]['coerencia']:7.3f}"
                  f"   perplexidade (validação) = {busca[-1]['perplexidade_val']:9.1f}")
        melhor = max(busca, key=lambda b: b["coerencia"])
        k, lda = melhor["K"], melhor["modelo"]
        print(f"  -> K escolhido: {k} (maior coerência)")

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot([b["K"] for b in busca], [b["coerencia"] for b in busca], "o-")
        ax1.set_xlabel("número de tópicos (K)")
        ax1.set_ylabel("coerência NPMI (maior = melhor)")
        ax1.set_title("Escolha do número de tópicos")
        ax1.axvline(k, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig("tematica_k.png", dpi=150)
        plt.close(fig)

    print(f"\n  Palavras mais prováveis de cada tópico (K = {k}):")
    for t, pesos in enumerate(lda.components_):
        topo = [vocabulario[i] for i in pesos.argsort()[::-1][:N_PALAVRAS_TOPO]]
        print(f"  T{t:<3} {', '.join(topo)}")

    # ------------------------------------------------ 2. Distribuição dos assuntos
    print("\n" + "=" * 70)
    print("2. Distribuição dos assuntos")
    registros, proporcao_media, dominantes = [], {}, {}
    for p in PARTICOES:
        theta = lda.transform(X[p])                 # mistura de tópicos de cada artigo
        proporcao_media[p] = theta.mean(axis=0)
        dominantes[p] = np.bincount(theta.argmax(axis=1), minlength=k) / len(theta)
        for titulo, mistura in zip(dados[p][0], theta):
            h, h_norm = entropia(mistura, k)
            registros.append({"particao": p, "titulo": titulo,
                              "topico_dominante": int(mistura.argmax()),
                              "peso_dominante": float(mistura.max()),
                              "entropia_bits": h, "entropia_norm": h_norm})
    df = pd.DataFrame(registros)
    df.to_csv("tematica_artigos.csv", index=False)

    tabela = pd.DataFrame({f"{p} (%)": 100 * dominantes[p] for p in PARTICOES})
    tabela.index = [f"T{t}" for t in range(k)]
    print("  Porcentagem de artigos em que cada tópico é o dominante:")
    print(tabela.round(1).to_string())
    print(f"\n  Tópicos que dominam pelo menos 1 artigo (treino): "
          f"{int((dominantes['train'] > 0).sum())} de {k}")

    # ------------------------------------------------ 3. Entropia temática
    print("\n" + "=" * 70)
    print("3. Entropia temática")
    print("  Por artigo (mistura de tópicos dentro de cada artigo):")
    print(df.groupby("particao", sort=False)["entropia_norm"]
          .agg(media="mean", mediana="median", desvio="std", minimo="min", maximo="max")
          .round(3).to_string())

    print("\n  Por partição (como os artigos se distribuem entre os tópicos):")
    print(f"  {'Partição':<12}{'H (bits)':>10}{'H normalizada':>16}")
    for p in PARTICOES:
        h, h_norm = entropia(dominantes[p] + 1e-12, k)
        print(f"  {p:<12}{h:>10.2f}{h_norm:>16.3f}")
    print(f"  (máximo possível: log2({k}) = {np.log2(k):.2f} bits)")

    print("\n  Artigos mais focados (menor entropia) e mais mistos (maior entropia):")
    treino = df[df["particao"] == "train"]
    for _, l in treino.nsmallest(3, "entropia_norm").iterrows():
        print(f"    focado  {l['entropia_norm']:.2f}  {l['titulo']}  (T{l['topico_dominante']})")
    for _, l in treino.nlargest(3, "entropia_norm").iterrows():
        print(f"    misto   {l['entropia_norm']:.2f}  {l['titulo']}")

    # ------------------------------------------------ gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    largura = 0.27
    for i, p in enumerate(PARTICOES):
        ax1.bar(np.arange(k) + (i - 1) * largura, 100 * proporcao_media[p], largura, label=p)
    ax1.set_xticks(range(k))
    ax1.set_xticklabels([f"T{t}" for t in range(k)], rotation=90 if k > 15 else 0)
    ax1.set_ylabel("% médio do conteúdo")
    ax1.set_title("Distribuição dos assuntos por partição")
    ax1.legend()

    for p in PARTICOES:
        ax2.hist(df.loc[df["particao"] == p, "entropia_norm"], bins=25,
                 range=(0, 1), density=True, alpha=0.5, label=p)
    ax2.set_xlabel("entropia temática normalizada do artigo")
    ax2.set_ylabel("densidade")
    ax2.set_title("Entropia temática por artigo")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("tematica.png", dpi=150)
    plt.close(fig)
    print("\nArquivos salvos: tematica_artigos.csv, tematica.png"
          + ("" if K_FIXO else ", tematica_k.png"))


if __name__ == "__main__":
    main()