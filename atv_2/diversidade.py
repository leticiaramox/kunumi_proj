"""
Diversidade lexical do WikiText-2: TTR, Distinct-1, Distinct-2 e MTLD.

Uso:
    python diversidade.py

Precisa do separar_artigos.py e dos arquivos wiki2_*.txt na mesma pasta.
Requer: pip install lexicalrichness pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from lexicalrichness import LexicalRichness

from separar_artigos import PARTICOES, separar_artigos, nivel_do_titulo

LIMIAR_MTLD = 0.72   # valor padrão proposto por McCarthy e Jarvis (2010)


def palavras_do_artigo(artigo):
    """Palavras do artigo, em minúsculas e sem pontuação nem marcações como '@-@'.

    Pontuação e maiúsculas são removidas para que "The" e "the" contem como a
    mesma palavra e para que "," e "." não inflem as repetições.
    """
    palavras = []
    for linha in artigo["linhas"]:
        if not linha.strip() or nivel_do_titulo(linha)[0] is not None:
            continue
        palavras.extend(p.lower() for p in linha.split() if any(c.isalnum() for c in p))
    return palavras


def distinct_n(palavras, n):
    """Distinct-N: n-gramas distintos / total de n-gramas."""
    ngramas = [tuple(palavras[i:i + n]) for i in range(len(palavras) - n + 1)]
    return len(set(ngramas)) / len(ngramas) if ngramas else 0.0


def medidas(palavras):
    """TTR e MTLD pela lexicalrichness; Distinct-N calculado à parte."""
    # Passamos a lista já tokenizada, sem o pré-processamento da biblioteca.
    lex = LexicalRichness(palavras, preprocessor=None, tokenizer=None)
    return {
        "palavras": lex.words,
        "vocabulario": lex.terms,
        "ttr": lex.ttr,
        "distinct_1": distinct_n(palavras, 1),
        "distinct_2": distinct_n(palavras, 2),
        "mtld": lex.mtld(threshold=LIMIAR_MTLD),
    }


def tabela_por_artigo():
    """Uma linha por artigo com as quatro medidas."""
    registros = []
    for particao in PARTICOES:
        for art in separar_artigos(f"wiki2_{particao}.txt"):
            palavras = palavras_do_artigo(art)
            if len(palavras) < 2:
                continue
            registros.append({"particao": particao, "titulo": art["titulo"], **medidas(palavras)})
    return pd.DataFrame(registros)


def tabela_por_particao():
    """As mesmas medidas calculadas sobre cada partição inteira (todos os artigos juntos)."""
    linhas = []
    for particao in PARTICOES:
        palavras = []
        for art in separar_artigos(f"wiki2_{particao}.txt"):
            palavras.extend(palavras_do_artigo(art))
        linhas.append({"particao": particao, **medidas(palavras)})
    return pd.DataFrame(linhas).set_index("particao")


def graficos(df):
    """Boxplots das medidas por partição e TTR/MTLD em função do tamanho do artigo."""
    fig, eixos = plt.subplots(2, 3, figsize=(15, 8.5))

    # Linha de cima: distribuição de cada medida por partição.
    for ax, (coluna, nome) in zip(eixos[0], [("ttr", "TTR"), ("distinct_2", "Distinct-2"), ("mtld", "MTLD")]):
        dados = [df.loc[df["particao"] == p, coluna] for p in PARTICOES]
        try:
            ax.boxplot(dados, tick_labels=PARTICOES)   # matplotlib 3.9+
        except TypeError:
            ax.boxplot(dados, labels=PARTICOES)        # versões mais antigas
        ax.set_title(f"{nome} por artigo")
        ax.set_ylabel(nome)

    # Linha de baixo: efeito do tamanho do texto em cada medida.
    for ax, (coluna, nome) in zip(eixos[1], [("ttr", "TTR"), ("distinct_2", "Distinct-2"), ("mtld", "MTLD")]):
        for particao in PARTICOES:
            sub = df[df["particao"] == particao]
            ax.scatter(sub["palavras"], sub[coluna], s=10, alpha=0.5, label=particao)
        ax.set_xscale("log")
        ax.set_xlabel("palavras no artigo (log)")
        ax.set_ylabel(nome)
        ax.set_title(f"{nome} × tamanho do artigo")
        ax.legend()

    fig.tight_layout()
    fig.savefig("diversidade.png", dpi=150)
    plt.close(fig)
    print("Gráfico salvo em diversidade.png")


def main():
    df = tabela_por_artigo()
    df.to_csv("diversidade_artigos.csv", index=False)
    print("Tabela por artigo salva em diversidade_artigos.csv\n")

    colunas = ["ttr", "distinct_1", "distinct_2", "mtld"]

    print("=" * 70)
    print("Média por artigo (± desvio-padrão)")
    media = df.groupby("particao", sort=False)[colunas].mean()
    desvio = df.groupby("particao", sort=False)[colunas].std()
    print((media.round(3).astype(str) + " ± " + desvio.round(3).astype(str)).to_string())

    print("\nMediana por artigo")
    print(df.groupby("particao", sort=False)[colunas].median().round(3).to_string())

    print("\n" + "=" * 70)
    print("Partição inteira (todos os artigos juntos)")
    print(tabela_por_particao()[["palavras", "vocabulario"] + colunas].round(3).to_string())

    # Correlação com o tamanho: mostra que o TTR cai com o tamanho e o MTLD não.
    print("\n" + "=" * 70)
    print("Correlação de Spearman com o nº de palavras do artigo")
    for coluna in colunas:
        print(f"  {coluna:<11} {df[coluna].corr(df['palavras'], method='spearman'):+.2f}")

    print("\nArtigos com menor e maior MTLD:")
    for _, linha in df.nsmallest(3, "mtld").iterrows():
        print(f"  mín {linha['mtld']:7.1f}  {linha['titulo']} ({linha['particao']})")
    for _, linha in df.nlargest(3, "mtld").iterrows():
        print(f"  máx {linha['mtld']:7.1f}  {linha['titulo']} ({linha['particao']})")

    print()
    graficos(df)


if __name__ == "__main__":
    main()