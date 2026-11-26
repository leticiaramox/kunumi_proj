"""
Conta palavras e frases por artigo do WikiText-2 e gera estatísticas e gráficos.

Uso:
    python estatisticas_artigos.py

Precisa do separar_artigos.py e dos arquivos wiki2_*.txt na mesma pasta.
Requer: pip install pandas matplotlib
"""

import re
import pandas as pd
import matplotlib.pyplot as plt

from separar_artigos import PARTICOES, separar_artigos, nivel_do_titulo

# Fim de frase: um ponto final isolado por espaços (" . "), ou no fim da linha.
RE_FIM_FRASE = re.compile(r"\s\.(?:\s|$)")


def contar_artigo(artigo):
    """Devolve (nº de palavras, nº de frases) de um artigo, ignorando títulos e linhas vazias."""
    palavras = 0
    frases = 0
    for linha in artigo["linhas"]:
        if not linha.strip() or nivel_do_titulo(linha)[0] is not None:
            continue
        palavras += len(linha.split())
        # Cada pedaço com texto entre pontos finais conta como uma frase.
        frases += sum(1 for parte in RE_FIM_FRASE.split(linha) if parte.strip())
    return palavras, frases


def montar_tabela():
    """Uma linha por artigo, com partição, título, palavras e frases."""
    registros = []
    for particao in PARTICOES:
        for art in separar_artigos(f"wiki2_{particao}.txt"):
            palavras, frases = contar_artigo(art)
            registros.append({
                "particao": particao,
                "titulo": art["titulo"],
                "palavras": palavras,
                "frases": frases,
            })
    return pd.DataFrame(registros)


def resumo(df, coluna):
    """Média, mediana, desvio-padrão e extremos por partição."""
    tabela = df.groupby("particao", sort=False)[coluna].agg(
        artigos="count", total="sum", media="mean", mediana="median",
        desvio="std", minimo="min", maximo="max",
    )
    return tabela.round(1)


def mostrar_extremos(df, coluna):
    """Artigos com o menor e o maior valor em cada partição."""
    for particao, grupo in df.groupby("particao", sort=False):
        menor = grupo.loc[grupo[coluna].idxmin()]
        maior = grupo.loc[grupo[coluna].idxmax()]
        print(f"  {particao:<11} mín: {menor[coluna]:>6} ({menor['titulo']})")
        print(f"  {'':<11} máx: {maior[coluna]:>6} ({maior['titulo']})")


def graficos(df, coluna, rotulo):
    """Histograma e boxplot por partição, salvos em PNG."""
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(13, 4.5))

    for particao in PARTICOES:
        valores = df.loc[df["particao"] == particao, coluna]
        ax_hist.hist(valores, bins=40, alpha=0.55, density=True, label=particao)
    ax_hist.set_title(f"Histograma de {rotulo} por artigo")
    ax_hist.set_xlabel(f"{rotulo} por artigo")
    ax_hist.set_ylabel("densidade")
    ax_hist.legend()

    dados = [df.loc[df["particao"] == p, coluna] for p in PARTICOES]
    try:
        ax_box.boxplot(dados, tick_labels=PARTICOES)   # matplotlib 3.9+
    except TypeError:
        ax_box.boxplot(dados, labels=PARTICOES)        # versões mais antigas
    ax_box.set_title(f"Boxplot de {rotulo} por artigo")
    ax_box.set_ylabel(f"{rotulo} por artigo")

    fig.tight_layout()
    nome = f"dist_{coluna}.png"
    fig.savefig(nome, dpi=150)
    print(f"  Gráfico salvo em {nome}")
    plt.close(fig)


def main():
    df = montar_tabela()
    df.to_csv("artigos_wikitext2.csv", index=False)
    print("Tabela por artigo salva em artigos_wikitext2.csv\n")

    for coluna, rotulo in [("palavras", "palavras"), ("frases", "frases")]:
        print("=" * 70)
        print(f"Estatísticas de {rotulo} por artigo")
        print(resumo(df, coluna).to_string())
        print("\nExtremos:")
        mostrar_extremos(df, coluna)
        graficos(df, coluna, rotulo)
        print()

    # Média de palavras por frase, útil para comparar o estilo das partições.
    df["palavras_por_frase"] = df["palavras"] / df["frases"].where(df["frases"] > 0)
    print("Média de palavras por frase:")
    print(df.groupby("particao", sort=False)["palavras_por_frase"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
