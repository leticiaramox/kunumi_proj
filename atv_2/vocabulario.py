"""
Frequência das palavras, vocabulário, Lei de Zipf e taxa de OOV no WikiText-2.

Uso:
    python vocabulario.py

Precisa do separar_artigos.py e dos arquivos wiki2_*.txt na mesma pasta.
Requer: pip install numpy matplotlib
"""

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from separar_artigos import separar_artigos, nivel_do_titulo

# True para tratar "The" e "the" como a mesma palavra.
MINUSCULAS = False

# True para descartar pontuação e marcações como "@-@" (fica só o que tem letra ou número).
SO_PALAVRAS = False


def tokens_da_particao(particao):
    """Lista de todas as palavras da partição, sem títulos e linhas vazias."""
    tokens = []
    for art in separar_artigos(f"wiki2_{particao}.txt"):
        for linha in art["linhas"]:
            if not linha.strip() or nivel_do_titulo(linha)[0] is not None:
                continue
            palavras = linha.split()
            if MINUSCULAS:
                palavras = [p.lower() for p in palavras]
            if SO_PALAVRAS:
                palavras = [p for p in palavras if any(ch.isalnum() for ch in p)]
            tokens.extend(palavras)
    return tokens


def analisar_vocabulario(freq):
    """Tamanho do vocabulário, total de tokens e palavras que aparecem uma única vez."""
    total = sum(freq.values())
    hapax = sum(1 for c in freq.values() if c == 1)

    print("=" * 70)
    print("Vocabulário do treino")
    print(f"  Total de tokens:            {total:>10,}")
    print(f"  Tamanho do vocabulário:     {len(freq):>10,}")
    print(f"  Palavras que aparecem 1 vez: {hapax:>9,}"
          f"  ({hapax / len(freq):.1%} do vocabulário, {hapax / total:.2%} dos tokens)")

    print("\n  20 palavras mais comuns:")
    for i, (palavra, c) in enumerate(freq.most_common(20), 1):
        print(f"  {i:>4}. {palavra:<12}{c:>9,}  ({c / total:.2%})")

    # Cobertura: que fração dos tokens as N palavras mais comuns cobrem.
    contagens = np.array(sorted(freq.values(), reverse=True))
    acumulado = np.cumsum(contagens) / total
    print("\n  Cobertura das N palavras mais comuns:")
    for n in [10, 100, 1000, 10000]:
        if n <= len(contagens):
            print(f"    top {n:>6,}: {acumulado[n - 1]:.1%} dos tokens")


def grafico_zipf(freq):
    """Gráfico log-log de frequência por rank, com a reta ajustada."""
    frequencias = np.array(sorted(freq.values(), reverse=True))
    ranks = np.arange(1, len(frequencias) + 1)

    # Ajuste de reta em log-log: log(f) = log(C) - s * log(r).
    # A lei de Zipf "pura" prevê inclinação s ≈ 1.
    inclinacao, intercepto = np.polyfit(np.log10(ranks), np.log10(frequencias), 1)
    print("\n" + "=" * 70)
    print("Lei de Zipf")
    print(f"  Inclinação ajustada: {inclinacao:.2f}  (Zipf ideal ≈ -1)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, frequencias, ".", markersize=3, label="WikiText-2 (treino)")
    ax.loglog(ranks, 10 ** intercepto * ranks ** inclinacao, "-", linewidth=2,
              label=f"ajuste: inclinação = {inclinacao:.2f}")
    ax.loglog(ranks, frequencias[0] / ranks, "--", linewidth=1.5,
              label="Zipf ideal (f = f₁ / r)")
    ax.set_xlabel("rank (log)")
    ax.set_ylabel("frequência (log)")
    ax.set_title("Frequência por rank – Lei de Zipf")
    ax.legend()
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig("zipf.png", dpi=150)
    plt.close(fig)
    print("  Gráfico salvo em zipf.png")


def taxa_oov(vocab_treino, particao):
    """Palavras da partição que não aparecem no treino, por tipo e por token."""
    freq = Counter(tokens_da_particao(particao))
    tipos_oov = [p for p in freq if p not in vocab_treino]
    tokens_oov = sum(freq[p] for p in tipos_oov)
    total = sum(freq.values())

    print(f"  {particao:<11}"
          f" tipos: {len(tipos_oov):>6,} de {len(freq):>6,} ({len(tipos_oov) / len(freq):.2%})"
          f" | tokens: {tokens_oov:>6,} de {total:>7,} ({tokens_oov / total:.2%})")
    exemplos = sorted(tipos_oov, key=lambda p: -freq[p])[:10]
    print(f"  {'':<11} exemplos: {', '.join(exemplos)}")


def main():
    freq_treino = Counter(tokens_da_particao("train"))

    analisar_vocabulario(freq_treino)
    grafico_zipf(freq_treino)

    print("\n" + "=" * 70)
    print("Palavras fora do vocabulário do treino (OOV)")
    vocab_treino = set(freq_treino)
    for particao in ["validation", "test"]:
        taxa_oov(vocab_treino, particao)


if __name__ == "__main__":
    main()
