"""
Diversidade semântica do WikiText-2: Self-BLEU e similaridade por embeddings.

Uso:
    python semantica.py

Precisa do separar_artigos.py e dos arquivos wiki2_*.txt na mesma pasta.
Requer: pip install nltk sentence-transformers numpy
"""

import random
import re

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer

from separar_artigos import PARTICOES, separar_artigos, nivel_do_titulo

N_FRASES = 500        # frases sorteadas por partição
MIN_PALAVRAS = 5      # descarta frases muito curtas
SEMENTE = 42

RE_FIM_FRASE = re.compile(r"\s\.(?:\s|$)")
RE_MARCACAO = re.compile(r" @(.)@ ")   # " @-@ " -> "-"


def amostra_de_frases(particao):
    """Sorteia N_FRASES frases da partição."""
    frases = []
    for art in separar_artigos(f"wiki2_{particao}.txt"):
        for linha in art["linhas"]:
            if not linha.strip() or nivel_do_titulo(linha)[0] is not None:
                continue
            linha = RE_MARCACAO.sub(r"\1", linha)
            for frase in RE_FIM_FRASE.split(linha):
                if len(frase.split()) >= MIN_PALAVRAS:
                    frases.append(frase.strip())
    return random.Random(SEMENTE).sample(frases, min(N_FRASES, len(frases)))


def self_bleu(frases):
    """Média do BLEU-4 de cada frase contra todas as outras. Alto = pouca diversidade."""
    tokens = [[p.lower() for p in f.split() if any(c.isalnum() for c in p)] for f in frases]
    suavizar = SmoothingFunction().method1
    notas = [
        sentence_bleu(tokens[:i] + tokens[i + 1:], hip, smoothing_function=suavizar)
        for i, hip in enumerate(tokens)
    ]
    return float(np.mean(notas))


def similaridade_embeddings(frases, modelo):
    """Cosseno médio entre todos os pares de frases. Alto = pouca diversidade."""
    vetores = modelo.encode(frases, normalize_embeddings=True)
    matriz = vetores @ vetores.T
    i, j = np.triu_indices(len(frases), k=1)   # pares distintos, sem a diagonal
    return float(matriz[i, j].mean())


def main():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"{'Partição':<12}{'Self-BLEU':>12}{'Similaridade':>15}")
    for particao in PARTICOES:
        frases = amostra_de_frases(particao)
        print(f"{particao:<12}{self_bleu(frases):>12.3f}"
              f"{similaridade_embeddings(frases, modelo):>15.3f}")


if __name__ == "__main__":
    main()