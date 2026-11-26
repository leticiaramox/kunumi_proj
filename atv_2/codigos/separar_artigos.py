"""
Separa os artigos do WikiText-2 (versão raw) e conta os títulos por nível.

Uso:
    python separar_artigos.py

Espera os arquivos wiki2_train.txt, wiki2_validation.txt e wiki2_test.txt
na mesma pasta em que o script é executado.
"""

import re
import random
from collections import Counter

PARTICOES = ["train", "validation", "test"]

# Qualquer título: um ou mais "=" separados por espaço, o texto, e os mesmos "=" no fim.
#   " = Título = "            -> nível 1
#   " = = Seção = = "         -> nível 2
#   " = = = Subseção = = = "  -> nível 3
RE_TITULO = re.compile(r"^(=(?: =)*) (.+?) \1$")


def nivel_do_titulo(linha):
    """Devolve (nível, texto do título) se a linha for um título, senão (None, None)."""
    m = RE_TITULO.match(linha.strip())
    if not m:
        return None, None
    nivel = m.group(1).count("=")
    return nivel, m.group(2)


def separar_artigos(caminho):
    """Agrupa as linhas em artigos: cada título de nível 1 abre um artigo novo."""
    artigos = []
    atual = None

    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            nivel, titulo = nivel_do_titulo(linha)

            if nivel == 1:
                atual = {"titulo": titulo, "linhas": []}
                artigos.append(atual)
            elif atual is not None:
                # Seções, subseções, parágrafos e linhas vazias ficam no artigo atual.
                atual["linhas"].append(linha.rstrip("\n"))
            elif linha.strip():
                # Texto antes do primeiro título (não deveria acontecer).
                print(f"Aviso: linha fora de artigo em {caminho}: {linha[:60]!r}")

    return artigos


def contar_titulos(artigos):
    """Conta os títulos por nível e os títulos de seção mais comuns."""
    por_nivel = Counter({1: len(artigos)})
    secoes = Counter()

    for art in artigos:
        for linha in art["linhas"]:
            nivel, titulo = nivel_do_titulo(linha)
            if nivel is not None:
                por_nivel[nivel] += 1
                secoes[titulo] += 1

    return por_nivel, secoes


def mostrar_amostra(artigos, n=3, semente=42):
    """Imprime o título, o tamanho e o início de alguns artigos sorteados."""
    random.seed(semente)
    for art in random.sample(artigos, min(n, len(artigos))):
        texto = [l for l in art["linhas"] if l.strip()]
        # Conta palavras só dos parágrafos, sem os títulos de seção.
        n_palavras = sum(len(l.split()) for l in texto if nivel_do_titulo(l)[0] is None)
        print(f"\n  Título: {art['titulo']}")
        print(f"  Linhas não vazias: {len(texto)} | Palavras: {n_palavras}")
        for l in texto[:3]:
            print(f"    {l.strip()[:100]}")
        print(f"    ...")
        print(f"    {texto[-1].strip()[:100]}" if texto else "    (artigo vazio!)")


def main():
    resultados = {}

    for particao in PARTICOES:
        artigos = separar_artigos(f"wiki2_{particao}.txt")
        resultados[particao] = artigos

        vazios = [a["titulo"] for a in artigos if not any(l.strip() for l in a["linhas"])]

        print("=" * 70)
        print(f"Partição: {particao}")
        print(f"  Artigos encontrados: {len(artigos)}")
        print(f"  Artigos vazios: {len(vazios)} {vazios[:5] if vazios else ''}")
        print(f"  Amostra:")
        mostrar_amostra(artigos)

    # Contador de títulos
    print("\n" + "=" * 70)
    print("Contador de títulos por nível")
    print(f"  {'Partição':<12}{'Nível 1':>10}{'Nível 2':>10}{'Nível 3':>10}{'Nível 4+':>10}")
    for particao, artigos in resultados.items():
        por_nivel, _ = contar_titulos(artigos)
        nivel4mais = sum(v for k, v in por_nivel.items() if k >= 4)
        print(f"  {particao:<12}{por_nivel[1]:>10}{por_nivel[2]:>10}{por_nivel[3]:>10}{nivel4mais:>10}")

    _, secoes = contar_titulos(resultados["train"])
    print("\nTítulos de seção mais comuns (treino):")
    for titulo, qtd in secoes.most_common(10):
        print(f"  {qtd:>5}  {titulo}")


if __name__ == "__main__":
    main()
