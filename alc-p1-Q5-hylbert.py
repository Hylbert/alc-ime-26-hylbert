"""
Prova 1 - Algebra Linear Computacional (2026.2) - Questao 5
Aluno: Hylbert Bentes Rocha Rodrigues

Resolve Ax = b por decomposicao LU sem pivoteamento (A = LU), em que
L e triangular inferior com 1 na diagonal e U e triangular superior.
"""
import numpy as np

# Valores com modulo abaixo disso sao considerados pivos nulos
TOL = 1e-12


def decomposicao_lu(A):
    """Constroi explicitamente L e U tais que A = LU (sem pivoteamento)."""
    n = A.shape[0]
    L = np.eye(n)                  # diagonal de L igual a 1
    U = np.array(A, dtype=float)   # copia de A, que vira U

    for k in range(n):             # k = coluna do pivo
        pivo = U[k, k]
        if abs(pivo) < TOL:
            raise Exception(
                f"Pivo nulo encontrado na posicao ({k}, {k}). A "
                "decomposicao LU sem pivoteamento nao pode continuar. "
                "Utilize uma funcao alternativa para resolver o sistema, "
                "por exemplo, uma eliminacao de Gauss com pivoteamento "
                "parcial (PA = LU)."
            )

        for i in range(k + 1, n):  # linhas abaixo do pivo
            # Multiplicador da operacao L_i <- L_i - m * L_k
            m = U[i, k] / pivo
            # O multiplicador e guardado em L, na mesma posicao (i, k)
            L[i, k] = m

            U[i, k] = 0.0          # elemento eliminado
            for j in range(k + 1, n):
                U[i, j] = U[i, j] - m * U[k, j]

    return L, U


def substituicao_progressiva(L, b):
    """Resolve Ly = b, com L triangular inferior, de cima para baixo."""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = (b[i] - soma) / L[i, i]
    return y


def substituicao_regressiva(U, y):
    """Resolve Ux = y, com U triangular superior, de baixo para cima."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]
    return x


def resolve_lu(A, b):
    """Resolve Ax = b por decomposicao LU.

    Retorna, nesta ordem: L, U e o vetor solucao x.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A deve ser uma matriz quadrada (n x n).")
    n = A.shape[0]

    # Aceita b como vetor (n,) ou como vetor coluna (n, 1)
    if b.ndim == 2 and b.shape[1] == 1:
        b = b[:, 0]
    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError(f"b deve ter {n} elementos.")

    L, U = decomposicao_lu(A)
    y = substituicao_progressiva(L, b)   # Ly = b
    x = substituicao_regressiva(U, y)    # Ux = y

    return L, U, x


if __name__ == "__main__":
    A = [[2, 1, 1],
         [4, -6, 0],
         [-2, 7, 2]]
    b = [5, -2, 9]

    L, U, x = resolve_lu(A, b)
    print("L =\n", L)
    print("U =\n", U)
    print("x =", x)

    # Exemplo com pivo nulo: A[0, 0] = 0
    try:
        resolve_lu([[0, 1], [1, 1]], [1, 2])
    except Exception as erro:
        print("\nExcecao:", erro)
