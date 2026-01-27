from typing import List
import math

Vector = List[float]

def dot_product(a: Vector, b: Vector) -> float:
    """
    Formula:
        dot(a, b) = Σ (a_i * b_i)

    Higher value => more similar (magnitude matters).
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")

    return sum(x * y for x, y in zip(a, b))


# =====================================================
# Cosine Similarity
# =====================================================

def cosine_similarity(a: Vector, b: Vector) -> float:
    """
    Formula:
        cos(a, b) = dot(a, b) / (||a|| * ||b||)

    Result range:
        [-1, 1]
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")

    dot = dot_product(a, b)
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return dot / (norm_a * norm_b)


def vector_norm(v: Vector) -> float:
    """
    ||v|| = sqrt(Σ v_i^2)
    """
    return math.sqrt(sum(x * x for x in v))
