from pinecone.core.openapi.inference.model.sparse_embedding import SparseEmbedding
from knowledge_base.processing.vector_normalizer import VectorNormalizer, VectorType
from typing import List
import pytest


class TestVectorNormalizer:

    def test_normalize_dense_successful(self, raw_dense_embeddings, normalized_dense_embeddings):
        """
        Verifies dense normalization using a clean 4-value vector from fixtures.

        Input: [[2.0, 4.0, 4.0, 8.0]]
        Math:
          - Sum of squares: 4 + 16 + 16 + 64 = 100
          - Magnitude (Norm): sqrt(100) = 10.0
          - Calculation: [2/10, 4/10, 4/10, 8/10]

        Expected Output: [[0.2, 0.4, 0.4, 0.8]]
        Verifies both value accuracy and output data types.
        """
        res = VectorNormalizer.normalize(raw_dense_embeddings, VectorType.DENSE)

        # pytest.approx does not support nested data structures
        # Using pytest.approx for rounding
        assert res[0] == pytest.approx(normalized_dense_embeddings[0])

        # Verify the function is returning List[List[float]]
        assert isinstance(res, list)
        assert isinstance(res[0], list)
        assert isinstance(res[0][0], float)

        # Check dimensions
        assert len(res) == 1
        assert len(res[0]) == 4

    def test_normalize_dense_zero_vector(self):
        """
        Verifies the edge case where a vector has zero magnitude.

        Logic:
          - A zero vector has a norm of 0.0.
          - Division by zero must be prevented.
          - The vector should be returned unchanged (as all zeros).

        Input: [[0.0, 0.0]]
        Expected: [[0.0, 0.0]]
        """
        vectors = [[0.0, 0.0]]
        expected_result = [[0.0, 0.0]]

        res = VectorNormalizer.normalize(vectors, VectorType.DENSE)

        assert res[0] == pytest.approx(expected_result[0])

    def test_normalize_dense_batch(self, raw_dense_embeddings, normalized_dense_embeddings):
        """
        Verifies that multiple vectors in a batch are normalized independently.

        Input Batch:
          1. Fixture Vector: [2.0, 4.0, 4.0, 8.0] (Norm 10)
          2. Manual Vector:  [1.0, 5.0, 5.0, 7.0] (Norm 10)

        Expected Output:
          1. [0.2, 0.4, 0.4, 0.8]
          2. [0.1, 0.5, 0.5, 0.7]
        """
        # Combining lists
        vectors = raw_dense_embeddings + [[1.0, 5.0, 5.0, 7.0]]
        expected_result = normalized_dense_embeddings + [[0.1, 0.5, 0.5, 0.7]]

        res = VectorNormalizer.normalize(vectors, VectorType.DENSE)

        assert res[0] == pytest.approx(expected_result[0])
        assert res[1] == pytest.approx(expected_result[1])

    def test_normalize_dense_incorrect_vector_type(self, raw_dense_embeddings, normalized_dense_embeddings):
        """
        Verifies that the normalizer robustly handles a single flat vector input.

        The normalize method expects a batch (List[List[float]]),
        but this test passes a single flat list (List[float]) to ensure the
        function can handle the "incorrect" nesting depth without crashing.

        Input: [2.0, 4.0, 4.0, 8.0] (Flat list, not wrapped as a batch)
        Expected: [[0.2, 0.4, 0.4, 0.8]] (Returns correctly normalized vector)
        """
        # Passing in List[float] rather than List[List[float]]
        vector = raw_dense_embeddings[0]

        res = VectorNormalizer.normalize(vector, VectorType.DENSE)

        assert res[0] == pytest.approx(normalized_dense_embeddings[0])

    def test_normalize_sparse_successful(self, raw_sparse_embeddings, normalized_sparse_embeddings):
        """
        Verifies successful normalization of a single SparseEmbedding object.

        Checks:
          - 'sparse_values' are normalized correctly (divided by Euclidean norm).
          - 'sparse_indices' remain strictly unchanged.
          - 'vector_type' attribute is preserved.
          - The return type is a list of SparseEmbedding objects.
        """
        res = VectorNormalizer.normalize(raw_sparse_embeddings, VectorType.SPARSE)

        assert res[0].sparse_values == pytest.approx(normalized_sparse_embeddings[0].sparse_values)  # type: ignore
        assert res[0].vector_type == normalized_sparse_embeddings[0].vector_type  # type: ignore
        # Verify that the indices remain unchanged
        assert res[0].sparse_indices == normalized_sparse_embeddings[0].sparse_indices  # type: ignore

        # Verify dimensions
        assert len(res) == 1
        assert len(res[0].sparse_values) == 4  # type: ignore

        # Verify List[SparseEmbedding]
        assert isinstance(res, list)
        assert isinstance(res[0], SparseEmbedding)

    def test_normalize_sparse_batch(self, raw_sparse_embeddings, normalized_sparse_embeddings):
        """
        Verifies batch normalization for SparseEmbedding objects.

        Ensures that when a list containing multiple SparseEmbedding objects is provided:
          1. The fixture object is normalized correctly.
          2. A second manually created object is normalized independently.
          3. Metadata (indices, vector_type) for both objects remains intact.
        """
        second_vector = SparseEmbedding(
            sparse_values=[2.0, 4.0, 4.0, 8.0],
            sparse_indices=[744372458, 2165993515, 3261080123, 3508911095],
            vector_type="sparse",
        )

        expected_second_vector = SparseEmbedding(
            sparse_values=[0.2, 0.4, 0.4, 0.8],
            sparse_indices=[744372458, 2165993515, 3261080123, 3508911095],
            vector_type="sparse",
        )

        # Combining lists
        batch = [raw_sparse_embeddings[0], second_vector]
        expected_results = [normalized_sparse_embeddings[0], expected_second_vector]

        res = VectorNormalizer.normalize(batch, VectorType.SPARSE)

        assert len(res) == 2

        assert res[0].sparse_values == pytest.approx(expected_results[0].sparse_values)  # type: ignore
        assert res[0].vector_type == expected_results[0].vector_type  # type: ignore
        assert res[0].sparse_indices == expected_results[0].sparse_indices  # type: ignore

        assert res[1].sparse_values == pytest.approx(expected_results[1].sparse_values)  # type: ignore
        assert res[1].vector_type == expected_results[1].vector_type  # type: ignore
        assert res[1].sparse_indices == expected_results[1].sparse_indices  # type: ignore

    def test_normalize_sparse_zero_vector(self):
        """
        Verifies handling of a SparseEmbedding object with zero-magnitude values.

        Logic:
          - Norm is 0.0.
          - Division by zero is prevented.
          - Returns the sparse values as [0.0, 0.0] without error.
        """
        vector = SparseEmbedding(sparse_values=[0.0, 0.0], sparse_indices=[1, 2], vector_type="sparse")

        res = VectorNormalizer.normalize([vector], VectorType.SPARSE)  # type: ignore

        assert res[0].sparse_values == [0.0, 0.0]  # type: ignore
        assert res[0].vector_type == "sparse"  # type: ignore

        assert len(res) == 1
        assert len(res[0].sparse_values) == 2  # type: ignore
