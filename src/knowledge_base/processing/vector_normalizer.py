import numpy as np
from enum import Enum
from typing import List, Union
from pinecone.core.openapi.inference.model.sparse_embedding import SparseEmbedding

class VectorType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"

class VectorNormalizer:
    @staticmethod
    def normalize(
        vectors: Union[List[List[float]], List[SparseEmbedding]],
        vector_type: VectorType,
    ) -> Union[List[List[float]], List[SparseEmbedding]]:
        """
        Normalizes a batch of vectors using Euclidean (L2) normalization.

        This method handles both dense vectors (lists of floats) and sparse vectors 
        (SparseEmbedding objects). It applies L2 normalization so that the magnitude 
        of each vector becomes 1.0.

        Args:
            vectors (Union[List[List[float]], List[SparseEmbedding]]): A list containing 
                either dense vectors (as lists of floats) or Pinecone SparseEmbedding objects.
            vector_type (VectorType): An enum indicating the type of vectors provided 
                (VectorType.DENSE or VectorType.SPARSE).

        Returns:
            Union[List[List[float]], List[SparseEmbedding]]: A list of normalized vectors 
            in the same format as the input. 
            - For dense vectors: Returns a list of lists of floats.
            - For sparse vectors: Returns a list of new SparseEmbedding objects with updated values.

        Raises:
            ValueError: If the `vector_type` provided is not a supported member of the 
                VectorType enum.
        
        Note:
            - Zero-vectors (magnitude of 0) are handled safely to avoid division by zero errors.
            - For dense zero-vectors, the output remains a zero vector (effectively).
            - For sparse zero-vectors, the output values remain unchanged (0.0).
        """
        match vector_type:
            case VectorType.DENSE:
                return VectorNormalizer._normalize_dense(vectors) #type: ignore
            case VectorType.SPARSE:
                return VectorNormalizer._normalize_sparse(vectors) #type: ignore
            case _:
                raise ValueError(f"Unsupported vector type: {vector_type}")
    @staticmethod
    def _normalize_dense(vectors: List[List[float]]) -> List[List[float]]:
        """
        Batch normalization for dense vectors.
        """
        arr = np.array(vectors)

        # Reshape single vector List[float] into List[List[float]]
        # Type hinting should catch this, but for extra safety
        if arr.ndim == 1: 
            arr = arr.reshape(1,-1) # Dynamically adjust col # based on vector dims (should be 768)
        
        # axis = 1 calculates norm row-wise, returning a single value for each row.
        # keepdims ensures that the value returned is still a column vector (x, 1) for x vectors passed.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)

        norms[norms == 0] = 1e-10 # Avoid division by zero errors for zero vectors
        return (arr / norms).tolist()
    @staticmethod
    def _normalize_sparse(vectors: List[SparseEmbedding]) -> List[SparseEmbedding]:
        """
        Batch normalization for sparse vectors.
        """
        normalized_list = []

        for vector in vectors:
            # Values and Indices are both lists
            values = vector.sparse_values

            np_values = np.array(values)
            norm = np.linalg.norm(np_values)

            if norm > 0:
                new_values = (np_values / norm).tolist()
            else:
                new_values = values
            
            # Create new SparseEmbedding object to attach the normalized vectors
            new_vec = SparseEmbedding(
                sparse_values = new_values,
                sparse_indices = vector.sparse_indices,
                vector_type = vector.vector_type
            )

            normalized_list.append(new_vec)
        
        return normalized_list

# Example usage 
if __name__ == "__main__": # pragma: no cover
    try:
        # Replicates List[SparseEmbedding]
        sparse_vector = [SparseEmbedding(
            sparse_values = [0.1, 0.5, 4.3, 8.0],
            sparse_indices = [744372458, 2165993515, 3261080123, 3508911095],
            vector_type = "sparse"
        )]

        print(f"Sparse Vector: {sparse_vector}")
        res = VectorNormalizer.normalize(sparse_vector, VectorType.SPARSE) #type: ignore
        print(f"Normalized Sparse Vector: {res}")

    except Exception as e:
        print(f"Test run failed: {e}")


