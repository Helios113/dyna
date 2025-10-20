import hashlib

from generate_standard_inputs import generate_standard_inputs

DEFAULT_HASH = "f2c792f21615cfde8830e3f2f45e638cd26f80a583d5b29d24ae51cf5dd8e716"


def test_generate_standard_inputs():
    # --- 3. Calculate the hash of the new tensor ---
    regenerated_tensor = generate_standard_inputs()
    # Convert the tensor to a byte string
    tensor_bytes = regenerated_tensor.numpy().tobytes()

    # Calculate the SHA-256 hash
    sha256_hash = hashlib.sha256(tensor_bytes).hexdigest()
    assert (
        sha256_hash == DEFAULT_HASH
    ), """Hashes of generated tensor do not match the pre-calculated hash,
        therefore results cannot be trusted."""
