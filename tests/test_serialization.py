import numpy as np

from qxg_platform.serialization import (
    decode_array,
    encode_array,
    relation_keys_from_json,
    relation_keys_to_json,
)


def test_array_roundtrip() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(decode_array(encode_array(array)), array)


def test_relation_key_roundtrip() -> None:
    relations = {(0, 2): {"distance": "close"}}
    assert relation_keys_from_json(relation_keys_to_json(relations)) == relations
