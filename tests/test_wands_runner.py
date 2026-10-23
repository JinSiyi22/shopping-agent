import numpy as np

from benchmarks.wands.run_benchmark import (
    rrf_fuse,
    top_k_indices,
)


def test_top_k_indices():

    scores = np.array(
        [
            0.1,
            0.8,
            0.3,
            0.9,
        ]
    )

    result = (
        top_k_indices(
            scores,
            2,
        )
    )

    assert (
        result.tolist()
        == [
            3,
            1,
        ]
    )


def test_rrf_fuse():

    first = np.array(
        [
            0,
            1,
            2,
        ]
    )

    second = np.array(
        [
            0,
            2,
            3,
        ]
    )

    result = (
        rrf_fuse(
            [
                first,
                second,
            ],
            top_k=4,
            rrf_k=60,
        )
    )

    assert (
        result[0]
        == 0
    )

    assert set(
        result.tolist()
    ) == {
        0,
        1,
        2,
        3,
    }
