
import pytest
import torch

from sc_exp_design.utils import match_shapes


class TestUtils:
    def test_match_shapes(
        self,
    ) -> None:        
    # dummy test
    batch_size = 64
    spatial_dim = 2
    target = torch.zeros((batch_size, spatial_dim))

    # case 0
    input_case0 = 0.0
    output_case0 = match_shapes(input_case0, target)
    assert output_case0.shape == (batch_size, 1)

    # case 1
    input_case1 = torch.zeros(())
    output_case1 = match_shapes(input_case1, target)
    assert output_case1.shape == (batch_size, 1)

    # case 2
    input_case2 = torch.zeros((1, ))
    output_case2 = match_shapes(input_case2, target)
    assert output_case2.shape == (batch_size, 1)

    # case 3
    input_case3 = torch.zeros((batch_size, ))
    output_case3 = match_shapes(input_case3, target)
    assert output_case3.shape == (batch_size, 1)

    # case 4
    input_case4 = torch.zeros((batch_size, 1))
    output_case4 = match_shapes(input_case4, target)
    assert output_case4.shape == (batch_size, 1)

    # case 5
    try:
        input_case5 = torch.zeros((batch_size, 3)) # should fail
        output_case5 = match_shapes(input_case5, target)
        raise RuntimeError
    except AssertionError as e:
        ...

    # case 6
    try:
        input_case6 = torch.zeros((batch_size + 1, )) # should fail
        output_case6 = match_shapes(input_case6, target)
        raise RuntimeError
    except AssertionError as e:
        ...

    # case 7
    try:
        input_case7 = torch.zeros((batch_size + 1, 1)) # should fail
        output_case7 = match_shapes(input_case7, target)
        raise RuntimeError
    except AssertionError as e:
        ...

    # case 8
    try:
        input_case8 = torch.zeros((batch_size + 1, 3)) # should fail
        output_case8 = match_shapes(input_case8, target)
        raise RuntimeError
    except AssertionError as e:
        ...

    # case 9
    try:
        input_case8 = torch.zeros((batch_size, 1, 1)) # should fail
        output_case8 = match_shapes(input_case8, target)
        raise RuntimeError
    except ValueError as e:
        ...

