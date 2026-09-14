"""CPU reference follows NATTEN v0.17.1's non-causal window/RPB indexing."""
import pytest
import torch
from allin1.models.natten_compat import na1d_qk, na1d_av, na2d_qk, na2d_av


def neighbors(index, length, kernel, dilation):
    # Independent scalar reference: center a full window in the query's
    # dilation residue class, shifting it inwards at either boundary.
    lane = list(range(index % dilation, length, dilation))
    position = lane.index(index)
    start = min(max(position - kernel // 2, 0), len(lane) - kernel)
    return lane[start:start + kernel]


@pytest.mark.parametrize("length,kernel,dilation", [(7, 3, 1), (11, 5, 2), (17, 3, 4)])
@pytest.mark.parametrize("bias", [False, True])
def test_1d_matches_scalar_reference(length, kernel, dilation, bias):
    torch.manual_seed(7)
    q, k, v = [torch.randn(2, 2, length, 3, dtype=torch.float64) for _ in range(3)]
    rpb = torch.randn(2, 2 * kernel - 1, dtype=q.dtype) if bias else None
    expected = torch.empty(2, 2, length, kernel, dtype=q.dtype)
    for i in range(length):
        for slot, j in enumerate(neighbors(i, length, kernel, dilation)):
            expected[:, :, i, slot] = (q[:, :, i] * k[:, :, j]).sum(-1)
            if bias:
                expected[:, :, i, slot] += rpb[:, kernel - 1 + (j - i) // dilation]
    actual = na1d_qk(q, k, kernel, dilation, rpb=rpb)
    torch.testing.assert_close(actual, expected)
    weights = expected.softmax(-1)
    expected_value = torch.zeros_like(v)
    for i in range(length):
        for slot, j in enumerate(neighbors(i, length, kernel, dilation)):
            expected_value[:, :, i] += weights[:, :, i, slot, None] * v[:, :, j]
    torch.testing.assert_close(na1d_av(weights, v, kernel, dilation), expected_value)


@pytest.mark.parametrize("shape,kernel,dilation", [((5, 7), (3, 3), (1, 1)), ((7, 11), (3, 5), (2, 2))])
def test_2d_matches_scalar_reference(shape, kernel, dilation):
    torch.manual_seed(8)
    height, width = shape
    kh, kw = kernel
    dh, dw = dilation
    q, k, v = [torch.randn(1, 2, height, width, 2, dtype=torch.float64) for _ in range(3)]
    rpb = torch.randn(2, 2 * kh - 1, 2 * kw - 1, dtype=q.dtype)
    expected = torch.empty(1, 2, height, width, kh * kw, dtype=q.dtype)
    neighborhoods = {}
    for i in range(height):
        for j in range(width):
            ns = [(x, y) for x in neighbors(i, height, kh, dh) for y in neighbors(j, width, kw, dw)]
            neighborhoods[i, j] = ns
            for slot, (x, y) in enumerate(ns):
                expected[:, :, i, j, slot] = (q[:, :, i, j] * k[:, :, x, y]).sum(-1)
                expected[:, :, i, j, slot] += rpb[:, kh - 1 + (x-i)//dh, kw - 1 + (y-j)//dw]
    torch.testing.assert_close(na2d_qk(q, k, kernel, dilation, rpb), expected)
    weights = expected.softmax(-1)
    expected_value = torch.zeros_like(v)
    for (i, j), ns in neighborhoods.items():
        for slot, (x, y) in enumerate(ns):
            expected_value[:, :, i, j] += weights[:, :, i, j, slot, None] * v[:, :, x, y]
    torch.testing.assert_close(na2d_av(weights, v, kernel, dilation), expected_value)


@pytest.mark.parametrize("dim", [1, 2])
def test_attention_gradients(dim):
    shape = (1, 1, 7, 2) if dim == 1 else (1, 1, 7, 7, 2)
    q, k, v = [torch.randn(shape, dtype=torch.double, requires_grad=True) for _ in range(3)]
    rpb = torch.randn((1, 5) if dim == 1 else (1, 5, 5), dtype=torch.double, requires_grad=True)
    qk, av = (na1d_qk, na1d_av) if dim == 1 else (na2d_qk, na2d_av)
    assert torch.autograd.gradcheck(
        lambda q, k, v, r: av(qk(q, k, 3, 2, r).softmax(-1), v, 3, 2),
        (q, k, v, rpb), fast_mode=True,
    )


def test_scalar_2d_parameters_without_bias():
    q = torch.randn(1, 1, 6, 6, 2)
    torch.testing.assert_close(na2d_qk(q, q, 3, 2), na2d_qk(q, q, (3, 3), (2, 2)))
    weights = na2d_qk(q, q, 3, 2).softmax(-1)
    torch.testing.assert_close(na2d_av(weights, q, 3, 2), na2d_av(weights, q, (3, 3), (2, 2)))


@pytest.mark.parametrize("length,kernel,dilation", [(4, 3, 2), (7, 2, 1), (7, 3, 0)])
def test_invalid_neighborhood_is_rejected(length, kernel, dilation):
    q = torch.zeros(1, 1, length, 1)
    with pytest.raises(ValueError):
        na1d_qk(q, q, kernel, dilation)
