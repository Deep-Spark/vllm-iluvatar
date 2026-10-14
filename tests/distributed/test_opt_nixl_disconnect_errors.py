import pytest

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.worker import (
    _is_terminal_xfer_query_error,
)


class nixlRemoteDisconnectError(RuntimeError):
    pass


class nixlNotFoundError(RuntimeError):
    pass


class nixlCancelledError(RuntimeError):
    pass


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"),
        RuntimeError("backend returned NIXL_ERR_NOT_FOUND"),
        RuntimeError("NIXL_ERR_CANCELED"),
        nixlRemoteDisconnectError(),
        nixlNotFoundError(),
        nixlCancelledError(),
    ],
)
def test_terminal_xfer_query_error(error: BaseException) -> None:
    assert _is_terminal_xfer_query_error(error)


def test_nonterminal_xfer_query_error() -> None:
    assert not _is_terminal_xfer_query_error(RuntimeError("temporary query failure"))
