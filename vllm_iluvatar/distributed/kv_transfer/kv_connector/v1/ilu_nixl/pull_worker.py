# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull worker for the Ilu NIXL connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.logger import init_logger

from .base_worker import IluNixlWorkerMixin

logger = init_logger(__name__)


class IluNixlPullConnectorWorker(IluNixlWorkerMixin, NixlPullConnectorWorker):
    def _get_new_notifs(self) -> set[str]:
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                # Notification payloads are "<req_id>:<tp_size>" for the
                # official format and "<req_id>:<tp_size>:<pp_consumers>" for
                # Ilu heterogeneous PP. Split from the right so legacy req_ids
                # containing ":" remain valid.
                parts = msg.rsplit(":", 2)
                if len(parts) == 3:
                    req_id, tp_size, pp_consumers = parts
                    n_consumers = int(tp_size)
                    pp_consumers_per_producer = int(pp_consumers)
                else:
                    req_id, tp_size = msg.rsplit(":", 1)
                    n_consumers = int(tp_size)
                    pp_consumers_per_producer = 1

                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.error(
                        "Potentially invalid KV blocks for unrecognized "
                        "request %s were retrieved by a decode worker. "
                        "They may have expired.",
                        req_id,
                    )
                    continue

                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
                tp_consumers_per_producer = (
                    -tp_ratio if n_consumers > self.world_size else 1
                )
                consumers_per_producer = (
                    tp_consumers_per_producer * pp_consumers_per_producer
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == consumers_per_producer
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids
