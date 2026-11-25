# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull worker for the Ilu NIXL connector."""

import time
from collections import defaultdict

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.logger import init_logger

from .base_worker import IluNixlWorkerMixin
from .metadata import decode_structured_notif

logger = init_logger(__name__)


class IluNixlPullConnectorWorker(IluNixlWorkerMixin, NixlPullConnectorWorker):
    def get_transfer_results(self):
        if getattr(self, "_io_thread_started", False):
            return self._drain_io_transfer_results()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorTransferResults,
        )

        if getattr(self, "_io_thread_enabled", False):
            results = KVConnectorTransferResults()
        else:
            done_sending, done_recving = super().get_finished()
            results = KVConnectorTransferResults(
                finished_sending=done_sending,
                finished_recving=done_recving,
            )
        now = time.perf_counter()
        if results.finished_sending:
            lease_starts = getattr(self, "_send_lease_started_at", None)
            stats = getattr(self, "xfer_stats", None)
            if (
                isinstance(lease_starts, dict)
                and stats is not None
                and hasattr(stats, "record_lease_hold_duration")
            ):
                for rid in results.finished_sending:
                    started = lease_starts.pop(rid, None)
                    if started is not None:
                        stats.record_lease_hold_duration(now - started)
        if hasattr(self, "_progress_draining_sessions"):
            self._progress_draining_sessions(now)
        return results

    def get_finished(self) -> tuple[set[str], set[str]]:
        if getattr(self, "_io_thread_started", False):
            results = self.get_transfer_results()
            return results.finished_sending, results.finished_recving
        return super().get_finished()

    def _clear_notif_state_for_req(self, req_id: str) -> None:
        self.consumer_notification_counts_by_req.pop(req_id, None)
        expected_map = getattr(self, "expected_consumer_notifications_by_req", None)
        if isinstance(expected_map, dict):
            expected_map.pop(req_id, None)
        seen_map = getattr(self, "_notif_seen_senders_by_req", None)
        if isinstance(seen_map, dict):
            seen_map.pop(req_id, None)
        self._reqs_to_process.discard(req_id)
        self._reqs_to_send.pop(req_id, None)

    def _record_consumer_notif(
        self,
        req_id: str,
        *,
        expected_consumers: int,
        status: str = "ok",
        sender_tag: str | None = None,
    ) -> bool:
        """Advance send-lease state for ``req_id``; return True when settled."""
        if status == "fail":
            logger.warning(
                "Consumer reported failed KV READ for request %s (sender=%s); "
                "releasing producer lease immediately.",
                req_id,
                sender_tag or "unknown",
            )
            self._record_stat_failed_transfer()
            self._clear_notif_state_for_req(req_id)
            return True

        if expected_consumers <= 0:
            self._clear_notif_state_for_req(req_id)
            return True

        if not hasattr(self, "_notif_seen_senders_by_req"):
            self._notif_seen_senders_by_req = defaultdict(set)
        if not hasattr(self, "expected_consumer_notifications_by_req"):
            self.expected_consumer_notifications_by_req = {}

        if sender_tag is not None:
            seen_senders = self._notif_seen_senders_by_req[req_id]
            if sender_tag in seen_senders:
                logger.warning(
                    "Ignoring duplicate consumer notification for request %s from %s",
                    req_id,
                    sender_tag,
                )
                return False
            seen_senders.add(sender_tag)

        prev_expected = self.expected_consumer_notifications_by_req.get(req_id)
        if prev_expected is None:
            self.expected_consumer_notifications_by_req[req_id] = expected_consumers
        elif prev_expected != expected_consumers:
            logger.error(
                "Inconsistent expected consumer count for request %s: "
                "previous=%d, current=%d (sender=%s)",
                req_id,
                prev_expected,
                expected_consumers,
                sender_tag,
            )
            expected_consumers = max(prev_expected, expected_consumers)
            self.expected_consumer_notifications_by_req[req_id] = expected_consumers

        self.consumer_notification_counts_by_req[req_id] += 1
        if self.consumer_notification_counts_by_req[req_id] >= expected_consumers:
            self._clear_notif_state_for_req(req_id)
            return True
        return False

    def _get_new_notifs(self) -> set[str]:
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()

        # Empty producer ranks expect 0 consumer notifications; settle them
        # through the exact same state machine with expected_consumers=0.
        if getattr(self, "_empty_producer_rank", False) and self._reqs_to_send:
            for req_id in list(self._reqs_to_send):
                if self._record_consumer_notif(
                    req_id, expected_consumers=0, status="ok"
                ):
                    notified_req_ids.add(req_id)

        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                try:
                    structured = decode_structured_notif(msg)
                except ValueError:
                    logger.exception("Failed to parse structured NIXL notification")
                    continue

                if structured is not None:
                    req_id, status, sender_tag, expected_consumers = structured
                    if (
                        req_id not in self._reqs_to_send
                        and req_id not in self._reqs_to_process
                    ):
                        if status == "ok":
                            logger.error(
                                "Potentially invalid KV blocks for unrecognized "
                                "request %s were retrieved by a decode worker "
                                "(sender=%s). They may have expired.",
                                req_id,
                                sender_tag,
                            )
                        continue
                    if self._record_consumer_notif(
                        req_id,
                        expected_consumers=expected_consumers,
                        status=status,
                        sender_tag=sender_tag,
                    ):
                        notified_req_ids.add(req_id)
                    continue

                # Legacy payloads: "<req_id>:<tp_size>" (official) or
                # "<req_id>:<tp_size>:<pp_consumers>" (Ilu v1). Split from the
                # right so legacy req_ids containing ":" remain valid.
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
                if self._record_consumer_notif(
                    req_id,
                    expected_consumers=consumers_per_producer,
                    status="ok",
                ):
                    notified_req_ids.add(req_id)
        return notified_req_ids
