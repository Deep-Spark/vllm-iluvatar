"""E25 MM identity proof and safe legacy fallback, independent from GPU work."""

import hashlib
from types import SimpleNamespace as NS

import msgspec
import pytest
from test_source_state import PacketTest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

from vllm_iluvatar.v1.worker.mm_replay.identity import (
    CERTIFIED_PREFIX,
    feature_descriptors,
    request_identity,
)
from vllm_iluvatar.v1.worker.prefill_state import source_state as pub
from vllm_iluvatar.v1.worker.prefill_state.runtime import PrefixRuntime


class MMProofTest(PacketTest):
    def test_versioned_complete_image_identity_and_safe_text_wire(self):
        rid = "image-request"
        n = 129
        proof = hashlib.sha256(
            b"processed-image-geometry-position-and-model"
        ).hexdigest()
        rows = self.shards(rid, n)
        directory = pub.request_dir(self.source, rid)
        for row in rows:
            row["mm_input_digest"] = proof
            pub.save(directory / f"auto_{row['rank']}.json", row)
            pub.save(
                directory / f"observed_{row['rank']}.json",
                dict(admission=1, mm_input_digest=proof),
            )
        self.publisher().poll(10)
        identity = dict(
            version=pub.MM_PROTOCOL_VERSION,
            token_sha256=self.digest,
            mm_input_digest=proof,
        )
        message = (pub.GET_STATE, self.engine, rid, 0, n, identity)
        reply, raw = pub.serve_state(self.wire, self.engine, message)
        self.assertEqual(reply["status"], "READY")
        self.assertEqual(reply["version"], 4)
        self.assertEqual(reply["metadata"]["token_sha256"], self.digest)
        self.assertEqual(reply["metadata"]["mm_input_digest"], proof)
        self.assertEqual(reply["source"]["mm_input_digest"], proof)
        bad = (
            pub.GET_STATE,
            self.engine,
            rid,
            0,
            n,
            dict(identity, mm_input_digest="b" * 64),
        )
        self.assertEqual(
            pub.serve_state(self.wire, self.engine, bad)[0]["status"],
            "IDENTITY_MISMATCH",
        )
        # A token-only query must not import an image-bound packet.
        self.assertEqual(
            pub.serve_state(self.wire, self.engine, self.request(rid, n=n))[0][
                "status"
            ],
            "IDENTITY_MISMATCH",
        )

        class Socket:
            def send(self, data):
                self.sent = msgspec.msgpack.decode(data)

            def recv_multipart(self):
                return [msgspec.msgpack.encode(reply), raw]

        socket = Socket()
        fetched, payload = pub.fetch_state(
            socket, self.engine, rid, 0, n, self.digest, mm_input_digest=proof
        )
        self.assertEqual(payload, raw)
        self.assertEqual(socket.sent[-1], identity)

    def test_image_shard_disagreement_cannot_publish(self):
        rid = "mixed-images"
        rows = self.shards(rid, 129)
        for row in rows:
            row["mm_input_digest"] = "a" * 64
        rows[-1]["mm_input_digest"] = "b" * 64
        with pytest.raises(ValueError, match="capture mismatch"):
            pub.join_packets(
                self.wire, self.source, self.engine, self.nonce, rid, 1, rows
            )


def test_mm_eligibility_and_digest_include_actual_images():
    features = [
        MultiModalFeatureSpec(
            None, "image", CERTIFIED_PREFIX + "a" * 64, PlaceholderRange(5, 8)
        )
    ]
    request = NS(
        prompt_token_ids=[1] * 129,
        prefill_token_ids=[1] * 129,
        prompt_len=129,
        mm_features=features,
        prompt_embeds=None,
        lora_request=None,
    )
    assert not PrefixRuntime.request_eligible(request)
    assert PrefixRuntime.request_eligible(request, allow_mm=True)
    first = request_identity("model", "tokens", 129, feature_descriptors(features))
    features[0].identifier = CERTIFIED_PREFIX + "b" * 64
    assert first != request_identity(
        "model", "tokens", 129, feature_descriptors(features)
    )
    request.lora_request = object()
    assert not PrefixRuntime.request_eligible(request, allow_mm=True)
    request.lora_request = None
    features[0].identifier = "user-uuid"
    assert not PrefixRuntime.request_eligible(request, allow_mm=True)
