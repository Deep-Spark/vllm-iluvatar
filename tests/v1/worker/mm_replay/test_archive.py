"""Standalone CPU/TCP correctness of P-only image leases and range replay."""

import multiprocessing as mp
import struct
import tempfile
import unittest
from pathlib import Path

from vllm_iluvatar.v1.worker.mm_replay import archive as a


def feature(key, start=3, rows=20):
    return dict(
        key=key, start=start, rows=rows, input=b"input-" + key.encode(), fields=[]
    )


def encoded(rows, hidden=8):
    return b"".join(
        struct.pack("<H", (r * hidden + c) % 65536)
        for r in range(rows)
        for c in range(hidden)
    )


def message(action="GET", generation="g1", digest="a" * 64, start=10, end=20):
    return (
        a.MM_OP,
        1,
        action,
        "P-engine",
        "request1",
        generation,
        digest,
        dict(start=start, end=end),
    )


def serve(root, pipe):
    import zmq

    archive = a.SourceArchive(
        Path(root), "P-engine", "session1", budget_bytes=20000, hidden_size=8
    )
    archive.admit("request1", "g1", "a" * 64, 129, [feature("img")])
    archive.publish("request1", "g1", "img", encoded(20))
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    pipe.send(port)
    try:
        for _ in range(2):
            import msgspec

            meta, raw = archive.handle(msgspec.msgpack.decode(sock.recv()))
            sock.send_multipart([msgspec.msgpack.encode(meta), raw])
    finally:
        sock.close(0)
        ctx.term()


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = a.SourceArchive(
            self.root, "P-engine", "session1", budget_bytes=20000, hidden_size=8
        )
        self.f = feature("img")
        self.store.admit("request1", "g1", "a" * 64, 129, [self.f])

    def tearDown(self):
        self.tmp.cleanup()

    def test_pending_and_exact_boundaries(self):
        self.assertEqual(self.store.handle(message())[0]["status"], "PENDING")
        self.store.publish("request1", "g1", "img", encoded(20))
        for start, end in [(0, 3), (2, 4), (3, 4), (10, 20), (22, 24), (0, 129)]:
            reply, raw = self.store.handle(message(start=start, end=end))
            self.assertTrue(
                a.validate_reply(
                    reply,
                    raw,
                    engine="P-engine",
                    rid="request1",
                    generation="g1",
                    digest="a" * 64,
                    start=start,
                    end=end,
                    hidden_size=8,
                    features=[self.f],
                )
            )
            lo, hi = max(3, start) - 3, min(23, end) - 3
            self.assertEqual(
                raw, encoded(20)[max(0, lo) * 16 : max(0, hi) * 16] if lo < hi else b""
            )

    def test_generation_and_digest_fences(self):
        for m in [message(generation="old"), message(digest="b" * 64)]:
            self.assertEqual(
                self.store.handle(m)[0]["status"], "IDENTITY_OR_PAYLOAD_ERROR"
            )
        self.store.handle(message(action="RELEASE", generation="old"))
        self.assertIsNotNone(self.store.request("request1", "g1"))
        self.assertEqual(
            self.store.handle(message(action="RELEASE"))[0]["status"], "RELEASED"
        )
        with self.assertRaises(ValueError):
            self.store.admit("request1", "g1", "a" * 64, 129, [self.f])
        self.assertTrue(self.store.admit("request1", "g2", "b" * 64, 129, [self.f]))
        self.assertEqual(
            self.store.handle(message(action="RELEASE"))[0]["status"],
            "IDENTITY_OR_PAYLOAD_ERROR",
        )
        self.assertIsNotNone(self.store.request("request1", "g2"))

    def test_wire_corruption_and_missing_coverage(self):
        self.store.publish("request1", "g1", "img", encoded(20))
        reply, raw = self.store.handle(message())
        expected = dict(
            engine="P-engine",
            rid="request1",
            generation="g1",
            digest="a" * 64,
            start=10,
            end=20,
            hidden_size=8,
            features=[self.f],
        )
        for bad in [
            dict(reply, records=[]),
            dict(reply, session="other"),
            dict(reply, byte_count=0),
        ]:
            with self.assertRaises(ValueError):
                a.validate_reply(bad, raw, session="session1", **expected)
        with self.assertRaises(ValueError):
            a.validate_reply(reply, raw[:-1] + bytes([raw[-1] ^ 1]), **expected)

    def test_budget_and_active_leases(self):
        root = self.root / "small"
        s = a.SourceArchive(
            root, "P-engine", "session1", budget_bytes=680, hidden_size=8
        )
        f = feature("A")
        g = feature("B")
        h = feature("C")
        self.assertTrue(s.admit("a", "g", "1" * 64, 129, [f]))
        self.assertTrue(s.admit("b", "g", "2" * 64, 129, [g]))
        self.assertFalse(s.admit("c", "g", "3" * 64, 129, [h]))
        s.release("a", "g", "1" * 64)
        self.assertTrue(s.admit("c", "g", "3" * 64, 129, [h]))
        self.assertLessEqual(s.stats()["reserved_bytes"], 680)
        self.assertEqual(s.stats()["leased_images"], 2)

    def test_same_image_multiple_positions(self):
        f = feature("img", start=3)
        g = feature("img", start=100)
        s = a.SourceArchive(
            self.root / "dup", "P-engine", "session1", budget_bytes=1000, hidden_size=8
        )
        self.assertTrue(s.admit("request1", "g1", "a" * 64, 129, [f, g]))
        s.publish("request1", "g1", "img", encoded(20))
        reply, raw = s.handle(message(start=0, end=129))
        self.assertEqual(raw, encoded(20) * 2)
        self.assertTrue(
            a.validate_reply(
                reply,
                raw,
                engine="P-engine",
                rid="request1",
                generation="g1",
                digest="a" * 64,
                start=0,
                end=129,
                hidden_size=8,
                features=[f, g],
            )
        )

    def test_private_directory_tcp(self):
        import msgspec
        import zmq

        ctxmp = mp.get_context("spawn")
        left, right = ctxmp.Pipe()
        p = ctxmp.Process(
            target=serve, args=(str(self.root / "producer-private"), right)
        )
        p.start()
        port = left.recv()
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        try:
            for _ in range(2):
                sock.send(msgspec.msgpack.encode(message()))
                meta, raw = sock.recv_multipart()
                reply = msgspec.msgpack.decode(meta)
                self.assertTrue(
                    a.validate_reply(
                        reply,
                        raw,
                        engine="P-engine",
                        rid="request1",
                        generation="g1",
                        digest="a" * 64,
                        start=10,
                        end=20,
                        hidden_size=8,
                        features=[self.f],
                    )
                )
                self.assertNotIn(str(self.root).encode(), meta)
        finally:
            sock.close(0)
            ctx.term()
            p.join(5)
        self.assertEqual(p.exitcode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
