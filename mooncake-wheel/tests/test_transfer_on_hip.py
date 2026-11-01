import multiprocessing
import socket
import unittest

import torch


def get_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


SUBALLOC_HOLDER_BYTES = 4 * 1024 * 1024
SUBALLOC_OFFSETS = (1024 * 1024 + 4096, 3 * 1024 * 1024)


def _suballoc_receiver(conn, nbytes):
    """Peer process for the HIP IPC sub-allocation test.

    Registers two views at nonzero offsets inside one hipMalloc segment, waits
    for the writer, and reports what landed where.
    """
    from mooncake.engine import TransferEngine

    try:
        engine = TransferEngine()
        hostname = get_ip()
        if engine.initialize(hostname, "P2PHANDSHAKE", "hip", "") != 0:
            conn.send(None)
            return
        if ":" in hostname:
            target_name = f"[{hostname}]:{engine.get_rpc_port()}"
        else:
            target_name = f"{hostname}:{engine.get_rpc_port()}"

        holder = torch.zeros(SUBALLOC_HOLDER_BYTES, dtype=torch.uint8,
                             device="cuda:0")
        views = [holder[off:off + nbytes] for off in SUBALLOC_OFFSETS]
        register_rc = [engine.register_memory(v.data_ptr(), nbytes)
                       for v in views]
        conn.send((target_name, [v.data_ptr() for v in views], register_rc))

        conn.recv()
        torch.cuda.synchronize()
        expected = torch.zeros_like(holder)
        for i, off in enumerate(SUBALLOC_OFFSETS):
            expected[off:off + nbytes] = i + 1
        mismatched = int((holder != expected).sum().item())
        unregister_rc = [engine.unregister_memory(v.data_ptr()) for v in views]
        reregister_rc = [engine.register_memory(v.data_ptr(), nbytes)
                         for v in views]
        for v in views:
            engine.unregister_memory(v.data_ptr())
        conn.send((mismatched, unregister_rc, reregister_rc))
    except Exception as e:
        conn.send(repr(e))


class TestTransferOnHip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mooncake.engine import TransferEngine

        if not torch.cuda.is_available():
            raise unittest.SkipTest("ROCm device not available")
        if not getattr(torch.version, "hip", None):
            raise unittest.SkipTest("PyTorch is not built with HIP support")

        cls.engine = TransferEngine()
        cls.local_hostname = get_ip()
        ret = cls.engine.initialize(cls.local_hostname, "P2PHANDSHAKE", "hip", "")
        if ret != 0:
            raise unittest.SkipTest(f"Failed to initialize HIP transport (code {ret})")

        if ":" in cls.local_hostname:
            cls.target_name = f"[{cls.local_hostname}]:{cls.engine.get_rpc_port()}"
        else:
            cls.target_name = f"{cls.local_hostname}:{cls.engine.get_rpc_port()}"

        cls.device = torch.device("cuda:0")
        cls.numel = 1024 * 1024
        cls.src_tensor = torch.ones(cls.numel, dtype=torch.uint8, device=cls.device)
        cls.dst_tensor = torch.zeros(cls.numel, dtype=torch.uint8, device=cls.device)
        cls.nbytes = cls.src_tensor.numel() * cls.src_tensor.element_size()

        ret1 = cls.engine.register_memory(cls.src_tensor.data_ptr(), cls.nbytes)
        ret2 = cls.engine.register_memory(cls.dst_tensor.data_ptr(), cls.nbytes)
        if ret1 != 0 or ret2 != 0:
            raise RuntimeError("Failed to register HIP buffers")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            if hasattr(cls, "src_tensor"):
                cls.engine.unregister_memory(cls.src_tensor.data_ptr())
            if hasattr(cls, "dst_tensor"):
                cls.engine.unregister_memory(cls.dst_tensor.data_ptr())

    def test_transfer_sync_write(self):
        self.src_tensor.fill_(123)
        self.dst_tensor.zero_()

        ret = self.engine.transfer_sync_write(
            self.target_name,
            self.src_tensor.data_ptr(),
            self.dst_tensor.data_ptr(),
            self.nbytes,
        )
        self.assertEqual(ret, 0)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(self.src_tensor, self.dst_tensor))

    def test_transfer_sync_read(self):
        self.src_tensor.zero_()
        self.dst_tensor.fill_(77)

        ret = self.engine.transfer_sync_read(
            self.target_name,
            self.src_tensor.data_ptr(),
            self.dst_tensor.data_ptr(),
            self.nbytes,
        )
        self.assertEqual(ret, 0)
        torch.cuda.synchronize()
        self.assertTrue(torch.all(self.src_tensor == 77))

    def test_ipc_write_into_suballocation(self):
        # Caching allocators hand out tensors inside larger hipMalloc
        # segments, and a HIP IPC handle always maps the whole segment, so a
        # write to a buffer at a nonzero offset must land at that offset in
        # the peer process.
        nbytes = 256 * 1024
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        peer = ctx.Process(target=_suballoc_receiver, args=(child_conn, nbytes))
        peer.start()
        try:
            self.assertTrue(parent_conn.poll(60), "peer did not start")
            info = parent_conn.recv()
            if info is None:
                self.skipTest("peer failed to initialize HIP transport")
            self.assertIsInstance(info, tuple, info)
            target_name, dst_ptrs, register_rc = info
            self.assertEqual(register_rc, [0, 0])

            for i, dst_ptr in enumerate(dst_ptrs):
                self.src_tensor[:nbytes].fill_(i + 1)
                torch.cuda.synchronize()
                ret = self.engine.transfer_sync_write(
                    target_name, self.src_tensor.data_ptr(), dst_ptr, nbytes
                )
                self.assertEqual(ret, 0)
            parent_conn.send("written")

            self.assertTrue(parent_conn.poll(60), "peer did not report")
            result = parent_conn.recv()
            self.assertIsInstance(result, tuple, result)
            mismatched, unregister_rc, reregister_rc = result
            self.assertEqual(mismatched, 0)
            self.assertEqual(unregister_rc, [0, 0])
            self.assertEqual(reregister_rc, [0, 0])
        finally:
            peer.join(timeout=30)
            if peer.is_alive():
                peer.terminate()


if __name__ == "__main__":
    unittest.main()
