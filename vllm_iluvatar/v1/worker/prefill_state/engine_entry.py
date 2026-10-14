"""P/D bootstrap for service-owned prefix states and consumer KV receipts."""
import os
import runpy
import uuid
from .scheduler_receipt import install

# Multiprocessing spawn imports __mp_main__: install before engine creation too.
# Each independently launched P or D service owns its own nonce and local files.
# Spawned children inherit it; the router never needs to know it.
if __name__ == '__main__':
    os.environ['P21_RUN_UUID'] = str(uuid.uuid4())
install()
if __name__=='__main__':
    runpy.run_module('vllm.entrypoints.cli.main',run_name='__main__')
