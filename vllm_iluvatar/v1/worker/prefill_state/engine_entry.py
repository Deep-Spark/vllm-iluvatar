"""D engine bootstrap for bounded current-request CPU ownership receipts."""
import runpy
from .scheduler_receipt import install

# Multiprocessing spawn imports __mp_main__: install before engine creation too.
install()
if __name__=='__main__':
    runpy.run_module('vllm.entrypoints.cli.main',run_name='__main__')
