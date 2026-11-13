"""CPU-only checks of optional platform telemetry; no model imports or downloads."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1] / '_klein_backend.py'
spec = importlib.util.spec_from_file_location('backend_compat', BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

class BackendCompatibilityTests(unittest.TestCase):
    def test_resource_unavailable(self):
        with patch.object(backend, 'resource', None):
            self.assertIsNone(backend.cpu_peak_kib())

    def test_resource_error_is_nonfatal(self):
        def fail(_):
            raise OSError('telemetry unavailable')
        with patch.object(backend, 'resource', SimpleNamespace(RUSAGE_SELF=0, getrusage=fail)):
            self.assertIsNone(backend.cpu_peak_kib())

    def test_platform_units(self):
        fake = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=4096))
        with patch.object(backend, 'resource', fake):
            with patch.object(backend.sys, 'platform', 'linux'):
                self.assertEqual(backend.cpu_peak_kib(), 4096)
            with patch.object(backend.sys, 'platform', 'darwin'):
                self.assertEqual(backend.cpu_peak_kib(), 4)

    def test_help_without_resource_module(self):
        code = '''
import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError("No module named 'resource'")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
path = sys.argv[1]
sys.argv = [path, '--help']
runpy.run_path(path, run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-c', code, str(BACKEND)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--nf4-compute', result.stdout)

if __name__ == '__main__':
    unittest.main()
