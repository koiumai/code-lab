"""Model-free wrapper contract tests. These do NOT validate GPU inference."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import harmonize

class WrapperContractTests(unittest.TestCase):
    def run_case(self, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            src = np.full((64,96,3), 160, np.uint8)
            mask = np.zeros((64,96), np.uint8); mask[16:48,32:64] = 255
            Image.fromarray(src).save(d/'input.png')
            Image.fromarray(mask).save(d/'mask.png')
            def backend(cmd, check):
                self.assertTrue(check)
                self.assertEqual(Path(cmd[1]).name, '_klein_backend.py')
                for key,val in [('--mode','nf4'),('--nf4-compute','fp16'),('--budget-gib','6.5'),('--max-side','768')]:
                    self.assertEqual(cmd[cmd.index(key)+1],val)
                if fail: raise subprocess.CalledProcessError(1,cmd)
                out=Path(cmd[cmd.index('--out')+1]); out.mkdir()
                seed=cmd[cmd.index('--seeds')+1]
                Image.fromarray(src//2).save(out/f'seed{seed}-raw.png')
            args=['harmonize.py','--image',str(d/'input.png'),'--mask',str(d/'mask.png'),'--output',str(d/'result.png'),'--seed','123']
            with patch.object(sys,'argv',args), patch.object(harmonize.subprocess,'run',side_effect=backend):
                if fail:
                    with self.assertRaises(subprocess.CalledProcessError): harmonize.main()
                    self.assertFalse((d/'result.png').exists())
                else:
                    harmonize.main()
                    result=np.asarray(Image.open(d/'result.png'))
                    allowed=np.asarray(Image.open(d/'result-assets/allowed-union.png'))>0
                    np.testing.assert_array_equal(result[~allowed],src[~allowed])
            report=json.loads((d/'result-assets/report.json').read_text())
            if fail: self.assertIn('CalledProcessError',report['error'])
            else: self.assertEqual(report['outside_allowed_changed'],0)
    def test_backend_command_and_composition(self): self.run_case()
    def test_backend_failure_is_reported(self): self.run_case(fail=True)

if __name__=='__main__': unittest.main()
