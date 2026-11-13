import unittest
from pathlib import Path
import sys
import tempfile
import subprocess
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compositing import compose

class ObjectModeTests(unittest.TestCase):
    def setUp(self):
        self.src=np.full((32,40,3),100,dtype=np.uint8)
        self.ref=np.full_like(self.src,[160,70,120])
        self.obj=np.zeros((32,40),np.float32); self.obj[5:20,10:25]=1
        self.shadow=np.zeros_like(self.obj); self.shadow[20:25,10:28]=1
    def run_mode(self,mode,strength):
        return compose(self.src,self.ref,self.obj,self.shadow,object_mode=mode,object_strength=strength)[0]
    def test_all_modes_outside_exact(self):
        allowed=(self.obj>0)|(self.shadow>0)
        for mode in ('gain','delta','blend'):
            for strength in (0,.5,1):
                out=self.run_mode(mode,strength)
                np.testing.assert_array_equal(out[~allowed],self.src[~allowed])
    def test_zero_preserves_object_and_shadow_is_independent(self):
        baseline=self.run_mode('gain',1)
        for mode in ('gain','delta','blend'):
            for strength in (0,.5,1):
                out=self.run_mode(mode,strength)
                np.testing.assert_array_equal(out[self.obj==0],baseline[self.obj==0])
                if strength==0:
                    np.testing.assert_array_equal(out[self.obj>0],self.src[self.obj>0])
    def test_blend_endpoints_and_half(self):
        for strength in (0,.5,1):
            out=self.run_mode('blend',strength)
            expected=np.uint8(np.rint(self.src*(1-strength)+self.ref*strength))
            np.testing.assert_array_equal(out[self.obj>0],expected[self.obj>0])
    def test_blend_soft_coverage(self):
        self.obj[5:20,10:25]=.5
        out=self.run_mode('blend',1)
        expected=np.uint8(np.rint(self.src*.5+self.ref*.5))
        np.testing.assert_array_equal(out[self.obj>0],expected[self.obj>0])
    def test_delta_constant_reaches_reference(self):
        out=self.run_mode('delta',1)
        np.testing.assert_array_equal(out[self.obj>0],self.ref[self.obj>0])
    def test_gain_default_matches_old_formula(self):
        s=self.src.astype(np.float32); r=self.ref.astype(np.float32)
        radius=max(2,min(self.obj.shape)*.015)
        weight=gaussian_filter(self.obj,radius)+1e-6
        smooth=lambda x: np.stack([gaussian_filter(x[:,:,c]*self.obj,radius)/weight for c in range(3)],-1)
        gain=np.clip((smooth(r)+16)/(smooth(s)+16),.5,1.7)
        expected=np.uint8(np.rint(np.clip(s*(1+self.obj[:,:,None]*(gain-1)),0,255)))
        default,_=compose(self.src,self.ref,self.obj,self.shadow)
        np.testing.assert_array_equal(default[self.obj>0],expected[self.obj>0])
        np.testing.assert_array_equal(default,self.run_mode('gain',1))
    def test_invalid_options(self):
        for mode,strength in [('bad',1),('gain',-.1),('delta',1.1),('blend',float('nan'))]:
            with self.assertRaises(ValueError): self.run_mode(mode,strength)
    def test_cli_reports_mode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            Image.fromarray(self.src).save(p/'src.png')
            Image.fromarray(self.ref).save(p/'ref.png')
            Image.fromarray(np.uint8(self.obj*255)).save(p/'mask.png')
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'harmonize.py'),
                '--image',str(p/'src.png'),'--mask',str(p/'mask.png'),'--reference',str(p/'ref.png'),
                '--output',str(p/'out.png'),'--object-mode','blend','--object-strength','.5','--no-shadow'],check=True,capture_output=True)
            report=json.loads((p/'out-assets/report.json').read_text())
            self.assertEqual(report['object_mode'],'blend')
            self.assertEqual(report['object_strength'],.5)

if __name__=='__main__':unittest.main()
