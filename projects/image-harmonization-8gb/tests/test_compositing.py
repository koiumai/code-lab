import unittest
import tempfile
import subprocess
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compositing import compose, load_mask, default_shadow_mask

class CompositionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(5)
        self.src=rng.integers(30,200,(48,64,3),dtype=np.uint8)
        self.obj=np.zeros((48,64),np.float32);self.obj[10:30,20:40]=1
        self.shadow=default_shadow_mask(self.obj)
    def test_exact_copy_and_dimensions(self):
        out,report=compose(self.src,self.src//2,self.obj,self.shadow)
        outside=~((self.obj>0)|(self.shadow>0))
        np.testing.assert_array_equal(out[outside],self.src[outside])
        self.assertEqual(out.shape,self.src.shape)
        self.assertEqual(report['outside_allowed_changed'],0)
        self.assertTrue(np.any(out!=self.src))
    def test_no_shadow(self):
        out,_=compose(self.src,self.src//2,self.obj,np.zeros_like(self.obj))
        np.testing.assert_array_equal(out[self.obj==0],self.src[self.obj==0])
    def test_identity(self):
        out,_=compose(self.src,self.src,self.obj,self.shadow)
        np.testing.assert_array_equal(out,self.src)
    def test_shadow_never_brightens(self):
        ref=np.full_like(self.src,255)
        out,_=compose(self.src,ref,self.obj,self.shadow)
        np.testing.assert_array_equal(out[self.obj==0],self.src[self.obj==0])
    def test_soft_edges_and_overlap(self):
        obj=self.obj.copy();obj[10,:]*=.1
        a,_=compose(self.src,self.src//2,obj,np.ones_like(obj))
        b,_=compose(self.src,self.src//2,obj,np.zeros_like(obj))
        np.testing.assert_array_equal(a[obj>0],b[obj>0])
    def test_contact_shadow_has_no_feather_gap(self):
        src=np.full((24,24,3),120,np.uint8);ref=np.full_like(src,60)
        obj=np.zeros((24,24),np.float32);obj[5:19,5:12]=1
        sh=np.zeros_like(obj);sh[3:21,12:21]=1
        out,_=compose(src,ref,obj,sh,strength=1,feather=3)
        self.assertLess(int(out[12,12,0]),90)  # No bright gap along object edge.
        self.assertEqual(int(out[12,11,0]),int(out[12,8,0]))  # No bright object rim.
    def test_invalid_arrays(self):
        for obj in (np.zeros_like(self.obj),self.obj-1,np.full_like(self.obj,np.nan)):
            with self.assertRaises(ValueError):compose(self.src,self.src,obj,self.shadow)
        with self.assertRaises(ValueError):compose(self.src,self.src,self.obj,self.shadow,strength=2)
    def test_mask_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'m.png'
            for scale in (1,255):
                Image.fromarray(np.uint8(self.obj*scale)).save(f)
                np.testing.assert_array_equal(load_mask(f,(64,48)),self.obj)
            with self.assertRaises(ValueError):load_mask(f,(1,1))
            Image.fromarray(self.src).save(f)
            with self.assertRaises(ValueError):load_mask(f,(64,48))
    def test_default_mask_excludes_object(self):
        self.assertTrue(np.any(self.shadow))
        self.assertFalse(np.any((self.obj>0)&(self.shadow>0)))
    def test_model_free_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            Image.fromarray(self.src).save(d/'s.png')
            Image.fromarray(self.src//2).save(d/'r.png')
            Image.fromarray(np.uint8(self.obj*255)).save(d/'m.png')
            args=[sys.executable,str(Path(__file__).resolve().parents[1]/'harmonize.py'),'--image',str(d/'s.png'),'--mask',str(d/'m.png'),'--reference',str(d/'r.png'),'--output',str(d/'final.png'),'--no-shadow']
            subprocess.run(args,check=True,capture_output=True)
            out=np.asarray(Image.open(d/'final.png'))
            np.testing.assert_array_equal(out[self.obj==0],self.src[self.obj==0])
            self.assertTrue((d/'final-assets'/'shadow-allowed.png').exists())
            self.assertEqual(json.loads((d/'final-assets'/'report.json').read_text())['outside_allowed_changed'],0)
            self.assertNotEqual(subprocess.run(args,capture_output=True).returncode,0)

if __name__=='__main__':unittest.main()
