import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from PIL import Image
from harmonize_anchor import compose,read_inputs

class Tests(unittest.TestCase):
    def test_fixed_background(self):
        src=Image.fromarray(np.full((10,10,3),100,np.uint8));base=Image.fromarray(np.full((10,10,3),90,np.uint8))
        m=np.zeros((10,10),np.uint8);m[2:5,2:5]=255
        result=compose(src,Image.fromarray(m),base,Image.fromarray(np.full((5,5,3),60,np.uint8)),blend=1)
        self.assertTrue(np.all(result[m==0]==90));self.assertTrue(np.all(result[m>0]==60))
    def test_cli_reference_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src=np.full((12,16,3),140,np.uint8);mask=np.zeros((12,16),np.uint8);mask[3:9,4:12]=1
            Image.fromarray(src).save(p/'s.png');Image.fromarray(mask).save(p/'m.png');Image.fromarray(src//2).save(p/'r.png')
            cmd=[sys.executable,str(Path(__file__).with_name('harmonize_anchor.py')),'--image',str(p/'s.png'),'--mask',str(p/'m.png'),'--reference',str(p/'r.png'),'--output',str(p/'o.png'),'--prompt','test']
            subprocess.run(cmd,check=True,capture_output=True)
            out=np.asarray(Image.open(p/'o.png'));np.testing.assert_array_equal(out[mask==0],src[mask==0]);self.assertTrue(np.all(out[mask>0]==105))
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
    def test_mask_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);Image.new('RGB',(4,4)).save(p/'s.png');Image.new('RGB',(4,4)).save(p/'m.png')
            with self.assertRaises(ValueError):read_inputs(p/'s.png',p/'m.png')
            Image.new('L',(3,3),255).save(p/'m.png')
            with self.assertRaises(ValueError):read_inputs(p/'s.png',p/'m.png')
            Image.new('L',(4,4),0).save(p/'m.png')
            with self.assertRaises(ValueError):read_inputs(p/'s.png',p/'m.png')

if __name__=='__main__':unittest.main()

class BlendAndBackendTests(unittest.TestCase):
    def test_blend_endpoints_and_invalid(self):
        src=Image.fromarray(np.full((3,3,3),100,np.uint8));ref=Image.fromarray(np.full((3,3,3),20,np.uint8));m=Image.new('L',(3,3),255)
        for blend,value in [(0,100),(.5,60),(1,20)]:
            self.assertTrue(np.all(compose(src,m,src,ref,blend)==value))
        for blend in [-1,2,float('nan')]:
            with self.assertRaises(ValueError):compose(src,m,src,ref,blend)
    def test_backend_command_contract(self):
        import harmonize_anchor as cli
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);Image.new('RGB',(16,16),(100,100,100)).save(p/'s.png');Image.new('L',(16,16),255).save(p/'m.png')
            def fake_run(cmd,check):
                self.assertEqual(cmd[cmd.index('--mode')+1],'nf4')
                self.assertEqual(cmd[cmd.index('--nf4-compute')+1],'fp16')
                self.assertIn('--anchor-mask',cmd)
                d=Path(cmd[cmd.index('--out')+1]);d.mkdir()
                Image.new('RGB',(16,16),(20,20,20)).save(d/'seed42-anchor1-raw.png')
            with patch.object(sys,'argv',['harmonize_anchor.py','--image',str(p/'s.png'),'--mask',str(p/'m.png'),'--output',str(p/'out.png'),'--prompt','test']),patch.object(cli.subprocess,'run',side_effect=fake_run):
                cli.main()
            self.assertTrue(np.all(np.asarray(Image.open(p/'out.png'))==60))
