import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
from output_utils import restore_outside, refuse_overwrite, write_exclusive
from extract_weights_only import extract

class TestCPU(unittest.TestCase):
 def setUp(self):
  self.original=np.full((3,4,3),50,dtype=np.uint8)
  self.pred=np.full((3,4,3),200,dtype=np.uint8)
  self.mask=np.array([[0,100,101,255]]*3,dtype=np.uint8)
 def test_preserve_original_outside(self):
  out=restore_outside(self.pred,self.original,self.mask)
  np.testing.assert_array_equal(out[self.mask<=100],self.original[self.mask<=100])
  np.testing.assert_array_equal(out[self.mask>100],self.pred[self.mask>100])
 def test_baseline_only_outside(self):
  baseline=np.full_like(self.original,25)
  out=restore_outside(self.pred,self.original,self.mask,baseline)
  np.testing.assert_array_equal(out[self.mask<=100],baseline[self.mask<=100])
  np.testing.assert_array_equal(out[self.mask>100],self.pred[self.mask>100])
 def test_inputs_unmodified(self):
  saved=[v.copy() for v in [self.pred,self.original,self.mask]]
  restore_outside(self.pred,self.original,self.mask)
  for x,y in zip(saved,[self.pred,self.original,self.mask]):np.testing.assert_array_equal(x,y)
 def test_bad_baseline_dimensions(self):
  with self.assertRaises(ValueError):restore_outside(self.pred,self.original,self.mask,np.zeros((4,4,3)))
 def test_bad_mask_dimensions(self):
  with self.assertRaises(ValueError):restore_outside(self.pred,self.original,np.zeros((1,1)))
 def test_output_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'a.png';refuse_overwrite(p);write_exclusive(p,b'keep')
   with self.assertRaises(FileExistsError):refuse_overwrite(p)
   with self.assertRaises(FileExistsError):write_exclusive(p,b'bad')
   self.assertEqual(p.read_bytes(),b'keep')
 def test_sidecar_overwrite_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'a.png';p.with_suffix('.json').write_text('{}')
   with self.assertRaises(FileExistsError):refuse_overwrite(p)
 def test_output_png_required(self):
  with self.assertRaises(ValueError):refuse_overwrite('a.jpg')
 def test_unknown_checkpoint_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'bad.pth';p.write_bytes(b'not the pinned model')
   with self.assertRaises(ValueError):extract(p,Path(t)/'out.pth')
   self.assertFalse((Path(t)/'out.pth').exists())

if __name__=='__main__':unittest.main()
