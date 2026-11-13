import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preflight import assess, memory_info, GIB

class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.report={'platform':'Linux','packages':{'torch':'test'},'ram':{'available_bytes':32*GIB},
                     'cuda':{'available':True,'compute_capability':[7,5], 'torch_arch_list':['sm_75'],
                             'free_bytes':7.5*GIB,'total_bytes':8*GIB}}
    def test_target_ready(self):
        self.assertEqual(assess(self.report,6.5),([],[]))
    def test_missing_cuda_and_package(self):
        self.report['cuda']['available']=False
        self.report['packages']['torch']=None
        errors,_=assess(self.report,6.5)
        self.assertEqual(len(errors),2)
    def test_low_memory_warnings(self):
        self.report['cuda']['free_bytes']=4*GIB
        self.report['ram']['available_bytes']=8*GIB
        errors,warnings=assess(self.report,6.5)
        self.assertFalse(errors)
        self.assertEqual(len(warnings),2)
    def test_unknown_arch_warning_not_false_failure(self):
        self.report['cuda']['torch_arch_list']=['sm_70','compute_70']
        errors,warnings=assess(self.report,6.5)
        self.assertFalse(errors)
        self.assertIn('sm_75',warnings[0])
    def test_too_small_gpu(self):
        self.report['cuda']['total_bytes']=4*GIB
        self.assertTrue(assess(self.report,6.5)[0])
    def test_ram_parser(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'meminfo'
            p.write_text('MemTotal: 2000 kB\nMemAvailable: 1000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n')
            self.assertEqual(memory_info(p)['available_bytes'],1024000)
            self.assertIsNone(memory_info(Path(d)/'missing'))
    def test_cuda_inspection_error(self):
        self.report['cuda']['error']='inspection failed'
        self.assertTrue(assess(self.report,6.5)[0])

if __name__=='__main__': unittest.main()
