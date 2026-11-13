import unittest
import sys
from pathlib import Path
import subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prompt_profiles import build_prompt
from compositing import default_shadow_mask, compose

class ProfileTests(unittest.TestCase):
    def test_legacy_exact(self):
        base=('Adjust the pasted foreground object lighting and color to match the surrounding scene. '
              'Preserve every object shape, material, texture and background geometry exactly. ')
        self.assertEqual(build_prompt(),base+'Add a subtle natural contact and cast shadow attached to the object on the ground, consistent with scene lighting.')
        self.assertEqual(build_prompt(no_shadow=True),base+'Do not add any shadows.')
        self.assertEqual(build_prompt(custom_prompt='custom'),'custom')
    def test_photometric_exact_tested_prompts(self):
        scenes=[('pink teddy bear','soft overcast daylight on the mountain road'),
                ('woman standing on the paved path','the existing sunlight and environmental color on the park path')]
        for subject,lighting in scenes:
            expected=(f'Relight only the {subject} to match {lighting}. '
                      'Adjust the foreground subject exposure, white balance, color cast, and shading so that it looks photographed in this scene, not separately lit. '
                      'Its current brightness and color cast are allowed to change; keep its identity, material colors, silhouette, pose and fine detail. '
                      'Keep the background exposure, white balance, colors, contrast, scenery and camera framing unchanged. '
                      'Do not apply a global color grade to the photograph. ')
            for no_shadow,suffix in [(True,'Do not add or alter cast shadows in this test.'),
                                    (False,'Also add a subtle natural contact shadow attached to its feet or base, consistent with the existing light direction.')]:
                self.assertEqual(build_prompt('photometric',subject,lighting,no_shadow),expected+suffix)
    def test_reject_ambiguous_profiles(self):
        cases=[{'subject':'bear'}, {'lighting':'sun'}, {'profile':'photometric'},
               {'profile':'photometric','subject':'  ','lighting':'sun'},
               {'profile':'photometric','subject':'bear','lighting':'\t'},
               {'profile':'photometric','subject':'bear','lighting':'sun','custom_prompt':''}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):build_prompt(**kwargs)
    def test_cli_validation_before_reading_inputs(self):
        script=Path(__file__).resolve().parents[1]/'harmonize.py'
        base=[sys.executable,str(script),'--image','MISSING','--mask','MISSING','--output','MISSING.png']
        cases=[['--lighting-profile','photometric'],['--subject','bear'],
               ['--lighting-profile','photometric','--subject','bear','--lighting','sun','--prompt','custom'],
               ['--shadow-vertical-scale','2','--no-shadow'],
               ['--shadow-vertical-scale','2','--shadow-mask','MISSING'],
               ['--shadow-vertical-scale','nan'],['--shadow-vertical-scale','0.5'],['--shadow-vertical-scale','4.1']]
        for options in cases:
            result=subprocess.run(base+options,capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertNotIn('Traceback',result.stderr)

class ShadowScaleTests(unittest.TestCase):
    def setUp(self):
        self.obj=np.zeros((120,100),np.float32);self.obj[35:65,40:60]=1
    def test_default_old_formula_exact(self):
        yy,xx=np.mgrid[:120,:100]
        old=((((xx-49.5)/14)**2+((yy-(64+.08*30))/5.4)**2)<=1)&(self.obj==0)
        np.testing.assert_array_equal(default_shadow_mask(self.obj),old.astype(np.float32))
    def test_expansion_superset_and_object_not_changed(self):
        original=self.obj.copy();small=default_shadow_mask(self.obj);large=default_shadow_mask(self.obj,3)
        self.assertFalse(np.any((small>0)&(large==0)))
        self.assertFalse(np.any((large>0)&(self.obj>0)))
        self.assertLess(np.where(large)[0].min(),np.where(small)[0].min())
        self.assertGreater(np.where(large)[0].max(),np.where(small)[0].max())
        np.testing.assert_array_equal(self.obj,original)
    def test_edge_clipping_and_invalid(self):
        obj=np.zeros_like(self.obj);obj[-10:,30:50]=1
        result=default_shadow_mask(obj,4)
        self.assertEqual(result.shape,obj.shape)
        self.assertFalse(np.any(result[obj>0]))
        for scale in [0,.9,4.01,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):default_shadow_mask(obj,scale)
    def test_extended_region_outside_exact(self):
        src=np.full((120,100,3),150,np.uint8);ref=np.full_like(src,80)
        for scale in [1,2,4]:
            shadow=default_shadow_mask(self.obj,scale)
            for mode in ['gain','delta','blend']:
                out,_=compose(src,ref,self.obj,shadow,object_mode=mode)
                outside=(self.obj==0)&(shadow==0)
                np.testing.assert_array_equal(out[outside],src[outside])

if __name__=='__main__':unittest.main()
