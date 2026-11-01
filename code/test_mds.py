import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from mds import distances, classical_mds, parse_q

class TestMDS(unittest.TestCase):
    def test_handout_triangle(self):
        x=np.array([[0,0],[4,0],[0,3]])
        d=distances(x,2)
        np.testing.assert_allclose(d,[[0,4,3],[4,0,5],[3,5,0]])
        z,diag=classical_mds(d,2)
        np.testing.assert_allclose(distances(z,2),d,atol=1e-12)
        expected=np.array([[25,-23,-2],[-23,73,-50],[-2,-50,52]])/9
        np.testing.assert_allclose(z@z.T,expected,atol=1e-12)
        self.assertEqual(diag['negative_count'],0)
    def test_square_and_negative_spectrum(self):
        x=np.array([[0,0],[1,0],[0,1],[1,1]])
        z,diag=classical_mds(distances(x,2),3)
        np.testing.assert_allclose(diag['eigenvalues'],[1,1,0,0],atol=1e-12)
        np.testing.assert_allclose(distances(z,2),distances(x,2),atol=1e-12)
        _,l1=classical_mds(distances(x,1),2)
        np.testing.assert_allclose(l1['eigenvalues'],[2,2,0,-1],atol=1e-12)
        self.assertEqual(l1['negative_count'],1)
        self.assertAlmostEqual(l1['negative_mass'],1)
    def test_minkowski_and_extreme_q(self):
        x=np.array([[0,0],[3,4],[3,4]])
        self.assertEqual(distances(x,1)[0,1],7)
        self.assertEqual(distances(x,np.inf)[0,1],4)
        self.assertAlmostEqual(distances(x,1e6)[0,1],4)
        self.assertEqual(distances(x,2)[1,2],0)
        np.testing.assert_allclose(distances(x,1.5)[0,1],(3**1.5+4**1.5)**(1/1.5))
    def test_degenerate(self):
        z,diag=classical_mds(np.zeros((3,3)),3)
        np.testing.assert_array_equal(z,np.zeros((3,3)))
        self.assertIsNone(diag['positive_mass_fraction_2D'])
        self.assertEqual(diag['normalized_distance_error'],0)
    def test_q_and_cli(self):
        for value in ['2','2.0','2.00','2e0']: self.assertEqual(parse_q(value)[1],'2')
        self.assertEqual(parse_q('1.500')[1],'1.5')
        for value in ['0','-1','nan','Infinity','1e9999','garbage']:
            with self.assertRaises(ValueError):parse_q(value)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=root/'Mixed_Case.csv'
            x.write_text('x,y\n0,0\n3,0\n0,4\n')
            cmd=[sys.executable,str(Path(__file__).with_name('mds.py')),'--input',str(x),'--q','2.00','--dimensions','2']
            run=subprocess.run(cmd,cwd=tmp,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            diag=json.loads(run.stdout);self.assertEqual(len(diag['eigenvalues']),3)
            self.assertTrue((root/'Mixed_Case_mds_q2_2D.png').exists())
            self.assertTrue((root/'Mixed_Case_mds_q2_2D.csv').exists())
            bad=subprocess.run(cmd[:-1]+['4'],cwd=tmp,capture_output=True)
            self.assertNotEqual(bad.returncode,0)
            x.write_text('x,y\n0,nan\n3,0\n')
            self.assertNotEqual(subprocess.run(cmd,cwd=tmp,capture_output=True).returncode,0)

if __name__=='__main__':unittest.main(verbosity=2)
