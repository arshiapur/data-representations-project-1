import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mds', ROOT/'code/mds.py')
mds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mds)

class MDSTests(unittest.TestCase):
    def setUp(self):
        self.square = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])

    def test_square_spectrum_and_distances(self):
        d = mds.distances(self.square, 2)
        np.testing.assert_allclose(d[0], [0,1,np.sqrt(2),1])
        y, r = mds.classical_mds(d, 2)
        np.testing.assert_allclose(r['eigenvalues'], [1,1,0,0], atol=1e-14)
        np.testing.assert_allclose(mds.distances(y, 2), d, atol=1e-14)
        self.assertLess(r['normalized_distance_rmse'], 1e-14)

    def test_handout_triangle_gram(self):
        x = np.array([[0.,0.],[4.,0.],[0.,3.]])
        y, r = mds.classical_mds(mds.distances(x,2),2)
        expected = np.array([[25,-23,-2],[-23,73,-50],[-2,-50,52]])/9
        np.testing.assert_allclose(y@y.T, expected, atol=1e-13)
        np.testing.assert_allclose(r['eigenvalues'][:2], [12.9641479965,3.7025186701], atol=1e-9)

    def test_minkowski(self):
        x = np.array([[0.,0.],[3.,4.]])
        self.assertEqual(mds.distances(x,1)[0,1],7)
        self.assertEqual(mds.distances(x,np.inf)[0,1],4)
        self.assertAlmostEqual(mds.distances(x,1.5)[0,1], (3**1.5+4**1.5)**(1/1.5))
        self.assertEqual(mds.distances(x,1e300)[0,1],4)
        self.assertAlmostEqual(mds.distances(x*1e100,1000)[0,1]/1e100,4)

    def test_negative_eigenvalue(self):
        _, r = mds.classical_mds(mds.distances(self.square,1),2)
        np.testing.assert_allclose(r['eigenvalues'],[2,2,0,-1],atol=1e-14)
        self.assertEqual(r['negative_count'],1)
        self.assertAlmostEqual(r['negative_mass'],1)

    def test_rank_deficiency_and_zero(self):
        for x in (np.array([[0.],[1.],[2.]]),np.zeros((3,2))):
            y,r = mds.classical_mds(mds.distances(x,2),3)
            self.assertEqual(y.shape,(3,3))
            self.assertTrue((y[:,1:]==0).all())
            self.assertEqual(r['negative_count'],0)

    def test_cli_outputs_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'My_Iris.csv'
            np.savetxt(p,self.square,delimiter=',')
            for q,token in [('2','2'),('2.0','2'),('2.00','2'),('1.50','1.5'),('inf','inf')]:
                result = subprocess.run([sys.executable,str(ROOT/'code/mds.py'),'--input',str(p),'--q',q,'--dimensions','3'],cwd=tmp,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                r = json.loads(result.stdout)
                self.assertEqual(r['coordinate_csv'],f'My_Iris_mds_q{token}_3D.csv')
                self.assertTrue((Path(tmp)/r['plot_png']).is_file())
                self.assertEqual(np.loadtxt(Path(tmp)/r['coordinate_csv'],delimiter=',',skiprows=1).shape,(4,3))

    def test_invalid_cli_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'x.csv'
            p.write_text('1,2\n3,4\n')
            base = [sys.executable,str(ROOT/'code/mds.py'),'--input',str(p)]
            for q in ('0','-1','nan','Infinity','INF','1e999','bad'):
                r = subprocess.run(base+['--q',q,'--dimensions','2'],cwd=tmp,capture_output=True)
                self.assertNotEqual(r.returncode,0)
            r = subprocess.run(base+['--q','2','--dimensions','4'],cwd=tmp,capture_output=True)
            self.assertNotEqual(r.returncode,0)
            for content in ('','a,b\n1,nan\n2,3\n','1,2\n3\n','a,b\n1,cat\n2,dog\n','1,2\n','1,label\n2,3\n'):
                p.write_text(content)
                with self.assertRaises(ValueError):
                    mds.read_features(p)

if __name__ == '__main__':
    unittest.main(verbosity=2)
