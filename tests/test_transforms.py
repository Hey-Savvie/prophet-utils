import numpy as np
import pandas as pd
import unittest

from prophet_utils import transforms


class AbstractTestTransform(unittest.TestCase):

    def _test_roundtrip(self, real: pd.Series):
        work = self.transform.to_work_series(real)
        back = self.transform.to_real_series(work)
        pd.testing.assert_series_equal(real, back, rtol=1e-12, atol=1e-12)

    def _test_to_work_finite_inputs(self, real: pd.Series):
        work = self.transform.to_work_series(real)
        self.assertGreater(work.min(), -np.infty)
        self.assertLess(work.max(), np.infty)
        self.assertEqual(0, work.isna().sum())
        self.assertEqual(len(real), len(work))
        for i in range(1, len(real)):
            self.assertEqual(real[i - 1] <= real[i], work[i - 1] <= work[i])

    def _test_invalid_real_inputs(self):
        if self.transform.lower_bound > -np.infty:
            with self.assertRaises(ValueError):
                self.transform.to_work_series(pd.Series([self.transform.lower_bound - 1e-3]))
        if self.transform.upper_bound < np.infty:
            with self.assertRaises(ValueError):
                self.transform.to_work_series(pd.Series([self.transform.upper_bound + 1e-3]))


class TestLogarithmic(AbstractTestTransform):

    @classmethod
    def setUpClass(cls):
        cls.transform = transforms.Logarithmic(1e-3)

    def test_roundtrip(self):
        real = pd.Series([0, 1e-4, 0.5, 10, np.infty])
        self._test_roundtrip(real)

    def test_to_work(self):
        real = pd.Series([0, 2e-4, 0.25, 12])
        self._test_to_work_finite_inputs(real)

    def test_to_real(self):
        work = pd.Series([-np.infty, np.infty])
        expected = pd.Series([0., np.infty])
        actual = self.transform.to_real_series(work)
        pd.testing.assert_series_equal(expected, actual, rtol=1e-12, atol=1e-12)

    def test_invalid_real_inputs(self):
        self._test_invalid_real_inputs()


class TestLogit(AbstractTestTransform):

    @classmethod
    def setUpClass(cls):
        cls.transform = transforms.Logit(1e-3)

    def test_roundtrip(self):
        real = pd.Series([0, 1e-4, 0.5, 0.9999, 1])
        self._test_roundtrip(real)

    def test_to_work(self):
        real = pd.Series([0, 2e-4, 0.25, 1])
        self._test_to_work_finite_inputs(real)

    def test_to_real(self):
        work = pd.Series([-np.infty, np.infty])
        expected = pd.Series([0., 1.])
        actual = self.transform.to_real_series(work)
        pd.testing.assert_series_equal(expected, actual, rtol=1e-12, atol=1e-12)

    def test_invalid_real_inputs(self):
        self._test_invalid_real_inputs()


if __name__ == '__main__':
    unittest.main()
