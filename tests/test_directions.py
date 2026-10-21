import unittest

import numpy as np

from cfra.analysis.common import limit_fit, unit
from cfra.directions import fit


class DirectionTests(unittest.TestCase):
    def test_m1_identity(self) -> None:
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(3), 20)
        values = unit(rng.standard_normal((60, 12)))
        operator, directions, classes = fit(values, labels, operator="M1")

        expected_classes = np.arange(3)
        np.testing.assert_array_equal(classes, expected_classes)
        means = np.stack([values[labels == class_id].mean(axis=0) for class_id in expected_classes])
        origin = unit(means).mean(axis=0)
        expected = unit(means - origin)

        np.testing.assert_allclose(operator.mean, origin, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(directions, expected, rtol=1e-12, atol=1e-12)

    def test_retired_w_form_is_not_m1(self) -> None:
        rng = np.random.default_rng(11)
        labels = np.repeat(np.arange(4), 8)
        values = rng.standard_normal((32, 10))
        _operator, directions, classes = fit(values, labels, operator="M1")
        means = np.stack([unit(values[labels == class_id]).mean(axis=0) for class_id in classes])
        origin = unit(means).mean(axis=0)
        retired = unit(unit(means) - origin)
        self.assertGreater(float(np.max(np.abs(retired - directions))), 1e-3)

    def test_limit_fit_uses_sha256_order(self) -> None:
        labels = np.repeat(np.arange(3), 10)
        object_ids = np.asarray([f"obj-{index:02d}" for index in range(30)])
        mask = np.ones(30, dtype=bool)
        selected = limit_fit(mask, labels, object_ids, limit=4)
        self.assertEqual(int(selected.sum()), 12)
        for class_id in range(3):
            self.assertEqual(int(selected[labels == class_id].sum()), 4)


if __name__ == "__main__":
    unittest.main()
