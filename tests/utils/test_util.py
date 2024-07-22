import unittest
import numpy as np
from streamauc.utils import _all_equal, onehot_encode


class TestAllEqual(unittest.TestCase):
    def test_all_equal(self):
        # Test case with all equal elements
        self.assertTrue(
            _all_equal([1, 1, 1, 1]),
            "All elements are equal, " "should return True",
        )
        self.assertTrue(
            _all_equal(["a", "a", "a", "a"]),
            "All elements are equal, " "should return True",
        )
        self.assertTrue(
            _all_equal([True, True, True]),
            "All elements are equal, " "should return True",
        )

    def test_not_all_equal(self):
        # Test case with not all equal elements
        self.assertFalse(
            _all_equal([1, 2, 1, 1]),
            "Not all elements are equal, " "should return False",
        )
        self.assertFalse(
            _all_equal(["a", "b", "a", "a"]),
            "Not all elements are equal, " "should return False",
        )
        self.assertFalse(
            _all_equal([True, False, True]),
            "Not all elements are equal, " "should return False",
        )

    def test_empty_iterable(self):
        # Test case with an empty list
        self.assertTrue(_all_equal([]), "Empty list, should return True")

    def test_single_element(self):
        # Test case with a single element
        self.assertTrue(
            _all_equal([1]), "Single element list, should return True"
        )
        self.assertTrue(
            _all_equal(["a"]), "Single element list, should return True"
        )

    def test_mixed_types(self):
        # Test case with different types
        self.assertFalse(
            _all_equal([1, "1", 1.0]), "Different types, should return False"
        )
        self.assertTrue(
            _all_equal([1.0, 1.0, 1.0]),
            "All floats are equal, should return True",
        )

    def test_large_input(self):
        # Test case with large input
        large_list = [1] * 1000000
        self.assertTrue(
            _all_equal(large_list),
            "Large list with all equal elements, " "should return True",
        )
        large_list[999999] = 2
        self.assertFalse(
            _all_equal(large_list),
            "Large list with one different element, " "should return False",
        )


class TestOneHotEncodeOptimized(unittest.TestCase):
    def test_onehot_encode_optimized(self):
        int_masks = np.array([0, 1, 2, 0, 1, 0])  # Sample integer masks
        int_masks_before = np.copy(int_masks)
        num_classes = 3  # Number of classes

        # Call the onehot_encode function
        onehot_masks = onehot_encode(int_masks, num_classes)

        # Define the expected output
        expected_output = np.array(
            [
                [1, 0, 0],  # Class 0
                [0, 1, 0],  # Class 1
                [0, 0, 1],  # Class 2
                [1, 0, 0],  # Class 0
                [0, 1, 0],  # Class 1
                [1, 0, 0],  # Class 0
            ]
        )
        # Perform assertions to validate the output
        self.assertTrue(np.array_equal(onehot_masks, expected_output))
        self.assertEqual(onehot_masks.shape, (len(int_masks), num_classes))
        self.assertTrue(np.all(int_masks_before == int_masks))
