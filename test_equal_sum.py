import unittest

from equal_sum import (
    find_pairs_with_equal_sum,
    find_pairs_with_equal_sum_multiprocessing,
    find_pairs_with_equal_sum_numpy,
    find_pairs_with_equal_sum_optimized,
)


class TestPairsWithEqualSum(unittest.TestCase):

    def setUp(self):
        """Set up test implementations to avoid duplication"""
        self.implementations = {
            "standard": find_pairs_with_equal_sum,
            "numpy": find_pairs_with_equal_sum_numpy,
            "optimized": find_pairs_with_equal_sum_optimized,
            "multiprocessing": find_pairs_with_equal_sum_multiprocessing,
        }

    def _test_pairs_with_equal_sum(self, arr, expected_pairs, impl_name):
        """Helper method to test both implementations with the same logic

        Args:
            arr: Input array to test
            expected_pairs: Dictionary of expected sums and their pairs
            impl_name: Name of the implementation to test ("standard" or "numpy")
        """
        implementation = self.implementations[impl_name]

        # Set special parameters for multiprocessing implementation to ensure
        # we're testing the actual multiprocessing logic
        if impl_name == "multiprocessing":
            pairs_by_sum = implementation(arr, batch_size=2, small_array_threshold=4)
        else:
            pairs_by_sum = implementation(arr)

        # Verify all expected sums are present in the result
        for sum_val, expected_sum_pairs in expected_pairs.items():
            self.assertIn(
                sum_val,
                pairs_by_sum,
                f"Sum {sum_val} not found in result with {impl_name} implementation",
            )
            self.assertEqual(
                len(pairs_by_sum[sum_val]),
                len(expected_sum_pairs),
                f"Expected {len(expected_sum_pairs)} pairs for sum {sum_val}, "
                f"got {len(pairs_by_sum[sum_val])} with {impl_name} implementation",
            )

            # Verify all expected pairs for each sum are present
            for pair in expected_sum_pairs:
                pair_exists = (
                    pair in pairs_by_sum[sum_val]
                    or (pair[1], pair[0]) in pairs_by_sum[sum_val]
                )
                self.assertTrue(
                    pair_exists,
                    f"Pair {pair} not found for sum {sum_val} with {impl_name} implementation",
                )

    def _test_empty_result(self, arr, impl_name):
        """Helper method to test cases where empty result is expected

        Args:
            arr: Input array to test
            impl_name: Name of the implementation to test ("standard" or "numpy")
        """
        implementation = self.implementations[impl_name]
        pairs_by_sum = implementation(arr)
        self.assertEqual(
            len(pairs_by_sum),
            0,
            f"Expected empty result with {impl_name} implementation",
        )

    def test_example1(self):
        """Test both implementations with example 1"""
        arr = [6, 4, 12, 10, 22, 54, 32, 42, 21, 11]
        expected_pairs = {
            16: [(4, 12), (6, 10)],
            32: [(10, 22), (21, 11)],
            33: [(12, 21), (22, 11)],
            43: [(22, 21), (32, 11)],
            53: [(32, 21), (42, 11)],
            54: [(12, 42), (22, 32)],
            64: [(10, 54), (22, 42)],
        }

        for impl_name in self.implementations:
            with self.subTest(implementation=impl_name):
                self._test_pairs_with_equal_sum(arr, expected_pairs, impl_name)

    def test_empty_array(self):
        """Test both implementations with empty array"""
        arr = []
        for impl_name in self.implementations:
            with self.subTest(implementation=impl_name):
                self._test_empty_result(arr, impl_name)

    def test_no_pairs(self):
        """Test both implementations with array having no pairs with equal sum"""
        arr = [1, 2, 3]  # No two pairs can have the same sum
        for impl_name in self.implementations:
            with self.subTest(implementation=impl_name):
                self._test_empty_result(arr, impl_name)

    def test_example2(self):
        """Test both implementations with example 2"""
        arr = [4, 23, 65, 67, 24, 12, 86]
        expected_pairs = {90: [(4, 86), (23, 67)]}

        for impl_name in self.implementations:
            with self.subTest(implementation=impl_name):
                self._test_pairs_with_equal_sum(arr, expected_pairs, impl_name)
