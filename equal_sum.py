import argparse
import math
import multiprocessing as mp
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def find_pairs_with_equal_sum(arr: List[int]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Find all pairs of elements in the array that have the same sum.

    Args:
        arr: Input array of integers

    Returns:
        Dictionary mapping sum values to lists of pairs having that sum
        (only includes sums with at least two pairs)

    Time Complexity: O(n²) where n is the length of the array
    Space Complexity: O(n²) in the worst case
    """
    # Dictionary to store pairs grouped by their sum
    pairs_by_sum = {}

    # Generate all possible pairs from the sorted array
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = arr[i] + arr[j]

            # Add the pair to the dictionary
            if pair_sum not in pairs_by_sum:
                pairs_by_sum[pair_sum] = []
            pairs_by_sum[pair_sum].append((arr[i], arr[j]))

    # Filter out sums that have only one pair
    return {sum_val: pairs for sum_val, pairs in pairs_by_sum.items() if len(pairs) > 1}


def find_pairs_with_equal_sum_optimized(
    arr: List[int],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Find all pairs of elements in the array that have the same sum using a
    sort + counter optimization.

    Args:
        arr: Input array of integers

    Returns:
        Dictionary mapping sum values to lists of pairs having that sum
        (only includes sums with at least two pairs)

    Time Complexity: O(n²) where n is the length of the array
    Space Complexity: O(n²) in the worst case, but often better
    """
    n = len(arr)
    if n <= 1:
        return {}

    # Step 1: Sort the array - O(n log n)
    # This helps with cache locality in the next steps
    sorted_arr = sorted(arr)

    # Step 2: First pass - count pair sums to identify candidates
    # This avoids storing unnecessary pairs
    pair_sum_counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = sorted_arr[i] + sorted_arr[j]
            pair_sum_counter[pair_sum] += 1

    # Filter to sums that appear in multiple pairs
    sums_with_multiple_pairs = {s for s, count in pair_sum_counter.items() if count > 1}

    # If no sums with multiple pairs, return empty dictionary
    if not sums_with_multiple_pairs:
        return {}

    # Step 3: Second pass - only collect pairs for sums we care about
    result = {sum_val: [] for sum_val in sums_with_multiple_pairs}
    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = sorted_arr[i] + sorted_arr[j]
            if pair_sum in sums_with_multiple_pairs:
                result[pair_sum].append((sorted_arr[i], sorted_arr[j]))

    return result


def find_pairs_with_equal_sum_numpy(arr: List[int]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Find all pairs of elements in the array that have the same sum.

    Args:
        arr: Input array of integers

    Returns:
        Dictionary mapping sum values to lists of pairs having that sum
        (only includes sums with at least two pairs)

    Time Complexity: O(n²)
    Space Complexity: O(n²)
    """
    if len(arr) <= 1:
        return {}

    # Convert to NumPy array
    np_arr = np.array(arr)
    n = len(np_arr)

    # Create a matrix of all possible sums using broadcasting
    sum_matrix = np_arr.reshape(1, n) + np_arr.reshape(n, 1)

    # Get upper triangular indices (to avoid duplicate pairs)
    i_indices, j_indices = np.triu_indices(n, k=1)

    # Get all unique pair sums
    sums = sum_matrix[i_indices, j_indices]

    # Find sums that appear multiple times
    unique_sums, counts = np.unique(sums, return_counts=True)
    sums_with_multiple_pairs = unique_sums[counts > 1]

    # Create result dictionary
    pairs_by_sum = {}

    # For each sum with multiple pairs, find all corresponding pairs
    for sum_val in sums_with_multiple_pairs:
        # Find indices where sum equals sum_val
        pair_indices = np.where(sums == sum_val)[0]

        # Get the corresponding pairs
        pairs = [(arr[i_indices[idx]], arr[j_indices[idx]]) for idx in pair_indices]
        pairs_by_sum[int(sum_val)] = pairs

    return pairs_by_sum


def find_pairs_with_equal_sum_multiprocessing(
    arr: List[int],
    num_workers: int = None,
    batch_size: int = 10,
    small_array_threshold: int = 20,
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Find pairs with equal sum using a queue-based multiprocessing approach.

    Args:
        arr: Input array of integers
        num_workers: Number of worker processes (defaults to CPU count)
        batch_size: Number of pairs per batch

    Returns:
        Dictionary mapping sum values to lists of pairs with that sum

    Time Complexity: O(n²) but distributed across multiple cores
    Space Complexity: O(n²) in worst case
    """
    # Sort array and remove duplicates
    arr = sorted(set(arr))

    n = len(arr)
    if n <= 1:
        return {}

    # Skip multiprocessing for small arrays (overhead would exceed benefit)
    if n <= small_array_threshold:
        return find_pairs_with_equal_sum_optimized(arr)

    if num_workers is None:
        num_workers = mp.cpu_count()

    # Set up queues
    task_queue = mp.Queue()  # For distributing work
    result_queue = mp.Queue()  # For collecting interim results
    final_result_queue = mp.Queue()  # For the merged result

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=_worker_process, args=(task_queue, result_queue))
        p.daemon = True
        p.start()
        workers.append(p)

    # Start merger process
    merger = mp.Process(
        target=_merger_process, args=(result_queue, final_result_queue, num_workers)
    )
    merger.daemon = True
    merger.start()

    # Generate and distribute pairs in batches
    batch = []
    for i in range(n):
        for j in range(i + 1, n):
            batch.append((arr[i], arr[j]))
            # When batch reaches desired size, send it to task queue
            if len(batch) >= batch_size:
                task_queue.put(batch)
                batch = []

    # Send any remaining pairs
    if batch:
        task_queue.put(batch)

    # Signal end of tasks
    for _ in range(num_workers):
        task_queue.put(None)

    # Get final merged result
    final_result = final_result_queue.get()

    # Clean up processes
    for worker in workers:
        worker.join()
    merger.join()

    # Filter sums with multiple pairs
    return {sum_val: pairs for sum_val, pairs in final_result.items() if len(pairs) > 1}


def _worker_process(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Worker that processes batches of pairs and produces intermediate results."""
    try:
        while True:
            batch = task_queue.get()
            if batch is None:  # Sentinel value
                break

            # Process this batch
            result = {}
            try:
                for pair in batch:
                    pair_sum = pair[0] + pair[1]
                    if pair_sum not in result:
                        result[pair_sum] = []
                    result[pair_sum].append(pair)
            except Exception as e:
                # Log error but continue processing other batches
                print(f"Error processing batch: {e}")
                continue

            # Send results
            result_queue.put(result)
    except Exception as e:
        print(f"Worker process encountered an error: {e}")
    finally:
        # Signal completion
        result_queue.put(None)


def _merger_process(
    result_queue: mp.Queue, final_result_queue: mp.Queue, num_workers: int
) -> None:
    """Collects and merges results from all workers."""
    merged = {}
    workers_completed = 0

    try:
        while workers_completed < num_workers:
            try:
                result = result_queue.get()

                if result is None:  # Worker finished
                    workers_completed += 1
                    continue

                # Merge this batch's results
                for sum_val, pairs in result.items():
                    if sum_val not in merged:
                        merged[sum_val] = []
                    merged[sum_val].extend(pairs)
            except Exception as e:
                print(f"Error processing result in merger: {e}")
                # Continue trying to process other results
                continue

        # Send final merged result back to main process
        final_result_queue.put(merged)
    except Exception as e:
        print(f"Merger process encountered an error: {e}")
        # Ensure we send at least an empty result if something goes wrong
        final_result_queue.put({})
    finally:
        # Make sure we don't leave the main process hanging if something fails
        if workers_completed < num_workers:
            print(
                f"Warning: Merger process terminated after processing only {workers_completed}/{num_workers} workers"
            )


def print_pairs_with_equal_sum(
    arr: List[int], implementation: str = "standard"
) -> None:
    """
    Print all pairs in the array with equal sum.

    Args:
        arr: Input array of integers
        implementation: The implementation to use ("standard", "numpy", "optimized", or "multiprocessing")
    """
    implementations = {
        "standard": find_pairs_with_equal_sum,
        "numpy": find_pairs_with_equal_sum_numpy,
        "optimized": find_pairs_with_equal_sum_optimized,
        "multiprocessing": find_pairs_with_equal_sum_multiprocessing,
    }
    find_pairs_function = implementations.get(implementation)
    if find_pairs_function is None:
        raise ValueError(f"Unknown implementation: {implementation}")
    pairs_by_sum = find_pairs_function(arr)

    # Print pairs grouped by sum in ascending order
    for sum_val, pairs in sorted(pairs_by_sum.items()):
        pairs_str = " ".join([f"( {pair[0]}, {pair[1]})" for pair in pairs])
        print(f"Pairs : {pairs_str} have sum : {sum_val}")


# Test with examples
if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Find pairs with equal sum in a list of integers"
    )
    parser.add_argument(
        "-i",
        "--implementation",
        choices=["standard", "numpy", "optimized", "multiprocessing"],
        default="standard",
        help="Implementation to use (standard, numpy, optimized, or multiprocessing)",
    )
    parser.add_argument(
        "numbers", nargs="*", type=int, help="List of integers to process"
    )

    args = parser.parse_args()

    # Use command-line arguments if provided, otherwise run examples
    if args.numbers:
        print_pairs_with_equal_sum(args.numbers, args.implementation)
    else:
        # Example 1
        print("Example 1:")
        arr1 = [6, 4, 12, 10, 22, 54, 32, 42, 21, 11]
        print_pairs_with_equal_sum(arr1, "standard")

        print("\nExample 2:")
        # Example 2
        arr2 = [4, 23, 65, 67, 24, 12, 86]
        print_pairs_with_equal_sum(arr2, "numpy")

        print("\nExample 3 (with multiprocessing):")
        # Example 3 with multiprocessing
        arr3 = [6, 4, 12, 10, 22, 54, 32, 42, 21, 11, 44, 54, 65, 76, 32, 45, 89, 34]
        print_pairs_with_equal_sum(arr3, "multiprocessing")

        # Show usage hint
        print(
            "\nUsage: python equal_sum.py [-i {standard,numpy,optimized,multiprocessing}] [numbers ...]"
        )
        print(
            "Example: python equal_sum.py -i multiprocessing 6 4 12 10 22 54 32 42 21 11"
        )
