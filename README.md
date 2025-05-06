# Equal Sum Pairs Finder

A Python utility for finding pairs of numbers with equal sums in an array.

## Problem Statement

Given an unsorted array of integers, this utility identifies and outputs all unique pairs of elements that have the same sum.

For example, in the array `[6, 4, 12, 10]`:
- Pairs `(4, 12)` and `(6, 10)` both have the sum `16`

## Features

- Four different implementations:
  - **Standard**: Pure Python implementation using nested loops
  - **NumPy**: Vectorized implementation using NumPy for improved performance
  - **Optimized**: Two-pass algorithm with early filtering
  - **Multiprocessing**: Parallel implementation that distributes work across multiple CPU cores

- Command-line interface for easy usage
- Comprehensive unit tests for each implementation

## Installation

### Prerequisites

- Python 3.11 or higher
- Dependencies listed in Pipfile

### Setup

1. Clone this repository
2. Install dependencies using pipenv:

```bash
pipenv install
```

## Usage

### Command Line Interface

```bash
python equal_sum.py [-i {standard,numpy,optimized,multiprocessing}] [numbers ...]
```

#### Options:

- `-i, --implementation`: Implementation to use (standard, numpy, optimized, or multiprocessing). Default is "standard".
- `numbers`: Space-separated list of integers to process.

#### Examples:

```bash
# Using standard implementation with provided numbers
python equal_sum.py 6 4 12 10 22 54 32 42 21 11

# Using NumPy implementation
python equal_sum.py -i numpy 4 23 65 67 24 12 86

# Using optimized implementation
python equal_sum.py -i optimized 6 4 12 10 22 54 32 42 21 11

# Using multiprocessing implementation
python equal_sum.py -i multiprocessing 6 4 12 10 22 54 42 21 11 44 65 76 32 45 89 34
```

### As a Module

```python
from equal_sum import find_pairs_with_equal_sum, find_pairs_with_equal_sum_numpy, find_pairs_with_equal_sum_optimized, find_pairs_with_equal_sum_multiprocessing

# Using standard implementation
result = find_pairs_with_equal_sum([6, 4, 12, 10, 22, 54, 32, 42, 21, 11])

# Using NumPy implementation
result = find_pairs_with_equal_sum_numpy([4, 23, 65, 67, 24, 12, 86])

# Using optimized implementation
result = find_pairs_with_equal_sum_optimized([6, 4, 12, 10, 22, 54, 32, 42, 21, 11])

# Using multiprocessing implementation
result = find_pairs_with_equal_sum_multiprocessing([6, 4, 12, 10, 22, 54, 32, 42, 21, 11, 44, 54, 65, 76, 32, 45, 89, 34])
```

## Algorithm Details

### Standard Implementation
- **Time Complexity**: O(n²) where n is the length of the array
- **Space Complexity**: O(n²) in the worst case

### NumPy Implementation
- Uses vectorized operations for better performance
- **Time Complexity**: O(n²) 
- **Space Complexity**: O(n²)
- Generally faster for larger arrays due to optimized C implementations in NumPy

### Optimized Implementation
- Two-pass algorithm:
  1. First identifies sums with multiple pairs
  2. Only collects pairs for relevant sums
- **Time Complexity**: Still O(n²) but with better constants
- **Space Complexity**: O(n²) in the worst case, but often better

### Multiprocessing Implementation
- Distributes pair computation across multiple CPU cores
- Automatically falls back to optimized implementation for small arrays (n < 20)
- **Time Complexity**: O(n²) but distributed across multiple cores
- **Space Complexity**: O(n²) in the worst case
- Best for large datasets where parallelization can provide significant speedups
- Uses Python's multiprocessing module for true parallel execution

## Testing

Run the included unit tests with:

```bash
python -m unittest test_equal_sum.py
```

## Example Outputs

### Example 1:
Input: `[6, 4, 12, 10, 22, 54, 32, 42, 21, 11]`

Output:
```
Pairs : (4, 12) (6, 10) have sum : 16
Pairs : (10, 22) (21, 11) have sum : 32
Pairs : (12, 21) (22, 11) have sum : 33
Pairs : (22, 21) (32, 11) have sum : 43
Pairs : (32, 21) (42, 11) have sum : 53
Pairs : (12, 42) (22, 32) have sum : 54
Pairs : (10, 54) (22, 42) have sum : 64
```

### Example 2:
Input: `[4, 23, 65, 67, 24, 12, 86]`

Output:
```
Pairs : (4, 86) (23, 67) have sum : 90
```

### Example 3 (with multiprocessing):
Input: `[6, 4, 12, 10, 22, 54, 32, 42, 21, 11, 44, 54, 65, 76, 32, 45, 89, 34]`

Output:
```
Pairs : (4, 12) (6, 10) have sum : 16
Pairs : (10, 22) (11, 21) have sum : 32
Pairs : (11, 22) (12, 21) have sum : 33
Pairs : (11, 32) (21, 22) have sum : 43
Pairs : (11, 42) (21, 32) have sum : 53
Pairs : (12, 42) (22, 32) have sum : 54
Pairs : (10, 54) (22, 42) have sum : 64
Pairs : (32, 32) (10, 54) have sum : 64
Pairs : (11, 54) (21, 44) have sum : 65
Pairs : (11, 65) (32, 44) have sum : 76
Pairs : (32, 54) (42, 44) have sum : 86
Pairs : (11, 76) (32, 55) have sum : 87
```
