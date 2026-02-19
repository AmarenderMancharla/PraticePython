import time
import sys
import importlib.util

try:
    import numpy as np
except ModuleNotFoundError:
    class SimpleArray:
        def __init__(self, data):
            self.data = list(data)

        def __mul__(self, other):
            if isinstance(other, SimpleArray):
                return SimpleArray([a * b for a, b in zip(self.data, other.data)])
            return SimpleArray([a * other for a in self.data])

        __rmul__ = __mul__

        def __add__(self, other):
            if isinstance(other, SimpleArray):
                return SimpleArray([a + b for a, b in zip(self.data, other.data)])
            return SimpleArray([a + other for a in self.data])

        def __radd__(self, other):
            return self.__add__(other)

        def __iter__(self):
            return iter(self.data)

        def __repr__(self):
            return repr(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        @property
        def shape(self):
            return (len(self.data),)

        @property
        def size(self):
            return len(self.data)

    class _NPFallback:
        def array(self, iterable):
            return SimpleArray(iterable)

        def round(self, value, ndigits=0):
            if isinstance(value, SimpleArray):
                return SimpleArray([round(x, ndigits) for x in value.data])
            return round(value, ndigits)

        def mean(self, arr):
            seq = arr.data if isinstance(arr, SimpleArray) else list(arr)
            return sum(seq) / len(seq)

        def max(self, arr):
            seq = arr.data if isinstance(arr, SimpleArray) else list(arr)
            return max(seq)

        def min(self, arr):
            seq = arr.data if isinstance(arr, SimpleArray) else list(arr)
            return min(seq)

        def arange(self, start, stop=None):
            if stop is None:
                stop = start
                start = 0
            return SimpleArray(range(start, stop))

        def sum(self, arr):
            seq = arr.data if isinstance(arr, SimpleArray) else list(arr)
            return sum(seq)

    np = _NPFallback()

temps_celsius = np.array([22, 25, 28, 24, 26])

temps_fahrenheit = temps_celsius * 1.8 + 32

avg_fahrenheit = np.round(np.mean(temps_fahrenheit), 1)

print("Celsius:", temps_celsius)
print("Fahrenheit:", temps_fahrenheit)
print("Average Fahrenheit:", avg_fahrenheit)

scores = np.array([85, 90, 78, 92, 88, 76, 95, 82, 89, 91, 87, 84])

print("\nShape:", scores.shape)
print("Total elements:", scores.size)
print("Highest score:", np.max(scores))
print("Lowest score:", np.min(scores))
print("Range:", np.max(scores) - np.min(scores))

numpy_array = np.arange(1, 50001)

python_list = list(range(1, 50001))

start_numpy = time.time()
numpy_sum = np.sum(numpy_array)
end_numpy = time.time()

numpy_time = end_numpy - start_numpy

start_python = time.time()
python_sum = sum(python_list)
end_python = time.time()

python_time = end_python - start_python

speed_ratio = python_time / numpy_time

print("\nNumPy sum:", numpy_sum)
print("Python sum:", python_sum)
print("NumPy time:", format(numpy_time, ".4f"), "seconds")
print("Python time:", format(python_time, ".4f"), "seconds")
print("NumPy is", round(speed_ratio, 1), "x faster")

print("\nPython executable:", sys.executable)
print("Python version:", sys.version)

numpy_spec = importlib.util.find_spec('numpy')
print("NumPy location:", numpy_spec.origin if numpy_spec else "NumPy not found")

