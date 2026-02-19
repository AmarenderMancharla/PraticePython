try:
    import numpy as np
    np.random.seed(42)
except Exception:
    # Minimal fallback for environments without numpy installed.
    import random

    class SimpleArray:
        def __init__(self, data):
            # Normalize to 2D list for matrices, 1D list for vectors
            if isinstance(data, SimpleArray):
                self.data = data.data
            else:
                if any(isinstance(x, (list, tuple)) for x in data):
                    self.data = [list(x) for x in data]
                else:
                    self.data = list(data)

        @property
        def shape(self):
            if len(self.data) == 0:
                return (0,)
            if isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)

        def mean(self, axis=0):
            if axis == 0:
                rows, cols = len(self.data), len(self.data[0])
                return [sum(self.data[r][c] for r in range(rows)) / rows for c in range(cols)]
            raise NotImplementedError

        def max(self, axis=1, keepdims=False):
            if axis == 1:
                res = [max(row) for row in self.data]
                if keepdims:
                    return SimpleArray([[v] for v in res])
                return res
            raise NotImplementedError

        def min(self, axis=1, keepdims=False):
            if axis == 1:
                res = [min(row) for row in self.data]
                if keepdims:
                    return SimpleArray([[v] for v in res])
                return res
            raise NotImplementedError

        def flatten(self):
            return [v for row in self.data for v in row]

        def __add__(self, other):
            # support broadcasting of 1D list/array across columns
            if isinstance(other, SimpleArray):
                other_data = other.data
            else:
                other_data = other
            if isinstance(other_data[0], list):
                return SimpleArray([[self.data[i][j] + other_data[i][j]
                                     for j in range(len(self.data[0]))]
                                    for i in range(len(self.data))])
            else:
                return SimpleArray([[self.data[i][j] + other_data[j]
                                     for j in range(len(self.data[0]))]
                                    for i in range(len(self.data))])

        def __sub__(self, other):
            if isinstance(other, SimpleArray):
                other_data = other.data
            else:
                other_data = other
            # other may be column vector (n x 1)
            if isinstance(other_data[0], list) and len(other_data[0]) == 1:
                return SimpleArray([[self.data[i][j] - other_data[i][0]
                                     for j in range(len(self.data[0]))]
                                    for i in range(len(self.data))])
            if not isinstance(other_data[0], list):
                return SimpleArray([[self.data[i][j] - other_data[j]
                                     for j in range(len(self.data[0]))]
                                    for i in range(len(self.data))])
            return SimpleArray([[self.data[i][j] - other_data[i][j]
                                 for j in range(len(self.data[0]))]
                                for i in range(len(self.data))])

        def __truediv__(self, other):
            if isinstance(other, SimpleArray):
                other_data = other.data
            else:
                other_data = other
            if isinstance(other_data[0], list) and len(other_data[0]) == 1:
                return SimpleArray([[self.data[i][j] / other_data[i][0] if other_data[i][0] != 0 else 0
                                     for j in range(len(self.data[0]))]
                                    for i in range(len(self.data))])
            if not isinstance(other_data[0], list):
                return SimpleArray([[self.data[i][j] / other_data[j] if other_data[j] != 0 else 0
                                     for j in range(len(self.data[0]))]
                                    for i in range(len(self.data))])
            return SimpleArray([[self.data[i][j] / other_data[i][j] if other_data[i][j] != 0 else 0
                                 for j in range(len(self.data[0]))]
                                for i in range(len(self.data))])

        def __gt__(self, val):
            return [[self.data[i][j] > val for j in range(len(self.data[0]))]
                    for i in range(len(self.data))]

        def __getitem__(self, key):
            # support tuple indexing like [i, j]
            if isinstance(key, tuple):
                return self.data[key[0]][key[1]]
            # boolean mask (nested list) -> return flattened selection
            if isinstance(key, list):
                res = []
                for i in range(len(self.data)):
                    for j in range(len(self.data[0])):
                        if key[i][j]:
                            res.append(self.data[i][j])
                return res
            return self.data[key]

        def __repr__(self):
            return repr(self.data)

    class _Random:
        def seed(self, s):
            random.seed(s)

        def randint(self, low, high, size=None):
            # numpy randint upper bound is exclusive; original code used 101 expecting inclusive of 100
            if isinstance(size, tuple):
                rows, cols = size
                return SimpleArray([[random.randint(low, high - 1) for _ in range(cols)] for _ in range(rows)])
            return random.randint(low, high - 1)

    def _round(vals, decimals=0):
        if isinstance(vals, SimpleArray):
            # return list for 1D or SimpleArray for 2D
            if isinstance(vals.data[0], list):
                return SimpleArray([[round(x, decimals) for x in row] for row in vals.data])
            return [round(x, decimals) for x in vals.data]
        if isinstance(vals, list):
            return [round(x, decimals) for x in vals]
        return round(vals, decimals)

    def _clip(arr, amin, amax):
        if isinstance(arr, SimpleArray):
            if isinstance(arr.data[0], list):
                return SimpleArray([[min(x, amax) if amax is not None else x for x in row] for row in arr.data])
            return SimpleArray([min(x, amax) if amax is not None else x for x in arr.data])
        return arr

    def _argmax(arr):
        if isinstance(arr, SimpleArray):
            flat = arr.flatten()
        elif any(isinstance(x, list) for x in arr):
            flat = [v for row in arr for v in row]
        else:
            flat = list(arr)
        return flat.index(max(flat))

    def _unravel_index(idx, shape):
        rows, cols = shape
        return (idx // cols, idx % cols)

    class _NP:
        pass

    np = _NP()
    np.random = _Random()
    np.array = lambda x: SimpleArray(x)
    np.round = _round
    np.clip = _clip
    np.argmax = _argmax
    np.unravel_index = _unravel_index

scores = np.random.randint(50, 101, size=(5, 4))

print("Original Scores:\n", scores)

print("\nScore of 3rd student in 2nd subject:",
      scores[2, 1])  
print("\nScores of last 2 students:\n",
      scores[-2:, :])
print("\nFirst 3 students, subjects 2 & 3:\n",
      scores[:3, 1:3])
column_mean = np.round(scores.mean(axis=0), 2)
print("\nColumn-wise mean (per subject):",
      column_mean)
curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve
curved_scores = np.clip(curved_scores, None, 100)

print("\nCurved Scores:\n", curved_scores)
row_max = curved_scores.max(axis=1)
print("\nBest subject score per student:",
      row_max)

row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)

normalized = (curved_scores - row_min) / (row_max - row_min)

print("\nNormalized Scores:\n", normalized)

max_index = np.unravel_index(np.argmax(normalized), normalized.shape)
print("\nHighest normalized value at (student_index, subject_index):",
      max_index)
above_90 = curved_scores[curved_scores > 90]
print("\nScores strictly above 90:\n", above_90)
