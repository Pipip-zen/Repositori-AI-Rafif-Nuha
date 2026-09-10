def sortarray(xs):
    arr = list(xs)
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

data = [5, 1, 4, 2, 3, 0]
t = sortarray(data)
print(t)