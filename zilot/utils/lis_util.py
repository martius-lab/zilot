from bisect import bisect_left


def lis_update(x, dp):
    """dp[k-1] = lowest x_k s.t. x_1, ..., x_k is increasing subsequence of data seen up to now"""

    i = bisect_left(dp, x)
    if i == len(dp):
        dp.append(x)
    else:
        dp[i] = min(dp[i], x)
    return dp


if __name__ == "__main__":
    arr = [3, 3, 3, 1, 2, 4, 5, 3, 4, 5]
    dp = []
    for a in arr:
        dp = lis_update(a, dp)
        print(dp)
