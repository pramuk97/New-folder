# Dynamic Programming DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common dynamic programming DSA interview questions.

---

## 1. Climbing Stairs
**Problem:** Count ways to climb n stairs (1 or 2 steps at a time).

```csharp
public int ClimbStairs(int n)
{
    if (n <= 2) return n;
    int a = 1, b = 2;
    for (int i = 3; i <= n; i++)
    {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}
```
**Explanation:** DP, Fibonacci sequence.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: n=5
a=1, b=2
i=3: c=3, a=2, b=3
i=4: c=5, a=3, b=5
i=5: c=8, a=5, b=8
Output: 8

---

## 2. House Robber
**Problem:** Max money robbed without adjacent houses.

```csharp
public int Rob(int[] nums)
{
    if (nums.Length == 0) return 0;
    if (nums.Length == 1) return nums[0];
    int prev1 = 0, prev2 = 0;
    foreach (var num in nums)
    {
        int temp = prev1;
        prev1 = Math.Max(prev2 + num, prev1);
        prev2 = temp;
    }
    return prev1;
}
```
**Explanation:** DP, choose to rob or skip each house.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: [2,7,9,3,1]
prev1=0, prev2=0
num=2: prev1=max(0+2,0)=2, prev2=0
num=7: prev1=max(0+7,2)=7, prev2=2
num=9: prev1=max(2+9,7)=11, prev2=7
num=3: prev1=max(7+3,11)=11, prev2=11
num=1: prev1=max(11+1,11)=12, prev2=11
Output: 12

---

## 3. Coin Change
**Problem:** Minimum coins to make up a given amount.

```csharp
public int CoinChangeMethod(int[] coins, int amount)
{
    int[] dp = new int[amount + 1];
    Array.Fill(dp, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++)
        foreach (var coin in coins)
            if (i - coin >= 0)
                dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
    return dp[amount] > amount ? -1 : dp[amount];
}
```
**Explanation:** DP, build up minimum coins for each amount.
**Time Complexity:** O(amount * n)
**Space Complexity:** O(amount)

**Dry Run Example:**
Input: coins=[1,2,5], amount=11
dp=[0,12,12,12,12,12,12,12,12,12,12,12]
i=1: dp[1]=1
i=2: dp[2]=1
i=3: dp[3]=2
... i=11: dp[11]=3
Output: 3

---

## 4. Longest Increasing Subsequence
**Problem:** Length of longest increasing subsequence.

```csharp
public int LengthOfLIS(int[] nums)
{
    if (nums.Length == 0) return 0;
    int[] dp = new int[nums.Length];
    Array.Fill(dp, 1);
    int maxLen = 1;
    for (int i = 1; i < nums.Length; i++)
        for (int j = 0; j < i; j++)
            if (nums[i] > nums[j])
                dp[i] = Math.Max(dp[i], dp[j] + 1);
    foreach (var len in dp) maxLen = Math.Max(maxLen, len);
    return maxLen;
}
```
**Explanation:** DP, track LIS ending at each index.
**Time Complexity:** O(n^2)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: [10,9,2,5,3,7,101,18]
dp=[1,1,1,1,1,1,1,1]
i=1: j=0, 9>10? no
i=2: j=0,2>10? no; j=1,2>9? no
i=3: j=0,5>10? no; j=1,5>9? no; j=2,5>2? yes, dp[3]=2
... (continue)
Output: 4

---

## 5. Unique Paths
**Problem:** Number of unique paths in m x n grid.

```csharp
public int UniquePathsMethod(int m, int n)
{
    int[,] dp = new int[m, n];
    for (int i = 0; i < m; i++) dp[i, 0] = 1;
    for (int j = 0; j < n; j++) dp[0, j] = 1;
    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            dp[i, j] = dp[i - 1, j] + dp[i, j - 1];
    return dp[m - 1, n - 1];
}
```
**Explanation:** DP, number of ways to reach each cell.
**Time Complexity:** O(mn)
**Space Complexity:** O(mn)

**Dry Run Example:**
Input: m=3, n=2
dp=[[1,1],[1,0],[1,0]]
i=1,j=1: dp[1,1]=dp[0,1]+dp[1,0]=1+1=2
i=2,j=1: dp[2,1]=dp[1,1]+dp[2,0]=2+1=3
Output: 3

---

## 6. Edit Distance
**Problem:** Minimum operations to convert word1 to word2.

```csharp
public int MinDistance(string word1, string word2)
{
    int m = word1.Length, n = word2.Length;
    int[,] dp = new int[m + 1, n + 1];
    for (int i = 0; i <= m; i++) dp[i, 0] = i;
    for (int j = 0; j <= n; j++) dp[0, j] = j;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (word1[i - 1] == word2[j - 1])
                dp[i, j] = dp[i - 1, j - 1];
            else
                dp[i, j] = 1 + Math.Min(dp[i - 1, j - 1], Math.Min(dp[i - 1, j], dp[i, j - 1]));
    return dp[m, n];
}
```
**Explanation:** DP, minimum operations to convert word1 to word2.
**Time Complexity:** O(mn)
**Space Complexity:** O(mn)

**Dry Run Example:**
Input: word1="horse", word2="ros"
dp[0,0]=0, dp[1,0]=1, dp[0,1]=1, ...
dp[5,3]=3
Output: 3

---

## 7. Maximum Product Subarray
**Problem:** Find the contiguous subarray with the largest product.

```csharp
public int MaxProduct(int[] nums)
{
    int maxProd = nums[0], minProd = nums[0], res = nums[0];
    for (int i = 1; i < nums.Length; i++)
    {
        if (nums[i] < 0)
        {
            int temp = maxProd;
            maxProd = minProd;
            minProd = temp;
        }
        maxProd = Math.Max(nums[i], maxProd * nums[i]);
        minProd = Math.Min(nums[i], minProd * nums[i]);
        res = Math.Max(res, maxProd);
    }
    return res;
}
```
**Explanation:** Track max/min product for negative numbers.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: [2,3,-2,4]
maxProd=2, minProd=2, res=2
i=1: 3>0, maxProd=max(3,2*3)=6, minProd=min(3,2*3)=3, res=6
i=2: -2<0, swap max/min, maxProd=max(-2,3*-2)=-2, minProd=min(-2,6*-2)=-12, res=6
i=3: 4>0, maxProd=max(4,-2*4)=4, minProd=min(4,-12*4)=-48, res=6
Output: 6

---

## 8. Word Break
**Problem:** Check if a string can be segmented into dictionary words.

```csharp
public bool WordBreakMethod(string s, IList<string> wordDict)
{
    var set = new HashSet<string>(wordDict);
    bool[] dp = new bool[s.Length + 1];
    dp[0] = true;
    for (int i = 1; i <= s.Length; i++)
        for (int j = 0; j < i; j++)
            if (dp[j] && set.Contains(s.Substring(j, i - j)))
                dp[i] = true;
    return dp[s.Length];
}
```
**Explanation:** DP, check if string can be segmented.
**Time Complexity:** O(n^2)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s="leetcode", wordDict=["leet","code"]
dp=[T,F,F,F,T,F,F,F,F]
i=4: j=0, "leet" in dict, dp[4]=T
i=8: j=4, "code" in dict, dp[8]=T
Output: true

---

## 9. Decode Ways
**Problem:** Count the number of ways to decode a string.

```csharp
public int NumDecodings(string s)
{
    if (string.IsNullOrEmpty(s)) return 0;
    int n = s.Length;
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = s[0] != '0' ? 1 : 0;
    for (int i = 2; i <= n; i++)
    {
        if (s[i - 1] != '0') dp[i] += dp[i - 1];
        int twoDigit = int.Parse(s.Substring(i - 2, 2));
        if (twoDigit >= 10 && twoDigit <= 26) dp[i] += dp[i - 2];
    }
    return dp[n];
}
```
**Explanation:** DP, count ways to decode using previous results.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s="226"
dp=[1,1,0,0]
i=2: s[1]='2', dp[2]=dp[1]=1, twoDigit=22, dp[2]+=dp[0]=2
i=3: s[2]='6', dp[3]=dp[2]=2, twoDigit=26, dp[3]+=dp[1]=3
Output: 3

---

## 10. Partition Equal Subset Sum
**Problem:** Can array be partitioned into two subsets with equal sum?

```csharp
public bool CanPartition(int[] nums)
{
    int sum = 0;
    foreach (var num in nums) sum += num;
    if (sum % 2 != 0) return false;
    int target = sum / 2;
    var dp = new bool[target + 1];
    dp[0] = true;
    foreach (var num in nums)
        for (int i = target; i >= num; i--)
            dp[i] = dp[i] || dp[i - num];
    return dp[target];
}
```
**Explanation:** DP, subset sum problem.
**Time Complexity:** O(n * sum)
**Space Complexity:** O(sum)

**Dry Run Example:**
Input: [1,5,11,5]
sum=22, target=11
dp=[T,F,F,F,F,F,F,F,F,F,F,F]
num=1: dp[1]=T
num=5: dp[6]=T, dp[5]=T
num=11: dp[11]=T, ...
Output: true

---
