// DSA Solutions - Dynamic Programming
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.DynamicProgramming
{
    // 1. Climbing Stairs
    public class ClimbingStairs
    {
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
        // Explanation: DP, Fibonacci sequence.
    }

    // 2. House Robber
    public class HouseRobber
    {
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
        // Explanation: DP, choose to rob or skip each house.
    }

    // 3. Coin Change
    public class CoinChange
    {
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
        // Explanation: DP, build up minimum coins for each amount.
    }

    // 4. Longest Increasing Subsequence
    public class LongestIncreasingSubsequence
    {
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
        // Explanation: DP, track LIS ending at each index.
    }

    // 5. Unique Paths
    public class UniquePaths
    {
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
        // Explanation: DP, number of ways to reach each cell.
    }

    // 6. Edit Distance
    public class EditDistance
    {
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
        // Explanation: DP, minimum operations to convert word1 to word2.
    }

    // 7. Maximum Product Subarray
    public class MaximumProductSubarray
    {
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
        // Explanation: Track max/min product for negative numbers.
    }

    // 8. Word Break
    public class WordBreak
    {
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
        // Explanation: DP, check if string can be segmented.
    }

    // 9. Decode Ways
    public class DecodeWays
    {
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
        // Explanation: DP, count ways to decode using previous results.
    }

    // 10. Partition Equal Subset Sum
    public class PartitionEqualSubsetSum
    {
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
        // Explanation: DP, subset sum problem.
    }
}
