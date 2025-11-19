// DSA Solutions - Arrays
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.Arrays
{
    // 1. Two Sum
    // Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    // Solution uses a dictionary for O(n) time complexity.
    public class TwoSumSolution
    {
        public int[] TwoSum(int[] nums, int target)
        {
            var dict = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                int complement = target - nums[i];
                if (dict.ContainsKey(complement))
                {
                    // Found the pair
                    return new int[] { dict[complement], i };
                }
                dict[nums[i]] = i;
            }
            throw new ArgumentException("No two sum solution");
        }
        // Explanation:
        // Iterate through the array, for each element, check if its complement (target - current) exists in the dictionary.
        // If found, return indices. Otherwise, add current element to dictionary.
    }

    // 2. Best Time to Buy and Sell Stock
    // Find the maximum profit from one transaction (buy/sell).
    public class BestTimeToBuySellStock
    {
        public int MaxProfit(int[] prices)
        {
            int minPrice = int.MaxValue;
            int maxProfit = 0;
            foreach (var price in prices)
            {
                if (price < minPrice) minPrice = price;
                else if (price - minPrice > maxProfit) maxProfit = price - minPrice;
            }
            return maxProfit;
        }
        // Explanation:
        // Track the minimum price so far and the maximum profit by selling at current price.
    }

    // 3. Maximum Subarray (Kadane's Algorithm)
    public class MaximumSubarray
    {
        public int MaxSubArray(int[] nums)
        {
            int maxSoFar = nums[0];
            int currMax = nums[0];
            for (int i = 1; i < nums.Length; i++)
            {
                currMax = Math.Max(nums[i], currMax + nums[i]);
                maxSoFar = Math.Max(maxSoFar, currMax);
            }
            return maxSoFar;
        }
        // Explanation:
        // At each step, decide whether to start a new subarray or extend the previous one.
    }

    // 4. Merge Intervals
    public class MergeIntervals
    {
        public int[][] Merge(int[][] intervals)
        {
            if (intervals.Length == 0) return intervals;
            Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
            var merged = new List<int[]>();
            int[] current = intervals[0];
            foreach (var interval in intervals)
            {
                if (interval[0] <= current[1])
                {
                    current[1] = Math.Max(current[1], interval[1]);
                }
                else
                {
                    merged.Add(current);
                    current = interval;
                }
            }
            merged.Add(current);
            return merged.ToArray();
        }
        // Explanation:
        // Sort intervals, then merge overlapping ones by comparing starts and ends.
    }

    // 5. Product of Array Except Self
    public class ProductOfArrayExceptSelf
    {
        public int[] ProductExceptSelf(int[] nums)
        {
            int n = nums.Length;
            int[] answer = new int[n];
            int left = 1;
            for (int i = 0; i < n; i++)
            {
                answer[i] = left;
                left *= nums[i];
            }
            int right = 1;
            for (int i = n - 1; i >= 0; i--)
            {
                answer[i] *= right;
                right *= nums[i];
            }
            return answer;
        }
        // Explanation:
        // Use two passes: left product and right product, without division.
    }

    // 6. 3Sum
    public class ThreeSum
    {
        public IList<IList<int>> ThreeSumMethod(int[] nums)
        {
            Array.Sort(nums);
            var res = new List<IList<int>>();
            for (int i = 0; i < nums.Length - 2; i++)
            {
                if (i > 0 && nums[i] == nums[i - 1]) continue;
                int left = i + 1, right = nums.Length - 1;
                while (left < right)
                {
                    int sum = nums[i] + nums[left] + nums[right];
                    if (sum == 0)
                    {
                        res.Add(new List<int> { nums[i], nums[left], nums[right] });
                        while (left < right && nums[left] == nums[left + 1]) left++;
                        while (left < right && nums[right] == nums[right - 1]) right--;
                        left++; right--;
                    }
                    else if (sum < 0) left++;
                    else right--;
                }
            }
            return res;
        }
        // Explanation:
        // Sort and use two pointers to find triplets that sum to zero, skipping duplicates.
    }

    // 7. Container With Most Water
    public class ContainerWithMostWater
    {
        public int MaxArea(int[] height)
        {
            int left = 0, right = height.Length - 1, maxArea = 0;
            while (left < right)
            {
                int area = Math.Min(height[left], height[right]) * (right - left);
                maxArea = Math.Max(maxArea, area);
                if (height[left] < height[right]) left++;
                else right--;
            }
            return maxArea;
        }
        // Explanation:
        // Use two pointers, move the shorter line inward to maximize area.
    }

    // 8. Set Matrix Zeroes
    public class SetMatrixZeroes
    {
        public void SetZeroes(int[][] matrix)
        {
            int m = matrix.Length, n = matrix[0].Length;
            bool[] rows = new bool[m];
            bool[] cols = new bool[n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    if (matrix[i][j] == 0)
                    {
                        rows[i] = true;
                        cols[j] = true;
                    }
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    if (rows[i] || cols[j]) matrix[i][j] = 0;
        }
        // Explanation:
        // Mark rows and columns to be zeroed, then update the matrix.
    }

    // 9. Rotate Array
    public class RotateArray
    {
        public void Rotate(int[] nums, int k)
        {
            k %= nums.Length;
            Reverse(nums, 0, nums.Length - 1);
            Reverse(nums, 0, k - 1);
            Reverse(nums, k, nums.Length - 1);
        }
        private void Reverse(int[] nums, int start, int end)
        {
            while (start < end)
            {
                int temp = nums[start];
                nums[start] = nums[end];
                nums[end] = temp;
                start++; end--;
            }
        }
        // Explanation:
        // Reverse the whole array, then reverse first k and last n-k elements.
    }

    // 10. Missing Number
    public class MissingNumber
    {
        public int FindMissingNumber(int[] nums)
        {
            int n = nums.Length;
            int total = n * (n + 1) / 2;
            int sum = 0;
            foreach (var num in nums) sum += num;
            return total - sum;
        }
        // Explanation:
        // Use sum formula for first n natural numbers, subtract actual sum.
    }
}
