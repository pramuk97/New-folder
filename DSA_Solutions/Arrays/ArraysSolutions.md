# Arrays DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common array-based DSA interview questions.

---

## 1. Two Sum
**Problem:** Find indices of two numbers in an array that add up to a target.

```csharp
public int[] TwoSum(int[] nums, int target)
{
    var dict = new Dictionary<int, int>();
    for (int i = 0; i < nums.Length; i++)
    {
        int complement = target - nums[i];
        if (dict.ContainsKey(complement))
            return new int[] { dict[complement], i };
        dict[nums[i]] = i;
    }
    throw new ArgumentException("No two sum solution");
}
```
**Explanation:**
- Use a dictionary to store previously seen numbers and their indices.
- For each number, check if its complement exists in the dictionary.
- If found, return the pair of indices.

**Time Complexity:**
- Best/Worst: O(n)
**Space Complexity:**
- O(n)

**Dry Run Example:**
Input: nums = [2, 7, 11, 15], target = 9
Step-by-step:
- i=0: dict={}, complement=7 (not found), add 2:0
- i=1: dict={2:0}, complement=2 (found at 0), return [0,1]
Output: [0, 1]

---

## 2. Best Time to Buy and Sell Stock
**Problem:** Find the maximum profit from one buy/sell transaction.

```csharp
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
```
**Explanation:**
- Track the minimum price so far and the maximum profit by selling at the current price.

**Time Complexity:**
- Best/Worst: O(n)
**Space Complexity:**
- O(1)

**Dry Run Example:**
Input: prices = [7, 1, 5, 3, 6, 4]
Step-by-step:
- minPrice=7, maxProfit=0
- price=1: minPrice=1
- price=5: maxProfit=4
- price=3: maxProfit=4
- price=6: maxProfit=5
- price=4: maxProfit=5
Output: 5

---

## 3. Maximum Subarray (Kadane's Algorithm)
**Problem:** Find the contiguous subarray with the largest sum.

```csharp
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
```
**Explanation:**
- At each step, decide whether to start a new subarray or extend the previous one.

**Time Complexity:**
- Best/Worst: O(n)
**Space Complexity:**
- O(1)

**Dry Run Example:**
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Step-by-step:
- currMax=-2, maxSoFar=-2
- i=1: currMax=1, maxSoFar=1
- i=2: currMax=-2, maxSoFar=1
- i=3: currMax=4, maxSoFar=4
- i=4: currMax=3, maxSoFar=4
- i=5: currMax=5, maxSoFar=5
- i=6: currMax=6, maxSoFar=6
- i=7: currMax=1, maxSoFar=6
- i=8: currMax=5, maxSoFar=6
Output: 6

---

## 4. Merge Intervals
**Problem:** Merge all overlapping intervals.

```csharp
public int[][] Merge(int[][] intervals)
{
    if (intervals.Length == 0) return intervals;
    Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
    var merged = new List<int[]>();
    int[] current = intervals[0];
    foreach (var interval in intervals)
    {
        if (interval[0] <= current[1])
            current[1] = Math.Max(current[1], interval[1]);
        else
        {
            merged.Add(current);
            current = interval;
        }
    }
    merged.Add(current);
    return merged.ToArray();
}
```
**Explanation:**
- Sort intervals, then merge overlapping ones by comparing starts and ends.

**Time Complexity:**
- Best: O(n log n) (sorting)
- Worst: O(n log n)
**Space Complexity:**
- O(n)

**Dry Run Example:**
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Step-by-step:
- Sort: [[1,3],[2,6],[8,10],[15,18]]
- current=[1,3], interval=[2,6]: merge to [1,6]
- interval=[8,10]: add [1,6], current=[8,10]
- interval=[15,18]: add [8,10], current=[15,18]
- Add last: [15,18]
Output: [[1,6],[8,10],[15,18]]

---

## 5. Product of Array Except Self
**Problem:** Return an array where each element is the product of all other elements.

```csharp
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
```
**Explanation:**
- Use two passes: left product and right product, without division.

**Time Complexity:**
- Best/Worst: O(n)
**Space Complexity:**
- O(1) (excluding output array)

**Dry Run Example:**
Input: nums = [1,2,3,4]
Step-by-step:
- Left pass: answer=[1,1,2,6]
- Right pass: answer=[24,12,8,6]
Output: [24,12,8,6]

---

## 6. 3Sum
**Problem:** Find all unique triplets that sum to zero.

```csharp
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
```
**Explanation:**
- Sort and use two pointers to find triplets that sum to zero, skipping duplicates.

**Time Complexity:**
- Best: O(n^2)
- Worst: O(n^2)
**Space Complexity:**
- O(n)

**Dry Run Example:**
Input: nums = [-1,0,1,2,-1,-4]
Step-by-step:
- Sort: [-4,-1,-1,0,1,2]
- i=0: -4, left=1, right=5: sum<0, left++
- i=1: -1, left=2, right=5: sum=0, add [-1,-1,2], skip duplicates
- i=1: left=3, right=4: sum=0, add [-1,0,1]
Output: [[-1,-1,2],[-1,0,1]]

---

## 7. Container With Most Water
**Problem:** Find the maximum area of water that can be contained.

```csharp
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
```
**Explanation:**
- Use two pointers, move the shorter line inward to maximize area.

**Time Complexity:**
- Best/Worst: O(n)
**Space Complexity:**
- O(1)

**Dry Run Example:**
Input: height = [1,8,6,2,5,4,8,3,7]
Step-by-step:
- left=0, right=8: area=8, maxArea=8
- left=1, right=8: area=49, maxArea=49
- ...
Output: 49

---

## 8. Set Matrix Zeroes
**Problem:** Set entire row and column to zero if an element is zero.

```csharp
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
```
**Explanation:**
- Mark rows and columns to be zeroed, then update the matrix.

**Time Complexity:**
- Best/Worst: O(mn)
**Space Complexity:**
- O(m + n)

**Dry Run Example:**
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Step-by-step:
- Mark row 1 and col 1
- Set row 1 and col 1 to zero
Output: [[1,0,1],[0,0,0],[1,0,1]]

---

## 9. Rotate Array
**Problem:** Rotate the array to the right by k steps.

```csharp
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
```
**Explanation:**
- Reverse the whole array, then reverse first k and last n-k elements.

**Time Complexity:**
- Best/Worst: O(n)
**Space Complexity:**
- O(1)

**Dry Run Example:**
Input: nums = [1,2,3,4,5,6,7], k = 3
Step-by-step:
- Reverse all: [7,6,5,4,3,2,1]
- Reverse first 3: [5,6,7,4,3,2,1]
- Reverse last 4: [5,6,7,1,2,3,4]
Output: [5,6,7,1,2,3,4]

---

## 10. Missing Number
**Problem:** Find the missing number in an array containing n distinct numbers from 0 to n.

```csharp
public int FindMissingNumber(int[] nums)
{
    int n = nums.Length;
    int total = n * (n + 1) / 2;
    int sum = 0;
    foreach (var num in nums) sum += num;
    return total - sum;
}
```
**Explanation:**
- Use sum formula for first n natural numbers, subtract actual sum.

**Time Complexity:**
- Best/Worst: O(n)
**Space Complexity:**
- O(1)

**Dry Run Example:**
Input: nums = [3,0,1]
Step-by-step:
- n=3, total=6, sum=4
- missing=6-4=2
Output: 2

---
