# NeetCode 150 Solutions

---

## 1. Two Sum

**Problem Statement:**
Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

**Solution (C#):**
```csharp
public int[] TwoSum(int[] nums, int target) {
    // Dictionary to store value and its index
    Dictionary<int, int> map = new Dictionary<int, int>();
    for (int i = 0; i < nums.Length; i++) {
        int complement = target - nums[i]; // Find the complement
        if (map.ContainsKey(complement)) {
            // If complement exists, return indices
            return new int[] { map[complement], i };
        }
        // Store the current value and its index
        map[nums[i]] = i;
    }
    return new int[0]; // No solution found
}
```

**Explanation & Dry Run:**
- Iterate through the array, for each number, check if its complement (target - number) exists in the map.
- If found, return indices.
- Example 1: nums = [2,7,11,15], target = 9
  - i=0: 2, complement=7, not in map
  - i=1: 7, complement=2, found at index 0 → return [0,1]
- Example 2: nums = [3,2,4], target = 6
  - i=0: 3, complement=3, not in map
  - i=1: 2, complement=4, not in map
  - i=2: 4, complement=2, found at index 1 → return [1,2]

**Complexity:**
- Time: O(n)
  - We traverse the array once, and dictionary operations (insert, lookup) are O(1) on average.
- Space: O(n)
  - In the worst case, we store every element in the dictionary.

---

## 2. Best Time to Buy and Sell Stock

**Problem Statement:**
Given an array `prices` where `prices[i]` is the price of a stock on day i, find the maximum profit you can achieve. You may only buy once and sell once.

**Solution (C#):**
```csharp
public int MaxProfit(int[] prices) {
    int minPrice = int.MaxValue; // Track the minimum price so far
    int maxProfit = 0; // Track the maximum profit
    foreach (int price in prices) {
        if (price < minPrice) minPrice = price; // Update min price
        else if (price - minPrice > maxProfit) maxProfit = price - minPrice; // Update max profit
    }
    return maxProfit;
}
```

**Explanation & Dry Run:**
- Track the minimum price so far and the max profit.
- Example 1: prices = [7,1,5,3,6,4]
  - minPrice=7, maxProfit=0
  - price=1 → minPrice=1
  - price=5 → maxProfit=4
  - price=6 → maxProfit=5
- Example 2: prices = [7,6,4,3,1]
  - minPrice=7, maxProfit=0
  - price=6 → minPrice=6
  - price=4 → minPrice=4
  - price=3 → minPrice=3
  - price=1 → minPrice=1
  - No profit possible, return 0

**Complexity:**
- Time: O(n)
  - We iterate through the array once, updating minPrice and maxProfit in constant time per element.
- Space: O(1)
  - Only two variables are used regardless of input size.

---

## 3. Contains Duplicate

**Problem Statement:**
Given an array of integers, find if the array contains any duplicates.

**Solution (C#):**
```csharp
public bool ContainsDuplicate(int[] nums) {
    HashSet<int> set = new HashSet<int>(); // Store unique elements
    foreach (int num in nums) {
        if (!set.Add(num)) // If num already exists, Add returns false
            return true; // Duplicate found
    }
    return false; // No duplicates
}
```

**Explanation & Dry Run:**
- Add each number to a set. If already present, return true.
- Example 1: nums = [1,2,3,1]
  - Add 1 → set: {1}
  - Add 2 → set: {1,2}
  - Add 3 → set: {1,2,3}
  - Add 1 → already in set → return true
- Example 2: nums = [1,2,3,4]
  - Add 1 → set: {1}
  - Add 2 → set: {1,2}
  - Add 3 → set: {1,2,3}
  - Add 4 → set: {1,2,3,4}
  - No duplicates, return false

**Complexity:**
- Time: O(n)
  - Each insertion and lookup in a HashSet is O(1) on average, and we process each element once.
- Space: O(n)
  - In the worst case, all elements are unique and stored in the set.

---

## 4. Product of Array Except Self

**Problem Statement:**
Given an array `nums`, return an array `answer` such that `answer[i]` is the product of all the elements of `nums` except `nums[i]`.

**Solution (C#):**
```csharp
public int[] ProductExceptSelf(int[] nums) {
    int n = nums.Length;
    int[] answer = new int[n];
    int left = 1;
    // First pass: product of all elements to the left of i
    for (int i = 0; i < n; i++) {
        answer[i] = left;
        left *= nums[i];
    }
    int right = 1;
    // Second pass: product of all elements to the right of i
    for (int i = n - 1; i >= 0; i--) {
        answer[i] *= right;
        right *= nums[i];
    }
    return answer;
}
```

**Explanation & Dry Run:**
- First pass: fill answer[i] with product of all elements to the left.
- Second pass: multiply by product of all elements to the right.
- Example 1: nums = [1,2,3,4]
  - Left pass: [1,1,2,6]
  - Right pass: [24,12,8,6]
- Example 2: nums = [2,3,4,5]
  - Left pass: [1,2,6,24]
  - Right pass: [60,40,30,24]

**Complexity:**
- Time: O(n)
  - We traverse the array twice, each in O(n) time.
- Space: O(1) (excluding output)
  - Only a few variables are used for calculation; output array is not counted as extra space.

---

## 5. Maximum Subarray

**Problem Statement:**
Given an integer array `nums`, find the contiguous subarray with the largest sum.

**Solution (C#):**
```csharp
public int MaxSubArray(int[] nums) {
    int maxSum = nums[0]; // Store the maximum sum found so far
    int currSum = nums[0]; // Store the current subarray sum
    for (int i = 1; i < nums.Length; i++) {
        // Decide to start new subarray or extend previous
        currSum = Math.Max(nums[i], currSum + nums[i]);
        maxSum = Math.Max(maxSum, currSum); // Update maxSum if needed
    }
    return maxSum;
}
```

**Explanation & Dry Run:**
- Use Kadane's algorithm: at each step, decide to start new subarray or extend previous.
- Example 1: nums = [-2,1,-3,4,-1,2,1,-5,4]
  - maxSum progresses: 1, 4, 6, 7
- Example 2: nums = [5,4,-1,7,8]
  - maxSum progresses: 5, 9, 8, 15, 23

**Complexity:**
- Time: O(n)
  - We iterate through the array once, updating sums in constant time per element.
- Space: O(1)
  - Only two variables are used regardless of input size.

---

## 6. Minimum in Rotated Sorted Array

**Problem Statement:**
Given a rotated sorted array, find the minimum element.

**Solution (C#):**
```csharp
public int FindMin(int[] nums) {
    int left = 0, right = nums.Length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        // If mid element is greater than right, min is in right half
        if (nums[mid] > nums[right]) left = mid + 1;
        else right = mid;
    }
    return nums[left]; // The minimum element
}
```

**Explanation & Dry Run:**
- Use binary search to find the inflection point.
- Example 1: nums = [3,4,5,1,2]
  - min is 1
- Example 2: nums = [4,5,6,7,0,1,2]
  - min is 0

**Complexity:**
- Time: O(log n) — Binary search halves the search space each time.
- Space: O(1) — Only pointers used.

---

## 7. Search in Rotated Sorted Array

**Problem Statement:**
Given a rotated sorted array and a target, return its index or -1 if not found.

**Solution (C#):**
```csharp
public int Search(int[] nums, int target) {
    int left = 0, right = nums.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        // Left sorted portion
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        } else {
            // Right sorted portion
            if (nums[mid] < target && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}
```

**Explanation & Dry Run:**
- Use binary search, check which half is sorted.
- Example 1: nums = [4,5,6,7,0,1,2], target = 0 → returns 4
- Example 2: nums = [4,5,6,7,0,1,2], target = 3 → returns -1

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 8. 3Sum

**Problem Statement:**
Given an array, find all unique triplets that sum to zero.

**Solution (C#):**
```csharp
public IList<IList<int>> ThreeSum(int[] nums) {
    Array.Sort(nums); // Sort the array
    var res = new List<IList<int>>();
    for (int i = 0; i < nums.Length - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue; // Skip duplicates
        int left = i + 1, right = nums.Length - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum == 0) {
                res.Add(new List<int> { nums[i], nums[left], nums[right] });
                left++; right--;
                while (left < right && nums[left] == nums[left - 1]) left++; // Skip duplicates
                while (left < right && nums[right] == nums[right + 1]) right--; // Skip duplicates
            } else if (sum < 0) left++;
            else right--;
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- Sort, then use two pointers for each i.
- Example 1: nums = [-1,0,1,2,-1,-4] → [[-1,-1,2],[-1,0,1]]
- Example 2: nums = [0,1,1] → []

**Complexity:**
- Time: O(n^2)
- Space: O(k) for output

---

## 9. Container With Most Water

**Problem Statement:**
Given n non-negative integers representing heights, find two lines that together with the x-axis form a container, such that the container contains the most water.

**Solution (C#):**
```csharp
public int MaxArea(int[] height) {
    int left = 0, right = height.Length - 1, maxArea = 0;
    while (left < right) {
        int area = Math.Min(height[left], height[right]) * (right - left);
        maxArea = Math.Max(maxArea, area);
        // Move the shorter line inward
        if (height[left] < height[right]) left++;
        else right--;
    }
    return maxArea;
}
```

**Explanation & Dry Run:**
- Use two pointers, move the shorter line.
- Example 1: height = [1,8,6,2,5,4,8,3,7] → max area = 49
- Example 2: height = [1,1] → max area = 1

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 10. Trapping Rain Water

**Problem Statement:**
Given n non-negative integers representing an elevation map, compute how much water it can trap after raining.

**Solution (C#):**
```csharp
public int Trap(int[] height) {
    int left = 0, right = height.Length - 1;
    int leftMax = 0, rightMax = 0, res = 0;
    while (left < right) {
        if (height[left] < height[right]) {
            leftMax = Math.Max(leftMax, height[left]);
            res += leftMax - height[left];
            left++;
        } else {
            rightMax = Math.Max(rightMax, height[right]);
            res += rightMax - height[right];
            right--;
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- Track max heights from both ends.
- Example 1: height = [0,1,0,2,1,0,1,3,2,1,2,1] → trapped water = 6
- Example 2: height = [4,2,0,3,2,5] → trapped water = 9

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 11. Valid Parentheses

**Problem Statement:**
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

**Solution (C#):**
```csharp
public bool IsValid(string s) {
    Stack<char> stack = new Stack<char>();
    foreach (char c in s) {
        if (c == '(' || c == '{' || c == '[') stack.Push(c);
        else {
            if (stack.Count == 0) return false;
            char top = stack.Pop();
            if ((c == ')' && top != '(') ||
                (c == '}' && top != '{') ||
                (c == ']' && top != '[')) return false;
        }
    }
    return stack.Count == 0;
}
```

**Explanation & Dry Run:**
- Use stack to match opening and closing brackets.
- Example 1: s = "()[]{}" → true
- Example 2: s = "(]" → false

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 12. Merge Intervals

**Problem Statement:**
Given an array of intervals, merge all overlapping intervals.

**Solution (C#):**
```csharp
public int[][] Merge(int[][] intervals) {
    if (intervals.Length == 0) return intervals;
    Array.Sort(intervals, (a, b) => a[0] - b[0]);
    var merged = new List<int[]>();
    int[] current = intervals[0];
    foreach (var interval in intervals) {
        // If intervals overlap, merge them
        if (interval[0] <= current[1]) current[1] = Math.Max(current[1], interval[1]);
        else {
            merged.Add(current);
            current = interval;
        }
    }
    merged.Add(current);
    return merged.ToArray();
}
```

**Explanation & Dry Run:**
- Sort intervals, merge overlapping ones.
- Example 1: intervals = [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]
- Example 2: intervals = [[1,4],[4,5]] → [[1,5]]

**Complexity:**
- Time: O(n log n)
- Space: O(n)

---

## 13. Insert Interval

**Problem Statement:**
Given a set of non-overlapping intervals and a new interval, insert the new interval into the intervals (merge if necessary).

**Solution (C#):**
```csharp
public int[][] Insert(int[][] intervals, int[] newInterval) {
    var result = new List<int[]>();
    int i = 0, n = intervals.Length;
    // Add intervals before newInterval
    while (i < n && intervals[i][1] < newInterval[0]) result.Add(intervals[i++]);
    // Merge overlapping intervals
    while (i < n && intervals[i][0] <= newInterval[1]) {
        newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
        newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
        i++;
    }
    result.Add(newInterval);
    // Add remaining intervals
    while (i < n) result.Add(intervals[i++]);
    return result.ToArray();
}
```

**Explanation & Dry Run:**
- Add intervals before, merge overlapping, add after.
- Example 1: intervals = [[1,3],[6,9]], newInterval = [2,5] → [[1,5],[6,9]]
- Example 2: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8] → [[1,2],[3,10],[12,16]]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 14. Group Anagrams

**Problem Statement:**
Given an array of strings, group anagrams together.

**Solution (C#):**
```csharp
public IList<IList<string>> GroupAnagrams(string[] strs) {
    var map = new Dictionary<string, List<string>>();
    foreach (var s in strs) {
        char[] chars = s.ToCharArray();
        Array.Sort(chars); // Sort to get the key
        string key = new string(chars);
        if (!map.ContainsKey(key)) map[key] = new List<string>();
        map[key].Add(s);
    }
    return map.Values.ToList<IList<string>>();
}
```

**Explanation & Dry Run:**
- Sort each string, use as key in map.
- Example 1: strs = ["eat","tea","tan","ate","nat","bat"] → [["eat","tea","ate"],["tan","nat"],["bat"]]
- Example 2: strs = [""] → [[""]]

**Complexity:**
- Time: O(n k log k) — n strings, each sorted (k = max length)
- Space: O(n k)

---

## 15. Top K Frequent Elements

**Problem Statement:**
Given a non-empty array of integers, return the k most frequent elements.

**Solution (C#):**
```csharp
public int[] TopKFrequent(int[] nums, int k) {
    var freq = new Dictionary<int, int>();
    foreach (var num in nums) {
        if (!freq.ContainsKey(num)) freq[num] = 0;
        freq[num]++;
    }
    // Bucket sort by frequency
    List<int>[] buckets = new List<int>[nums.Length + 1];
    foreach (var pair in freq) {
        int count = pair.Value;
        if (buckets[count] == null) buckets[count] = new List<int>();
        buckets[count].Add(pair.Key);
    }
    var res = new List<int>();
    for (int i = buckets.Length - 1; i >= 0 && res.Count < k; i--) {
        if (buckets[i] != null) res.AddRange(buckets[i]);
    }
    return res.Take(k).ToArray();
}
```

**Explanation & Dry Run:**
- Count frequency, bucket sort, collect top k.
- Example 1: nums = [1,1,1,2,2,3], k = 2 → [1,2]
- Example 2: nums = [1], k = 1 → [1]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 16. Kth Largest Element in an Array

**Problem Statement:**
Find the kth largest element in an unsorted array.

**Solution (C#):**
```csharp
public int FindKthLargest(int[] nums, int k) {
    // Use a min-heap of size k
    var pq = new SortedSet<(int val, int idx)>();
    for (int i = 0; i < nums.Length; i++) {
        pq.Add((nums[i], i));
        if (pq.Count > k) pq.Remove(pq.Min);
    }
    return pq.Min.val;
}
```

**Explanation & Dry Run:**
- Maintain a min-heap of size k.
- Example 1: nums = [3,2,1,5,6,4], k = 2 → 5
- Example 2: nums = [3,2,3,1,2,4,5,5,6], k = 4 → 4

**Complexity:**
- Time: O(n log k)
- Space: O(k)

---

## 17. Longest Consecutive Sequence

**Problem Statement:**
Given an unsorted array, find the length of the longest consecutive elements sequence.

**Solution (C#):**
```csharp
public int LongestConsecutive(int[] nums) {
    var set = new HashSet<int>(nums);
    int longest = 0;
    foreach (int num in nums) {
        // Only start counting if num is the start of a sequence
        if (!set.Contains(num - 1)) {
            int curr = num;
            int streak = 1;
            while (set.Contains(curr + 1)) {
                curr++;
                streak++;
            }
            longest = Math.Max(longest, streak);
        }
    }
    return longest;
}
```

**Explanation & Dry Run:**
- Use a set for O(1) lookups, only start at sequence starts.
- Example 1: nums = [100,4,200,1,3,2] → 4 ([1,2,3,4])
- Example 2: nums = [0,3,7,2,5,8,4,6,0,1] → 9 ([0-8])

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 18. Course Schedule

**Problem Statement:**
Given numCourses and prerequisites, determine if you can finish all courses.

**Solution (C#):**
```csharp
public bool CanFinish(int numCourses, int[][] prerequisites) {
    var graph = new List<int>[numCourses];
    for (int i = 0; i < numCourses; i++) graph[i] = new List<int>();
    foreach (var p in prerequisites) graph[p[1]].Add(p[0]);
    var visited = new int[numCourses]; // 0=unvisited, 1=visiting, 2=visited
    bool dfs(int node) {
        if (visited[node] == 1) return false;
        if (visited[node] == 2) return true;
        visited[node] = 1;
        foreach (var nei in graph[node]) if (!dfs(nei)) return false;
        visited[node] = 2;
        return true;
    }
    for (int i = 0; i < numCourses; i++) if (!dfs(i)) return false;
    return true;
}
```

**Explanation & Dry Run:**
- DFS for cycle detection.
- Example 1: numCourses = 2, prerequisites = [[1,0]] → true
- Example 2: numCourses = 2, prerequisites = [[1,0],[0,1]] → false

**Complexity:**
- Time: O(V+E)
- Space: O(V+E)

---

## 19. Valid Palindrome

**Problem Statement:**
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

**Solution (C#):**
```csharp
public bool IsPalindrome(string s) {
    int left = 0, right = s.Length - 1;
    while (left < right) {
        while (left < right && !char.IsLetterOrDigit(s[left])) left++;
        while (left < right && !char.IsLetterOrDigit(s[right])) right--;
        if (char.ToLower(s[left]) != char.ToLower(s[right])) return false;
        left++; right--;
    }
    return true;
}
```

**Explanation & Dry Run:**
- Two pointers, skip non-alphanumeric.
- Example 1: s = "A man, a plan, a canal: Panama" → true
- Example 2: s = "race a car" → false

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 20. Valid Anagram

**Problem Statement:**
Given two strings, determine if they are anagrams of each other.

**Solution (C#):**
```csharp
public bool IsAnagram(string s, string t) {
    if (s.Length != t.Length) return false;
    int[] count = new int[26];
    for (int i = 0; i < s.Length; i++) {
        count[s[i] - 'a']++;
        count[t[i] - 'a']--;
    }
    foreach (int c in count) if (c != 0) return false;
    return true;
}
```

**Explanation & Dry Run:**
- Count characters, compare counts.
- Example 1: s = "anagram", t = "nagaram" → true
- Example 2: s = "rat", t = "car" → false

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 21. Binary Search

**Problem Statement:**
Given a sorted array and a target, return its index or -1 if not found.

**Solution (C#):**
```csharp
public int Search(int[] nums, int target) {
    int left = 0, right = nums.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

**Explanation & Dry Run:**
- Standard binary search.
- Example 1: nums = [-1,0,3,5,9,12], target = 9 → 4
- Example 2: nums = [-1,0,3,5,9,12], target = 2 → -1

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 22. Search a 2D Matrix

**Problem Statement:**
Given a 2D matrix, search for a target value.

**Solution (C#):**
```csharp
public bool SearchMatrix(int[][] matrix, int target) {
    int m = matrix.Length, n = matrix[0].Length;
    int left = 0, right = m * n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int val = matrix[mid / n][mid % n];
        if (val == target) return true;
        else if (val < target) left = mid + 1;
        else right = mid - 1;
    }
    return false;
}
```

**Explanation & Dry Run:**
- Treat matrix as a flat array.
- Example 1: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3 → true
- Example 2: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13 → false

**Complexity:**
- Time: O(log(mn))
- Space: O(1)

---

## 23. Pow(x, n)

**Problem Statement:**
Implement pow(x, n), which calculates x raised to the power n.

**Solution (C#):**
```csharp
public double MyPow(double x, int n) {
    long N = n;
    if (N < 0) { x = 1 / x; N = -N; }
    double res = 1;
    while (N > 0) {
        if ((N & 1) == 1) res *= x;
        x *= x;
        N >>= 1;
    }
    return res;
}
```

**Explanation & Dry Run:**
- Fast exponentiation (binary).
- Example 1: x = 2.0, n = 10 → 1024.0
- Example 2: x = 2.1, n = 3 → 9.261

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 24. Climbing Stairs

**Problem Statement:**
You can climb 1 or 2 steps at a time. Given n steps, how many distinct ways can you climb to the top?

**Solution (C#):**
```csharp
public int ClimbStairs(int n) {
    if (n <= 2) return n;
    int a = 1, b = 2;
    for (int i = 3; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}
```

**Explanation & Dry Run:**
- Fibonacci sequence.
- Example 1: n = 2 → 2
- Example 2: n = 3 → 3

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 25. Coin Change

**Problem Statement:**
Given coins of different denominations and a total amount, find the fewest coins needed to make up that amount.

**Solution (C#):**
```csharp
public int CoinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Array.Fill(dp, amount + 1);
    dp[0] = 0;
    for (int coin = 0; coin < coins.Length; coin++) {
        for (int i = coins[coin]; i <= amount; i++) {
            dp[i] = Math.Min(dp[i], dp[i - coins[coin]] + 1);
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```

**Explanation & Dry Run:**
- Dynamic programming.
- Example 1: coins = [1,2,5], amount = 11 → 3 ([5,5,1])
- Example 2: coins = [2], amount = 3 → -1

**Complexity:**
- Time: O(amount * coins.Length)
- Space: O(amount)

---

## 26. Longest Increasing Subsequence

**Problem Statement:**
Given an unsorted array, find the length of the longest increasing subsequence.

**Solution (C#):**
```csharp
public int LengthOfLIS(int[] nums) {
    int[] dp = new int[nums.Length];
    Array.Fill(dp, 1);
    for (int i = 1; i < nums.Length; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) dp[i] = Math.Max(dp[i], dp[j] + 1);
        }
    }
    return dp.Max();
}
```

**Explanation & Dry Run:**
- Dynamic programming, dp[i] = length of LIS ending at i.
- Example 1: nums = [10,9,2,5,3,7,101,18] → 4 ([2,3,7,101])
- Example 2: nums = [0,1,0,3,2,3] → 4 ([0,1,2,3])

**Complexity:**
- Time: O(n^2)
  - For each element, we compare it with all previous elements, leading to a nested loop.
- Space: O(n)
  - We use a dp array of size n to store the LIS length for each index.

---

## 27. Longest Common Subsequence

**Problem Statement:**
Given two strings, find the length of their longest common subsequence.

**Solution (C#):**
```csharp
public int LongestCommonSubsequence(string text1, string text2) {
    int m = text1.Length, n = text2.Length;
    int[,] dp = new int[m + 1, n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i - 1] == text2[j - 1])
                dp[i, j] = dp[i - 1, j - 1] + 1;
            else
                dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
        }
    }
    return dp[m, n];
}
```

**Explanation & Dry Run:**
- DP table, compare characters.
- Example 1: text1 = "abcde", text2 = "ace" → 3
- Example 2: text1 = "abc", text2 = "abc" → 3

**Complexity:**
- Time: O(mn)
  - We fill a table of size m x n, where m and n are the lengths of the two strings.
- Space: O(mn)
  - The DP table requires m x n space.

---

## 28. Word Break

**Problem Statement:**
Given a string and a dictionary, determine if the string can be segmented into a space-separated sequence of dictionary words.

**Solution (C#):**
```csharp
public bool WordBreak(string s, IList<string> wordDict) {
    var set = new HashSet<string>(wordDict);
    bool[] dp = new bool[s.Length + 1];
    dp[0] = true;
    for (int i = 1; i <= s.Length; i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && set.Contains(s.Substring(j, i - j))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[s.Length];
}
```

**Explanation & Dry Run:**
- DP, dp[i] = true if s[0..i] can be segmented.
- Example 1: s = "leetcode", wordDict = ["leet","code"] → true
- Example 2: s = "applepenapple", wordDict = ["apple","pen"] → true

**Complexity:**
- Time: O(n^3)
  - For each position, we check all previous positions and substring operations.
- Space: O(n)
  - The dp array is of size n+1.

---

## 29. Combination Sum

**Problem Statement:**
Given an array of distinct integers and a target, find all unique combinations that sum to the target.

**Solution (C#):**
```csharp
public IList<IList<int>> CombinationSum(int[] candidates, int target) {
    var res = new List<IList<int>>();
    void Backtrack(int start, int sum, List<int> comb) {
        if (sum == target) {
            res.Add(new List<int>(comb));
            return;
        }
        if (sum > target) return;
        for (int i = start; i < candidates.Length; i++) {
            comb.Add(candidates[i]);
            Backtrack(i, sum + candidates[i], comb);
            comb.RemoveAt(comb.Count - 1);
        }
    }
    Backtrack(0, 0, new List<int>());
    return res;
}
```

**Explanation & Dry Run:**
- Backtracking, try all combinations.
- Example 1: candidates = [2,3,6,7], target = 7 → [[7],[2,2,3]]
- Example 2: candidates = [2,3,5], target = 8 → [[2,2,2,2],[2,3,3],[3,5]]

**Complexity:**
- Time: O(2^n)
  - Each candidate can be included or not, leading to exponential combinations.
- Space: O(target)
  - The recursion stack can go as deep as the target value.

---

## 30. Permutations

**Problem Statement:**
Given a collection of distinct numbers, return all possible permutations.

**Solution (C#):**
```csharp
public IList<IList<int>> Permute(int[] nums) {
    var res = new List<IList<int>>();
    void Backtrack(List<int> perm, bool[] used) {
        if (perm.Count == nums.Length) {
            res.Add(new List<int>(perm));
            return;
        }
        for (int i = 0; i < nums.Length; i++) {
            if (used[i]) continue;
            used[i] = true;
            perm.Add(nums[i]);
            Backtrack(perm, used);
            perm.RemoveAt(perm.Count - 1);
            used[i] = false;
        }
    }
    Backtrack(new List<int>(), new bool[nums.Length]);
    return res;
}
```

**Explanation & Dry Run:**
- Backtracking, build permutations.
- Example 1: nums = [1,2,3] → [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
- Example 2: nums = [0,1] → [[0,1],[1,0]]

**Complexity:**
- Time: O(n!)
  - There are n! possible permutations for n elements.
- Space: O(n!)
  - All permutations are stored in the result list.

---

## 31. Rotate Image

**Problem Statement:**
Given an n x n 2D matrix, rotate the image by 90 degrees (clockwise).

**Solution (C#):**
```csharp
public void Rotate(int[][] matrix) {
    int n = matrix.Length;
    // Transpose
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
    // Reverse each row
    for (int i = 0; i < n; i++) {
        Array.Reverse(matrix[i]);
    }
}
```

**Explanation & Dry Run:**
- Transpose, then reverse each row.
- Example 1: matrix = [[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]]
- Example 2: matrix = [[1,2],[3,4]] → [[3,1],[4,2]]

**Complexity:**
- Time: O(n^2)
  - We visit each element twice: once for transpose, once for reverse.
- Space: O(1)
  - The rotation is done in-place, no extra space used.

---

## 32. Set Matrix Zeroes

**Problem Statement:**
Given a matrix, if an element is 0, set its entire row and column to 0.

**Solution (C#):**
```csharp
public void SetZeroes(int[][] matrix) {
    int m = matrix.Length, n = matrix[0].Length;
    bool[] rows = new bool[m];
    bool[] cols = new bool[n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] == 0) {
                rows[i] = true;
                cols[j] = true;
            }
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (rows[i] || cols[j]) matrix[i][j] = 0;
        }
    }
}
```

**Explanation & Dry Run:**
- Mark rows and columns, then set zeroes.
- Example 1: matrix = [[1,1,1],[1,0,1],[1,1,1]] → [[1,0,1],[0,0,0],[1,0,1]]
- Example 2: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]] → [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

**Complexity:**
- Time: O(mn)
  - We scan the matrix twice: once to mark, once to set zeroes.
- Space: O(m+n)
  - We use two arrays to track rows and columns to zero.

---

## 33. Spiral Matrix

**Problem Statement:**
Given an m x n matrix, return all elements in spiral order.

**Solution (C#):**
```csharp
public IList<int> SpiralOrder(int[][] matrix) {
    var res = new List<int>();
    int m = matrix.Length, n = matrix[0].Length;
    int top = 0, bottom = m - 1, left = 0, right = n - 1;
    while (top <= bottom && left <= right) {
        for (int i = left; i <= right; i++) res.Add(matrix[top][i]);
        top++;
        for (int i = top; i <= bottom; i++) res.Add(matrix[i][right]);
        right--;
        if (top <= bottom) {
            for (int i = right; i >= left; i--) res.Add(matrix[bottom][i]);
            bottom--;
        }
        if (left <= right) {
            for (int i = bottom; i >= top; i--) res.Add(matrix[i][left]);
            left++;
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- Traverse boundaries in spiral order.
- Example 1: matrix = [[1,2,3],[4,5,6],[7,8,9]] → [1,2,3,6,9,8,7,4,5]
- Example 2: matrix = [[1,2],[3,4]] → [1,2,4,3]

**Complexity:**
- Time: O(mn)
  - Each element is visited once.
- Space: O(1) (excluding output)
  - No extra space except for the output list.

---

## 34. Jump Game

**Problem Statement:**
Given an array where each element represents your maximum jump length, determine if you can reach the last index.

**Solution (C#):**
```csharp
public bool CanJump(int[] nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.Length; i++) {
        if (i > maxReach) return false;
        maxReach = Math.Max(maxReach, i + nums[i]);
    }
    return true;
}
```

**Explanation & Dry Run:**
- Track the farthest index you can reach.
- Example 1: nums = [2,3,1,1,4]
  - maxReach=0
  - i=0: 2, maxReach=2
  - i=1: 3, maxReach=4
- Example 2: nums = [3,2,1,0,4]
  - maxReach=0
  - i=0: 3, maxReach=3
  - i=1: 2, maxReach=3
  - i=2: 1, maxReach=3
  - i=3: 0, maxReach=3
  - i=4: 4, can't reach → return false

**Complexity:**
- Time: O(n)
  - We iterate through the array once, updating maxReach in constant time per element.
- Space: O(1)
  - Only a few variables are used.

---

## 35. Merge Sorted Array

**Problem Statement:**
Given two sorted arrays, merge them into one sorted array in-place.

**Solution (C#):**
```csharp
public void Merge(int[] nums1, int m, int[] nums2, int n) {
    int i = m - 1, j = n - 1, k = m + n - 1;
    while (i >= 0 && j >= 0) {
        if (nums1[i] > nums2[j]) nums1[k--] = nums1[i--];
        else nums1[k--] = nums2[j--];
    }
    while (j >= 0) nums1[k--] = nums2[j--];
}
```

**Explanation & Dry Run:**
- Merge from the end to avoid overwriting.
- Example 1: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3 → [1,2,2,3,5,6]
- Example 2: nums1 = [1], m = 1, nums2 = [], n = 0 → [1]

**Complexity:**
- Time: O(m+n)
  - Each element from both arrays is processed once.
- Space: O(1)
  - The merge is done in-place, no extra space used.

---

## 36. Find Minimum Window Substring

**Problem Statement:**
Given two strings s and t, return the minimum window in s which contains all the characters in t.

**Solution (C#):**
```csharp
public string MinWindow(string s, string t) {
    if (t.Length > s.Length) return "";
    var map = new Dictionary<char, int>();
    foreach (char c in t) {
        if (!map.ContainsKey(c)) map[c] = 0;
        map[c]++;
    }
    int left = 0, right = 0, minLen = int.MaxValue, minStart = 0, count = t.Length;
    while (right < s.Length) {
        if (map.ContainsKey(s[right]) && map[s[right]] > 0) count--;
        if (map.ContainsKey(s[right])) map[s[right]]--;
        right++;
        while (count == 0) {
            if (right - left < minLen) {
                minLen = right - left;
                minStart = left;
            }
            if (map.ContainsKey(s[left])) {
                map[s[left]]++;
                if (map[s[left]] > 0) count++;
            }
            left++;
        }
    }
    return minLen == int.MaxValue ? "" : s.Substring(minStart, minLen);
}
```

**Explanation & Dry Run:**
- Use sliding window and hash map to track required characters.
- Example 1: root = [1,2,3,4,5,6,7], k = 3 → "BANC"
- Example 2: root = [] → ""

**Complexity:**
- Time: O(n + m)
  - Each character is processed at most twice.
- Space: O(m)
  - Hash map stores counts for t's characters.

---

## 37. Longest Substring Without Repeating Characters

**Problem Statement:**
Given a string, find the length of the longest substring without repeating characters.

**Solution (C#):**
```csharp
public int LengthOfLongestSubstring(string s) {
    var set = new HashSet<char>();
    int left = 0, maxLen = 0;
    for (int right = 0; right < s.Length; right++) {
        while (set.Contains(s[right])) {
            set.Remove(s[left++]);
        }
        set.Add(s[right]);
        maxLen = Math.Max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

**Explanation & Dry Run:**
- Use sliding window and set to track unique characters.
- Example 1: s = "abcabcbb" → 3 ("abc")
- Example 2: s = "bbbbb" → 1 ("b")

**Complexity:**
- Time: O(n)
  - Each character is visited at most twice.
- Space: O(min(n, m))
  - Set stores up to m unique characters (alphabet size).

---

## 38. Longest Repeating Character Replacement

**Problem Statement:**
Given a string and an integer k, find the length of the longest substring containing the same letter you can get after replacing at most k characters.

**Solution (C#):**
```csharp
public int CharacterReplacement(string s, int k) {
    int[] count = new int[26];
    int left = 0, maxCount = 0, maxLen = 0;
    for (int right = 0; right < s.Length; right++) {
        count[s[right] - 'A']++;
        maxCount = Math.Max(maxCount, count[s[right] - 'A']);
        while (right - left + 1 - maxCount > k) {
            count[s[left] - 'A']--;
            left++;
        }
        maxLen = Math.Max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

**Explanation & Dry Run:**
- Sliding window, track max frequency in window.
- Example 1: s = "ABAB", k = 2 → 4
- Example 2: s = "AABABBA", k = 1 → 4

**Complexity:**
- Time: O(n)
  - Each character is processed at most twice.
- Space: O(1)
  - Fixed array for 26 letters.

---

## 39. Minimum Size Subarray Sum

**Problem Statement:**
Given an array of positive integers and a target, find the minimal length of a contiguous subarray of which the sum ≥ target.

**Solution (C#):**
```csharp
public int MinSubArrayLen(int target, int[] nums) {
    int left = 0, sum = 0, minLen = int.MaxValue;
    for (int right = 0; right < nums.Length; right++) {
        sum += nums[right];
        while (sum >= target) {
            minLen = Math.Min(minLen, right - left + 1);
            sum -= nums[left++];
        }
    }
    return minLen == int.MaxValue ? 0 : minLen;
}
```

**Explanation & Dry Run:**
- Sliding window, shrink window when sum ≥ target.
- Example 1: nums = [2,3,1,2,4,3], target = 7 → 2 ([4,3])
- Example 2: nums = [1,4,4], target = 4 → 1 ([4])

**Complexity:**
- Time: O(n)
  - Each element is added and removed at most once.
- Space: O(1)
  - Only variables used.

---

## 40. Sliding Window Maximum

**Problem Statement:**
Given an array and a window size k, return the max in each window.

**Solution (C#):**
```csharp
public int[] MaxSlidingWindow(int[] height, int k) {
    var deque = new LinkedList<int>();
    var res = new List<int>();
    for (int i = 0; i < height.Length; i++) {
        while (deque.Count > 0 && deque.First.Value <= i - k) deque.RemoveFirst();
        while (deque.Count > 0 && height[deque.Last.Value] < height[i]) deque.RemoveLast();
        deque.AddLast(i);
        if (i >= k - 1) res.Add(height[deque.First.Value]);
    }
    return res.ToArray();
}
```

**Explanation & Dry Run:**
- Use deque to keep indices of useful elements.
- Example 1: height = [1,8,6,2,5,4,8,3,7], k = 3 → [8,8,8,7]
- Example 2: height = [1,1] → [1]

**Complexity:**
- Time: O(n)
  - Each element is added and removed at most once.
- Space: O(k)
  - Deque stores up to k indices.

---

## 41. Intersection of Two Arrays

**Problem Statement:**
Given two arrays, return their intersection as a set.

**Solution (C#):**
```csharp
public int[] Intersection(int[] nums1, int[] nums2) {
    var set1 = new HashSet<int>(nums1);
    var set2 = new HashSet<int>(nums2);
    set1.IntersectWith(set2);
    return set1.ToArray();
}
```

**Explanation & Dry Run:**
- Use sets to find common elements.
- Example 1: nums1 = [1,2,2,1], nums2 = [2,2] → [2]
- Example 2: nums1 = [4,9,5], nums2 = [9,4,9,8,4] → [9,4]

**Complexity:**
- Time: O(n + m)
  - Build sets and intersect.
- Space: O(n + m)
  - Sets store all elements.

---

## 42. Happy Number

**Problem Statement:**
Determine if a number is happy (replace by sum of squares of digits, repeat until 1 or loop).

**Solution (C#):**
```csharp
public bool IsHappy(int n) {
    var set = new HashSet<int>();
    while (n != 1 && !set.Contains(n)) {
        set.Add(n);
        int sum = 0;
        while (n > 0) {
            int d = n % 10;
            sum += d * d;
            n /= 10;
        }
        n = sum;
    }
    return n == 1;
}
```

**Explanation & Dry Run:**
- Use set to detect cycles.
- Example 1: n = 19 → true
- Example 2: n = 2 → false

**Complexity:**
- Time: O(log n)
  - Each iteration reduces n, but may loop for unhappy numbers.
- Space: O(log n)
  - Set stores seen numbers.

---

## 43. Linked List Cycle

**Problem Statement:**
Given a linked list, determine if it has a cycle.

**Solution (C#):**
```csharp
public bool HasCycle(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.Next != null) {
        slow = slow.Next;
        fast = fast.Next.Next;
        if (slow == fast) return true;
    }
    return false;
}
```

**Explanation & Dry Run:**
- Use two pointers (Floyd's algorithm).
- Example 1: head = [3,2,0,-4], pos = 1 → true
- Example 2: head = [1,2], pos = 0 → true

**Complexity:**
- Time: O(n)
  - Each node is visited at most twice.
- Space: O(1)
  - Only pointers used.

---

## 44. Reverse Linked List

**Problem Statement:**
Reverse a singly linked list.

**Solution (C#):**
```csharp
public ListNode ReverseList(ListNode head) {
    ListNode prev = null, curr = head;
    while (curr != null) {
        ListNode next = curr.Next;
        curr.Next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```

**Explanation & Dry Run:**
- Iteratively reverse pointers.
- Example 1: head = [1,2,3,4,5] → [5,4,3,2,1]
- Example 2: head = [1,2] → [2,1]

**Complexity:**
- Time: O(n)
  - Each node is visited once.
- Space: O(1)
  - Only pointers used.

---

## 45. Merge Two Sorted Lists

**Problem Statement:**
Merge two sorted linked lists and return as a new sorted list.

**Solution (C#):**
```csharp
public ListNode MergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0), curr = dummy;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            curr.Next = l1;
            l1 = l1.Next;
        } else {
            curr.Next = l2;
            l2 = l2.Next;
        }
        curr = curr.Next;
    }
    curr.Next = l1 ?? l2;
    return dummy.Next;
}
```

**Explanation & Dry Run:**
- Use a dummy node and merge by comparing values.
- Example 1: l1 = [1,2,4], l2 = [1,3,4] → [1,1,2,3,4,4]
- Example 2: l1 = [], l2 = [] → []

**Complexity:**
- Time: O(n + m)
  - Each node from both lists is processed once.
- Space: O(1)
  - Only pointers used.

---

## 46. Remove Nth Node From End of List

**Problem Statement:**
Given a linked list, remove the nth node from the end and return its head.

**Solution (C#):**
```csharp
public ListNode RemoveNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0) { Next = head };
    ListNode curr = dummy;
    // Find the length of the list
    int length = 0;
    while (curr != null) {
        length++;
        curr = curr.Next;
    }
    // Find the node before the one to be removed
    curr = dummy;
    for (int i = 0; i < length - n; i++) {
        curr = curr.Next;
    }
    // Remove the nth node from the end
    curr.Next = curr.Next.Next;
    return dummy.Next;
}
```

**Explanation & Dry Run:**
- Calculate length, find node before target, remove target.
- Example 1: head = [1,2,3,4,5], n = 2 → [1,2,3,5]
- Example 2: head = [1], n = 1 → []

**Complexity:**
- Time: O(L)
  - L is the length of the list, each node is visited at most twice.
- Space: O(1)
  - Only pointers used.

---

## 47. Valid Perfect Square

**Problem Statement:**
Given a positive integer num, return true if it is a perfect square.

**Solution (C#):**
```csharp
public bool IsPerfectSquare(int num) {
    int left = 1, right = num;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        long sq = (long)mid * mid;
        if (sq == num) return true;
        else if (sq < num) left = mid + 1;
        else right = mid - 1;
    }
    return false;
}
```

**Explanation & Dry Run:**
- Binary search for square root.
- Example 1: num = 16 → true
- Example 2: num = 14 → false

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 48. Implement Trie (Prefix Tree)

**Problem Statement:**
Implement a trie with insert, search, and startsWith methods.

**Solution (C#):**
```csharp
public class Trie {
    private TrieNode root;
    public Trie() { root = new TrieNode(); }
    public void Insert(string word) {
        var node = root;
        foreach (char c in word) {
            if (!node.Children.ContainsKey(c)) node.Children[c] = new TrieNode();
            node = node.Children[c];
        }
        node.IsWord = true;
    }
    public bool Search(string word) {
        var node = root;
        foreach (char c in word) {
            if (!node.Children.ContainsKey(c)) return false;
            node = node.Children[c];
        }
        return node.IsWord;
    }
    public bool StartsWith(string prefix) {
        var node = root;
        foreach (char c in prefix) {
            if (!node.Children.ContainsKey(c)) return false;
            node = node.Children[c];
        }
        return true;
    }
    private class TrieNode {
        public Dictionary<char, TrieNode> Children = new Dictionary<char, TrieNode>();
        public bool IsWord = false;
    }
}
```

**Explanation & Dry Run:**
- Use nodes with children dictionary.
- Example 1: Insert "apple", Search "apple" → true
- Example 2: Search "app" → false

**Complexity:**
- Time: O(L) per operation (L = word length)
- Space: O(NL) for N words of length L

---

## 49. Implement strStr()

**Problem Statement:**
Return the index of the first occurrence of needle in haystack, or -1.

**Solution (C#):**
```csharp
public int StrStr(string haystack, string needle) {
    if (needle == "") return 0;
    for (int i = 0; i <= haystack.Length - needle.Length; i++) {
        if (haystack.Substring(i, needle.Length) == needle) return i;
    }
    return -1;
}
```

**Explanation & Dry Run:**
- Check each substring.
- Example 1: haystack = "hello", needle = "ll" → 2
- Example 2: haystack = "aaaaa", needle = "bba" → -1

**Complexity:**
- Time: O((N-M)M)
- Space: O(1)

---

## 50. Longest Palindromic Substring

**Problem Statement:**
Given a string, find the longest palindromic substring.

**Solution (C#):**
```csharp
public string LongestPalindrome(string s) {
    int start = 0, end = 0;
    for (int i = 0; i < s.Length; i++) {
        int len1 = Expand(s, i, i);
        int len2 = Expand(s, i, i + 1);
        int len = Math.Max(len1, len2);
        if (len > end - start) {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }
    }
    return s.Substring(start, end - start + 1);
}
private int Expand(string s, int left, int right) {
    while (left >= 0 && right < s.Length && s[left] == s[right]) {
        left--; right++;
    }
    return right - left - 1;
}
```

**Explanation & Dry Run:**
- Expand around center for each index.
- Example 1: s = "babad" → "bab" or "aba"
- Example 2: s = "cbbd" → "bb"

**Complexity:**
- Time: O(n^2)
- Space: O(1)

---

## 51. Palindromic Substrings

**Problem Statement:**
Count how many palindromic substrings in a string.

**Solution (C#):**
```csharp
public int CountSubstrings(string s) {
    int count = 0;
    for (int i = 0; i < s.Length; i++) {
        count += Expand(s, i, i);
        count += Expand(s, i, i + 1);
    }
    return count;
}
private int Expand(string s, int left, int right) {
    int res = 0;
    while (left >= 0 && right < s.Length && s[left] == s[right]) {
        res++; left--; right++;
    }
    return res;
}
```

**Explanation & Dry Run:**
- Expand around center for each index.
- Example 1: s = "abc" → 3
- Example 2: s = "aaa" → 6

**Complexity:**
- Time: O(n^2)
- Space: O(1)

---

## 52. Subarray Sum Equals K

**Problem Statement:**
Given an array and an integer k, find the total number of continuous subarrays whose sum equals to k.

**Solution (C#):**
```csharp
public int SubarraySum(int[] nums, int k) {
    var map = new Dictionary<int, int> { [0] = 1 };
    int sum = 0, count = 0;
    foreach (int num in nums) {
        sum += num;
        if (map.ContainsKey(sum - k)) count += map[sum - k];
        if (!map.ContainsKey(sum)) map[sum] = 0;
        map[sum]++;
    }
    return count;
}
```

**Explanation & Dry Run:**
- Use prefix sum and hash map.
- Example 1: nums = [1,1,1,2,2,3], k = 2 → 2
- Example 2: nums = [1,2,3], k = 3 → 2

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 53. Maximum Product Subarray

**Problem Statement:**
Find the contiguous subarray within an array which has the largest product.

**Solution (C#):**
```csharp
public int MaxProduct(int[] nums) {
    int max = nums[0], min = nums[0], res = nums[0];
    for (int i = 1; i < nums.Length; i++) {
        if (nums[i] < 0) (max, min) = (min, max);
        max = Math.Max(nums[i], max * nums[i]);
        min = Math.Min(nums[i], min * nums[i]);
        res = Math.Max(res, max);
    }
    return res;
}
```

**Explanation & Dry Run:**
- Track max and min product at each step.
- Example 1: nums = [2,3,-2,4] → 6
- Example 2: nums = [-2,0,-1] → 0

**Complexity:**
- Time: O(n)
  - We iterate through the array once, updating max and min in constant time per element.
- Space: O(1)
  - Only a few variables are used.

---

## 54. Find Minimum in Rotated Sorted Array II

**Problem Statement:**
Find the minimum in a rotated sorted array that may contain duplicates.

**Solution (C#):**
```csharp
public int FindMin(int[] nums) {
    int left = 0, right = nums.Length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[right]) left = mid + 1;
        else if (nums[mid] < nums[right]) right = mid;
        else right--;
    }
    return nums[left];
}
```

**Explanation & Dry Run:**
- Shrink window when duplicates.
- Example 1: nums = [2,2,2,0,1] → 0
- Example 2: nums = [1,3,5] → 1

**Complexity:**
- Time: O(n) worst case
- Space: O(1)

---

## 55. Find K Closest Elements

**Problem Statement:**
Given a sorted array, two integers k and x, find the k closest elements to x.

**Solution (C#):**
```csharp
public IList<int> FindClosestElements(int[] arr, int k, int x) {
    int left = 0, right = arr.Length - k;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (x - arr[mid] > arr[mid + k] - x) left = mid + 1;
        else right = mid;
    }
    return arr.Skip(left).Take(k).ToList();
}
```

**Explanation & Dry Run:**
- Binary search for window.
- Example 1: arr = [1,2,3,4,5], k = 4, x = 3 → [1,2,3,4]
- Example 2: arr = [1,2,3,4,5], k = 4, x = -1 → [1,2,3,4]

**Complexity:**
- Time: O(log(n-k) + k)
- Space: O(k)

---

## 56. Find Smallest Letter Greater Than Target

**Problem Statement:**
Given a list of sorted characters and a target, find the smallest character greater than target.

**Solution (C#):**
```csharp
public char NextGreatestLetter(char[] letters, char target) {
    int left = 0, right = letters.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (letters[mid] <= target) left = mid + 1;
        else right = mid - 1;
    }
    return left < letters.Length ? letters[left] : letters[0];
}
```

**Explanation & Dry Run:**
- Binary search for next greatest letter.
- Example 1: letters = ['c','f','j'], target = 'a' → 'c'
- Example 2: letters = ['c','f','j'], target = 'c' → 'f'

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 57. Find Right Interval

**Problem Statement:**
Given a list of intervals, for each interval, find the interval with the smallest start point >= end point of current interval.

**Solution (C#):**
```csharp
public int[] FindRightInterval(int[][] intervals) {
    var map = new SortedDictionary<int, int>();
    for (int i = 0; i < intervals.Length; i++) map[intervals[i][0]] = i;
    int[] res = new int[intervals.Length];
    for (int i = 0; i < intervals.Length; i++) {
        var kv = map.FirstOrDefault(x => x.Key >= intervals[i][1]);
        res[i] = kv.Equals(default(KeyValuePair<int,int>)) ? -1 : kv.Value;
    }
    return res;
}
```

**Explanation & Dry Run:**
- Use sorted map for fast lookup.
- Example 1: intervals = [[1,2]] → [-1]
- Example 2: intervals = [[3,4],[2,3],[1,2]] → [-1,0,1]

**Complexity:**
- Time: O(n log n)
- Space: O(n)

---

## 58. Find Duplicate Subtrees

**Problem Statement:**
Given a binary tree, return all duplicate subtrees.

**Solution (C#):**
```csharp
public IList<TreeNode> FindDuplicateSubtrees(TreeNode root) {
    var map = new Dictionary<string, int>();
    var res = new List<TreeNode>();
    string Traverse(TreeNode node) {
        if (node == null) return "#";
        string serial = node.val + "," + Traverse(node.Left) + "," + Traverse(node.Right);
        if (map.ContainsKey(serial)) map[serial]++;
        else map[serial] = 1;
        if (map[serial] == 2) res.Add(node);
        return serial;
    }
    Traverse(root);
    return res;
}
```

**Explanation & Dry Run:**
- Serialize subtrees, count occurrences.
- Example 1: root = [1,2,3,4,null,5,6,null,null,7,4] → [2,4]
- Example 2: root = [1,2,3,4,null,5,6,null,null,7] → []

**Complexity:**
- Time: O(n^2)
- Space: O(n)

---

## 59. Search in Rotated Sorted Array II

**Problem Statement:**
Given a rotated sorted array that may contain duplicates, return true if target is found.

**Solution (C#):**
```csharp
public bool Search(int[] nums, int target) {
    int left = 0, right = nums.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return true;
        if (nums[left] == nums[mid] && nums[mid] == nums[right]) {
            left++; right--;
        } else if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return false;
}
```

**Explanation & Dry Run:**
- Handle duplicates by shrinking window.
- Example 1: nums = [2,5,6,0,0,1,2], target = 0 → true
- Example 2: nums = [2,5,6,0,0,1,2], target = 3 → false

**Complexity:**
- Time: O(n) worst case (all duplicates)
- Space: O(1)

---

## 60. Find Peak Element

**Problem Statement:**
Find a peak element in an array (greater than neighbors).

**Solution (C#):**
```csharp
public int FindPeakElement(int[] nums) {
    int left = 0, right = nums.Length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[mid + 1]) right = mid;
        else left = mid + 1;
    }
    return left;
}
```

**Explanation & Dry Run:**
- Binary search for peak.
- Example 1: nums = [1,2,3,1] → 2
- Example 2: nums = [1,2,1,3,5,6,4] → 5

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 61. Search Insert Position

**Problem Statement:**
Given a sorted array and a target, return the index if found, or where it would be inserted.

**Solution (C#):**
```csharp
public int SearchInsert(int[] nums, int target) {
    int left = 0, right = nums.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return left;
}
```

**Explanation & Dry Run:**
- Binary search for insert position.
- Example 1: nums = [1,3,5,6], target = 5 → 2
- Example 2: nums = [1,3,5,6], target = 2 → 1

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 62. Find First and Last Position of Element in Sorted Array

**Problem Statement:**
Given a sorted array, find the starting and ending position of a target value.

**Solution (C#):**
```csharp
public int[] SearchRange(int[] nums, int target) {
    int left = FindBound(nums, target, true);
    int right = FindBound(nums, target, false);
    return new int[] { left, right };
}
private int FindBound(int[] nums, int target, bool isFirst) {
    int left = 0, right = nums.Length - 1, bound = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            bound = mid;
            if (isFirst) right = mid - 1;
            else left = mid + 1;
        } else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return bound;
}
```

**Explanation & Dry Run:**
- Binary search for first and last occurrence.
- Example 1: nums = [5,7,7,8,8,10], target = 8 → [3,4]
- Example 2: nums = [5,7,7,8,8,10], target = 6 → [-1,-1]

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 63. Serialize and Deserialize Binary Tree

**Problem Statement:**
Design an algorithm to serialize and deserialize a binary tree.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(TreeNode root) {
        var sb = new StringBuilder();
        void Preorder(TreeNode node) {
            if (node == null) return;
            sb.Append(node.val + ",");
            Preorder(node.Left);
            Preorder(node.Right);
        }
        Preorder(root);
        return sb.ToString();
    }
    public TreeNode Deserialize(string data) {
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries);
        int i = 0;
        TreeNode DFS() {
            if (vals[i] == "#") { i++; return null; }
            var node = new TreeNode(int.Parse(vals[i++]));
            node.Left = DFS();
            node.Right = DFS();
            return node;
        }
        return DFS();
    }
}
```

**Explanation & Dry Run:**
- Preorder traversal, use '#' for nulls.
- Example 1: root = [1,2,3,4,null,5,6] → "1,2,#,#,3,4,#,#,5,#,#,"
- Example 2: root = [] → "#," 

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 64. Flatten Binary Tree to Linked List

**Problem Statement:**
Flatten a binary tree to a linked list in-place.

**Solution (C#):**
```csharp
public void Flatten(TreeNode root) {
    while (root != null) {
        if (root.Left != null) {
            var rightMost = root.Left;
            while (rightMost.Right != null) rightMost = rightMost.Right;
            rightMost.Right = root.Right;
            root.Right = root.Left;
            root.Left = null;
        }
        root = root.Right;
    }
}
```

**Explanation & Dry Run:**
- Move left subtree to right, attach original right subtree.
- Example 1: root = [1,2,5,3,4,null,6] → [1,2,3,4,5,6]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 65. Search Insert Position

**Problem Statement:**
Given a sorted array and a target, return the index if found, or where it would be inserted.

**Solution (C#):**
```csharp
public int SearchInsert(int[] nums, int target) {
    int left = 0, right = nums.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return left;
}
```

**Explanation & Dry Run:**
- Binary search for insert position.
- Example 1: nums = [1,3,5,6], target = 5 → 2
- Example 2: nums = [1,3,5,6], target = 2 → 1

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 66. Find First and Last Position of Element in Sorted Array

**Problem Statement:**
Given a sorted array, find the starting and ending position of a target value.

**Solution (C#):**
```csharp
public int[] SearchRange(int[] nums, int target) {
    int left = FindBound(nums, target, true);
    int right = FindBound(nums, target, false);
    return new int[] { left, right };
}
private int FindBound(int[] nums, int target, bool isFirst) {
    int left = 0, right = nums.Length - 1, bound = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            bound = mid;
            if (isFirst) right = mid - 1;
            else left = mid + 1;
        } else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return bound;
}
```

**Explanation & Dry Run:**
- Binary search for first and last occurrence.
- Example 1: nums = [5,7,7,8,8,10], target = 8 → [3,4]
- Example 2: nums = [5,7,7,8,8,10], target = 6 → [-1,-1]

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 67. Serialize and Deserialize Binary Tree

**Problem Statement:**
Design an algorithm to serialize and deserialize a binary tree.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(TreeNode root) {
        var sb = new StringBuilder();
        void Preorder(TreeNode node) {
            if (node == null) return;
            sb.Append(node.val + ",");
            Preorder(node.Left);
            Preorder(node.Right);
        }
        Preorder(root);
        return sb.ToString();
    }
    public TreeNode Deserialize(string data) {
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries);
        int i = 0;
        TreeNode DFS() {
            if (vals[i] == "#") { i++; return null; }
            var node = new TreeNode(int.Parse(vals[i++]));
            node.Left = DFS();
            node.Right = DFS();
            return node;
        }
        return DFS();
    }
}
```

**Explanation & Dry Run:**
- Preorder traversal, use '#' for nulls.
- Example 1: root = [1,2,3,4,null,5,6] → "1,2,#,#,3,4,#,#,5,#,#,"
- Example 2: root = [] → "#," 

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 68. Flatten Binary Tree to Linked List

**Problem Statement:**
Flatten a binary tree to a linked list in-place.

**Solution (C#):**
```csharp
public void Flatten(TreeNode root) {
    while (root != null) {
        if (root.Left != null) {
            var rightMost = root.Left;
            while (rightMost.Right != null) rightMost = rightMost.Right;
            rightMost.Right = root.Right;
            root.Right = root.Left;
            root.Left = null;
        }
        root = root.Right;
    }
}
```

**Explanation & Dry Run:**
- Move left subtree to right, attach original right subtree.
- Example 1: root = [1,2,5,3,4,null,6] → [1,2,3,4,5,6]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 69. Find Peak Element

**Problem Statement:**
Find a peak element in an array (greater than neighbors).

**Solution (C#):**
```csharp
public int FindPeakElement(int[] nums) {
    int left = 0, right = nums.Length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[mid + 1]) right = mid;
        else left = mid + 1;
    }
    return left;
}
```

**Explanation & Dry Run:**
- Binary search for peak.
- Example 1: nums = [1,2,3,1] → 2
- Example 2: nums = [1,2,1,3,5,6,4] → 5

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 70. Search Insert Position

**Problem Statement:**
Given a sorted array and a target, return the index if found, or where it would be inserted.

**Solution (C#):**
```csharp
public int SearchInsert(int[] nums, int target) {
    int left = 0, right = nums.Length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return left;
}
```

**Explanation & Dry Run:**
- Binary search for insert position.
- Example 1: nums = [1,3,5,6], target = 5 → 2
- Example 2: nums = [1,3,5,6], target = 2 → 1

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 71. Find First and Last Position of Element in Sorted Array

**Problem Statement:**
Given a sorted array, find the starting and ending position of a target value.

**Solution (C#):**
```csharp
public int[] SearchRange(int[] nums, int target) {
    int left = FindBound(nums, target, true);
    int right = FindBound(nums, target, false);
    return new int[] { left, right };
}
private int FindBound(int[] nums, int target, bool isFirst) {
    int left = 0, right = nums.Length - 1, bound = -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            bound = mid;
            if (isFirst) right = mid - 1;
            else left = mid + 1;
        } else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return bound;
}
```

**Explanation & Dry Run:**
- Binary search for first and last occurrence.
- Example 1: nums = [5,7,7,8,8,10], target = 8 → [3,4]
- Example 2: nums = [5,7,7,8,8,10], target = 6 → [-1,-1]

**Complexity:**
- Time: O(log n)
- Space: O(1)

---

## 72. Serialize and Deserialize Binary Tree

**Problem Statement:**
Design an algorithm to serialize and deserialize a binary tree.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(TreeNode root) {
        var sb = new StringBuilder();
        void Preorder(TreeNode node) {
            if (node == null) return;
            sb.Append(node.val + ",");
            Preorder(node.Left);
            Preorder(node.Right);
        }
        Preorder(root);
        return sb.ToString();
    }
    public TreeNode Deserialize(string data) {
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries);
        int i = 0;
        TreeNode DFS() {
            if (vals[i] == "#") { i++; return null; }
            var node = new TreeNode(int.Parse(vals[i++]));
            node.Left = DFS();
            node.Right = DFS();
            return node;
        }
        return DFS();
    }
}
```

**Explanation & Dry Run:**
- Preorder traversal, use '#' for nulls.
- Example 1: root = [1,2,3,4,null,5,6] → "1,2,#,#,3,4,#,#,5,#,#,"
- Example 2: root = [] → "#," 

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 73. Flatten Binary Tree to Linked List

**Problem Statement:**
Flatten a binary tree to a linked list in-place.

**Solution (C#):**
```csharp
public void Flatten(TreeNode root) {
    while (root != null) {
        if (root.Left != null) {
            var rightMost = root.Left;
            while (rightMost.Right != null) rightMost = rightMost.Right;
            rightMost.Right = root.Right;
            root.Right = root.Left;
            root.Left = null;
        }
        root = root.Right;
    }
}
```

**Explanation & Dry Run:**
- Move left subtree to right, attach original right subtree.
- Example 1: root = [1,2,5,3,4,null,6] → [1,2,3,4,5,6]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 74. Binary Tree Maximum Path Sum

**Problem Statement:**
Given a non-empty binary tree, find the maximum path sum.

**Solution (C#):**
```csharp
public int MaxPathSum(TreeNode root) {
    int maxSum = int.MinValue;
    int DFS(TreeNode node) {
        if (node == null) return 0;
        int left = Math.Max(DFS(node.Left), 0);
        int right = Math.Max(DFS(node.Right), 0);
        maxSum = Math.Max(maxSum, node.val + left + right);
        return node.val + Math.Max(left, right);
    }
    DFS(root);
    return maxSum;
}
```

**Explanation & Dry Run:**
- DFS, track max sum at each node.
- Example 1: root = [1,2,3] → 6
- Example 2: root = [-10,9,20,null,null,15,7] → 42

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 75. Binary Tree Level Order Traversal

**Problem Statement:**
Return the level order traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<IList<int>> LevelOrder(TreeNode root) {
    var res = new List<IList<int>>();
    if (root == null) return res;
    var queue = new Queue<TreeNode>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        int size = queue.Count;
        var level = new List<int>();
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            level.Add(node.val);
            if (node.Left != null) queue.Enqueue(node.Left);
            if (node.Right != null) queue.Enqueue(node.Right);
        }
        res.Add(level);
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, collect nodes at each level.
- Example 1: root = [3,9,20,null,null,15,7] → [[3],[9,20],[15,7]]
- Example 2: root = [1] → [[1]]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 76. Binary Tree Right Side View

**Problem Statement:**
Return the values of the nodes you can see from the right side.

**Solution (C#):**
```csharp
public IList<int> RightSideView(TreeNode root) {
    var res = new List<int>();
    if (root == null) return res;
    var queue = new Queue<TreeNode>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        int size = queue.Count;
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            if (i == size - 1) res.Add(node.val);
            if (node.Left != null) queue.Enqueue(node.Left);
            if (node.Right != null) queue.Enqueue(node.Right);
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, add last node at each level.
- Example 1: root = [1,2,3,null,5,null,4] → [1,3,4]
- Example 2: root = [1,null,3] → [1,3]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 77. Binary Tree Zigzag Level Order Traversal

**Problem Statement:**
Return the zigzag level order traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<IList<int>> ZigzagLevelOrder(TreeNode root) {
    var res = new List<IList<int>>();
    if (root == null) return res;
    var queue = new Queue<TreeNode>();
    queue.Enqueue(root);
    bool leftToRight = true;
    while (queue.Count > 0) {
        int size = queue.Count;
        var level = new LinkedList<int>();
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            if (leftToRight) level.AddLast(node.val);
            else level.AddFirst(node.val);
            if (node.Left != null) queue.Enqueue(node.Left);
            if (node.Right != null) queue.Enqueue(node.Right);
        }
        res.Add(level.ToList());
        leftToRight = !leftToRight;
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, alternate direction at each level.
- Example 1: root = [3,9,20,null,null,15,7] → [[3],[20,9],[15,7]]
- Example 2: root = [1] → [[1]]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 78. Binary Tree Inorder Traversal

**Problem Statement:**
Return the inorder traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> InorderTraversal(TreeNode root) {
    var res = new List<int>();
    var stack = new Stack<TreeNode>();
    while (root != null || stack.Count > 0) {
        while (root != null) {
            stack.Push(root);
            root = root.Left;
        }
        root = stack.Pop();
        res.Add(root.val);
        root = root.Right;
    }
    return res;
}
```

**Explanation & Dry Run:**
- Use stack for iterative inorder traversal.
- Example 1: root = [1,null,2,3] → [1,3,2]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 79. Binary Tree Preorder Traversal

**Problem Statement:**
Return the preorder traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> PreorderTraversal(TreeNode root) {
    var res = new List<int>();
    var stack = new Stack<TreeNode>();
    if (root != null) stack.Push(root);
    while (stack.Count > 0) {
        var node = stack.Pop();
        res.Add(node.val);
        if (node.Right != null) stack.Push(node.Right);
        if (node.Left != null) stack.Push(node.Left);
    }
    return res;
}
```

**Explanation & Dry Run:**
- Use stack for iterative preorder traversal.
- Example 1: root = [1,null,2,3] → [1,2,3]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 80. Binary Tree Postorder Traversal

**Problem Statement:**
Return the postorder traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> PostorderTraversal(TreeNode root) {
    var res = new List<int>();
    var stack = new Stack<TreeNode>();
    TreeNode lastVisited = null;
    while (root != null || stack.Count > 0) {
        if (root != null) {
            stack.Push(root);
            root = root.Left;
        } else {
            var peek = stack.Peek();
            if (peek.Right != null && lastVisited != peek.Right) {
                root = peek.Right;
            } else {
                res.Add(peek.val);
                lastVisited = stack.Pop();
            }
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- Use stack for iterative postorder traversal.
- Example 1: root = [1,null,2,3] → [3,2,1]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 81. Serialize and Deserialize BST

**Problem Statement:**
Design an algorithm to serialize and deserialize a BST.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(TreeNode root) {
        var sb = new StringBuilder();
        void Preorder(TreeNode node) {
            if (node == null) return;
            sb.Append(node.val + ",");
            Preorder(node.Left);
            Preorder(node.Right);
        }
        Preorder(root);
        return sb.ToString();
    }
    public TreeNode Deserialize(string data) {
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToList();
        int i = 0;
        TreeNode Helper(int min, int max) {
            if (i == vals.Count || vals[i] < min || vals[i] > max) return null;
            var node = new TreeNode(vals[i++]);
            node.Left = Helper(min, node.val - 1);
            node.Right = Helper(node.val + 1, max);
            return node;
        }
        return Helper(int.MinValue, int.MaxValue);
    }
}
```

**Explanation & Dry Run:**
- Preorder traversal, use value bounds for BST.
- Example 1: root = [2,1,3] → "2,1,3,"
- Example 2: root = [] → ""

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 82. BST Iterator

**Problem Statement:**
Implement an iterator over a BST.

**Solution (C#):**
```csharp
public class BSTIterator {
    private Stack<TreeNode> stack = new Stack<TreeNode>();
    public BSTIterator(TreeNode root) {
        PushLeft(root);
    }
    private void PushLeft(TreeNode node) {
        while (node != null) {
            stack.Push(node);
            node = node.Left;
        }
    }
    public int Next() {
        var node = stack.Pop();
        PushLeft(node.Right);
        return node.val;
    }
    public bool HasNext() {
        return stack.Count > 0;
    }
}
```

**Explanation & Dry Run:**
- Use stack to simulate inorder traversal.
- Example 1: root = [7,3,15,null,null,9,20] → 3,7,9,15,20
- Example 2: root = [1] → 1

**Complexity:**
- Time: O(1) average per operation
- Space: O(h)

---

## 83. Palindrome Linked List

**Problem Statement:**
Check if a linked list is a palindrome.

**Solution (C#):**
```csharp
public bool IsPalindrome(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.Next != null) {
        slow = slow.Next;
        fast = fast.Next.Next;
    }
    ListNode prev = null;
    while (slow != null) {
        ListNode next = slow.Next;
        slow.Next = prev;
        prev = slow;
        slow = next;
    }
    ListNode left = head, right = prev;
    while (right != null) {
        if (left.val != right.val) return false;
        left = left.Next;
        right = right.Next;
    }
    return true;
}
```

**Explanation & Dry Run:**
- Find middle, reverse second half, compare.
- Example 1: head = [1,2,2,1] → true
- Example 2: head = [1,2] → false

**Complexity:**
- Time: O(n)
  - Each node is visited at most twice.
- Space: O(1)
  - Only pointers used.

---

## 84. Remove Linked List Elements

**Problem Statement:**
Remove all elements from a linked list of integers that have value val.

**Solution (C#):**
```csharp
public ListNode RemoveElements(ListNode head, int val) {
    ListNode dummy = new ListNode(0) { Next = head };
    ListNode curr = dummy;
    while (curr.Next != null) {
        if (curr.Next.val == val) curr.Next = curr.Next.Next;
        else curr = curr.Next;
    }
    return dummy.Next;
}
```

**Explanation & Dry Run:**
- Use dummy node, skip nodes with value val.
- Example 1: head = [1,2,6,3,4,5,6], val = 6 → [1,2,3,4,5]
- Example 2: head = [], val = 1 → []

**Complexity:**
- Time: O(n)
  - Each node is visited once.
- Space: O(1)
  - Only pointers used.

---

## 85. Odd Even Linked List

**Problem Statement:**
Given a singly linked list, group all odd nodes together followed by even nodes.

**Solution (C#):**
```csharp
public ListNode OddEvenList(ListNode head) {
    if (head == null) return null;
    ListNode odd = head, even = head.Next, evenHead = even;
    while (even != null && even.Next != null) {
        odd.Next = even.Next;
        odd = odd.Next;
        even.Next = odd.Next;
        even = even.Next;
    }
    odd.Next = evenHead;
    return head;
}
```

**Explanation & Dry Run:**
- Separate odd and even nodes, then join.
- Example 1: head = [1,2,3,4,5] → [1,3,5,2,4]
- Example 2: head = [2,1,3,5,6,4,7] → [2,3,6,7,1,5,4]

**Complexity:**
- Time: O(n)
  - Each node is visited once.
- Space: O(1)
  - Only pointers used.

---

## 86. Binary Tree Maximum Path Sum

**Problem Statement:**
Given a non-empty binary tree, find the maximum path sum.

**Solution (C#):**
```csharp
public int MaxPathSum(TreeNode root) {
    int maxSum = int.MinValue;
    int DFS(TreeNode node) {
        if (node == null) return 0;
        int left = Math.Max(DFS(node.Left), 0);
        int right = Math.Max(DFS(node.Right), 0);
        maxSum = Math.Max(maxSum, node.val + left + right);
        return node.val + Math.Max(left, right);
    }
    DFS(root);
    return maxSum;
}
```

**Explanation & Dry Run:**
- DFS, track max sum at each node.
- Example 1: root = [1,2,3] → 6
- Example 2: root = [-10,9,20,null,null,15,7] → 42

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 87. Binary Tree Level Order Traversal

**Problem Statement:**
Return the level order traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<IList<int>> LevelOrder(TreeNode root) {
    var res = new List<IList<int>>();
    if (root == null) return res;
    var queue = new Queue<TreeNode>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        int size = queue.Count;
        var level = new List<int>();
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            level.Add(node.val);
            if (node.Left != null) queue.Enqueue(node.Left);
            if (node.Right != null) queue.Enqueue(node.Right);
        }
        res.Add(level);
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, collect nodes at each level.
- Example 1: root = [3,9,20,null,null,15,7] → [[3],[9,20],[15,7]]
- Example 2: root = [1] → [[1]]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 88. Binary Tree Right Side View

**Problem Statement:**
Return the values of the nodes you can see from the right side.

**Solution (C#):**
```csharp
public IList<int> RightSideView(TreeNode root) {
    var res = new List<int>();
    if (root == null) return res;
    var queue = new Queue<TreeNode>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        int size = queue.Count;
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            if (i == size - 1) res.Add(node.val);
            if (node.Left != null) queue.Enqueue(node.Left);
            if (node.Right != null) queue.Enqueue(node.Right);
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, add last node at each level.
- Example 1: root = [1,2,3,null,5,null,4] → [1,3,4]
- Example 2: root = [1,null,3] → [1,3]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 89. Binary Tree Zigzag Level Order Traversal

**Problem Statement:**
Return the zigzag level order traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<IList<int>> ZigzagLevelOrder(TreeNode root) {
    var res = new List<IList<int>>();
    if (root == null) return res;
    var queue = new Queue<TreeNode>();
    queue.Enqueue(root);
    bool leftToRight = true;
    while (queue.Count > 0) {
        int size = queue.Count;
        var level = new LinkedList<int>();
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            if (leftToRight) level.AddLast(node.val);
            else level.AddFirst(node.val);
            if (node.Left != null) queue.Enqueue(node.Left);
            if (node.Right != null) queue.Enqueue(node.Right);
        }
        res.Add(level.ToList());
        leftToRight = !leftToRight;
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, alternate direction at each level.
- Example 1: root = [3,9,20,null,null,15,7] → [[3],[20,9],[15,7]]
- Example 2: root = [1] → [[1]]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 90. Binary Tree Inorder Traversal

**Problem Statement:**
Return the inorder traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> InorderTraversal(TreeNode root) {
    var res = new List<int>();
    var stack = new Stack<TreeNode>();
    while (root != null || stack.Count > 0) {
        while (root != null) {
            stack.Push(root);
            root = root.Left;
        }
        root = stack.Pop();
        res.Add(root.val);
        root = root.Right;
    }
    return res;
}
```

**Explanation & Dry Run:**
- Use stack for iterative inorder traversal.
- Example 1: root = [1,null,2,3] → [1,3,2]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 91. Binary Tree Preorder Traversal

**Problem Statement:**
Return the preorder traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> PreorderTraversal(TreeNode root) {
    var res = new List<int>();
    var stack = new Stack<TreeNode>();
    if (root != null) stack.Push(root);
    while (stack.Count > 0) {
        var node = stack.Pop();
        res.Add(node.val);
        if (node.Right != null) stack.Push(node.Right);
        if (node.Left != null) stack.Push(node.Left);
    }
    return res;
}
```

**Explanation & Dry Run:**
- Use stack for iterative preorder traversal.
- Example 1: root = [1,null,2,3] → [1,2,3]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 92. Binary Tree Postorder Traversal

**Problem Statement:**
Return the postorder traversal of a binary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> PostorderTraversal(TreeNode root) {
    var res = new List<int>();
    var stack = new Stack<TreeNode>();
    TreeNode lastVisited = null;
    while (root != null || stack.Count > 0) {
        if (root != null) {
            stack.Push(root);
            root = root.Left;
        } else {
            var peek = stack.Peek();
            if (peek.Right != null && lastVisited != peek.Right) {
                root = peek.Right;
            } else {
                res.Add(peek.val);
                lastVisited = stack.Pop();
            }
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- Use stack for iterative postorder traversal.
- Example 1: root = [1,null,2,3] → [3,2,1]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 93. Serialize and Deserialize BST

**Problem Statement:**
Design an algorithm to serialize and deserialize a BST.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(TreeNode root) {
        var sb = new StringBuilder();
        void Preorder(TreeNode node) {
            if (node == null) return;
            sb.Append(node.val + ",");
            Preorder(node.Left);
            Preorder(node.Right);
        }
        Preorder(root);
        return sb.ToString();
    }
    public TreeNode Deserialize(string data) {
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToList();
        int i = 0;
        TreeNode Helper(int min, int max) {
            if (i == vals.Count || vals[i] < min || vals[i] > max) return null;
            var node = new TreeNode(vals[i++]);
            node.Left = Helper(min, node.val - 1);
            node.Right = Helper(node.val + 1, max);
            return node;
        }
        return Helper(int.MinValue, int.MaxValue);
    }
}
```

**Explanation & Dry Run:**
- Preorder traversal, use value bounds for BST.
- Example 1: root = [2,1,3] → "2,1,3,"
- Example 2: root = [] → ""

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 94. BST Iterator

**Problem Statement:**
Implement an iterator over a BST.

**Solution (C#):**
```csharp
public class BSTIterator {
    private Stack<TreeNode> stack = new Stack<TreeNode>();
    public BSTIterator(TreeNode root) {
        PushLeft(root);
    }
    private void PushLeft(TreeNode node) {
        while (node != null) {
            stack.Push(node);
            node = node.Left;
        }
    }
    public int Next() {
        var node = stack.Pop();
        PushLeft(node.Right);
        return node.val;
    }
    public bool HasNext() {
        return stack.Count > 0;
    }
}
```

**Explanation & Dry Run:**
- Use stack to simulate inorder traversal.
- Example 1: root = [7,3,15,null,null,9,20] → 3,7,9,15,20
- Example 2: root = [1] → 1

**Complexity:**
- Time: O(1) average per operation
- Space: O(h)

---

## 95. Populating Next Right Pointers in Each Node

**Problem Statement:**
Given a perfect binary tree, populate each next pointer to its right node.

**Solution (C#):**
```csharp
public void Connect(TreeNode root) {
    if (root == null) return;
    var leftmost = root;
    while (leftmost.Left != null) {
        var head = leftmost;
        while (head != null) {
            head.Left.next = head.Right;
            if (head.next != null) head.Right.next = head.next.Left;
            head = head.next;
        }
        leftmost = leftmost.Left;
    }
}
```

**Explanation & Dry Run:**
- Use next pointers to traverse levels.
- Example 1: root = [1,2,3,4,5,6,7] → next pointers set
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 96. Populating Next Right Pointers in Each Node II

**Problem Statement:**
Given a binary tree, populate each next pointer to its right node.

**Solution (C#):**
```csharp
public void Connect(TreeNode root) {
    var curr = root;
    while (curr != null) {
        var dummy = new TreeNode(0);
        var tail = dummy;
        while (curr != null) {
            if (curr.Left != null) { tail.next = curr.Left; tail = tail.next; }
            if (curr.Right != null) { tail.next = curr.Right; tail = tail.next; }
            curr = curr.next;
        }
        curr = dummy.next;
    }
}
```

**Explanation & Dry Run:**
- Use dummy node to build next level.
- Example 1: root = [1,2,3,4,5,null,7] → next pointers set
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 97. Construct Binary Tree from Preorder and Inorder Traversal

**Problem Statement:**
Given preorder and inorder traversal of a tree, construct the binary tree.

**Solution (C#):**
```csharp
public TreeNode BuildTree(int[] preorder, int[] inorder) {
    var map = new Dictionary<int, int>();
    for (int i = 0; i < inorder.Length; i++) map[inorder[i]] = i;
    int preIdx = 0;
    TreeNode Helper(int inLeft, int inRight) {
        if (inLeft > inRight) return null;
        int rootVal = preorder[preIdx++];
        var root = new TreeNode(rootVal);
        int idx = map[rootVal];
        root.Left = Helper(inLeft, idx - 1);
        root.Right = Helper(idx + 1, inRight);
        return root;
    }
    return Helper(0, inorder.Length - 1);
}
```

**Explanation & Dry Run:**
- Recursively build tree using preorder and inorder indices.
- Example 1: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7] → tree built
- Example 2: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7] → tree built

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 98. Construct Binary Search Tree from Preorder Traversal

**Problem Statement:**
Given preorder traversal of a tree, construct the binary search tree.

**Solution (C#):**
```csharp
public TreeNode BstFromPreorder(int[] preorder) {
    int i = 0;
    TreeNode Helper(int min, int max) {
        if (i == preorder.Length || preorder[i] < min || preorder[i] > max) return null;
        var node = new TreeNode(preorder[i++]);
        node.Left = Helper(min, node.val - 1);
        node.Right = Helper(node.val + 1, max);
        return node;
    }
    return Helper(int.MinValue, int.MaxValue);
}
```

**Explanation & Dry Run:**
- Recursively build BST using value bounds.
- Example 1: preorder = [8,5,1,7,10,14] → tree built
- Example 2: preorder = [1] → tree built

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 99. Recover Binary Search Tree

**Problem Statement:**
Recover a BST where two nodes are swapped by mistake.

**Solution (C#):**
```csharp
public void RecoverTree(TreeNode root) {
    TreeNode first = null, second = null, prev = null;
    void Inorder(TreeNode node) {
        if (node == null) return;
        Inorder(node.Left);
        if (prev != null && node.val < prev.val) {
            if (first == null) first = prev;
            second = node;
        }
        prev = node;
        Inorder(node.Right);
    }
    Inorder(root);
    int temp = first.val;
    first.val = second.val;
    second.val = temp;
}
```

**Explanation & Dry Run:**
- Inorder traversal, find swapped nodes.
- Example 1: root = [1,3,null,null,2] → [3,1,null,null,2] (after recovery)
- Example 2: root = [3,1,4,null,null,2] → [2,1,4,null,null,3] (after recovery)

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 100. Maximum Width of Binary Tree

**Problem Statement:**
Find the maximum width of a binary tree.

**Solution (C#):**
```csharp
public int WidthOfBinaryTree(TreeNode root) {
    if (root == null) return 0;
    int maxWidth = 0;
    var queue = new Queue<(TreeNode node, int idx)>();
    queue.Enqueue((root, 0));
    while (queue.Count > 0) {
        int size = queue.Count;
        int minIdx = queue.Peek().idx, maxIdx = minIdx;
        for (int i = 0; i < size; i++) {
            var (node, idx) = queue.Dequeue();
            maxIdx = idx;
            if (node.Left != null) queue.Enqueue((node.Left, 2 * idx));
            if (node.Right != null) queue.Enqueue((node.Right, 2 * idx + 1));
        }
        maxWidth = Math.Max(maxWidth, maxIdx - minIdx + 1);
    }
    return maxWidth;
}
```

**Explanation & Dry Run:**
- BFS, track indices for width.
- Example 1: root = [1,3,2,5,3,null,9] → 4
- Example 2: root = [1,3,null,5,3] → 2

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 101. Serialize and Deserialize N-ary Tree

**Problem Statement:**
Design an algorithm to serialize and deserialize an N-ary tree.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(Node root) {
        if (root == null) return "";
        var sb = new StringBuilder();
        void Preorder(Node node) {
            sb.Append(node.val + ",");
            sb.Append(node.children.Count + ",");
            foreach (var child in node.children) Preorder(child);
        }
        Preorder(root);
        return sb.ToString();
    }
    public Node Deserialize(string data) {
        if (string.IsNullOrEmpty(data)) return null;
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries);
        int i = 0;
        Node DFS() {
            var node = new Node(int.Parse(vals[i++]), new List<Node>());
            int size = int.Parse(vals[i++]);
            for (int j = 0; j < size; j++) node.children.Add(DFS());
            return node;
        }
        return DFS();
    }
}
```

**Explanation & Dry Run:**
- Store value and number of children.
- Example 1: root = [1,null,3,2,4,null,5,6] → serialized string
- Example 2: root = [] → ""

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 102. N-ary Tree Level Order Traversal

**Problem Statement:**
Return the level order traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<IList<int>> LevelOrder(Node root) {
    var res = new List<IList<int>>();
    if (root == null) return res;
    var queue = new Queue<Node>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        int size = queue.Count;
        var level = new List<int>();
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            level.Add(node.val);
            foreach (var child in node.children) queue.Enqueue(child);
        }
        res.Add(level);
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, collect nodes at each level.
- Example 1: root = [1,null,3,2,4,null,5,6] → [[1],[3,2,4],[5,6]]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 103. N-ary Tree Preorder Traversal

**Problem Statement:**
Return the preorder traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> Preorder(Node root) {
    var res = new List<int>();
    void DFS(Node node) {
        if (node == null) return;
        res.Add(node.val);
        foreach (var child in node.children) DFS(child);
    }
    DFS(root);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, visit node then children.
- Example 1: root = [1,null,3,2,4,null,5,6] → [1,3,5,6,2,4]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 104. N-ary Tree Postorder Traversal

**Problem Statement:**
Return the postorder traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> Postorder(Node root) {
    var res = new List<int>();
    void DFS(Node node) {
        if (node == null) return;
        foreach (var child in node.children) DFS(child);
        res.Add(node.val);
    }
    DFS(root);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, visit children then node.
- Example 1: root = [1,null,3,2,4,null,5,6] → [5,6,3,2,4,1]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 105. Maximum Depth of N-ary Tree

**Problem Statement:**
Given an N-ary tree, find its maximum depth.

**Solution (C#):**
```csharp
public int MaxDepth(Node root) {
    if (root == null) return 0;
    int max = 0;
    foreach (var child in root.children) max = Math.Max(max, MaxDepth(child));
    return max + 1;
}
```

**Explanation & Dry Run:**
- Recursively find max depth.
- Example 1: root = [1,null,3,2,4,null,5,6] → 3
- Example 2: root = [] → 0

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 106. Lowest Common Ancestor of a Binary Tree

**Problem Statement:**
Given a binary tree, find the lowest common ancestor of two nodes.

**Solution (C#):**
```csharp
public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;
    var left = LowestCommonAncestor(root.Left, p, q);
    var right = LowestCommonAncestor(root.Right, p, q);
    if (left != null && right != null) return root;
    return left ?? right;
}
```

**Explanation & Dry Run:**
- Recursively search for p and q in left and right subtrees.
- Example 1: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1 → 3
- Example 2: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4 → 5

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 107. Lowest Common Ancestor of a BST

**Problem Statement:**
Given a binary search tree, find the lowest common ancestor of two nodes.

**Solution (C#):**
```csharp
public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    while (root != null) {
        if (p.val < root.val && q.val < root.val) root = root.Left;
        else if (p.val > root.val && q.val > root.val) root = root.Right;
        else return root;
    }
    return null;
}
```

**Explanation & Dry Run:**
- Traverse BST, split when p and q are on different sides.
- Example 1: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8 → 6
- Example 2: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4 → 2

**Complexity:**
- Time: O(h)
- Space: O(1)

---

## 108. Validate Binary Search Tree

**Problem Statement:**
Check if a binary tree is a valid BST.

**Solution (C#):**
```csharp
public bool IsValidBST(TreeNode root) {
    return Helper(root, long.MinValue, long.MaxValue);
}
private bool Helper(TreeNode node, long min, long max) {
    if (node == null) return true;
    if (node.val <= min || node.val >= max) return false;
    return Helper(node.Left, min, node.val) && Helper(node.Right, node.val, max);
}
```

**Explanation & Dry Run:**
- Recursively check value bounds.
- Example 1: root = [2,1,3] → true
- Example 2: root = [5,1,4,null,null,3,6] → false

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 109. Symmetric Tree

**Problem Statement:**
Check if a binary tree is symmetric.

**Solution (C#):**
```csharp
public bool IsSymmetric(TreeNode root) {
    return IsMirror(root, root);
}
private bool IsMirror(TreeNode t1, TreeNode t2) {
    if (t1 == null && t2 == null) return true;
    if (t1 == null || t2 == null) return false;
    return t1.val == t2.val && IsMirror(t1.Left, t2.Right) && IsMirror(t1.Right, t2.Left);
}
```

**Explanation & Dry Run:**
- Recursively check mirror property.
- Example 1: root = [1,2,2,3,4,4,3] → true
- Example 2: root = [1,2,2,null,3,null,3] → false

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 110. Binary Tree Paths

**Problem Statement:**
Return all root-to-leaf paths in a binary tree.

**Solution (C#):**
```csharp
public IList<string> BinaryTreePaths(TreeNode root) {
    var res = new List<string>();
    void DFS(TreeNode node, string path) {
        if (node == null) return;
        if (node.Left == null && node.Right == null) res.Add(path + node.val);
        else {
            DFS(node.Left, path + node.val + "->");
            DFS(node.Right, path + node.val + "->");
        }
    }
    DFS(root, "");
    return res;
}
```

**Explanation & Dry Run:**
- DFS, build path string.
- Example 1: root = [1,2,3,4,null,5,6] → ["1->2->4","1->3->6"]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 111. Sum Root to Leaf Numbers

**Problem Statement:**
Given a binary tree, return the sum of all root-to-leaf numbers.

**Solution (C#):**
```csharp
public int SumNumbers(TreeNode root) {
    int DFS(TreeNode node, int curr) {
        if (node == null) return 0;
        curr = curr * 10 + node.val;
        if (node.Left == null && node.Right == null) return curr;
        return DFS(node.Left, curr) + DFS(node.Right, curr);
    }
    return DFS(root, 0);
}
```

**Explanation & Dry Run:**
- DFS, accumulate number at each node.
- Example 1: root = [1,2,3] → 25
- Example 2: root = [4,9,0,5,1] → 1026

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 112. Count Complete Tree Nodes

**Problem Statement:**
Count the number of nodes in a complete binary tree.

**Solution (C#):**
```csharp
public int CountNodes(TreeNode root) {
    if (root == null) return 0;
    int left = LeftHeight(root), right = RightHeight(root);
    if (left == right) return (1 << left) - 1;
    return 1 + CountNodes(root.Left) + CountNodes(root.Right);
}
private int LeftHeight(TreeNode node) {
    int h = 0;
    while (node != null) { h++; node = node.Left; }
    return h;
}
private int RightHeight(TreeNode node) {
    int h = 0;
    while (node != null) { h++; node = node.Right; }
    return h;
}
```

**Explanation & Dry Run:**
- Use tree height to optimize counting.
- Example 1: root = [1,2,3,4,5,6] → 6
- Example 2: root = [] → 0

**Complexity:**
- Time: O((log n)^2)
- Space: O(log n)

---

## 113. Invert Binary Tree

**Problem Statement:**
Invert a binary tree.

**Solution (C#):**
```csharp
public TreeNode InvertTree(TreeNode root) {
    if (root == null) return null;
    var left = InvertTree(root.Left);
    var right = InvertTree(root.Right);
    root.Left = right;
    root.Right = left;
    return root;
}
```

**Explanation & Dry Run:**
- Recursively swap left and right children.
- Example 1: root = [4,2,7,1,3,6,9] → [4,7,2,9,6,3,1]
- Example 2: root = [2,1,3] → [2,3,1]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 114. Maximum Width of Binary Tree

**Problem Statement:**
Find the maximum width of a binary tree.

**Solution (C#):**
```csharp
public int WidthOfBinaryTree(TreeNode root) {
    if (root == null) return 0;
    int maxWidth = 0;
    var queue = new Queue<(TreeNode node, int idx)>();
    queue.Enqueue((root, 0));
    while (queue.Count > 0) {
        int size = queue.Count;
        int minIdx = queue.Peek().idx, maxIdx = minIdx;
        for (int i = 0; i < size; i++) {
            var (node, idx) = queue.Dequeue();
            maxIdx = idx;
            if (node.Left != null) queue.Enqueue((node.Left, 2 * idx));
            if (node.Right != null) queue.Enqueue((node.Right, 2 * idx + 1));
        }
        maxWidth = Math.Max(maxWidth, maxIdx - minIdx + 1);
    }
    return maxWidth;
}
```

**Explanation & Dry Run:**
- BFS, track indices for width.
- Example 1: root = [1,3,2,5,3,null,9] → 4
- Example 2: root = [1,3,null,5,3] → 2

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 115. Construct Binary Tree from String

**Problem Statement:**
Construct a binary tree from a string with parentheses.

**Solution (C#):**
```csharp
public TreeNode Str2Tree(string s) {
    int i = 0;
    TreeNode Helper() {
        if (i >= s.Length) return null;
        int start = i;
        while (i < s.Length && (char.IsDigit(s[i]) || s[i] == '-')) i++;
        var node = new TreeNode(int.Parse(s.Substring(start, i - start)));
        if (i < s.Length && s[i] == '(') {
            i++; node.Left = Helper(); i++;
        }
        if (i < s.Length && s[i] == '(') {
            i++; node.Right = Helper(); i++;
        }
        return node;
    }
    return s == "" ? null : Helper();
}
```

**Explanation & Dry Run:**
- Parse string, recursively build tree.
- Example 1: s = "4(2(3)(1))(6(5))" → tree built
- Example 2: s = "" → null

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 116. Serialize and Deserialize N-ary Tree

**Problem Statement:**
Design an algorithm to serialize and deserialize an N-ary tree.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(Node root) {
        if (root == null) return "";
        var sb = new StringBuilder();
        void Preorder(Node node) {
            sb.Append(node.val + ",");
            sb.Append(node.children.Count + ",");
            foreach (var child in node.children) Preorder(child);
        }
        Preorder(root);
        return sb.ToString();
    }
    public Node Deserialize(string data) {
        if (string.IsNullOrEmpty(data)) return null;
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries);
        int i = 0;
        Node DFS() {
            var node = new Node(int.Parse(vals[i++]), new List<Node>());
            int size = int.Parse(vals[i++]);
            for (int j = 0; j < size; j++) node.children.Add(DFS());
            return node;
        }
        return DFS();
    }
}
```

**Explanation & Dry Run:**
- Store value and number of children.
- Example 1: root = [1,null,3,2,4,null,5,6] → serialized string
- Example 2: root = [] → ""

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 117. N-ary Tree Level Order Traversal

**Problem Statement:**
Return the level order traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<IList<int>> LevelOrder(Node root) {
    var res = new List<IList<int>>();
    if (root == null) return res;
    var queue = new Queue<Node>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        int size = queue.Count;
        var level = new List<int>();
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            level.Add(node.val);
            foreach (var child in node.children) queue.Enqueue(child);
        }
        res.Add(level);
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, collect nodes at each level.
- Example 1: root = [1,null,3,2,4,null,5,6] → [[1],[3,2,4],[5,6]]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 118. N-ary Tree Preorder Traversal

**Problem Statement:**
Return the preorder traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> Preorder(Node root) {
    var res = new List<int>();
    void DFS(Node node) {
        if (node == null) return;
        res.Add(node.val);
        foreach (var child in node.children) DFS(child);
    }
    DFS(root);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, visit node then children.
- Example 1: root = [1,null,3,2,4,null,5,6] → [1,3,5,6,2,4]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 119. N-ary Tree Postorder Traversal

**Problem Statement:**
Return the postorder traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> Postorder(Node root) {
    var res = new List<int>();
    void DFS(Node node) {
        if (node == null) return;
        foreach (var child in node.children) DFS(child);
        res.Add(node.val);
    }
    DFS(root);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, visit children then node.
- Example 1: root = [1,null,3,2,4,null,5,6] → [5,6,3,2,4,1]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 120. Maximum Depth of N-ary Tree

**Problem Statement:**
Given an N-ary tree, find its maximum depth.

**Solution (C#):**
```csharp
public int MaxDepth(Node root) {
    if (root == null) return 0;
    int max = 0;
    foreach (var child in root.children) max = Math.Max(max, MaxDepth(child));
    return max + 1;
}
```

**Explanation & Dry Run:**
- Recursively find max depth.
- Example 1: root = [1,null,3,2,4,null,5,6] → 3
- Example 2: root = [] → 0

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 121. Range Sum of BST

**Problem Statement:**
Given a BST and a range [low, high], return the sum of values in that range.

**Solution (C#):**
```csharp
public int RangeSumBST(TreeNode root, int low, int high) {
    if (root == null) return 0;
    if (root.val < low) return RangeSumBST(root.Right, low, high);
    if (root.val > high) return RangeSumBST(root.Left, low, high);
    return root.val + RangeSumBST(root.Left, low, high) + RangeSumBST(root.Right, low, high);
}
```

**Explanation & Dry Run:**
- Recursively sum values in range.
- Example 1: root = [10,5,15,3,7,null,18], low = 7, high = 15 → 32
- Example 2: root = [7,3,15,1,5,9,20], low = 6, high = 10 → 16

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 122. Validate Binary Tree Nodes

**Problem Statement:**
Check if n nodes form a valid binary tree.

**Solution (C#):**
```csharp
public bool ValidateBinaryTreeNodes(int n, int[] leftChild, int[] rightChild) {
    var parent = new int[n];
    Array.Fill(parent, -1);
    for (int i = 0; i < n; i++) {
        if (leftChild[i] != -1) {
            if (parent[leftChild[i]] != -1) return false;
            parent[leftChild[i]] = i;
        }
        if (rightChild[i] != -1) {
            if (parent[rightChild[i]] != -1) return false;
            parent[rightChild[i]] = i;
        }
    }
    int root = -1;
    for (int i = 0; i < n; i++) {
        if (parent[i] == -1) {
            if (root == -1) root = i;
            else return false;
        }
    }
    if (root == -1) return false;
    var visited = new bool[n];
    bool DFS(int node) {
        if (node == -1) return true;
        if (visited[node]) return false;
        visited[node] = true;
        return DFS(leftChild[node]) && DFS(rightChild[node]);
    }
    return DFS(root) && visited.All(v => v);
}
```

**Explanation & Dry Run:**
- Check parent and root, DFS for cycles.
- Example 1: n = 4, leftChild = [1,-1,3,-1], rightChild = [2,-1,-1,-1] → true
- Example 2: n = 2, leftChild = [1,0], rightChild = [-1,-1] → false

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 123. Maximum Binary Tree

**Problem Statement:**
Construct the maximum binary tree from an array.

**Solution (C#):**
```csharp
public TreeNode ConstructMaximumBinaryTree(int[] nums) {
    return Helper(nums, 0, nums.Length - 1);
}
private TreeNode Helper(int[] nums, int left, int right) {
    if (left > right) return null;
    int maxIdx = left;
    for (int i = left; i <= right; i++) {
        if (nums[i] > nums[maxIdx]) maxIdx = i;
    }
    var root = new TreeNode(nums[maxIdx]);
    root.Left = Helper(nums, left, maxIdx - 1);
    root.Right = Helper(nums, maxIdx + 1, right);
    return root;
}
```

**Explanation & Dry Run:**
- Recursively build tree from max element.
- Example 1: nums = [3,2,1,6,0,5] → tree built
- Example 2: nums = [3,2,1,5,6,4] → tree built

**Complexity:**
- Time: O(n^2)
- Space: O(n)

---

## 124. Construct Binary Search Tree from Preorder Traversal

**Problem Statement:**
Construct a BST from preorder traversal.

**Solution (C#):**
```csharp
public TreeNode BstFromPreorder(int[] preorder) {
    int i = 0;
    TreeNode Helper(int min, int max) {
        if (i == preorder.Length || preorder[i] < min || preorder[i] > max) return null;
        var node = new TreeNode(preorder[i++]);
        node.Left = Helper(min, node.val - 1);
        node.Right = Helper(node.val + 1, max);
        return node;
    }
    return Helper(int.MinValue, int.MaxValue);
}
```

**Explanation & Dry Run:**
- Recursively build BST using value bounds.
- Example 1: preorder = [8,5,1,7,10,14] → tree built
- Example 2: preorder = [1] → tree built

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 125. Recover Binary Search Tree

**Problem Statement:**
Recover a BST where two nodes are swapped by mistake.

**Solution (C#):**
```csharp
public void RecoverTree(TreeNode root) {
    TreeNode first = null, second = null, prev = null;
    void Inorder(TreeNode node) {
        if (node == null) return;
        Inorder(node.Left);
        if (prev != null && node.val < prev.val) {
            if (first == null) first = prev;
            second = node;
        }
        prev = node;
        Inorder(node.Right);
    }
    Inorder(root);
    int temp = first.val;
    first.val = second.val;
    second.val = temp;
}
```

**Explanation & Dry Run:**
- Inorder traversal, find swapped nodes.
- Example 1: root = [1,3,null,null,2] → [3,1,null,null,2] (after recovery)
- Example 2: root = [3,1,4,null,null,2] → [2,1,4,null,null,3] (after recovery)

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 126. Increasing Order Search Tree

**Problem Statement:**
Given a BST, rearrange the tree in increasing order so that the leftmost node is the root and every node has no left child.

**Solution (C#):**
```csharp
public TreeNode IncreasingBST(TreeNode root) {
    TreeNode dummy = new TreeNode(0), curr = dummy;
    void Inorder(TreeNode node) {
        if (node == null) return;
        Inorder(node.Left);
        curr.Right = new TreeNode(node.val);
        curr = curr.Right;
        Inorder(node.Right);
    }
    Inorder(root);
    return dummy.Right;
}
```

**Explanation & Dry Run:**
- Inorder traversal, build new tree.
- Example 1: root = [1,2,5,3,4,null,6] → [1,2,3,4,5,6]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 127. Validate Binary Tree Preorder Serialization

**Problem Statement:**
Check if a given preorder serialization of a binary tree is valid.

**Solution (C#):**
```csharp
public bool IsValidSerialization(string preorder) {
    int slots = 1;
    foreach (var node in preorder.Split(',')) {
        if (--slots < 0) return false;
        if (node != "#") slots += 2;
    }
    return slots == 0;
}
```

**Explanation & Dry Run:**
- Track available slots for nodes.
- Example 1: preorder = "9,3,4,#,#,1,#,#,2,#,6,#,#" → true
- Example 2: preorder = "1,#" → false

**Complexity:**
- Time: O(n)
- Space: O(1)

---

## 128. Serialize and Deserialize N-ary Tree

**Problem Statement:**
Design an algorithm to serialize and deserialize an N-ary tree.

**Solution (C#):**
```csharp
public class Codec {
    public string Serialize(Node root) {
        if (root == null) return "";
        var sb = new StringBuilder();
        void Preorder(Node node) {
            sb.Append(node.val + ",");
            sb.Append(node.children.Count + ",");
            foreach (var child in node.children) Preorder(child);
        }
        Preorder(root);
        return sb.ToString();
    }
    public Node Deserialize(string data) {
        if (string.IsNullOrEmpty(data)) return null;
        var vals = data.Split(',', StringSplitOptions.RemoveEmptyEntries);
        int i = 0;
        Node DFS() {
            var node = new Node(int.Parse(vals[i++]), new List<Node>());
            int size = int.Parse(vals[i++]);
            for (int j = 0; j < size; j++) node.children.Add(DFS());
            return node;
        }
        return DFS();
    }
}
```

**Explanation & Dry Run:**
- Store value and number of children.
- Example 1: root = [1,null,3,2,4,null,5,6] → serialized string
- Example 2: root = [] → ""

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 129. N-ary Tree Level Order Traversal

**Problem Statement:**
Return the level order traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<IList<int>> LevelOrder(Node root) {
    var res = new List<IList<int>>();
    if (root == null) return res;
    var queue = new Queue<Node>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        int size = queue.Count;
        var level = new List<int>();
        for (int i = 0; i < size; i++) {
            var node = queue.Dequeue();
            level.Add(node.val);
            foreach (var child in node.children) queue.Enqueue(child);
        }
        res.Add(level);
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, collect nodes at each level.
- Example 1: root = [1,null,3,2,4,null,5,6] → [[1],[3,2,4],[5,6]]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 130. N-ary Tree Preorder Traversal

**Problem Statement:**
Return the preorder traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> Preorder(Node root) {
    var res = new List<int>();
    void DFS(Node node) {
        if (node == null) return;
        res.Add(node.val);
        foreach (var child in node.children) DFS(child);
    }
    DFS(root);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, visit node then children.
- Example 1: root = [1,null,3,2,4,null,5,6] → [1,3,5,6,2,4]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 131. N-ary Tree Postorder Traversal

**Problem Statement:**
Return the postorder traversal of an N-ary tree's nodes' values.

**Solution (C#):**
```csharp
public IList<int> Postorder(Node root) {
    var res = new List<int>();
    void DFS(Node node) {
        if (node == null) return;
        foreach (var child in node.children) DFS(child);
        res.Add(node.val);
    }
    DFS(root);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, visit children then node.
- Example 1: root = [1,null,3,2,4,null,5,6] → [5,6,3,2,4,1]
- Example 2: root = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 132. Maximum Depth of N-ary Tree

**Problem Statement:**
Given an N-ary tree, find its maximum depth.

**Solution (C#):**
```csharp
public int MaxDepth(Node root) {
    if (root == null) return 0;
    int max = 0;
    foreach (var child in root.children) max = Math.Max(max, MaxDepth(child));
    return max + 1;
}
```

**Explanation & Dry Run:**
- Recursively find max depth.
- Example 1: root = [1,null,3,2,4,null,5,6] → 3
- Example 2: root = [] → 0

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 133. Employee Importance

**Problem Statement:**
Given a list of employees, find the total importance value of an employee and their subordinates.

**Solution (C#):**
```csharp
public int GetImportance(IList<Employee> employees, int id) {
    var map = employees.ToDictionary(e => e.id);
    int DFS(int eid) {
        var emp = map[eid];
        int total = emp.importance;
        foreach (var sub in emp.subordinates) total += DFS(sub);
        return total;
    }
    return DFS(id);
}
```

**Explanation & Dry Run:**
- DFS, sum importance recursively.
- Example 1: employees = [{1,5,[2,3]},{2,3,[]},{3,3,[]}], id = 1 → 11
- Example 2: employees = [{1,2,[5]},{5,3,[]}], id = 5 → 3

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 134. Clone N-ary Tree

**Problem Statement:**
Clone an N-ary tree.

**Solution (C#):**
```csharp
public Node CloneTree(Node root) {
    if (root == null) return null;
    var node = new Node(root.val, new List<Node>());
    foreach (var child in root.children) node.children.Add(CloneTree(child));
    return node;
}
```

**Explanation & Dry Run:**
- Recursively clone each node and its children.
- Example 1: root = [1,null,3,2,4,null,5,6] → cloned tree
- Example 2: root = [] → null

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 135. Flatten N-ary Tree to Linked List

**Problem Statement:**
Flatten an N-ary tree to a linked list in preorder.

**Solution (C#):**
```csharp
public Node Flatten(Node root) {
    if (root == null) return null;
    var dummy = new Node(0, new List<Node>());
    var curr = dummy;
    void Preorder(Node node) {
        if (node == null) return;
        curr.next = new Node(node.val, new List<Node>());
        curr = curr.next;
        foreach (var child in node.children) Preorder(child);
    }
    Preorder(root);
    return dummy.next;
}
```

**Explanation & Dry Run:**
- Preorder traversal, build linked list.
- Example 1: root = [1,null,3,2,4,null,5,6] → [1,3,5,6,2,4]
- Example 2: root = [] → null

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 136. Word Search

**Problem Statement:**
Given a 2D board and a word, find if the word exists in the grid.

**Solution (C#):**
```csharp
public bool Exist(char[][] board, string word) {
    int m = board.Length, n = board[0].Length;
    bool DFS(int i, int j, int k) {
        if (k == word.Length) return true;
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[k]) return false;
        char temp = board[i][j];
        board[i][j] = '#';
        bool found = DFS(i+1, j, k+1) || DFS(i-1, j, k+1) || DFS(i, j+1, k+1) || DFS(i, j-1, k+1);
        board[i][j] = temp;
        return found;
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (DFS(i, j, 0)) return true;
    return false;
}
```

**Explanation & Dry Run:**
- DFS, mark visited cells.
- Example 1: board = [[A,B,C,E],[S,F,C,S],[A,D,E,E]], word = "ABCCED" → true
- Example 2: board = [[A,B,C,E],[S,F,C,S],[A,D,E,E]], word = "SEE" → true

**Complexity:**
- Time: O(mn * 4^L)
- Space: O(L)

---

## 137. Number of Islands

**Problem Statement:**
Given a 2D grid, count the number of islands.

**Solution (C#):**
```csharp
public int NumIslands(char[][] grid) {
    int m = grid.Length, n = grid[0].Length, count = 0;
    void DFS(int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') return;
        grid[i][j] = '0';
        DFS(i+1, j); DFS(i-1, j); DFS(i, j+1); DFS(i, j-1);
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == '1') { count++; DFS(i, j); }
    return count;
}
```

**Explanation & Dry Run:**
- DFS, mark visited land.
- Example 1: grid = [[1,1,1,1,0],[1,1,0,1,0],[1,1,0,0,0],[0,0,0,0,0]] → 1
- Example 2: grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]] → 3

**Complexity:**
- Time: O(mn)
- Space: O(mn)

---

## 138. Surrounded Regions

**Problem Statement:**
Capture all regions surrounded by 'X' in a 2D board.

**Solution (C#):**
```csharp
public void Solve(char[][] board) {
    int m = board.Length, n = board[0].Length;
    void DFS(int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O') return;
        board[i][j] = 'E';
        DFS(i+1, j); DFS(i-1, j); DFS(i, j+1); DFS(i, j-1);
    }
    for (int i = 0; i < m; i++) {
        DFS(i, 0); DFS(i, n-1);
    }
    for (int j = 0; j < n; j++) {
        DFS(0, j); DFS(m-1, j);
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (board[i][j] == 'O') board[i][j] = 'X';
            else if (board[i][j] == 'E') board[i][j] = 'O';
}
```

**Explanation & Dry Run:**
- DFS from border, mark safe regions.
- Example 1: board = [[X,X,X,X],[X,O,O,X],[X,X,O,X],[X,O,X,X]] → [[X,X,X,X],[X,X,X,X],[X,X,X,X],[X,O,X,X]]
- Example 2: board = [[X]] → [[X]]

**Complexity:**
- Time: O(mn)
- Space: O(mn)

---

## 139. Walls and Gates

**Problem Statement:**
Fill each empty room with distance to its nearest gate.

**Solution (C#):**
```csharp
public void WallsAndGates(int[][] rooms) {
    int m = rooms.Length, n = rooms[0].Length;
    var queue = new Queue<(int, int)>();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (rooms[i][j] == 0) queue.Enqueue((i, j));
    int[][] dirs = { new[] {1,0}, new[] {-1,0}, new[] {0,1}, new[] {0,-1} };
    while (queue.Count > 0) {
        var (i, j) = queue.Dequeue();
        foreach (var d in dirs) {
            int ni = i + d[0], nj = j + d[1];
            if (ni < 0 || ni >= m || nj < 0 || nj >= n || rooms[ni][nj] != int.MaxValue) continue;
            rooms[ni][nj] = rooms[i][j] + 1;
            queue.Enqueue((ni, nj));
        }
    }
}
```

**Explanation & Dry Run:**
- BFS from all gates.
- Example 1: rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]] → filled distances
- Example 2: rooms = [[X]] → [[X]]

**Complexity:**
- Time: O(mn)
- Space: O(mn)

---

## 140. Number of Connected Components in an Undirected Graph

**Problem Statement:**
Given n nodes and edges, count the number of connected components.

**Solution (C#):**
```csharp
public int CountComponents(int n, int[][] edges) {
    var parent = Enumerable.Range(0, n).ToArray();
    int Find(int x) { return parent[x] == x ? x : parent[x] = Find(parent[x]); }
    foreach (var e in edges) {
        parent[Find(e[0])] = Find(e[1]);
    }
    return parent.Select(Find).Distinct().Count();
}
```

**Explanation & Dry Run:**
- Union-find to group components.
- Example 1: n = 5, edges = [[0,1],[1,2],[3,4]] → 2
- Example 2: n = 1, edges = [] → 1

**Complexity:**
- Time: O(n + e)
- Space: O(n)

---

## 141. Graph Valid Tree

**Problem Statement:**
Given n nodes and edges, check if the graph is a valid tree.

**Solution (C#):**
```csharp
public bool ValidTree(int n, int[][] edges) {
    if (edges.Length != n - 1) return false;
    var parent = Enumerable.Range(0, n).ToArray();
    int Find(int x) { return parent[x] == x ? x : parent[x] = Find(parent[x]); }
    foreach (var e in edges) {
        int px = Find(e[0]), py = Find(e[1]);
        if (px == py) return false;
        parent[px] = py;
    }
    return true;
}
```

**Explanation & Dry Run:**
- Union-find, check for cycles and connectivity.
- Example 1: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]] → true
- Example 2: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]] → false

**Complexity:**
- Time: O(n + e)
- Space: O(n)

---

## 142. Course Schedule II

**Problem Statement:**
Return the order to take courses given prerequisites.

**Solution (C#):**
```csharp
public int[] FindOrder(int numCourses, int[][] prerequisites) {
    var graph = new List<int>[numCourses];
    for (int i = 0; i < numCourses; i++) graph[i] = new List<int>();
    foreach (var p in prerequisites) graph[p[1]].Add(p[0]);
    var visited = new int[numCourses];
    var res = new List<int>();
    bool DFS(int node) {
        if (visited[node] == 1) return false;
        if (visited[node] == 2) return true;
        visited[node] = 1;
        foreach (var nei in graph[node]) if (!DFS(nei)) return false;
        visited[node] = 2;
        res.Add(node);
        return true;
    }
    for (int i = 0; i < numCourses; i++) if (!DFS(i)) return new int[0];
    return res.ToArray();
}
```

**Explanation & Dry Run:**
- DFS for topological sort.
- Example 1: numCourses = 2, prerequisites = [[1,0]] → [0,1]
- Example 2: numCourses = 2, prerequisites = [[1,0],[0,1]] → []

**Complexity:**
- Time: O(V+E)
- Space: O(V+E)

---

## 143. Alien Dictionary

**Problem Statement:**
Given a sorted dictionary of alien language, find the order of characters.

**Solution (C#):**
```csharp
public string AlienOrder(string[] words) {
    var graph = new Dictionary<char, HashSet<char>>();
    var indegree = new Dictionary<char, int>();
    foreach (var word in words)
        foreach (var c in word) {
            if (!graph.ContainsKey(c)) graph[c] = new HashSet<char>();
            if (!indegree.ContainsKey(c)) indegree[c] = 0;
        }
    for (int i = 0; i < words.Length - 1; i++) {
        string w1 = words[i], w2 = words[i+1];
        int minLen = Math.Min(w1.Length, w2.Length);
        for (int j = 0; j < minLen; j++) {
            if (w1[j] != w2[j]) {
                if (!graph[w1[j]].Contains(w2[j])) {
                    graph[w1[j]].Add(w2[j]);
                    indegree[w2[j]]++;
                }
                break;
            }
        }
    }
    var queue = new Queue<char>(indegree.Where(x => x.Value == 0).Select(x => x.Key));
    var res = new StringBuilder();
    while (queue.Count > 0) {
        var c = queue.Dequeue();
        res.Append(c);
        foreach (var nei in graph[c]) {
            indegree[nei]--;
            if (indegree[nei] == 0) queue.Enqueue(nei);
        }
    }
    return res.Length == indegree.Count ? res.ToString() : "";
}
```

**Explanation & Dry Run:**
- Build graph, topological sort.
- Example 1: words = ["wrt","wrf","er","ett","rftt"] → "wertf"
- Example 2: words = ["z","x"] → "zx"

**Complexity:**
- Time: O(N + K)
- Space: O(N + K)

---

## 144. Graph Clone

**Problem Statement:**
Clone an undirected graph.

**Solution (C#):**
```csharp
public Node CloneGraph(Node node) {
    if (node == null) return null;
    var map = new Dictionary<Node, Node>();
    Node DFS(Node n) {
        if (map.ContainsKey(n)) return map[n];
        var clone = new Node(n.val, new List<Node>());
        map[n] = clone;
        foreach (var nei in n.neighbors) clone.neighbors.Add(DFS(nei));
        return clone;
    }
    return DFS(node);
}
```

**Explanation & Dry Run:**
- DFS, clone nodes and neighbors.
- Example 1: node = [1,[2,4]] → cloned graph
- Example 2: node = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 145. Graph BFS

**Problem Statement:**
Perform BFS traversal on an undirected graph.

**Solution (C#):**
```csharp
public IList<int> BFS(Node node) {
    var res = new List<int>();
    if (node == null) return res;
    var queue = new Queue<Node>();
    var visited = new HashSet<Node>();
    queue.Enqueue(node);
    visited.Add(node);
    while (queue.Count > 0) {
        var curr = queue.Dequeue();
        res.Add(curr.val);
        foreach (var nei in curr.neighbors) {
            if (!visited.Contains(nei)) {
                queue.Enqueue(nei);
                visited.Add(nei);
            }
        }
    }
    return res;
}
```

**Explanation & Dry Run:**
- BFS, visit each node once.
- Example 1: node = [1,[2,4]] → [1,2,4]
- Example 2: node = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 146. Graph DFS

**Problem Statement:**
Perform DFS traversal on an undirected graph.

**Solution (C#):**
```csharp
public IList<int> DFS(Node node) {
    var res = new List<int>();
    var visited = new HashSet<Node>();
    void Helper(Node n) {
        if (n == null || visited.Contains(n)) return;
        visited.Add(n);
        res.Add(n.val);
        foreach (var nei in n.neighbors) Helper(nei);
    }
    Helper(node);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, visit each node once.
- Example 1: node = [1,[2,4]] → [1,2,4]
- Example 2: node = [] → []

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 147. Pacific Atlantic Water Flow

**Problem Statement:**
Given a matrix, find cells that can flow to both Pacific and Atlantic oceans.

**Solution (C#):**
```csharp
public IList<IList<int>> PacificAtlantic(int[][] heights) {
    int m = heights.Length, n = heights[0].Length;
    var pacific = new bool[m, n];
    var atlantic = new bool[m, n];
    void DFS(int i, int j, bool[,] ocean) {
        ocean[i, j] = true;
        int[][] dirs = { new[] {1,0}, new[] {-1,0}, new[] {0,1}, new[] {0,-1} };
        foreach (var d in dirs) {
            int ni = i + d[0], nj = j + d[1];
            if (ni < 0 || ni >= m || nj < 0 || nj >= n || ocean[ni, nj] || heights[ni][nj] < heights[i][j]) continue;
            DFS(ni, nj, ocean);
        }
    }
    for (int i = 0; i < m; i++) { DFS(i, 0, pacific); DFS(i, n-1, atlantic); }
    for (int j = 0; j < n; j++) { DFS(0, j, pacific); DFS(m-1, j, atlantic); }
    var res = new List<IList<int>>();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (pacific[i, j] && atlantic[i, j]) res.Add(new List<int> { i, j });
    return res;
}
```

**Explanation & Dry Run:**
- DFS from borders, mark reachable cells.
- Example 1: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]] → cells
- Example 2: heights = [[1]] → [[0,0]]

**Complexity:**
- Time: O(mn)
- Space: O(mn)

---

## 148. Redundant Connection

**Problem Statement:**
Given a tree with one extra edge, find the edge that can be removed to make it a tree.

**Solution (C#):**
```csharp
public int[] FindRedundantConnection(int[][] edges) {
    int n = edges.Length;
    var parent = Enumerable.Range(0, n).ToArray();
    int Find(int x) { return parent[x] == x ? x : parent[x] = Find(parent[x]); }
    foreach (var e in edges) {
        int px = Find(e[0]), py = Find(e[1]);
        if (px == py) return e;
        parent[px] = py;
    }
    return new int[0];
}
```

**Explanation & Dry Run:**
- Union-find, detect cycle.
- Example 1: edges = [[1,2],[1,3],[2,3]] → [2,3]
- Example 2: edges = [[1,2],[2,3],[3,4],[1,4],[1,5]] → [1,4]

**Complexity:**
- Time: O(n)
- Space: O(n)

---

## 149. Critical Connections in a Network

**Problem Statement:**
Find all critical connections (bridges) in a network.

**Solution (C#):**
```csharp
public IList<IList<int>> CriticalConnections(int n, IList<IList<int>> connections) {
    var graph = new List<int>[n];
    for (int i = 0; i < n; i++) graph[i] = new List<int>();
    foreach (var c in connections) {
        graph[c[0]].Add(c[1]);
        graph[c[1]].Add(c[0]);
    }
    var res = new List<IList<int>>();
    var disc = new int[n];
    Array.Fill(disc, -1);
    int time = 0;
    void DFS(int u, int parent) {
        disc[u] = time++;
        int low = disc[u];
        foreach (var v in graph[u]) {
            if (v == parent) continue;
            if (disc[v] == -1) {
                int childLow = DFS(v, u);
                low = Math.Min(low, childLow);
                if (childLow > disc[u]) res.Add(new List<int> { u, v });
            } else low = Math.Min(low, disc[v]);
        }
        return low;
    }
    DFS(0, -1);
    return res;
}
```

**Explanation & Dry Run:**
- DFS, track discovery times and low links.
- Example 1: n = 4, connections = [[0,1],[1,2],[2,0],[1,3],[1,4]] → [[1,3]]
- Example 2: n = 2, connections = [[0,1]] → [[0,1]]

**Complexity:**
- Time: O(n+e)
- Space: O(n+e)

---

## 150. Course Schedule III

**Problem Statement:**
Given n courses with durations and deadlines, find the maximum number of courses you can take.

**Solution (C#):**
```csharp
public int ScheduleCourse(int[][] courses) {
    Array.Sort(courses, (a, b) => a[1] - b[1]);
    var pq = new SortedSet<int>();
    int time = 0;
    foreach (var c in courses) {
        time += c[0];
        pq.Add(c[0]);
        if (time > c[1]) {
            time -= pq.Max;
            pq.Remove(pq.Max);
        }
    }
    return pq.Count;
}
```

**Explanation & Dry Run:**
- Sort by deadline, use max heap for durations.
- Example 1: courses = [[100,200],[200,1300],[1000,1250],[2000,3200]] → 3
- Example 2: courses = [[1,2]] → 1

**Complexity:**
- Time: O(n log n)
- Space: O(n)

---
