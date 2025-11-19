# DSA Solutions - All Topics

## Arrays
[This section contains solutions for array-based DSA problems.]

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

## Strings
[This section contains solutions for string-based DSA problems.]

# Strings DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common string-based DSA interview questions.

---

## 1. Longest Substring Without Repeating Characters
**Problem:** Find the length of the longest substring without repeating characters.

```csharp
public int LengthOfLongestSubstring(string s)
{
    var set = new HashSet<char>();
    int left = 0, maxLen = 0;
    for (int right = 0; right < s.Length; right++)
    {
        while (set.Contains(s[right]))
        {
            set.Remove(s[left++]);
        }
        set.Add(s[right]);
        maxLen = Math.Max(maxLen, right - left + 1);
    }
    return maxLen;
}
```
**Explanation:** Sliding window with a set to track unique characters.
**Time Complexity:** O(n)
**Space Complexity:** O(min(n, m))

**Dry Run Example:**
Input: s = "abcabcbb"
Step-by-step:
- right=0: set={'a'}, maxLen=1
- right=1: set={'a','b'}, maxLen=2
- right=2: set={'a','b','c'}, maxLen=3
- right=3: 'a' repeats, remove 'a', set={'b','c'}, add 'a', maxLen=3
... (continue)
Output: 3

---

## 2. Valid Anagram
**Problem:** Check if two strings are anagrams.

```csharp
public bool IsAnagram(string s, string t)
{
    if (s.Length != t.Length) return false;
    var count = new int[26];
    foreach (var c in s) count[c - 'a']++;
    foreach (var c in t) count[c - 'a']--;
    foreach (var n in count) if (n != 0) return false;
    return true;
}
```
**Explanation:** Count character frequencies and compare.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: s = "anagram", t = "nagaram"
Step-by-step:
- Count for s: [a:3, n:1, g:1, r:1, m:1]
- Subtract for t: all counts zero
Output: true

---

## 3. Group Anagrams
**Problem:** Group strings that are anagrams.

```csharp
public IList<IList<string>> GroupAnagramsMethod(string[] strs)
{
    var dict = new Dictionary<string, List<string>>();
    foreach (var str in strs)
    {
        var chars = str.ToCharArray();
        Array.Sort(chars);
        string key = new string(chars);
        if (!dict.ContainsKey(key)) dict[key] = new List<string>();
        dict[key].Add(str);
    }
    return new List<IList<string>>(dict.Values);
}
```
**Explanation:** Sort each string and group by sorted value.
**Time Complexity:** O(nk log k)
**Space Complexity:** O(nk)

**Dry Run Example:**
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Step-by-step:
- "eat" sorted: "aet" → group ["eat"]
- "tea" sorted: "aet" → group ["eat","tea"]
- ...
Output: [["eat","tea","ate"],["tan","nat"],["bat"]]

---

## 4. Longest Palindromic Substring
**Problem:** Find the longest palindromic substring.

```csharp
public string LongestPalindrome(string s)
{
    if (string.IsNullOrEmpty(s)) return "";
    int start = 0, end = 0;
    for (int i = 0; i < s.Length; i++)
    {
        int len1 = ExpandAroundCenter(s, i, i);
        int len2 = ExpandAroundCenter(s, i, i + 1);
        int len = Math.Max(len1, len2);
        if (len > end - start)
        {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }
    }
    return s.Substring(start, end - start + 1);
}
private int ExpandAroundCenter(string s, int left, int right)
{
    while (left >= 0 && right < s.Length && s[left] == s[right])
    {
        left--; right++;
    }
    return right - left - 1;
}
```
**Explanation:** Expand around center for each character.
**Time Complexity:** O(n^2)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: s = "babad"
Step-by-step:
- Center at 0: "b"
- Center at 1: "bab"
- Center at 2: "aba"
Output: "bab" or "aba"

---

## 5. String to Integer (atoi)
**Problem:** Implement atoi to convert a string to an integer.

```csharp
public int MyAtoi(string s)
{
    int i = 0, sign = 1, result = 0;
    while (i < s.Length && s[i] == ' ') i++;
    if (i < s.Length && (s[i] == '-' || s[i] == '+'))
        sign = s[i++] == '-' ? -1 : 1;
    while (i < s.Length && char.IsDigit(s[i]))
    {
        int digit = s[i++] - '0';
        if (result > (int.MaxValue - digit) / 10)
            return sign == 1 ? int.MaxValue : int.MinValue;
        result = result * 10 + digit;
    }
    return result * sign;
}
```
**Explanation:** Parse string, handle whitespace, sign, and overflow.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: s = "   -42"
Step-by-step:
- Skip spaces, sign=-1
- Read digits: 4, 2 → result=42
- Apply sign: -42
Output: -42

---

## 6. Reverse Words in a String
**Problem:** Reverse the order of words in a string.

```csharp
public string ReverseWords(string s)
{
    var words = s.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
    Array.Reverse(words);
    return string.Join(" ", words);
}
```
**Explanation:** Split, reverse, and join words.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s = "the sky is blue"
Step-by-step:
- Split: ["the","sky","is","blue"]
- Reverse: ["blue","is","sky","the"]
- Join: "blue is sky the"
Output: "blue is sky the"

---

## 7. Valid Parentheses
**Problem:** Check if the input string is valid parentheses.

```csharp
public bool IsValid(string s)
{
    var stack = new Stack<char>();
    var map = new Dictionary<char, char> { { ')', '(' }, { '}', '{' }, { ']', '[' } };
    foreach (var c in s)
    {
        if (map.ContainsValue(c)) stack.Push(c);
        else if (map.ContainsKey(c))
        {
            if (stack.Count == 0 || stack.Pop() != map[c]) return false;
        }
    }
    return stack.Count == 0;
}
```
**Explanation:** Use stack to match opening and closing brackets.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s = "()[]{}"
Step-by-step:
- Push '(', pop with ')'
- Push '[', pop with ']'
- Push '{', pop with '}'
Output: true

---

## 8. Implement strStr()
**Problem:** Find the first occurrence of a substring.

```csharp
public int StrStr(string haystack, string needle)
{
    if (needle == "") return 0;
    for (int i = 0; i <= haystack.Length - needle.Length; i++)
    {
        if (haystack.Substring(i, needle.Length) == needle) return i;
    }
    return -1;
}
```
**Explanation:** Brute force substring search.
**Time Complexity:** O((n-m)m)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: haystack = "hello", needle = "ll"
Step-by-step:
- i=0: "he" != "ll"
- i=1: "el" != "ll"
- i=2: "ll" == "ll" → return 2
Output: 2

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
Input: s = "226"
Step-by-step:
- dp[0]=1, dp[1]=1
- i=2: '2' valid, dp[2]=1, '22' valid, dp[2]+=1 → dp[2]=2
- i=3: '6' valid, dp[3]=2, '26' valid, dp[3]+=1 → dp[3]=3
Output: 3

---

## 10. Count and Say
**Problem:** Generate the nth term of the count-and-say sequence.

```csharp
public string CountAndSayMethod(int n)
{
    string res = "1";
    for (int i = 2; i <= n; i++)
    {
        res = Next(res);
    }
    return res;
}
private string Next(string s)
{
    var sb = new System.Text.StringBuilder();
    int count = 1;
    for (int i = 1; i <= s.Length; i++)
    {
        if (i < s.Length && s[i] == s[i - 1]) count++;
        else
        {
            sb.Append(count).Append(s[i - 1]);
            count = 1;
        }
    }
    return sb.ToString();
}
```
**Explanation:** Build next sequence by counting consecutive digits.
**Time Complexity:** O(n * m) (m = length of sequence)
**Space Complexity:** O(m)

**Dry Run Example:**
Input: n = 4
Step-by-step:
- 1: "1"
- 2: "one 1" → "11"
- 3: "two 1s" → "21"
- 4: "one 2, one 1" → "1211"
Output: "1211"

---

## Linked List
[This section contains solutions for linked list-based DSA problems.]

# Linked List DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common linked list-based DSA interview questions.

---

## 1. Reverse Linked List
**Problem:** Reverse a singly linked list.

```csharp
public ListNode ReverseList(ListNode head)
{
    ListNode prev = null, curr = head;
    while (curr != null)
    {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```
**Explanation:** Iteratively reverse pointers.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: 1 -> 2 -> 3 -> 4 -> 5
Step-by-step:
- prev=null, curr=1
- Reverse 1: next=2, 1->null, prev=1, curr=2
- Reverse 2: next=3, 2->1, prev=2, curr=3
- ...
Output: 5 -> 4 -> 3 -> 2 -> 1

---

## 2. Merge Two Sorted Lists
**Problem:** Merge two sorted linked lists.

```csharp
public ListNode MergeTwoLists(ListNode l1, ListNode l2)
{
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;
    while (l1 != null && l2 != null)
    {
        if (l1.val < l2.val)
        {
            curr.next = l1;
            l1 = l1.next;
        }
        else
        {
            curr.next = l2;
            l2 = l2.next;
        }
        curr = curr.next;
    }
    curr.next = l1 ?? l2;
    return dummy.next;
}
```
**Explanation:** Merge by comparing nodes.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: l1: 1->2->4, l2: 1->3->4
Step-by-step:
- Compare 1 and 1: add 1
- Compare 2 and 3: add 2
- Compare 4 and 3: add 3
- ...
Output: 1->1->2->3->4->4

---

## 3. Linked List Cycle
**Problem:** Detect if a cycle exists in a linked list.

```csharp
public bool HasCycle(ListNode head)
{
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null)
    {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}
```
**Explanation:** Floyd's Tortoise and Hare algorithm.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: 3 -> 2 -> 0 -> -4 (tail connects to node 2)
Step-by-step:
- slow=3, fast=3
- slow=2, fast=0
- slow=0, fast=2
- slow=-4, fast=-4 (meet)
Output: true

---

## 4. Remove Nth Node From End of List
**Problem:** Remove the nth node from the end of a linked list.

```csharp
public ListNode RemoveNthFromEnd(ListNode head, int n)
{
    ListNode dummy = new ListNode(0) { next = head };
    ListNode first = dummy, second = dummy;
    for (int i = 0; i <= n; i++) first = first.next;
    while (first != null)
    {
        first = first.next;
        second = second.next;
    }
    second.next = second.next.next;
    return dummy.next;
}
```
**Explanation:** Two pointer technique.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: 1->2->3->4->5, n=2
Step-by-step:
- Move first pointer n+1 steps ahead
- Move both until first is null
- Remove node after second
Output: 1->2->3->5

---

## 5. Intersection of Two Linked Lists
**Problem:** Find the intersection node of two linked lists.

```csharp
public ListNode GetIntersectionNode(ListNode headA, ListNode headB)
{
    ListNode a = headA, b = headB;
    while (a != b)
    {
        a = a == null ? headB : a.next;
        b = b == null ? headA : b.next;
    }
    return a;
}
```
**Explanation:** Traverse both lists, switch heads when reaching end.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: A: 4->1->8->4->5, B: 5->6->1->8->4->5 (intersect at 8)
Step-by-step:
- Traverse A and B, switch heads at end
- Meet at node 8
Output: Node with value 8

---

## 6. Add Two Numbers
**Problem:** Add two numbers represented by linked lists.

```csharp
public ListNode AddTwoNumbersMethod(ListNode l1, ListNode l2)
{
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;
    int carry = 0;
    while (l1 != null || l2 != null || carry != 0)
    {
        int sum = (l1?.val ?? 0) + (l2?.val ?? 0) + carry;
        carry = sum / 10;
        curr.next = new ListNode(sum % 10);
        curr = curr.next;
        l1 = l1?.next;
        l2 = l2?.next;
    }
    return dummy.next;
}
```
**Explanation:** Add digits, handle carry.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: l1: 2->4->3, l2: 5->6->4
Step-by-step:
- 2+5=7, 4+6=10 (carry 1), 3+4+1=8
Output: 7->0->8

---

## 7. Palindrome Linked List
**Problem:** Check if a linked list is a palindrome.

```csharp
public bool IsPalindrome(ListNode head)
{
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null)
    {
        slow = slow.next;
        fast = fast.next.next;
    }
    ListNode second = ReverseList(slow);
    ListNode first = head;
    while (second != null)
    {
        if (first.val != second.val) return false;
        first = first.next;
        second = second.next;
    }
    return true;
}
private ListNode ReverseList(ListNode head)
{
    ListNode prev = null;
    while (head != null)
    {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```
**Explanation:** Find middle, reverse second half, compare.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: 1->2->2->1
Step-by-step:
- Find middle, reverse second half: 1->2, 2->1
- Compare: 1==1, 2==2
Output: true

---

## 8. Copy List with Random Pointer
**Problem:** Deep copy a linked list with random pointer.

```csharp
public Node CopyRandomList(Node head)
{
    if (head == null) return null;
    var map = new Dictionary<Node, Node>();
    Node curr = head;
    while (curr != null)
    {
        map[curr] = new Node(curr.val);
        curr = curr.next;
    }
    curr = head;
    while (curr != null)
    {
        map[curr].next = curr.next == null ? null : map[curr.next];
        map[curr].random = curr.random == null ? null : map[curr.random];
        curr = curr.next;
    }
    return map[head];
}
```
**Explanation:** Use a map to copy nodes and pointers.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: 7->13->11->10->1 (with random pointers)
Step-by-step:
- Copy nodes, set next and random using map
Output: Deep copy of list with correct random pointers

---

## 9. Reorder List
**Problem:** Reorder a linked list as L0→Ln→L1→Ln-1→...

```csharp
public void ReorderListMethod(ListNode head)
{
    if (head == null || head.next == null) return;
    ListNode slow = head, fast = head;
    while (fast.next != null && fast.next.next != null)
    {
        slow = slow.next;
        fast = fast.next.next;
    }
    ListNode second = ReverseList(slow.next);
    slow.next = null;
    ListNode first = head;
    while (second != null)
    {
        ListNode temp1 = first.next, temp2 = second.next;
        first.next = second;
        second.next = temp1;
        first = temp1;
        second = temp2;
    }
}
private ListNode ReverseList(ListNode head)
{
    ListNode prev = null;
    while (head != null)
    {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```
**Explanation:** Split, reverse second half, merge alternately.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: 1->2->3->4
Step-by-step:
- Find middle, reverse second half: 3->4 becomes 4->3
- Merge: 1->4->2->3
Output: 1->4->2->3

---

## 10. Flatten a Multilevel Doubly Linked List
**Problem:** Flatten a multilevel doubly linked list.

```csharp
public Node Flatten(Node head)
{
    if (head == null) return null;
    var stack = new Stack<Node>();
    Node curr = head;
    while (curr != null)
    {
        if (curr.child != null)
        {
            if (curr.next != null) stack.Push(curr.next);
            curr.next = curr.child;
            curr.child.prev = curr;
            curr.child = null;
        }
        if (curr.next == null && stack.Count > 0)
        {
            curr.next = stack.Pop();
            curr.next.prev = curr;
        }
        curr = curr.next;
    }
    return head;
}
```
**Explanation:** Use stack to process child nodes.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: 1->2->3, 2 has child 4->5
Step-by-step:
- Visit 1, 2, push 3, go to child 4->5
- After 5, pop 3, continue
Output: 1->2->4->5->3

---

## Stack & Queue
[This section contains solutions for stack and queue-based DSA problems.]

# Stack & Queue DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common stack and queue-based DSA interview questions.

---

## 1. Valid Parentheses
**Problem:** Check if the input string is valid parentheses.

```csharp
public bool IsValid(string s)
{
    var stack = new Stack<char>();
    var map = new Dictionary<char, char> { { ')', '(' }, { '}', '{' }, { ']', '[' } };
    foreach (var c in s)
    {
        if (map.ContainsValue(c)) stack.Push(c);
        else if (map.ContainsKey(c))
        {
            if (stack.Count == 0 || stack.Pop() != map[c]) return false;
        }
    }
    return stack.Count == 0;
}
```
**Explanation:** Use stack to match opening and closing brackets.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s = "({[]})"
Stack: [ (
Stack: [ (, {
Stack: [ (, {, [
Stack: [ (, {
Stack: [ (
Stack: [ ]
Output: true (all brackets matched)

---

## 2. Min Stack
**Problem:** Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

```csharp
private Stack<int> stack = new Stack<int>();
private Stack<int> minStack = new Stack<int>();
public void Push(int x)
{
    stack.Push(x);
    if (minStack.Count == 0 || x <= minStack.Peek()) minStack.Push(x);
}
public void Pop()
{
    if (stack.Pop() == minStack.Peek()) minStack.Pop();
}
public int Top() => stack.Peek();
public int GetMin() => minStack.Peek();
```
**Explanation:** Maintain a stack for minimums.
**Time Complexity:** O(1)
**Space Complexity:** O(n)

**Dry Run Example:**
Push(3): stack=[3], minStack=[3]
Push(5): stack=[3,5], minStack=[3]
Push(2): stack=[3,5,2], minStack=[3,2]
GetMin(): 2
Pop(): stack=[3,5], minStack=[3]
GetMin(): 3

---

## 3. Next Greater Element I
**Problem:** Find the next greater element for each element in nums1 from nums2.

```csharp
public int[] NextGreaterElement(int[] nums1, int[] nums2)
{
    var map = new Dictionary<int, int>();
    var stack = new Stack<int>();
    foreach (var num in nums2)
    {
        while (stack.Count > 0 && stack.Peek() < num)
            map[stack.Pop()] = num;
        stack.Push(num);
    }
    while (stack.Count > 0) map[stack.Pop()] = -1;
    var res = new int[nums1.Length];
    for (int i = 0; i < nums1.Length; i++)
        res[i] = map[nums1[i]];
    return res;
}
```
**Explanation:** Use stack to track next greater elements.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
nums1 = [4,1,2], nums2 = [1,3,4,2]
Stack: []
num=1: stack=[1]
num=3: 1<3, map[1]=3, stack=[], stack=[3]
num=4: 3<4, map[3]=4, stack=[], stack=[4]
num=2: stack=[4,2]
Pop remaining: map[2]=-1, map[4]=-1
Result: [ -1, 3, -1 ]

---

## 4. Daily Temperatures
**Problem:** Find how many days until a warmer temperature.

```csharp
public int[] DailyTemperaturesMethod(int[] T)
{
    int n = T.Length;
    int[] res = new int[n];
    var stack = new Stack<int>();
    for (int i = 0; i < n; i++)
    {
        while (stack.Count > 0 && T[i] > T[stack.Peek()])
        {
            int idx = stack.Pop();
            res[idx] = i - idx;
        }
        stack.Push(i);
    }
    return res;
}
```
**Explanation:** Stack to track indices of temperatures.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
T = [73, 74, 75, 71, 69, 72, 76, 73]
Stack: []
i=0: stack=[0]
i=1: 74>73, res[0]=1, stack=[], stack=[1]
i=2: 75>74, res[1]=1, stack=[], stack=[2]
i=3: stack=[2,3]
i=4: stack=[2,3,4]
i=5: 72>69, res[4]=1, stack=[2,3], 72>71, res[3]=2, stack=[2], stack=[2,5]
i=6: 76>72, res[5]=1, stack=[2], 76>75, res[2]=4, stack=[], stack=[6]
i=7: stack=[6,7]
Result: [1,1,4,2,1,1,0,0]

---

## 5. Evaluate Reverse Polish Notation
**Problem:** Evaluate the value of an arithmetic expression in Reverse Polish Notation.

```csharp
public int EvalRPN(string[] tokens)
{
    var stack = new Stack<int>();
    foreach (var token in tokens)
    {
        if (int.TryParse(token, out int num)) stack.Push(num);
        else
        {
            int b = stack.Pop(), a = stack.Pop();
            switch (token)
            {
                case "+": stack.Push(a + b); break;
                case "-": stack.Push(a - b); break;
                case "*": stack.Push(a * b); break;
                case "/": stack.Push(a / b); break;
            }
        }
    }
    return stack.Pop();
}
```
**Explanation:** Stack for operands, pop and apply operator.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
tokens = ["2", "1", "+", "3", "*"]
Stack: []
Push 2: [2]
Push 1: [2,1]
Operator '+': Pop 1,2 -> 2+1=3, Push 3: [3]
Push 3: [3,3]
Operator '*': Pop 3,3 -> 3*3=9, Push 9: [9]
Output: 9

---

## 6. Implement Queue using Stacks
**Problem:** Implement a queue using stacks.

```csharp
private Stack<int> inStack = new Stack<int>();
private Stack<int> outStack = new Stack<int>();
public void Push(int x) => inStack.Push(x);
public int Pop()
{
    if (outStack.Count == 0)
        while (inStack.Count > 0) outStack.Push(inStack.Pop());
    return outStack.Pop();
}
public int Peek()
{
    if (outStack.Count == 0)
        while (inStack.Count > 0) outStack.Push(inStack.Pop());
    return outStack.Peek();
}
public bool Empty() => inStack.Count == 0 && outStack.Count == 0;
```
**Explanation:** Use two stacks to simulate queue.
**Time Complexity:** O(1) amortized
**Space Complexity:** O(n)

**Dry Run Example:**
Push(1): inStack=[1], outStack=[]
Push(2): inStack=[1,2], outStack=[]
Pop(): outStack empty, move all from inStack to outStack: inStack=[], outStack=[2,1], Pop 1
Push(3): inStack=[3], outStack=[2]
Peek(): outStack not empty, Peek 2
Empty(): false

---

## 7. Simplify Path
**Problem:** Simplify a Unix-style file path.

```csharp
public string SimplifyPathMethod(string path)
{
    var stack = new Stack<string>();
    foreach (var part in path.Split('/'))
    {
        if (part == "" || part == ".") continue;
        if (part == "..")
        {
            if (stack.Count > 0) stack.Pop();
        }
        else stack.Push(part);
    }
    return "/" + string.Join("/", stack.Reverse());
}
```
**Explanation:** Use stack to process path segments.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: "/a/./b/../../c/"
Split: ["", "a", ".", "b", "..", "..", "c", ""]
stack: []
part="a": stack=["a"]
part=".": skip
part="b": stack=["a","b"]
part="..": pop "b", stack=["a"]
part="..": pop "a", stack=[]
part="c": stack=["c"]
Output: "/c"

---

## 8. Basic Calculator
**Problem:** Implement a basic calculator to evaluate a string expression.

```csharp
public int Calculate(string s)
{
    int res = 0, sign = 1, num = 0;
    var stack = new Stack<int>();
    for (int i = 0; i < s.Length; i++)
    {
        char c = s[i];
        if (char.IsDigit(c))
        {
            num = num * 10 + (c - '0');
        }
        else if (c == '+')
        {
            res += sign * num;
            num = 0; sign = 1;
        }
        else if (c == '-')
        {
            res += sign * num;
            num = 0; sign = -1;
        }
        else if (c == '(')
        {
            stack.Push(res); stack.Push(sign);
            res = 0; sign = 1;
        }
        else if (c == ')')
        {
            res += sign * num;
            res *= stack.Pop();
            res += stack.Pop();
            num = 0;
        }
    }
    res += sign * num;
    return res;
}
```
**Explanation:** Stack to handle parentheses and signs.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: s = "1 + (2 - (3 + 4))"
res=0, sign=1, num=0, stack=[]
Read '1': num=1
Read '+': res=1, num=0, sign=1
Read '(': push res=1, sign=1, res=0, sign=1
Read '2': num=2
Read '-': res=2, num=0, sign=-1
Read '(': push res=2, sign=-1, res=0, sign=1
Read '3': num=3
Read '+': res=3, num=0, sign=1
Read '4': num=4
Read ')': res=7, pop sign=1, res=7, pop res=2, res=2-7=-5, num=0
Read ')': res=-5, pop sign=1, res=-5, pop res=1, res=1+(-5)=-4, num=0
Output: -4

---

## 9. Largest Rectangle in Histogram
**Problem:** Find the largest rectangle in a histogram.

```csharp
public int LargestRectangleArea(int[] heights)
{
    int n = heights.Length, maxArea = 0;
    var stack = new Stack<int>();
    for (int i = 0; i <= n; i++)
    {
        int h = (i == n) ? 0 : heights[i];
        while (stack.Count > 0 && h < heights[stack.Peek()])
        {
            int height = heights[stack.Pop()];
            int width = stack.Count == 0 ? i : i - stack.Peek() - 1;
            maxArea = Math.Max(maxArea, height * width);
        }
        stack.Push(i);
    }
    return maxArea;
}
```
**Explanation:** Stack to track indices, calculate area.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
heights = [2,1,5,6,2,3]
stack: []
i=0: h=2, stack=[0]
i=1: h=1, 2>1, pop 0, area=2*1=2, maxArea=2, stack=[], stack=[1]
i=2: h=5, stack=[1,2]
i=3: h=6, stack=[1,2,3]
i=4: h=2, 6>2, pop 3, area=6*1=6, maxArea=6, 5>2, pop 2, area=5*2=10, maxArea=10, stack=[1,4]
i=5: h=3, stack=[1,4,5]
i=6: h=0, 3>0, pop 5, area=3*1=3, 2>0, pop 4, area=2*4=8, stack=[1], pop 1, area=1*6=6
Output: 10

---

## 10. Sliding Window Maximum
**Problem:** Find the maximum in each sliding window of size k.

```csharp
public int[] MaxSlidingWindow(int[] nums, int k)
{
    var deque = new LinkedList<int>();
    var res = new List<int>();
    for (int i = 0; i < nums.Length; i++)
    {
        while (deque.Count > 0 && deque.First.Value <= i - k) deque.RemoveFirst();
        while (deque.Count > 0 && nums[deque.Last.Value] < nums[i]) deque.RemoveLast();
        deque.AddLast(i);
        if (i >= k - 1) res.Add(nums[deque.First.Value]);
    }
    return res.ToArray();
}
```
**Explanation:** Deque to maintain max in window.
**Time Complexity:** O(n)
**Space Complexity:** O(k)

**Dry Run Example:**
nums = [1,3,-1,-3,5,3,6,7], k=3
deque: []
i=0: deque=[0]
i=1: nums[1]=3>nums[0]=1, remove 0, deque=[], deque=[1]
i=2: nums[2]=-1<nums[1]=3, deque=[1,2]
i=2: i>=k-1, res=[3]
i=3: deque=[1,2,3], remove 1 (out of window), deque=[2,3]
i=4: nums[4]=5>nums[3]=-3, remove 3, nums[4]=5>nums[2]=-1, remove 2, deque=[4]
i=4: i>=k-1, res=[3,3,5]
i=5: deque=[4,5]
i=5: i>=k-1, res=[3,3,5,5]
i=6: nums[6]=6>nums[5]=3, remove 5, nums[6]=6>nums[4]=5, remove 4, deque=[6]
i=6: i>=k-1, res=[3,3,5,5,6]
i=7: nums[7]=7>nums[6]=6, remove 6, deque=[7]
i=7: i>=k-1, res=[3,3,5,5,6,7]
Output: [3,3,5,5,6,7]

---

## Dynamic Programming
[This section contains solutions for dynamic programming DSA problems.]

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
