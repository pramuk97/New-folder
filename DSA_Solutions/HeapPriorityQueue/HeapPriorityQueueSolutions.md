# Heap / Priority Queue DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common heap and priority queue DSA interview questions.

---

## 1. Kth Largest Element in an Array
**Problem:** Find the kth largest element in an array.

```csharp
public int FindKthLargest(int[] nums, int k)
{
    var pq = new SortedSet<(int, int)>();
    for (int i = 0; i < nums.Length; i++)
    {
        pq.Add((nums[i], i));
        if (pq.Count > k) pq.Remove(pq.Min);
    }
    return pq.Min.Item1;
}
```
**Explanation:** Min-heap of size k.
**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

**Dry Run Example:**
Input: nums=[3,2,1,5,6,4], k=2
pq: add 3,2,1,5,6,4 (keep size 2)
pq after all: [5,6]
Output: 5

---

## 2. Top K Frequent Elements
**Problem:** Find the k most frequent elements.

```csharp
public int[] TopKFrequent(int[] nums, int k)
{
    var dict = new Dictionary<int, int>();
    foreach (var num in nums) dict[num] = dict.GetValueOrDefault(num, 0) + 1;
    var pq = new SortedSet<(int, int)>();
    foreach (var kv in dict)
    {
        pq.Add((kv.Value, kv.Key));
        if (pq.Count > k) pq.Remove(pq.Min);
    }
    var res = new List<int>();
    foreach (var item in pq) res.Add(item.Item2);
    return res.ToArray();
}
```
**Explanation:** Min-heap to keep top k frequent elements.
**Time Complexity:** O(n log k)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: nums=[1,1,1,2,2,3], k=2
dict: {1:3, 2:2, 3:1}
pq: add (3,1), (2,2), (1,3) (keep size 2)
Output: [1,2]

---

## 3. Merge K Sorted Lists
**Problem:** Merge k sorted linked lists.

```csharp
public ListNode MergeKLists(ListNode[] lists)
{
    var pq = new SortedSet<(int, int, ListNode)>();
    for (int i = 0; i < lists.Length; i++)
        if (lists[i] != null) pq.Add((lists[i].val, i, lists[i]));
    var dummy = new ListNode(0);
    var curr = dummy;
    while (pq.Count > 0)
    {
        var (val, idx, node) = pq.Min; pq.Remove(pq.Min);
        curr.next = node;
        curr = curr.next;
        if (node.next != null) pq.Add((node.next.val, idx, node.next));
    }
    return dummy.next;
}
```
**Explanation:** Min-heap to merge k lists.
**Time Complexity:** O(N log k)
**Space Complexity:** O(k)

**Dry Run Example:**
Input: lists=[[1,4,5],[1,3,4],[2,6]]
pq: add 1,1,2
pop 1, add next from same list
Continue until all lists empty
Output: [1,1,2,3,4,4,5,6]

---

## 4. Find Median from Data Stream
**Problem:** Find median from a stream of numbers.

```csharp
private PriorityQueue<int, int> minHeap = new PriorityQueue<int, int>();
private PriorityQueue<int, int> maxHeap = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b - a));
public void AddNum(int num)
{
    maxHeap.Enqueue(num, num);
    minHeap.Enqueue(maxHeap.Dequeue(), maxHeap.Peek());
    if (minHeap.Count > maxHeap.Count) maxHeap.Enqueue(minHeap.Dequeue(), minHeap.Peek());
}
public double FindMedian()
{
    if (maxHeap.Count > minHeap.Count) return maxHeap.Peek();
    return (maxHeap.Peek() + minHeap.Peek()) / 2.0;
}
```
**Explanation:** Two heaps to maintain lower and upper halves.
**Time Complexity:** O(log n) per operation
**Space Complexity:** O(n)

**Dry Run Example:**
AddNum(1): maxHeap=[1], minHeap=[]
AddNum(2): maxHeap=[1], minHeap=[2]
AddNum(3): maxHeap=[2,1], minHeap=[3]
FindMedian(): (2+1)/2=1.5

---

## 5. Task Scheduler
**Problem:** Find least intervals to finish all tasks with cooldown.

```csharp
public int LeastInterval(char[] tasks, int n)
{
    var dict = new int[26];
    foreach (var t in tasks) dict[t - 'A']++;
    Array.Sort(dict);
    int max = dict[25] - 1, idle = max * n;
    for (int i = 24; i >= 0 && dict[i] > 0; i--)
        idle -= Math.Min(dict[i], max);
    return idle > 0 ? tasks.Length + idle : tasks.Length;
}
```
**Explanation:** Greedy, fill idle slots with most frequent tasks.
**Time Complexity:** O(n)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: tasks=[A,A,A,B,B,B], n=2
dict: [3,3,0,...]
max=2, idle=4
Fill idle with other tasks
Output: 8

---

## 6. Reorganize String
**Problem:** Rearrange string so no two adjacent chars are the same.

```csharp
var dict = new int[26];
foreach (var c in s) dict[c - 'a']++;
var pq = new PriorityQueue<(char, int), int>(Comparer<int>.Create((a, b) => b - a));
for (int i = 0; i < 26; i++)
    if (dict[i] > 0) pq.Enqueue(((char)(i + 'a'), dict[i]), dict[i]);
var sb = new System.Text.StringBuilder();
(char, int) prev = ('#', 0);
while (pq.Count > 0)
{
    var curr = pq.Dequeue();
    sb.Append(curr.Item1);
    if (prev.Item2 > 0) pq.Enqueue(prev, prev.Item2);
    curr.Item2--;
    prev = curr;
}
return sb.Length == s.Length ? sb.ToString() : "";
```
**Explanation:** Max-heap to always pick most frequent char.
**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

**Dry Run Example:**
Input: s="aab"
dict: [2,1,...]
pq: ('a',2),('b',1)
Build string: aba
Output: "aba"

---

## 7. K Closest Points to Origin
**Problem:** Find k closest points to the origin.

```csharp
public int[][] KClosest(int[][] points, int k)
{
    var pq = new SortedSet<(int, int, int)>();
    for (int i = 0; i < points.Length; i++)
    {
        int dist = points[i][0] * points[i][0] + points[i][1] * points[i][1];
        pq.Add((dist, points[i][0], points[i][1]));
        if (pq.Count > k) pq.Remove(pq.Max);
    }
    var res = new int[k][];
    int idx = 0;
    foreach (var item in pq) res[idx++] = new[] { item.Item2, item.Item3 };
    return res;
}
```
**Explanation:** Max-heap to keep k closest points.
**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

**Dry Run Example:**
Input: points=[[1,3],[-2,2]], k=1
pq: add (10,1,3), (8,-2,2) (keep size 1)
Output: [[-2,2]]

---

## 8. Smallest Range Covering Elements from K Lists
**Problem:** Find smallest range covering at least one element from each list.

```csharp
var pq = new PriorityQueue<(int val, int row, int idx), int>();
int max = int.MinValue, start = 0, end = int.MaxValue;
for (int i = 0; i < nums.Count; i++)
{
    pq.Enqueue((nums[i][0], i, 0), nums[i][0]);
    max = Math.Max(max, nums[i][0]);
}
while (pq.Count == nums.Count)
{
    var (val, row, idx) = pq.Dequeue();
    if (max - val < end - start)
    {
        start = val; end = max;
    }
    if (idx + 1 < nums[row].Count)
    {
        pq.Enqueue((nums[row][idx + 1], row, idx + 1), nums[row][idx + 1]);
        max = Math.Max(max, nums[row][idx + 1]);
    }
}
return new[] { start, end };
```
**Explanation:** Min-heap to track current range.
**Time Complexity:** O(n log k)
**Space Complexity:** O(k)

**Dry Run Example:**
Input: nums=[[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
pq: add 4,0,5
Track min/max, update range
Output: [20,24]

---

## 9. Sliding Window Median
**Problem:** Find median in each sliding window of size k.

*See C# implementation for two heaps approach.*

**Dry Run Example:**
Input: nums=[1,3,-1,-3,5,3,6,7], k=3
Add to heaps, balance, output median for each window

---

## 10. Ugly Number II
**Problem:** Find the nth ugly number.

```csharp
public int NthUglyNumber(int n)
{
    var ugly = new int[n]; ugly[0] = 1;
    int i2 = 0, i3 = 0, i5 = 0;
    for (int i = 1; i < n; i++)
    {
        int next = Math.Min(ugly[i2] * 2, Math.Min(ugly[i3] * 3, ugly[i5] * 5));
        ugly[i] = next;
        if (next == ugly[i2] * 2) i2++;
        if (next == ugly[i3] * 3) i3++;
        if (next == ugly[i5] * 5) i5++;
    }
    return ugly[n - 1];
}
```
**Explanation:** DP with pointers for 2, 3, 5 multiples.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: n=10
ugly=[1]
i2=0,i3=0,i5=0
next=min(2,3,5)=2, ugly=[1,2]
next=min(4,3,5)=3, ugly=[1,2,3]
... continue
Output: 12

---
