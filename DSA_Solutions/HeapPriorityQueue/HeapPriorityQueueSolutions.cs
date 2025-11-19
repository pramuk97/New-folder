// DSA Solutions - Heap / Priority Queue
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.HeapPriorityQueue
{
    // 1. Kth Largest Element in an Array
    public class KthLargestElementInArray
    {
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
        // Explanation: Min-heap of size k.
    }

    // 2. Top K Frequent Elements
    public class TopKFrequentElements
    {
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
        // Explanation: Min-heap to keep top k frequent elements.
    }

    // 3. Merge K Sorted Lists
    public class MergeKSortedLists
    {
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
        public class ListNode
        {
            public int val;
            public ListNode next;
            public ListNode(int x) { val = x; }
        }
        // Explanation: Min-heap to merge k lists.
    }

    // 4. Find Median from Data Stream
    public class MedianFinder
    {
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
        // Explanation: Two heaps to maintain lower and upper halves.
    }

    // 5. Task Scheduler
    public class TaskScheduler
    {
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
        // Explanation: Greedy, fill idle slots with most frequent tasks.
    }

    // 6. Reorganize String
    public class ReorganizeString
    {
        public string ReorganizeStringMethod(string s)
        {
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
        }
        // Explanation: Max-heap to always pick most frequent char.
    }

    // 7. K Closest Points to Origin
    public class KClosestPointsToOrigin
    {
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
        // Explanation: Max-heap to keep k closest points.
    }

    // 8. Smallest Range Covering Elements from K Lists
    public class SmallestRangeFromKLists
    {
        public int[] SmallestRange(IList<IList<int>> nums)
        {
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
        }
        // Explanation: Min-heap to track current range.
    }

    // 9. Sliding Window Median
    public class SlidingWindowMedian
    {
        // For brevity, use two heaps approach (not full implementation here)
        // Explanation: Use two heaps to maintain window medians.
    }

    // 10. Ugly Number II
    public class UglyNumberII
    {
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
        // Explanation: DP with pointers for 2, 3, 5 multiples.
    }
}
