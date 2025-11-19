// DSA Solutions - Stack & Queue
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.StackQueue
{
    // 1. Valid Parentheses
    public class ValidParentheses
    {
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
        // Explanation: Use stack to match opening and closing brackets.
    }

    // 2. Min Stack
    public class MinStack
    {
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
        // Explanation: Maintain a stack for minimums.
    }

    // 3. Next Greater Element I
    public class NextGreaterElementI
    {
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
        // Explanation: Use stack to track next greater elements.
    }

    // 4. Daily Temperatures
    public class DailyTemperatures
    {
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
        // Explanation: Stack to track indices of temperatures.
    }

    // 5. Evaluate Reverse Polish Notation
    public class EvaluateReversePolishNotation
    {
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
        // Explanation: Stack for operands, pop and apply operator.
    }

    // 6. Implement Queue using Stacks
    public class MyQueue
    {
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
        // Explanation: Use two stacks to simulate queue.
    }

    // 7. Simplify Path
    public class SimplifyPath
    {
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
        // Explanation: Use stack to process path segments.
    }

    // 8. Basic Calculator
    public class BasicCalculator
    {
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
        // Explanation: Stack to handle parentheses and signs.
    }

    // 9. Largest Rectangle in Histogram
    public class LargestRectangleInHistogram
    {
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
        // Explanation: Stack to track indices, calculate area.
    }

    // 10. Sliding Window Maximum
    public class SlidingWindowMaximum
    {
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
        // Explanation: Deque to maintain max in window.
    }
}
