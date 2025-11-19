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
