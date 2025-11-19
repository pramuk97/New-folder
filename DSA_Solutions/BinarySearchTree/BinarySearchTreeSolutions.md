# Binary Search Tree DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common BST-based DSA interview questions.

---

## 1. Validate Binary Search Tree
**Problem:** Check if a binary tree is a valid BST.

```csharp
public bool IsValidBST(TreeNode root)
{
    return IsValid(root, long.MinValue, long.MaxValue);
}
private bool IsValid(TreeNode node, long min, long max)
{
    if (node == null) return true;
    if (node.val <= min || node.val >= max) return false;
    return IsValid(node.left, min, node.val) && IsValid(node.right, node.val, max);
}
```
**Explanation:** Recursively check value bounds.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [2,1,3]
IsValid(2, -inf, inf): 2 in range, left=IsValid(1, -inf,2), right=IsValid(3,2,inf)
All checks true, Output: true

---

## 2. Insert into a BST
**Problem:** Insert a value into a BST.

```csharp
public TreeNode Insert(TreeNode root, int val)
{
    if (root == null) return new TreeNode(val);
    if (val < root.val) root.left = Insert(root.left, val);
    else root.right = Insert(root.right, val);
    return root;
}
```
**Explanation:** Recursively insert value.
**Time Complexity:** O(h)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [4,2,7,1,3], val=5
Insert(4,5): 5>4, go right
Insert(7,5): 5<7, go left
Insert(null,5): insert new node 5
Output: [4,2,7,1,3,5]

---

## 3. Delete Node in a BST
**Problem:** Delete a node from a BST.

```csharp
public TreeNode DeleteNode(TreeNode root, int key)
{
    if (root == null) return null;
    if (key < root.val) root.left = DeleteNode(root.left, key);
    else if (key > root.val) root.right = DeleteNode(root.right, key);
    else
    {
        if (root.left == null) return root.right;
        if (root.right == null) return root.left;
        TreeNode minNode = GetMin(root.right);
        root.val = minNode.val;
        root.right = DeleteNode(root.right, minNode.val);
    }
    return root;
}
private TreeNode GetMin(TreeNode node)
{
    while (node.left != null) node = node.left;
    return node;
}
```
**Explanation:** Handle three cases, use min node for replacement.
**Time Complexity:** O(h)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [5,3,6,2,4,null,7], key=3
DeleteNode(5,3): 3<5, go left
DeleteNode(3,3): found, has two children
GetMin(4): 4
Replace 3 with 4, delete 4 from right
Output: [5,4,6,2,null,null,7]

---

## 4. Lowest Common Ancestor of a BST
**Problem:** Find the LCA of two nodes in a BST.

```csharp
public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q)
{
    while (root != null)
    {
        if (p.val < root.val && q.val < root.val) root = root.left;
        else if (p.val > root.val && q.val > root.val) root = root.right;
        else return root;
    }
    return null;
}
```
**Explanation:** Use BST property to find LCA.
**Time Complexity:** O(h)
**Space Complexity:** O(1)

**Dry Run Example:**
Input: [6,2,8,0,4,7,9], p=2, q=8
root=6: p,q on both sides, return 6
Output: 6

---

## 5. Kth Smallest Element in a BST
**Problem:** Find the kth smallest element in a BST.

```csharp
public int KthSmallest(TreeNode root, int k)
{
    var stack = new Stack<TreeNode>();
    while (true)
    {
        while (root != null)
        {
            stack.Push(root);
            root = root.left;
        }
        root = stack.Pop();
        if (--k == 0) return root.val;
        root = root.right;
    }
}
```
**Explanation:** Inorder traversal to find kth smallest.
**Time Complexity:** O(h + k)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [3,1,4,null,2], k=1
stack: []
Push 3, push 1, pop 1 (k=0), return 1
Output: 1

---

## 6. BST Iterator
**Problem:** Implement an iterator over a BST.

```csharp
private Stack<TreeNode> stack = new Stack<TreeNode>();
public BSTIterator(TreeNode root)
{
    PushLeft(root);
}
private void PushLeft(TreeNode node)
{
    while (node != null)
    {
        stack.Push(node);
        node = node.left;
    }
}
public bool HasNext() => stack.Count > 0;
public int Next()
{
    var node = stack.Pop();
    PushLeft(node.right);
    return node.val;
}
```
**Explanation:** Use stack to simulate inorder traversal.
**Time Complexity:** O(1) amortized per operation
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [7,3,15,null,null,9,20]
BSTIterator(root): stack=[7,3]
Next(): pop 3, push right (null), return 3
Next(): pop 7, push 15,9, return 7
HasNext(): true

---

## 7. Convert Sorted Array to BST
**Problem:** Convert a sorted array to a height-balanced BST.

```csharp
public TreeNode SortedArrayToBST(int[] nums)
{
    return Build(nums, 0, nums.Length - 1);
}
private TreeNode Build(int[] nums, int left, int right)
{
    if (left > right) return null;
    int mid = left + (right - left) / 2;
    var node = new TreeNode(nums[mid]);
    node.left = Build(nums, left, mid - 1);
    node.right = Build(nums, mid + 1, right);
    return node;
}
```
**Explanation:** Recursively build BST from sorted array.
**Time Complexity:** O(n)
**Space Complexity:** O(log n)

**Dry Run Example:**
Input: nums=[-10,-3,0,5,9]
mid=2, node=0
Left: [-10,-3], mid=0, node=-10
Right: [5,9], mid=4, node=9
Output: [0,-10,9,null,-3,5]

---

## 8. Range Sum of BST
**Problem:** Sum values of all nodes within a range.

```csharp
public int RangeSumBST(TreeNode root, int low, int high)
{
    if (root == null) return 0;
    if (root.val < low) return RangeSumBST(root.right, low, high);
    if (root.val > high) return RangeSumBST(root.left, low, high);
    return root.val + RangeSumBST(root.left, low, high) + RangeSumBST(root.right, low, high);
}
```
**Explanation:** Use BST property to sum values in range.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [10,5,15,3,7,13,18], low=7, high=15
Node=10: in range, sum=10+sum(left)+sum(right)
Left=5: 5<7, skip left, go right
Right=15: in range, sum=15+sum(left)+sum(right)
Output: 10+15+7+13=45

---

## 9. Trim a BST
**Problem:** Trim a BST so that all its elements lie in [low, high].

```csharp
public TreeNode TrimBSTMethod(TreeNode root, int low, int high)
{
    if (root == null) return null;
    if (root.val < low) return TrimBSTMethod(root.right, low, high);
    if (root.val > high) return TrimBSTMethod(root.left, low, high);
    root.left = TrimBSTMethod(root.left, low, high);
    root.right = TrimBSTMethod(root.right, low, high);
    return root;
}
```
**Explanation:** Recursively trim nodes outside range.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [1,0,2], low=1, high=2
TrimBST(1): 1 in range, trim left/right
TrimBST(0): 0<1, go right (null)
TrimBST(2): 2 in range, trim left/right
Output: [1,null,2]

---

## 10. Recover Binary Search Tree
**Problem:** Recover a BST where two nodes are swapped by mistake.

```csharp
private TreeNode first, second, prev;
public void RecoverTree(TreeNode root)
{
    first = second = prev = null;
    Inorder(root);
    if (first != null && second != null)
    {
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }
}
private void Inorder(TreeNode node)
{
    if (node == null) return;
    Inorder(node.left);
    if (prev != null && node.val < prev.val)
    {
        if (first == null) first = prev;
        second = node;
    }
    prev = node;
    Inorder(node.right);
}
```
**Explanation:** Inorder traversal to find swapped nodes.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [1,3,null,null,2], swapped 3 and 2
Inorder: 3>2, first=3, second=2
Swap 3 and 2
Output: [1,2,3]

---
