# Binary Tree DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common binary tree-based DSA interview questions.

---

## 1. Binary Tree Inorder Traversal
**Problem:** Inorder traversal of a binary tree.

```csharp
public IList<int> InorderTraversal(TreeNode root)
{
    var res = new List<int>();
    var stack = new Stack<TreeNode>();
    TreeNode curr = root;
    while (curr != null || stack.Count > 0)
    {
        while (curr != null)
        {
            stack.Push(curr);
            curr = curr.left;
        }
        curr = stack.Pop();
        res.Add(curr.val);
        curr = curr.right;
    }
    return res;
}
```
**Explanation:** Iterative inorder traversal using stack.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: [1, null, 2, 3]
Stack: []
curr=1: push 1, curr=null
pop 1, add 1 to res, curr=1.right=2
push 2, curr=2.left=null
pop 2, add 2 to res, curr=2.right=3
push 3, curr=3.left=null
pop 3, add 3 to res, curr=3.right=null
Output: [1,2,3]

---

## 2. Maximum Depth of Binary Tree
**Problem:** Find the maximum depth of a binary tree.

```csharp
public int MaxDepth(TreeNode root)
{
    if (root == null) return 0;
    return 1 + Math.Max(MaxDepth(root.left), MaxDepth(root.right));
}
```
**Explanation:** Recursively find max depth of left and right subtrees.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [3,9,20,null,null,15,7]
Call MaxDepth(3): left=MaxDepth(9)=1, right=MaxDepth(20)=2
MaxDepth(20): left=MaxDepth(15)=1, right=MaxDepth(7)=1
Return: 1+max(1,2)=3

---

## 3. Invert Binary Tree
**Problem:** Invert a binary tree.

```csharp
public TreeNode InvertTree(TreeNode root)
{
    if (root == null) return null;
    var left = InvertTree(root.left);
    var right = InvertTree(root.right);
    root.left = right;
    root.right = left;
    return root;
}
```
**Explanation:** Recursively swap left and right children.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [4,2,7,1,3,6,9]
Call InvertTree(4): swap left/right
InvertTree(2): swap left/right
InvertTree(1): null
InvertTree(3): null
InvertTree(7): swap left/right
InvertTree(6): null
InvertTree(9): null
Output: [4,7,2,9,6,3,1]

---

## 4. Diameter of Binary Tree
**Problem:** Find the diameter of a binary tree.

```csharp
private int maxDiameter = 0;
public int DiameterOfBinaryTreeMethod(TreeNode root)
{
    Depth(root);
    return maxDiameter;
}
private int Depth(TreeNode node)
{
    if (node == null) return 0;
    int left = Depth(node.left);
    int right = Depth(node.right);
    maxDiameter = Math.Max(maxDiameter, left + right);
    return 1 + Math.Max(left, right);
}
```
**Explanation:** Track max path between any two nodes.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [1,2,3,4,5]
Call Depth(1): left=Depth(2), right=Depth(3)
Depth(2): left=Depth(4)=1, right=Depth(5)=1, maxDiameter=2
Depth(3): left/right=null=0
maxDiameter=3
Output: 3

---

## 5. Balanced Binary Tree
**Problem:** Check if a binary tree is height-balanced.

```csharp
public bool IsBalanced(TreeNode root)
{
    return Check(root) != -1;
}
private int Check(TreeNode node)
{
    if (node == null) return 0;
    int left = Check(node.left);
    int right = Check(node.right);
    if (left == -1 || right == -1 || Math.Abs(left - right) > 1) return -1;
    return 1 + Math.Max(left, right);
}
```
**Explanation:** Check height difference recursively.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [3,9,20,null,null,15,7]
Check(3): left=Check(9)=1, right=Check(20)=2, abs(1-2)=1
Check(20): left=Check(15)=1, right=Check(7)=1, abs(1-1)=0
Return: balanced

---

## 6. Path Sum
**Problem:** Check if a path with a given sum exists.

```csharp
public bool HasPathSum(TreeNode root, int sum)
{
    if (root == null) return false;
    if (root.left == null && root.right == null) return root.val == sum;
    return HasPathSum(root.left, sum - root.val) || HasPathSum(root.right, sum - root.val);
}
```
**Explanation:** Recursively check for path sum.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [5,4,8,11,null,13,4,7,2,null,null,null,1], sum=22
HasPathSum(5,22): left=HasPathSum(4,17), right=HasPathSum(8,17)
HasPathSum(4,17): left=HasPathSum(11,13)
HasPathSum(11,13): left=HasPathSum(7,2), right=HasPathSum(2,2)
HasPathSum(2,2): leaf, 2==2 -> true
Output: true

---

## 7. Lowest Common Ancestor
**Problem:** Find the lowest common ancestor of two nodes.

```csharp
public TreeNode LowestCommonAncestorMethod(TreeNode root, TreeNode p, TreeNode q)
{
    if (root == null || root == p || root == q) return root;
    var left = LowestCommonAncestorMethod(root.left, p, q);
    var right = LowestCommonAncestorMethod(root.right, p, q);
    if (left != null && right != null) return root;
    return left ?? right;
}
```
**Explanation:** Recursively find LCA in left and right subtrees.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: root=3, p=5, q=1
Call LCA(3,5,1): left=LCA(5,5,1)=5, right=LCA(1,5,1)=1
Both left/right not null, return 3
Output: 3

---

## 8. Serialize and Deserialize Binary Tree
**Problem:** Serialize and deserialize a binary tree.

```csharp
public string Serialize(TreeNode root)
{
    var sb = new System.Text.StringBuilder();
    SerializeHelper(root, sb);
    return sb.ToString();
}
private void SerializeHelper(TreeNode node, System.Text.StringBuilder sb)
{
    if (node == null)
    {
        sb.Append("null,");
        return;
    }
    sb.Append(node.val).Append(",");
    SerializeHelper(node.left, sb);
    SerializeHelper(node.right, sb);
}
public TreeNode Deserialize(string data)
{
    var nodes = new Queue<string>(data.Split(','));
    return DeserializeHelper(nodes);
}
private TreeNode DeserializeHelper(Queue<string> nodes)
{
    var val = nodes.Dequeue();
    if (val == "null" || val == "") return null;
    var node = new TreeNode(int.Parse(val));
    node.left = DeserializeHelper(nodes);
    node.right = DeserializeHelper(nodes);
    return node;
}
```
**Explanation:** Preorder traversal for serialization/deserialization.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: [1,2,3,null,null,4,5]
Serialize: "1,2,null,null,3,4,null,null,5,null,null,"
Deserialize: split to queue, build tree recursively
Output: Tree with structure [1,2,3,null,null,4,5]

---

## 9. Construct Binary Tree from Preorder and Inorder
**Problem:** Build a binary tree from preorder and inorder traversals.

```csharp
public TreeNode BuildTree(int[] preorder, int[] inorder)
{
    var map = new Dictionary<int, int>();
    for (int i = 0; i < inorder.Length; i++) map[inorder[i]] = i;
    return Build(preorder, 0, preorder.Length - 1, inorder, 0, inorder.Length - 1, map);
}
private TreeNode Build(int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd, Dictionary<int, int> map)
{
    if (preStart > preEnd || inStart > inEnd) return null;
    var root = new TreeNode(preorder[preStart]);
    int inRoot = map[root.val];
    int leftSize = inRoot - inStart;
    root.left = Build(preorder, preStart + 1, preStart + leftSize, inorder, inStart, inRoot - 1, map);
    root.right = Build(preorder, preStart + leftSize + 1, preEnd, inorder, inRoot + 1, inEnd, map);
    return root;
}
```
**Explanation:** Use preorder/inorder indices and hashmap.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Build root=3, left subtree from inorder[0:0], right from inorder[2:4]
Left: root=9, no children
Right: root=20, left=15, right=7
Output: [3,9,20,null,null,15,7]

---

## 10. Symmetric Tree
**Problem:** Check if a binary tree is symmetric.

```csharp
public bool IsSymmetric(TreeNode root)
{
    return root == null || IsMirror(root.left, root.right);
}
private bool IsMirror(TreeNode t1, TreeNode t2)
{
    if (t1 == null && t2 == null) return true;
    if (t1 == null || t2 == null) return false;
    return t1.val == t2.val && IsMirror(t1.left, t2.right) && IsMirror(t1.right, t2.left);
}
```
**Explanation:** Recursively check mirror symmetry.
**Time Complexity:** O(n)
**Space Complexity:** O(h)

**Dry Run Example:**
Input: [1,2,2,3,4,4,3]
IsMirror(2,2): t1.val==t2.val, check children
IsMirror(3,3): true, IsMirror(4,4): true
All checks true, Output: true

---
