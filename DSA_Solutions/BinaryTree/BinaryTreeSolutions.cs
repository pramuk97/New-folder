// DSA Solutions - Binary Tree
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.BinaryTree
{
    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int x) { val = x; }
    }

    // 1. Binary Tree Inorder Traversal
    public class BinaryTreeInorderTraversal
    {
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
        // Explanation: Iterative inorder traversal using stack.
    }

    // 2. Maximum Depth of Binary Tree
    public class MaximumDepthOfBinaryTree
    {
        public int MaxDepth(TreeNode root)
        {
            if (root == null) return 0;
            return 1 + Math.Max(MaxDepth(root.left), MaxDepth(root.right));
        }
        // Explanation: Recursively find max depth of left and right subtrees.
    }

    // 3. Invert Binary Tree
    public class InvertBinaryTree
    {
        public TreeNode InvertTree(TreeNode root)
        {
            if (root == null) return null;
            var left = InvertTree(root.left);
            var right = InvertTree(root.right);
            root.left = right;
            root.right = left;
            return root;
        }
        // Explanation: Recursively swap left and right children.
    }

    // 4. Diameter of Binary Tree
    public class DiameterOfBinaryTree
    {
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
        // Explanation: Track max path between any two nodes.
    }

    // 5. Balanced Binary Tree
    public class BalancedBinaryTree
    {
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
        // Explanation: Check height difference recursively.
    }

    // 6. Path Sum
    public class PathSum
    {
        public bool HasPathSum(TreeNode root, int sum)
        {
            if (root == null) return false;
            if (root.left == null && root.right == null) return root.val == sum;
            return HasPathSum(root.left, sum - root.val) || HasPathSum(root.right, sum - root.val);
        }
        // Explanation: Recursively check for path sum.
    }

    // 7. Lowest Common Ancestor
    public class LowestCommonAncestor
    {
        public TreeNode LowestCommonAncestorMethod(TreeNode root, TreeNode p, TreeNode q)
        {
            if (root == null || root == p || root == q) return root;
            var left = LowestCommonAncestorMethod(root.left, p, q);
            var right = LowestCommonAncestorMethod(root.right, p, q);
            if (left != null && right != null) return root;
            return left ?? right;
        }
        // Explanation: Recursively find LCA in left and right subtrees.
    }

    // 8. Serialize and Deserialize Binary Tree
    public class Codec
    {
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
        // Explanation: Preorder traversal for serialization/deserialization.
    }

    // 9. Construct Binary Tree from Preorder and Inorder
    public class ConstructBinaryTreeFromPreorderInorder
    {
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
        // Explanation: Use preorder/inorder indices and hashmap.
    }

    // 10. Symmetric Tree
    public class SymmetricTree
    {
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
        // Explanation: Recursively check mirror symmetry.
    }
}
