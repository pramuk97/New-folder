// DSA Solutions - Binary Search Tree
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.BinarySearchTree
{
    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int x) { val = x; }
    }

    // 1. Validate Binary Search Tree
    public class ValidateBinarySearchTree
    {
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
        // Explanation: Recursively check value bounds.
    }

    // 2. Insert into a BST
    public class InsertIntoBST
    {
        public TreeNode Insert(TreeNode root, int val)
        {
            if (root == null) return new TreeNode(val);
            if (val < root.val) root.left = Insert(root.left, val);
            else root.right = Insert(root.right, val);
            return root;
        }
        // Explanation: Recursively insert value.
    }

    // 3. Delete Node in a BST
    public class DeleteNodeInBST
    {
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
        // Explanation: Handle three cases, use min node for replacement.
    }

    // 4. Lowest Common Ancestor of a BST
    public class LowestCommonAncestorBST
    {
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
        // Explanation: Use BST property to find LCA.
    }

    // 5. Kth Smallest Element in a BST
    public class KthSmallestElementInBST
    {
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
        // Explanation: Inorder traversal to find kth smallest.
    }

    // 6. BST Iterator
    public class BSTIterator
    {
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
        // Explanation: Use stack to simulate inorder traversal.
    }

    // 7. Convert Sorted Array to BST
    public class ConvertSortedArrayToBST
    {
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
        // Explanation: Recursively build BST from sorted array.
    }

    // 8. Range Sum of BST
    public class RangeSumOfBST
    {
        public int RangeSumBST(TreeNode root, int low, int high)
        {
            if (root == null) return 0;
            if (root.val < low) return RangeSumBST(root.right, low, high);
            if (root.val > high) return RangeSumBST(root.left, low, high);
            return root.val + RangeSumBST(root.left, low, high) + RangeSumBST(root.right, low, high);
        }
        // Explanation: Use BST property to sum values in range.
    }

    // 9. Trim a BST
    public class TrimBST
    {
        public TreeNode TrimBSTMethod(TreeNode root, int low, int high)
        {
            if (root == null) return null;
            if (root.val < low) return TrimBSTMethod(root.right, low, high);
            if (root.val > high) return TrimBSTMethod(root.left, low, high);
            root.left = TrimBSTMethod(root.left, low, high);
            root.right = TrimBSTMethod(root.right, low, high);
            return root;
        }
        // Explanation: Recursively trim nodes outside range.
    }

    // 10. Recover Binary Search Tree
    public class RecoverBinarySearchTree
    {
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
        // Explanation: Inorder traversal to find swapped nodes.
    }
}
