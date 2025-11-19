// DSA Solutions - Linked List
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.LinkedList
{
    // 1. Reverse Linked List
    public class ReverseLinkedList
    {
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
        // Explanation: Iteratively reverse pointers.
    }

    // 2. Merge Two Sorted Lists
    public class MergeTwoSortedLists
    {
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
        // Explanation: Merge by comparing nodes.
    }

    // 3. Linked List Cycle
    public class LinkedListCycle
    {
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
        // Explanation: Floyd's Tortoise and Hare algorithm.
    }

    // 4. Remove Nth Node From End of List
    public class RemoveNthNodeFromEndOfList
    {
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
        // Explanation: Two pointer technique.
    }

    // 5. Intersection of Two Linked Lists
    public class IntersectionOfTwoLinkedLists
    {
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
        // Explanation: Traverse both lists, switch heads when reaching end.
    }

    // 6. Add Two Numbers
    public class AddTwoNumbers
    {
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
        // Explanation: Add digits, handle carry.
    }

    // 7. Palindrome Linked List
    public class PalindromeLinkedList
    {
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
        // Explanation: Find middle, reverse second half, compare.
    }

    // 8. Copy List with Random Pointer
    public class CopyListWithRandomPointer
    {
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
        // Explanation: Use a map to copy nodes and pointers.
    }

    // 9. Reorder List
    public class ReorderList
    {
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
        // Explanation: Split, reverse second half, merge alternately.
    }

    // 10. Flatten a Multilevel Doubly Linked List
    public class FlattenMultilevelDoublyLinkedList
    {
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
        // Explanation: Use stack to process child nodes.
    }

    // ListNode and Node class definitions
    public class ListNode
    {
        public int val;
        public ListNode next;
        public ListNode(int x) { val = x; }
    }
    public class Node
    {
        public int val;
        public Node next;
        public Node random;
        public Node child;
        public Node prev;
        public Node(int x) { val = x; }
    }
}
