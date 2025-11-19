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
