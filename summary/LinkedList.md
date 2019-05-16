# Linked List Review
链表结构通常想清楚current = current.next怎么往下走即可，场景技巧
1. Dummy Head，在in-place merge, 删除第一个节点等操作的时候有用。
2. 快慢指针。
3. 通过拷贝来"伪装"其他操作，因为没有反向的指针。


#### [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

>Example 1:
<code><pre>
Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
</code></pre>

>算法：
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {

        ListNode dummy = new ListNode(0);
        ListNode current = dummy;

        while ((l1 != null) || (l2 != null)) {
            if ((l1 != null) && (l2 != null)) {
                if (l1.val < l2.val) {
                    current.next = l1;
                    l1 = l1.next;
                } else {
                    current.next = l2;
                    l2 = l2.next;
                }
                current = current.next;
            } else if (l1 == null) {
                current.next = l2;
                break;
            } else {
                current.next = l1;
                break;
            }
        }
        return dummy.next;
    }
}
```
注意dummy node和判断条件。

#### [Sort List](https://leetcode.com/problems/sort-list/)
Sort a linked list in O(n log n) time using constant space complexity.

>Example 1:
<code><pre>
Input: 4->2->1->3
Output: 1->2->3->4
</code></pre>

>算法：
+ 时间复杂度O(NlogN)
+ 空间复杂度O(NlogN)

```java
class Solution {

     public ListNode sortList(ListNode head) {
        ListNode res = null;
        if (head != null) {
            res = this.mergeSort(head);
        }
        return res;

    }

     private ListNode mergeSort(ListNode head) {
        if (head.next==null){
            return head; // one element left
        }

        ListNode mid = head;
        ListNode fast = head;
        while ((fast.next != null) && (fast.next.next != null)) {
            mid = mid.next;
            fast = fast.next.next;
        }
        ListNode pre = mid;
        mid = mid.next;
        pre.next = null;
        ListNode sortedHead = this.mergeSort(head);
        ListNode sortedMid = this.mergeSort(mid);
        return this.mergeTwoLists(sortedHead, sortedMid);
    }

    private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // no need to check null
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;

        while ((l1 != null) || (l2 != null)) {

            if ((l1 != null) && (l2 != null)) {
                if (l1.val < l2.val) {
                    current.next = l1;
                    l1 = l1.next;
                } else {
                    current.next = l2;
                    l2 = l2.next;
                }
                current = current.next;
            } else if (l1 == null) {
                current.next = l2;
                break;
            } else {
                current.next = l1;
                break;
            }
        }
        return dummy.next;
    }
}
```
#### [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

>Example 1:
<code><pre>
Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.
</code></pre>

>算法：
1. 每个节点后面复制一个｀n'｀；
2. 给｀n'｀加上random（第一步的目的就是为了这一步可以用｀current.next.random = current.random.next｀来复制random）；
3. 把n, n'拆开还原；

+ 时间复杂度O(N)
+ 空间复杂度O(1)，不计copy

```java
class Solution {
    //0ms, 100%
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node current;
        Node temp;

        //make copy as interleaving
        current = head;
        while (current != null) {
            temp = current.next;
            current.next = new Node();
            current.next.val = current.val;
            current.next.next = temp; // may be null
            current = temp; //current = current.next.next;
        }

        //handle random for copy nodes
        current = head;
        while (current != null) {
            if (current.random != null) {
                current.next.random = current.random.next;
            }
            current = current.next.next; // current.next should not be null due to step 1.
        }

        //decouple the copy
        Node res = head.next;
        current = head;
        while ((current.next != null) && (current.next.next != null)) {
            temp = current.next;
            current.next = current.next.next;
            temp.next = temp.next.next;
            current = current.next;
        }
        current.next = null;
        return res;
    }
}
```
这个题目对于如何把链表实现的concise（记住实际的状态和情况，忽略一定不会出现的情况）有参考价值，实现不要需要额外多加不少判断条件。

### Template
#### []()


>Example 1:
<code><pre>

</code></pre>

>算法：
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    
   
}
```
