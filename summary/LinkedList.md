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