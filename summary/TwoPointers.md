# Pointers for Arrays Review
Two Pointers是线性数据结构常用的技巧，本质是在遍历过程中有左右、快慢等不同的集合选择状态需要记录和比较，以判断是否满足条件。这里只总结一些不太直观的问题，快慢指针这样简单的问题不再总结。

## Sliding Window

### Standard Sliding Window
个人觉得Sliding Window是leetcode最需要练习和总结的问题之一，原理简单但中间状态设计通常不是很直观，如果再加上子问题分解，例如[395](https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/)，会更加不直接。滑动窗口的算法模式很明确，通过缓存[left, right]之间的计算结果，减少[left_next, right_next]的计算代价到常数时间，[这里](https://medium.com/leetcode-patterns/leetcode-pattern-2-sliding-windows-for-strings-e19af105316b)加法的例子就是很好的说明，如何通过滑动窗口把复杂度从$O(n^2)$降低为$O(n)$，因为通常左边去掉和右边加上的开销是常量的。

复杂一些的问题，滑动窗口长度不定，过程可能需要辅助数据结构(keyset, wcTable, remainCount, uniqueCount等)来缓存中间结果。但问题本质其实都是左边减掉、右边增加、通过缓存窗口内的状态来实现$O(n)$复杂度。对于复杂滑动窗口类问题，移动的边界条件和状态的变化最好能事先勾勒出来。

#### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

>Example 1:
<code><pre>
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
</code></pre>

Note:
1. If there is no such window in S that covers all characters in T, return the empty string "".
2. If there is such window, you are guaranteed that there will always be only one unique minimum window in S.

>算法：滑动窗口，先右边移动，判断是否满足匹配条件，如果满足则往左trim()多余的字符，找到一个满足条件的窗口后，比较，把左边往前移动1.

>窗口内状态：
+ `keyset`: 统计是否是目标模式，实现中使用`pattern`数组；
+ `wcTable`: 统计每个char还缺几个；
+ `remainCount`: 记录还剩下需要匹配的distinct字符个数；
```
init keyset, wcTable and reaminCount;

while (right is valid){
    if (s[right] in keyset){

        wcTable[s[right]]--;
        update remainCount if wcTable[s[right]] is 0;

        if (remainCount == 0){
             move left if (s[left] is not in key set) or (wcTable[s[left]] < 0);
                update wcTable[s[left]];
             find a windown here!!
             move left at 1;
        }
    }
    move right at 1
}

```
这个过程的关键，是`keyset`, `wcTable`和`remainCount`三个状态量，移动的过程是先移动`right`，检查是否可以减少`remainCount`，达到匹配条件后移动`left`进行trim.
+ 时间复杂度$O(N)$
+ 空间复杂度$O(N)$

```java
class Solution {
    //2ms, 97.1%
    public String minWindow(String s, String t) {

        //prepare wc table, pattern table and remaining count
        int[] wcTable = new int[256];
        boolean[] pattern = new boolean[256];
        int remainCount = 0;
        for (int c : t.toCharArray()) {
            wcTable[c] += 1;
            pattern[c] = true;
            if (wcTable[c] == 1) {
                remainCount++;
            }
        }

        //sliding right and trim left
        int left = 0, right = 0;
        int resStart = 0, resEnd = 0, minLen = Integer.MAX_VALUE;
        char[] str = s.toCharArray();
        while (right < s.length()) {
            if (pattern[str[right]]) {

                wcTable[str[right]] -= 1;
                if (wcTable[str[right]] == 0) {
                    remainCount--;
                    if (remainCount == 0) {
                        //find all letters, now we trim left
                        while ((!pattern[str[left]]) || (wcTable[str[left]] < 0)) {
                            if (wcTable[str[left]] < 0) {
                                wcTable[str[left]] += 1;
                            }
                            left++;
                        }
                        //find a window
                        if (right - left < minLen) {
                            resStart = left;
                            resEnd = right + 1; // then no need to assert if matched (e.g., s="a", p="a")
                            minLen = right - left;
                        }
                        //slide left on
                        wcTable[str[left]] += 1;
                        remainCount++;
                        left++;
                    }
                }
            }
            right++;
        }
        return s.substring(resStart, resEnd);
    }
}
```
滑动窗口实现常用的技巧是用'wcTable'配合`remainCount`在判断是否满足条件，例如初始化的时候`wcTable[i]==1`则`remainCount++`, 滑动窗口过程中`wcTable[i]==0`则`remainCount--`等。

#### [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.

>Example 1:
<code><pre>
Input:
s: "cbaebabacd" p: "abc"
Output:
[0, 6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
</code></pre>

>Example 2:
<code><pre>
Input:
s: "abab" p: "ab"
Output:
[0, 1, 2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
</code></pre>

>算法：滑动窗口，固定窗口大小为`p.length()`，滑动过程中判断是否满足条件，如果满足则记录`left`到`res`集合。

>窗口内状态：
+ `keyset`记录pattern的字符集，实现中用`pattern`数组；
+ `wcTable`记录对于每个字符，还剩几个字符能匹配pattern；
+ `remainCount`记录还剩几个distinct的字符；

复杂度
+ 时间复杂度$O(N)$
+ 空间复杂度$O(N)$

```java
class Solution {
    //5ms, 99.62%
    public List<Integer> findAnagrams(String s, String p) {

        ArrayList<Integer> res = new ArrayList<>();
        //prepare wc table, pattern table and remaining count
        int[] wcTable = new int[256];
        boolean[] pattern = new boolean[256];
        int remainCount = 0;
        for (int c : p.toCharArray()) {
            wcTable[c] += 1;
            pattern[c] = true;
            if (wcTable[c] == 1) {
                remainCount++;
            }
        }

        char[] str = s.toCharArray();
        int left = -p.length();
        for (int right = 0; right <= s.length(); right++, left++) {
            if (left >= 0) {
                //find a solution
                if (remainCount == 0) {
                    res.add(left);
                }
                //remove left
                if (pattern[str[left]]) {
                    wcTable[str[left]]++;
                    if (wcTable[str[left]] == 1){
                        remainCount++;
                    }
                }
            }
            //add right
            if ((right < s.length()) && (pattern[str[right]])) {
                wcTable[str[right]]--;
                if (wcTable[str[right]]==0){
                    remainCount--;
                }
            }
        }
        return res;
    }

}
```
和第一个题目几乎完全一样，用`pattern`来存`keyset`, 单独维护一个`remainCount`来和`wcTable`配合一起判断是否满足匹配条件。

#### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
Given a string, find the length of the longest substring without repeating characters.

>Example 1:
<code><pre>
Input: "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
</code></pre>

>Example 2:
<code><pre>
Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
</code></pre>

>Example 3:
<code><pre>
Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
             Note that the answer must be a substring, "pwke" is a subsequence and not a
</code></pre>

>算法：滑动窗口，如果发现有重复数据，left移动到重复的地方
+ 时间复杂度$O(N)$
+ 空间复杂度$O(N)$

```java
class Solution {

    //2ms, 99.9%
    public int lengthOfLongestSubstring(String s) {
        int res = 0;
        if (s == null) {
            return res;
        }

        char[] str = s.toCharArray();
        int[] repeatPos = new int[256];
        int left = 0;
        for (int right = 0; right < str.length; right++) {

            if (left < repeatPos[str[right]]) {
                left = repeatPos[str[right]];
            }
            res = Math.max(res, right-left+1);
            repeatPos[str[right]] = right + 1;
        }
        return res;
    }
}
```
这里有个小的trick,如果左边比新进来比较的c位置小（证明新进来的数已经放入过repeatPos，那么left直接移动到重复数据后面。如果这个不太直观，用一个`boolean[]`作为哈希表也可以。

#### [395. Longest Substring with At Least K Repeating Characters](https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/)
Find the length of the longest substring T of a given string (consists of lowercase letters only) such that every character in T appears no less than k times.

>Example 1:
<code><pre>
Input:
s = "aaabb", k = 3
Output:
3
The longest substring is "aaa", as 'a' is repeated 3 times.
</code></pre>

>Example 2:
<code><pre>
Input:
s = "ababbc", k = 2
Output:
5
The longest substring is "ababb", as 'a' is repeated 2 times and 'b' is repeated 3 times.
</code></pre>
>算法：滑动窗口，用最大的不同char个数(1-26)来控制窗口左边滑动，这里不太直接的是通过unique来拆分问题，变成26个滑动窗口子问题。

> 窗口内状态:
+ `wcTable`：窗口内的字符计数；
+ `remainCount`：窗口内满足条件的剩余distict字符；
+ `uniqueCount`：本子问题最大distict的字符数，用来移动`left`以实现滑动窗口。

复杂度
+ 时间复杂度$O(26N)=O(N)$
+ 空间复杂度$O(1)$

```java
class Solution {

    //4ms, 62.61%
    public int longestSubstring(String s, int k) {

        int res = 0;
        if ((s == null) || (s.length() < 1) || (k < 0)) {
            return res;
        }
        char[] str = s.toCharArray();
        for (int uMax = 1; uMax <= 26; uMax++) {
            int left = 0;
            int right = 0;
            int[] wcTable = new int[26];
            Arrays.fill(wcTable, 0);
            int uniqueCount = 0;
            int remainCount = 0;
            while (right < str.length) {
                if (uniqueCount <= uMax) {
                    //move right
                    int index = str[right] - 'a';
                    wcTable[index]++;
                    if (wcTable[index] == 1) {
                        uniqueCount++;
                        remainCount++;
                    }
                    if (wcTable[index] == k) {
                        remainCount--;
                    }
                    right++;

                } else {
                    //move left
                    int index = str[left] - 'a';
                    if (wcTable[index] == k) {
                        remainCount++;
                    }
                    wcTable[index]--;
                    if (wcTable[index] == 0) {
                        uniqueCount--;
                        remainCount--;
                    }
                    left++;
                }

                if ((remainCount == 0)) {
                    res = Math.max(res, right - left);
                }
            }
        }
        return res;
    }
}
```
这个问题如果之前没有见过，用26个unique上限来分解为滑动窗口子问题可能不太容易想到，最差情况可以用暴力方法Two Pointers，复杂度是$O(n^2)$，只要用`wcTable`和`remainCount`来判断模式匹配条件是否满足，还是可以AC的。

### Sliding Window + Monotonic Queue/Stack
[Monotonic Queue](https://medium.com/algorithms-and-leetcode/monotonic-queue-explained-with-leetcode-problems-7db7c530c1d6)用来解决滑动窗口max/min的问题，复杂度是$O(n)$。单调队列的基本原理是右边进、左边出，进去的时候如果有在其前面肯定不会被max/min访问到的，则先把那些没用的干掉，使得整个队列保持单调。

#### [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

>Example 1:
<code><pre>
Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3
Output: [3,3,5,5,6,7]
Explanation:
Window position                Max
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
</code></pre>
Note:
You may assume k is always valid, 1 ≤ k ≤ input array's size for non-empty array.

>算法：使用memotonic queue, 如果左大右小，那么右边进（进的时候把比自己小的都干掉，因为在自己前面不能被getMax了)，左边出，出来的时候如果没被干掉，getMax会是自己。

复杂度：
+ 时间复杂度$O(N)$
+ 空间复杂度$O(N)$

```java
class Solution {

    //5ms, 90.1%
    class MonotonicQueue {

        LinkedList<Integer> deque = new LinkedList<>();

        public void offer(int e) {
            while (!deque.isEmpty() && (deque.getLast() < e)) {
                deque.removeLast();
            }
            deque.addLast(e);
        }

        public void poll(int e) {
            if (this.deque.getFirst() == e) {
                this.deque.removeFirst(); // it can only be used in limited case such as sliding window
            }
        }

        public int getMax() {
            return this.deque.getFirst();
        }
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0) {
            return nums;
        }

        int[] res = new int[nums.length - k + 1];
        MonotonicQueue mq = new MonotonicQueue();
        for (int left = -k, right = 0; right <= nums.length; right++, left++) {
            if (left >= 0) {
                res[left] = mq.getMax();
                mq.poll(nums[left]);
            }
            if (right<nums.length){
                mq.offer(nums[right]);
            }
        }
        return res;
    }

}
```
用`ArrayList`实现deque速度略快。
```java
class MonotonicQueue {

        ArrayList<Integer> deque = new ArrayList<>();

        public void offer(int e) {
            while (!deque.isEmpty() && (deque.get(deque.size()-1) < e)) {
                deque.remove(deque.size()-1);
            }
           deque.add(deque.size(), e);
        }

        public void poll(int e) {
            if (this.deque.get(0) == e) {
                this.deque.remove(0); // it can be used in limited case such as sliding window
            }
        }

        public int getMax() {
            return this.deque.get(0);
        }
}
```

#### [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.

>Example 1:
<code><pre>
Example:
Input: [2,1,5,6,2,3]
Output: 10
</code></pre>

>算法：类似monotic queue, 维护monotic stack遇到大的放到stack，直到遇到小的right，逐个从stack中弹出比right大的，计算可能的R，最后一个放回stack，其高度更新为right；
这个过程中保证stack中的所有元素的单调递增的。
+ 时间复杂度$O(N)$
+ 空间复杂度$O(N)$

```java
class Solution {
    //4ms, 87.5%
    public int largestRectangleArea(int[] heights) {
        int res = 0, leftPos = 0, height;
        if (heights.length > 0) {
            LinkedList<Integer> monoticStack = new LinkedList<>();
            monoticStack.addLast(0);

            for (int i = 1; i <= heights.length; i++) {
                height = (i == heights.length) ? 0 : heights[i];
                if (height > heights[i - 1]) {
                    monoticStack.addLast(i);
                } else if (height < heights[i - 1]) {
                    while ((monoticStack.size() > 0) && (heights[monoticStack.getLast()] >= height)) {
                        leftPos = monoticStack.removeLast();
                        res = Math.max(res, (i - leftPos) * (heights[leftPos]));
                    }
                    monoticStack.addLast(leftPos);
                    heights[leftPos] = height; ////update height to ensure mononic
                }//do nothing when ==
            }
        }
        return res;
    }
}
```
这里注意每次满足条件计算后，需要把左边height更新` heights[leftPos] = height`。

## Fast and Slow


## Other Two Pointers
Two Pointer通常用两个指针来记录处理的状态（当然也可能是三个指针或多个指针）。注意Sliding Window和快慢指针都可以看成是特殊的Two Pointers，我们在前面已经讨论过了，这里介绍其他常见的Two Pointers问题。

#### [15. 3Sum](https://leetcode.com/problems/3sum/)
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note:

The solution set must not contain duplicate triplets.

>Example 1:
<code><pre>
Given array nums = [-1, 0, 1, 2, -1, -4],
A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
</code></pre>

>算法：排序，然后从左往右逐一先固定起点(base)，后用left和right向中间逼近满足条件的解。
+ 时间复杂度$O(N^2)$
+ 空间复杂度$O(1)$

```java
class Solution {


}
```
此外，可以基于[1. 2SUMS](https://leetcode.com/problems/two-sum/)用哈希。


#### [283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)
Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

>Example 1:
<code><pre>
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
</code></pre>
You must do this in-place without making a copy of the array.
Minimize the total number of operations.

>算法：left和right两个指针, left指向下一个被向右交换的0的位置，right指向当前已经整理的位置。
+ 时间复杂度$O(N)$
+ 空间复杂度$O(N)$

```java
class Solution {
    //0ms, 100%
    public void moveZeroes(int[] nums) {
        if ((nums != null) && (nums.length > 0)) {
            int left = 0;
            int right = 0;
            while (right < nums.length){
                if (nums[right] !=0 ){
                    if (left < right){
                        nums[left] = nums[right];
                        nums[right] = 0;
                    }
                    left++;
                }
                right++;
            }
        }
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