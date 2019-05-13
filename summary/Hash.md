# Hash Review
Hash基本比较简单和直接，不要被2Sum, Longest Consecutive Sequence这样的卡住。

### [Two Sums](https://leetcode.com/problems/two-sum/)

Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

>Example 1:
<code><pre>
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
</code></pre>

>算法：
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
     public int[] twoSum(int[] nums, int target) {

        int[] res = new int[2];
        if (nums.length<1){
            return res;
        }

        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i=0;i<nums.length;i++){
            map.put(nums[i], i);
        }

        for (int i=0;i<nums.length;i++){
            if ((map.keySet().contains(target-nums[i])) && (i!=map.get(target-nums[i]))){
                res[0] = i;
                res[1] = map.get(target-nums[i]);
                return res;
            }
        }
        return res;
    }
}
```
注意自己不能包括自己的边界条件！

#### [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)
Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O(n) complexity.

>Example 1:
<code><pre>
Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
</code></pre>

>算法：
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        int maxLen = 0;
        if ((nums != null) && (nums.length > 0)) {
            maxLen = 1;
            HashSet<Integer> set = new HashSet<>();
            for (int n : nums) {
                set.add(n);
            }
            for (int n : nums) {
                if ((!set.contains(n - 1)) && (set.contains(n + maxLen))) {
                    int delta = 0;
                    while (set.contains(n + (++delta))) ;
                    maxLen = Math.max(maxLen, delta);
                }
            }
        }
        return maxLen;
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