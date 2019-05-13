# Stack Review
栈的特性是先进后出，使用栈都是利用这个顺序来辅助计算。一些可以用栈的数组问题，通常可以先画折线图，仔细观察并找出规律。利用栈进行递归，放在递归迭代实现里面总结，不在本章包括。

## 数值趋势类问题
### [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures `T = [73, 74, 75, 71, 69, 72, 76, 73]`, your output should be `[1, 1, 4, 2, 1, 1, 0, 0]`.

Note: The length of temperatures will be in the range `[1, 30000]`. Each temperature will be an integer in the range `[30, 100]`.


>算法：把温度画出来，发现对于每个元素
        如果T[n]>T[n-1]，则一直往栈里面找比T[n]小的pos，找到了res[pos]=n-pos；
        把`T[n]`加入stack；
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {

    public int[] dailyTemperatures(int[] T) {

        if ((T == null) || (T.length < 1)) {
            return T;
        }

        int[] res = new int[T.length];
        LinkedList<Integer> stack = new LinkedList<>();
        stack.addLast(0); // pos not value
        for (int pos = 1; pos < T.length; pos++) {
            while ((!stack.isEmpty()) && (T[pos] > T[stack.getLast()])) {
                int colderPos = stack.removeLast();
                res[colderPos] = pos - colderPos;
            }
            stack.addLast(pos);
        }
        return res;
    }


}
```

## Template
### []()


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