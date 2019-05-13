[TOC]

# Dynamic Programming Review
## Summary
[这里](https://www.cs.cmu.edu/~avrim/451f09/lectures/lect1001.pdf)是对动态规划的一个简介，核心思想是通过子问题的分解，重复的子问题只需要计算一次，进而把原本指数的复杂度降低为多项式时间，例如，对2D降低为O(mn)。子问题的最优解是可以被重复利用的，只用计算一次。因此，通常这个过程需要缓存子问题的状态（中间结果），缓存的状态个数和子问题数量相关（通常和时间复杂度也相关），因此这个分解过程是在用空间换时间。

因此，是否能使用动态规划通常有两个条件
+ 问题n能分解为n-1, n-2..等子问题；
+ 子问题的空间是有限且大量重复，通常n对应常数个子问题，或子问题空间是多项式时间的。

动态规划虽灵活多变，但核心是子问题分解+状态表设计（类似，backtracking抓住递归树和select状态剩下就是套路了），“以终为始”（可以从结果去反着思考退一步怎么去分解），当n步问题被分解表示为n-1, n-2, ...等子问题的组合，问题分解完成，进而能完成中间状态量的设计。剩下的事情手到擒来。

DP具体实现方式:
+ bottle-up迭代: 按1, 2, ...n的顺序计算，因为计算n的时候，依赖的前序子问题已经求解，所以比较直接，复杂度是多项式时间的，等于子问题的个数。
+ top-down递归: 按n, n-1, ...1的顺序计算，这里利用hash或者数组，在递归的过程中把地k步的结果缓存下来，如果以后要使用，一旦缓存有就不用重复去递归了，这样的top-down递归+缓存也称为memoizing（因此top-down dp也称为recursion with memoization）。如果没有缓存，递归的复杂度通常是指数的，但加入缓存，复杂度降为和bottle-up一样的多项式时间（等于子问题个数）。
>注: 
1. 两种方法的时间复杂度是一样的，递归多了递归栈的开销，好处是对于一些问题可能可以跳过一些sub-problems，在实际问题中哪个更直接就用哪个，区别并不大。
2. 对于memo的赋值，都是对于`momo[n]`递归完成后，对`n`赋值，不会是对`n-1`进行多个条件的赋值，参考Word Break, Target Sum. 
3. 有些不太直接的推导，“到当前位置”，值得尝试。

## 1-D
缓存设计:
+ 类型1：dp[]数组索引是问题空间， 元素值是子问题解空间，例如：Claiming Stairs, Stock Buy and Sale, etc.
+ 类型2：dp[]数组索引是约束条件，元素值是问题解空间，这类问题约束条件通常有明确的bound， 例如：Change Coins. 

注：
1. 有时候dp[]可以被优化为一两个变量O(1)，但是建议先写出dp[]的公式，再尝试优化，优化过程比较容易出错。
2. 类型1, 2的索引加一起组合，通常就是2D的动态规划，类型1+类型1例如UniquePaths, 类型1＋类型2例如背包问题。

1-D动态规划模板:
动态规划关键是状态的定义，本身并不需要什么模板。分析清楚状态转移，实现就是递归或遍历，另外注意初始条件的表达即可。

### 类型1：`dp[]`数组以步骤为索引，最优值（或相关状态）为值。
这类问题相对比较直接，识别清楚问题，推导出`dp[n]`怎么用`dp[n-1]`分情况表达即可。但对于数组，可能会被Pointer混淆。其实区别比较明显，Pointer通常需要看的是左边很多个元素，而dp只会看左边常数（通常就是1）个元素，分情况表达。

#### [Climbing Charis](https://leetcode.com/problems/climbing-stairs/)
You are climbing a stair case. It takes n steps to reach to the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.
>Example: 
<code><pre>
Input: 2
Output: 2
Explanation: There are two ways to climb to the top:1) 1 step + 1 step, 2) 2 steps
</code></pre>

>算法：爬楼梯是动态规划教科书第一题，第n格的爬法由n-1和n-2决定

>状态递推:
<pre><code>
dp[n] = dp[n-1] +dp[n-2]
n：第n格
dp[n]：第n格的爬法
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(n)，也可以是O(1)，只是为了易读性。

bottle-up
```java
public class Solution{
    public int climbStairs(int n) {
        if (n <= 1) {
            return n;
        }
        int[] dp = new int[n+1];
        dp[1] = 1;
        dp[2] = 2;
        for (int pos = 3; pos <= n; pos++){
            dp[pos] = dp[pos-1] + dp[pos-2];
        }
        return dp[n];
    }
}
```
top-down
```java
public class Solution{
        public int climbStairs(int n) {
            if (n <= 1) {
                return n;
            }
            int[] memo = new int[n+1];
            return this.helper(memo, n);
        }
    
        private int helper(int[] memo, int n){
            if (memo[n]!=0){
                return memo[n];
            }
            if ((n==1) || (n==2)){
                return n;
            }
            memo[n] = this.helper(memo, n-1)+this.helper(memo, n-2);
            return memo[n];
        }
}
```
#### [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
>Example: 
<code><pre>
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
</code></pre>

>算法：逐个统计到当前位置的max subarray

>状态递推:
<pre><code>
dp[n] = max(nums[n], dp[n-1])
n：数组第n位
dp[n]：以第n位为结尾的maxSubArray的sum
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(1)

bottle-up
```java
public class Solution{
    public int maxSubArray(int[] nums) {

        if ((nums == null) || (nums.length < 1)) {
            return 0;
        }

        int maxToCurrent = nums[0];
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            maxToCurrent = Math.max(nums[i], maxToCurrent + nums[i]);
            res = Math.max(res, maxToCurrent);
        }
        return res;
    }
}
```


#### [Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

>Example 1: 
<code><pre>
Input: "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()"
</code></pre>


>Example 2: 
<code><pre>
Input: ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()"
</code></pre>

>算法：逐个统计到当前位置的max valid ()

>状态递推:
<pre><code>
if s.charAt(n) == '(' 
    dp[n] = 0
else 
    if left没有'('
        dp[n]=0;
    else
        往回从n-1开始，根据dp[i]找到匹配'('的位置 
n：字符串数组第n位
dp[n]：以第n位为结尾的max valid ()
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(n)

bottle-up
```java
public class Solution{
    public int longestValidParentheses(String s) {

        int res = 0;
        if ((s == null) || (s.length() < 1)) {
            return res;
        }

        int left = 0;
        int dp[] = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                dp[i] = 0;
                left++;
            } else {
                if (left > 0) {
                    int findLeft = i - 1;
                    while (s.charAt(findLeft) == ')') {
                        findLeft -= dp[findLeft];
                    }
                    dp[i] = (i - findLeft + 1)
                            + ((findLeft >= 1) ? dp[findLeft - 1] : 0);
                    res = Math.max(dp[i], res);
                    left--;
                } else {
                    dp[i] = 0; // not valid
                }
            }
        }
        return res;
    }
}
```
这个和上面一个Maximum Subarray本质类似，都是“到当前位置"。有意思的是，记录`dp[n]`和`left`的个数，就能够一趟完成求解。

#### [House Robber](https://leetcode.com/problems/house-robber/)
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

>Example 1: 
<code><pre>
Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
</code></pre>

>Example 2: 
<code><pre>
Input: [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
             Total amount you can rob = 2 + 9 + 1 = 12.
</code></pre>

Note:You may assume that you have an infinite number of each kind of coin.

>算法：用rob对象索引为dp数组索引，则第n个对象可以由第n-2个（rob）和n-1个（not rob）凑得。

>状态递推:
<pre><code>
dp[n] =　max(dp[n-2]+nums[n], dp[n-1])
n：rob对象索引
dp[n]：到第n个对象位置，最大rob值
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(n)，可以优化为O(1)

bottle-up
```java
public class Solution{
    public int rob(int[] nums) {
        if ((nums == null) || (nums.length<1)){
            return 0;
        }

        int dp[] = new int[nums.length];
        dp[0] = nums[0];
        if (nums.length>1){
            dp[1] = Math.max(dp[0], nums[1]);
            for (int i=2;i<nums.length;i++){
                dp[i] = Math.max(dp[i-2]+nums[i], dp[i-1]);
            }
        }
        return dp[nums.length-1];
    }
}
```

#### [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
Given an unsorted array of integers, find the length of longest increasing subsequence.

>Example 1: 
<code><pre>
Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
</code></pre>

Note:
1. There may be more than one LIS combination, it is only necessary for you to return the length.
2. Your algorithm should run in `O(n^2)` complexity.

>算法：对于第n位，向前每一位j，对于所有比其小的，都dp[j]+1

>状态递推:
<pre><code>
dp[n] =　max(dp[n-1]+1, dp[n-2])+1, ..., dp[0]+1), if dp[n] > dp[j]
n：序列索引
dp[n]：到第n个对象位置，LIS
</code></pre>
+ 时间复杂度O(n^2)
+ 空间复杂度O(n)

bottle-up
```java
public class Solution{
   
    public int lengthOfLIS(int[] nums) {
        if ((nums == null) || (nums.length == 0)) {
            return 0;
        }

        int lisLen = 1;
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            lisLen = Math.max(lisLen, dp[i]);
        }
        return lisLen;
    }
}
```

#### [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.
>Example: 
<code><pre>
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
</code></pre>

>算法：逐个统计到当前位置的最低买入价

>状态递推:
<pre><code>
dp[n] = min(dp[n-1], nums[n])
n：第n天
dp[n]：到第n天为止的最低买入价
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(1)

bottle-up
```java
public class Solution{
    public int maxProfit(int[] prices) {
        int res = 0;
        if (prices.length < 1) {
            return res;
        }

        int minToNow = prices[0];
        for (int i = 1; i < prices.length; i++) {
            res = Math.max(res, prices[i] - minToNow);
            minToNow = Math.min(minToNow, prices[i]);
        }
        return res;
    }
}
```

#### [Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)

>Example: 
<code><pre>
Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]
</code></pre>

>算法：
存储到目前为止，如果最后操作是buy，最大盈利；如果最后操作是sell，最大盈利。那么当前的sell和buy最大盈利可以由前序两个状态推出。

>状态递推:
<pre><code>
dpBuy[i] = Math.max(dpBuy[i - 1], dpSell[i - 2] - prices[i]);
dpSell[i] = Math.max(dpSell[i - 1], dpBuy[i - 1] + prices[i]);
n：第n天
dp[n]：
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(1)，优化后

bottle-up
```java
public class Solution{
    public int maxProfit(int[] prices) {
        if ((prices == null) || (prices.length < 1)) {
            return 0;
        }
        int[] dpBuy = new int[prices.length];
        int[] dpSell = new int[prices.length];
        dpBuy[0] = -prices[0];
        dpSell[0] = 0;
        if (prices.length > 1) {
            dpBuy[1] = Math.max(dpBuy[0], -prices[1]);
            dpSell[1] = Math.max(dpSell[0], prices[1] - prices[0]);
        }
        for (int i = 2; i < prices.length; i++) {
            dpBuy[i] = Math.max(dpBuy[i - 1], dpSell[i - 2] - prices[i]);
            dpSell[i] = Math.max(dpSell[i - 1], dpBuy[i - 1] + prices[i]);
        }
        return Math.max(dpBuy[dpBuy.length-1], dpSell[dpSell.length-1]);
    }
}
```
注：这个和Maximum Product类似，都是需要缓存两个状态，空间O(1)只是实现上的优化。

#### [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)
Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.
>Example 1: 
<code><pre>
Input: [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
</code></pre>

>Example 2: 
<code><pre>
Input: [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
</code></pre>

>算法：统计到当前位置的最大和最小，作为下一个位置的计算依据。

>状态递推:
<pre><code>
dpMax[n] = max(dpMin[n-1]*nums[n], dpMax[n-1]*nums[n], nums[n])
dpMin[n] = min(dpMin[n-1]*nums[n], dpMax[n-1]*nums[n], nums[n])
n：第n位
dp[n]：以第n位为结束，子array的最大和最小乘积。类似Maximum Subarray的子问题拆解方法。
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(1)

>注：对于O(1)空间复杂度，不用dp[]来存状态，要注意中间变量的修改，比如这个例子求dpMax的时候dpMax已经变化，要是再拿来求dpMin就错啦。

bottle-up
```java
public class Solution{
    public int maxProduct(int[] nums) {
        int res = Integer.MIN_VALUE;
        if (nums.length > 0) {
            int maxToCurrent = nums[0];
            int minToCurrent = nums[0];
            int product1, product2;
            res = maxToCurrent;
            for (int i=1;i<nums.length;i++){
                product1 = maxToCurrent*nums[i];
                product2 = minToCurrent*nums[i];
                maxToCurrent = Math.max(nums[i], Math.max(product1, product2));
                minToCurrent = Math.min(nums[i], Math.min(product1, product2));
                res = Math.max(res, maxToCurrent);
           }
        }
        return res;
    }
}
```
#### [Counting Bits](https://leetcode.com/problems/counting-bits/)
Given a non negative integer number num. For every numbers i in the range `0 ≤ i ≤ num` calculate the number of 1's in their binary representation and return them as an array.

>Example 1: 
<code><pre>
Input: 2
Output: [0,1,1]
</code></pre>

>Example 2: 
<code><pre>
Input: 5
Output: [0,1,1,2,1,2]
</code></pre>

>算法： 观察bit, 位数, count，每增加一位，回到`dp[0]+1`

>状态递推:
<pre><code>
     num  bit   digits  '1'count
     0    0       1       0
     1    1       1       1
     2    10      2       1
     3    11      2       2
     4    100     3       1
     5    101     3       2
     6    110     3       2
     7    111     3       3
     8    1000    4       1
     9    1001    4       2
     10   1010    4       2
     11   1011    4       3
     12   1100    4       2
     13   1101    4       3
     14   1110    4       3
     15   1111    4       4
n：第n位
dp[n]：n的二进制
每增加一位二进制，后面全部清理，回到dp[0]
因此，dp[i] = dp[prev] +1, prev在增加位数的时候重置为0
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(n)

bottle-up
```java
public class CountingBits {
    public int[] countBits(int num) {
        if (num < 1) {
            return new int[]{0};
        }
        int[] dp = new int[num + 1];
        dp[0] = 0;
        dp[1] = 1;
        int digit = 2;
        int prev = 0;
        for (int i = 2; i <= num; i++) {
            dp[i] = dp[prev++] + 1;
            if (dp[i] == digit) {
                prev = 0;
                digit++;
            }
        }
        return dp;
    }
}
```

#### [Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)
Given `n`, how many structurally unique BST's (binary search trees) that store values `1 ... n`?

>Example 1: 
<code><pre>
Input: 3
Output: 5
Explanation:
Given n = 3, there are a total of 5 unique BST's
</code></pre>

>算法：假设n的唯一二叉树为`dp[n]`, 递推，我们3为例, {1, 2, 3}:
1. 取1，则左边没有，右边是{2,3}其dp等同于{1,2}都是`dp[2]`假设`dp[0]=1`，则`dp[0]*dp[2]`；
2. 同理，取2, 为`dp[1]*dp[1]`；
3. 取3, 为`dp[2]*dp[1]`；
因此, `dp[3]=dp[0]*dp[2]+dp[1]*dp[1]+dp[2]*dp[0]`。

>状态递推:
<pre><code>
dp[n]=dp[0]*dp[n-1]+dp[1]*dp[n-2]+....+dp[n-1]*dp[0]
n：第n位
dp[n]：n的二叉树个数
</code></pre>
+ 时间复杂度O(n^2)
+ 空间复杂度O(n)

bottle-up
```java
public class UniqueBinarySearchTrees {
    public int numTrees(int n) {
        if (n < 1) {
            return 0;
        }

        int[] dp = new int[n + 2];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        return dp[n];
    }
}
```

#### [Decode Ways](https://leetcode.com/problems/decode-ways/)
A message containing letters from A-Z is being encoded to numbers using the following mapping:
```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```
Given a non-empty string containing only digits, determine the total number of ways to decode it.

>Example 1:
<code><pre>
Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).
</code></pre>

>Example 2:
<code><pre>
Input: "226"
Output: 3
Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
</code></pre>

>算法：动态规划，`dp[n]`由`dp[n-1]`和`dp[n-2]`得出，如果对应取值合法。

>状态递推:
<pre><code>
dp[n] = dp[n-1] + dp[n-2] if substr(i,n) is valid
n：到s第n位
dp[n]：到第n位位置的合法组合个数
</code></pre>
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public int numDecodings(String s) {

        if ((s == null) || (s.length() < 1)) {
            return 0;
        }

        char[] schar = s.toCharArray();
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        if (this.isValid(schar, 0, 0)) {
            dp[1] = 1;
        } else {
            return 0;
        }

        for (int start = 1; start < schar.length; start++) {
            if (this.isValid(schar, start, start)) {
                dp[start + 1] += dp[start];
            }

            if (this.isValid(schar, start - 1, start)) {
                dp[start + 1] += dp[start - 1];
            }
        }
        return dp[s.length()];
    }
}
```

#### [Word Break](https://leetcode.com/problems/word-break/)
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Note:
1. The same word in the dictionary may be reused multiple times in the segmentation.
2. You may assume the dictionary does not contain duplicate words.

>Example 1: 
<code><pre>
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
</code></pre>

>Example 2: 
<code><pre>
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
</code></pre>

>Example 3: 
<code><pre>
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false
</code></pre>

>算法：回溯，对于wordDict每个能匹配的，进行dfs, 过程记录下不能满足条件的substring，作为memo来排除重复比较；这里回溯的状态用pos来控制。

>状态递推:
<pre><code>
memo[n] = false if each of the word in wordDict can not make memo[n-word.length] true
n：s第n位
dp[n]：n开始的substr是否满足条件
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(n)

top-down
```java
public class Solution {

    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] memo = new boolean[s.length()];
        Arrays.fill(memo, true);
        return this.helper(s.toCharArray(), 0, wordDict, memo);
    }

    private boolean helper(char[] str, int pos, List<String> wordDict, boolean[] memo) {
        if (pos == str.length) { // be aware of this condition
            return true;
        } else {
            if (memo[pos]) {
                for (String word : wordDict) {
                    if ((this.startWith(str, pos, word))
                            && (this.helper(str, pos + word.length(), wordDict, memo))) {
                        return true;
                    }
                }
            }
            memo[pos] = false;
            return false;
        }

    }

    //for performance
    private boolean startWith(char[] str, int pos, String word) {
        for (char c : word.toCharArray()) {
            if ((pos > str.length - 1) || (str[pos++] != c)) {
                return false;
            }
        }
        return true;
    }
}
```
这个题是回溯＋动态规划的典范。


### 类型2：`dp[]`数组以约束条件索引，通常这个值是有明确边界的，然后根据`dp[n]`到dp`[n-m]`的规则，找到最优解。
这类问题主要是搞清楚`dp[n-nums[i]]`怎么到`dp[n]`

#### [Coin Change](https://leetcode.com/problems/coin-change/)
You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
>Example 1: 
<code><pre>
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
</code></pre>

>Example 2: 
<code><pre>
Input: coins = [2], amount = 3
Output: -1
</code></pre>

Note:You may assume that you have an infinite number of each kind of coin.

>算法：用amount作为dp数组索引，则amount可以被索引为amount - coins[i]凑得。

>状态递推:
<pre><code>
dp[n] =　min(dp[n-coins[0]], dp[n-coins[1], .., dp[n-coins[k]]]) + 1, if dp[n-coins[i]] is valid
n：amount
dp[n]：amont对应的最小coin个数
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(n)

bottle-up
```java
 public class Solution{
    public int coinChange(int[] coins, int amount) {
      
        int res = 0;
        if ((coins.length < 1) || (amount<0)){
            return res;
        }

        int[] dp = new int[amount+1];
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for (int i=1;i<=amount;i++){
            for (int coin : coins){
                if ((i-coin >=0) && (dp[i-coin]>=0)){
                    if (dp[i] == -1){
                        dp[i] = dp[i-coin] +1;
                    }else {
                        dp[i] = Math.min(dp[i], dp[i-coin] +1);
                    }
                }
            }
        }
        return dp[amount];
    }
}
```
这里增加一个dummy的dp[0]作为起点。


#### [Perfect Squares](https://leetcode.com/problems/perfect-squares/)
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

>Example 1: 
<code><pre>
Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.
</code></pre>

>Example 2: 
<code><pre>
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
</code></pre>

>算法：从小到大凑，利用递推关系。

>状态递推:
<pre><code>
dp[n] = Math.min(dp[n], dp[n - i * i] + 1);
n：求和
dp[n]：最小平方数
</code></pre>
+ 时间复杂度O(n)
+ 空间复杂度O(n)

bottle-up
```java
 public class Solution{
      public int numSquares(int n) {
        if (n < 1) {
            return 0;
        }
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i * i <= n; i++) {
            for (int j = i * i; j <= n; j++) {
                dp[j] = Math.min(dp[j], dp[j - i * i] + 1);
            }
        }
        return dp[n];
    }
}
```



## 2-D

缓存设计：
+ x, y标号是问题空间, 元素值是子问题解空间，例如: Longest Common Substring (LCS), Edit Distance, Claiming Stairs, etc.
+ x标号是问题空间，y标号是约束条件，元素是状态转移（例如，选或者不选）条件下的最优值，这类问题解空间通常有明确的bound， 例如：0/1 Knapsack, etc.

从数组状态（一堆子问题的最优值）到最优解：通常都是从状态表的右下角开始，按状态转移规则，反着求每一步是否选择了。

### 类型１: `dp[][]`的二位索引是解空间，`dp[][]`是最优解。
这类问题比较直接，只要找到`dp[y][x]`和`dp[y-1][x]`, `dp[y][x-1]`, ...的关系即可。


#### [Unique Paths](https://leetcode.com/problems/unique-paths/)
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

Above is a 7 x 3 grid. How many possible unique paths are there?

Note: m and n will be at most 100.

>Example 1: 
<code><pre>
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
Right -> Right -> Down
Right -> Down -> Right
Down -> Right -> Right
</code></pre>

>Example 2: 
<code><pre>
Input: m = 7, n = 3
Output: 28
</code></pre>

>算法：找(y,x)和左上方格子之间的关系

>状态递推:
<pre><code>
dp[y][x] = dp[y-1][x] + dp[y][x-1];
x, y：坐标
dp[y][x]：路径和
</code></pre>
+ 时间复杂度O(m*n)
+ 空间复杂度O(m*n), 可以优化为O(1)

bottle-up
```java
public class Solution{
    public int uniquePaths(int m, int n) {

        if ((m < 1) || (n < 1)) {
            return 0;
        }

        int[][] dp = new int[m][n];
        for (int y = 0; y < dp.length; y++) {
            for (int x = 0; x < dp[0].length; x++) {
                if ((x == 0) || (y == 0)) {
                    dp[y][x] = 1;
                } else {
                    dp[y][x] = dp[y-1][x] + dp[y][x-1];
                }
            }
        }
        return dp[m-1][n-1];
    }
}
```

top-down
```java
public class Solution{
   public int uniquePaths(int m, int n) {

        if ((m < 1) || (n < 1)) {
            return 0;
        }

        int[][] memo = new int[m][n];
        return this.uniquePathsDp(m-1, n-1, memo);

   }


   public int uniquePathsDp(int m, int n, int[][] memo) {

        if (memo[m][n] != 0) {
            return memo[m][n];
        }

        if ((m == 0) || (n == 0)) {
            memo[m][n] = 1;
            return 1;
        }

        int numOfPaths = this.uniquePathsDp(m - 1, n, memo)
                + this.uniquePathsDp(m, n - 1, memo);
        memo[m][n] = numOfPaths; //no need to set dp[n][m] since it should check if [n, m] is valid.
        return numOfPaths;
    }
}
```
和Climbing Stairs一样，是教科书动态规划2D的第一题。


#### [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
Given a `m x n` grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

>Example 1: 
<code><pre>
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
</code></pre>

>算法：找(y,x)和左上方格子之间的关系

>状态递推:
<pre><code>
dp[y][x] = min(dp[y][x - 1], dp[y - 1][x]) + grid[y][x]
x, y：坐标
dp[y][x]：到当前坐标的最小和
</code></pre>
+ 时间复杂度O(m*n)
+ 空间复杂度O(m*n), 可以优化为O(1)

bottle-up
```java
public class Solution{
    public int minPathSum(int[][] grid) {
        if ((grid.length < 1) || (grid[0].length < 1)) {
            return 0;
        }

        int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0] = grid[0][0];
        for (int y = 0; y < dp.length; y++) {
            for (int x = 0; x < dp[0].length; x++) {
                if ((x == 0) && (y != 0)) {
                    dp[y][x] = dp[y - 1][x] + grid[y][x];
                } else if ((x != 0) && (y == 0)) {
                    dp[y][x] = dp[y][x - 1] + grid[y][x];
                } else if ((x != 0) && (y != 0)) {
                    dp[y][x] = Math.min(dp[y][x - 1], dp[y - 1][x]) + grid[y][x];
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
}
```
和Unique Path一样，先完成左边和上边的边界，然后递推。


#### [Maximal Square](https://leetcode.com/problems/maximal-square/)
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

>Example 1: 
<code><pre>
Input: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Output: 4
</code></pre>

>算法：找(y,x)和左上方格子之间的关系

>状态递推:
<pre><code>
dp[y][x] = min(dp[y - 1][x], dp[y][x - 1], dp[y - 1][x - 1])+1
x, y：坐标
dp[y][x]：到当前坐标的最大正方形边长
</code></pre>
+ 时间复杂度O(m*n)
+ 空间复杂度O(m*n), 可以优化为O(n)

bottle-up
```java
public class Solution{
    public int maximalSquare(char[][] matrix) {
        if ((matrix == null) || (matrix.length < 1 || (matrix[0].length < 1))) {
            return 0;
        }
        int dp[][] = new int[matrix.length][matrix[0].length];
        int maxLen=0;
        for (int y = 0; y < dp.length; y++) {
            for (int x = 0; x < dp[0].length; x++) {
                if (matrix[y][x] == '1') {
                    if ((x == 0) || (y == 0)) {
                        dp[y][x] = 1;
                    } else {
                        dp[y][x] = Math.min(Math.min(dp[y - 1][x], dp[y][x - 1]), dp[y - 1][x - 1]) + 1;
                    }
                    maxLen = Math.max(maxLen, dp[y][x]);
                } else {
                    dp[y][x] = 0;
                }
            }
        }
        return maxLen*maxLen;
    }
}
```
和Unique Path一样，先完成左边和上边的边界，然后递推，边上可以用条件`(x == 0) || (y == 0)`来完成。

空间O(n)优化
>状态递推:
<pre><code>
dp[j]=min(dp[j−1],dp[j],prev)
prev, dp[i], dp[i-1]和new_dp[i]分别为最近四个方格
</code></pre>
```java
public class Solution{
    public int maximalSquare(char[][] matrix) {
        int rows = matrix.length, cols = rows > 0 ? matrix[0].length : 0;
        int[] dp = new int[cols + 1];
        int maxsqlen = 0, prev = 0;
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                int temp = dp[j];
                if (matrix[i - 1][j - 1] == '1') {
                    dp[j] = Math.min(Math.min(dp[j - 1], prev), dp[j]) + 1;
                    maxsqlen = Math.max(maxsqlen, dp[j]);
                } else {
                    dp[j] = 0;
                }
                prev = temp;
            }
        }
        return maxsqlen * maxsqlen;
    }
}
```
#### [Burst Balloons](https://leetcode.com/problems/burst-balloons/)
Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. You are asked to burst all the balloons. If the you burst balloon i you will get `nums[left] * nums[i] * nums[right]` coins. Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.
Find the maximum coins you can collect by bursting the balloons wisely.

Note:
+ You may imagine `nums[-1] = nums[n] = 1`. They are not real therefore you can not burst them.
+ `0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100`

>Example 1:
<code><pre>
Input: [3,1,5,8]
Output: 167 
Explanation: nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
             coins =  3\*1\*5 + 3\*5\*8 + 1\*3\*8 + 1\*8\*1 = 167
</code></pre>

>算法：递归，对于一个气球，如果左边和右边的都先破了，那么左边、右边和这个气球总和是
`dp[left][right] = dp[left][pos - 1] + nums[left - 1]*nums[pos]*nums[right + 1] + dp[pos+1][right];`
这里假设pos位置的气球最后破。
+ 时间复杂度O(N*N)
+ 空间复杂度O(N*N)

```java
class Solution {
    
    public int maxCoins(int[] nums) {
        int res = 0;
        if ((nums == null) || (nums.length < 1)) {
            return res;
        }

        int[] numsWide = new int[nums.length + 2];
        numsWide[0] = 1;
        numsWide[nums.length + 1] = 1;
        for (int i = 1; i <= nums.length; i++) {
            numsWide[i] = nums[i - 1];
        }
        int[][] memo = new int[numsWide.length][numsWide.length];

        res = this.helper(numsWide, 1, nums.length, memo);
        return res;

    }

    private int helper(int[] nums, int left, int right, int[][] memo) {

        if (memo[left][right] != 0) {
            return memo[left][right];
        }

        if (left == right) {
            return nums[left - 1] * nums[left] * nums[left + 1];
        }

        int res = 0;
        for (int pos = left; pos <= right; pos++) {
            int curValue = this.helper(nums, left, pos - 1, memo)
                    + nums[left - 1] * nums[pos] * nums[right + 1]
                    + this.helper(nums, pos + 1, right, memo);
            res = Math.max(res, curValue);
        }
        memo[left][right] = res;
        return res;
    }    
}
```


### 类型2:背包问题0/1 kanpsack，dp[][]的y是约束条件空间，x是最优解空间（选或不选）
这类问题关键是y索引的设计，经常是sum或者sum/2等技巧。如果索引和递推方法设计不当，可能有非常多容易出错的边界条件。

#### [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note: Each of the array element will not exceed 100.
      The array size will not exceed 200.

>Example 1: 
<code><pre>
Input: [1, 5, 11, 5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
</code></pre>

>Example 2: 
<code><pre>
Input: [1, 2, 3, 5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
</code></pre>

>算法：转为背包问题，约束条件是和等于sum/2, 找(y, x)左边(y, x-1)和(y-nums[x], x-1)

>状态递推:
<pre><code>
dp[y][x] = dp[y-nums[x-1][x-1] || dp[y][x-1] 
x: nums的选项
y: 选项对应求和
</code></pre>
+ 时间复杂度O(m*n)
+ 空间复杂度O(m*n), 可以优化为O(m)

bottle-up

```java
public class Solution {
    
    public boolean canPartition(int[] nums) {
        if ((nums == null) || (nums.length < 2)) {
            return false;
        }

        int sum = 0;
        for (int n : nums) {
            sum += n;
        }
        if (sum % 2 != 0) {
            return false;
        }

        sum = sum / 2;
        boolean[][] dp = new boolean[sum + 1][nums.length];
        dp[0][0] = true;
        dp[nums[0]][0] = true;
        for (int x = 0; x < dp[0].length - 1; x++) {
            for (int y = 0; y < dp.length; y++) {
                if (dp[y][x]) {
                    dp[y][x + 1] = true;
                    if (y + nums[x + 1] < dp.length) {
                        dp[y + nums[x + 1]][x + 1] = true;
                    }
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }
}
```
这个题目注意边界条件，左边往右不容易错，另外`<length`不是`<length-1`

可以进一步优化为空间O(n),注意如果只有一个dp[], 那么左边一列到右边一列的递推过程，左边可能是会不断变化的。这个优化不仅把空间复杂度降为O(n)，而且还节省了“不选”时候“重复”赋值的开销。

top-down

这个问题由于只是找有没有，因此dfs可以在最好情况下加速。我们从`[sum/2][nums.length-1]`出发，通过递归来深度优先搜索，关键是记录一个memo表，已经访问过不行的点记录下来，下次递归到相同点就不再继续了。没有这个memo，会超时。
```java
public class Solution {

        public boolean canPartition(int[] nums) {
    
            if ((nums == null) || (nums.length < 2)) {
                return false;
            }
    
            int sum = 0;
            for (int n : nums) {
                sum += n;
            }
            if (sum % 2 != 0) {
                return false;
            }
    
            sum = sum / 2;
            boolean[][] memo = new boolean[sum + 1][nums.length];
            return helper(nums, memo, nums.length-1, sum);
        }
    
        private boolean helper(int[]nums, boolean[][] memo, int pos, int y){
            if (memo[y][pos] == true){
                return false;
            }
            if (pos == 0){
                return ((y == 0) || (y == nums[0]));
            }else{
                if (this.helper(nums, memo, pos-1, y)){
                    return true;
                }
                if ((y-nums[pos]>=0) && (this.helper(nums, memo, pos-1, y-nums[pos]))){
                    return true;
                }
            }
            memo[y][pos] = true;
            return false;
        }
}
```

#### [Target Sum](https://leetcode.com/problems/target-sum/)

You are given a list of non-negative integers, `a1, a2, ..., an`, and a target, `S`. Now you have 2 symbols `+` and `-`. For each integer, you should choose one from `+` and `-` as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target `S`.

>Example 1: 
<code><pre>
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
There are 5 ways to assign symbols to make the sum of nums be target 3.
</code></pre>

>算法：转为背包问题，约束条件是sum，索引需要通过2sum把负数转化过来。

>状态递推:
<pre><code>
dp[y][x] = dp[y-nums[x-1][x-1] || dp[y+nums[x-1]][x-1] 
x: nums的选项
y: 选项对应求和
</code></pre>
+ 时间复杂度O(m*n)
+ 空间复杂度O(m*n), 可以优化为O(m)

bottle-up
```java
public class Solution {
    
     public int findTargetSumWays(int[] nums, int target) {
    
            int res = 0;
            if ((nums != null) && (nums.length > 0)) {
                int sum = 0;
                for (int n : nums) {
                    sum += n;
                }
                if (sum < target) {
                    return 0;
                }
    
                int[][] dp = new int[sum * 2 + 1][nums.length + 1];
                dp[sum][0] = 1;
                for (int x = 0; x < dp[0].length - 1; x++) {
                    for (int y = 0; y < dp.length; y++) {
                        if (dp[y][x] > 0) {
                            int number = nums[x];
                            dp[y + number][x + 1] += dp[y][x];
                            dp[y - number][x + 1] += dp[y][x];
                        }
                    }
                }
                res = dp[sum + target][nums.length];
    
            }
            return res;
    
        }

}
```
这里的`dp[][]`也可以被优化为`O(m)`。 
 

top-down

这里memorizion的缓存用一个{-1, 0, k}三个状态表示，-1表示已经试过，不通，k表示已经试过，可以到达，其中k是可行路径个数；0表示还未尝试过。
由于尾递归会访问所有路径，所以一次可以把k求完，后续如果再访问到这个节点，直接加上k。
由于必须memoization路径个数，所以不可以把路径个数作为helper返回值，这个top-down的方法过于复杂。
```java
public class Solution {
    
    public int findTargetSumWays(int[] nums, int target) {
        if ((nums == null) || (nums.length < 0)) {
            return 0;
        }
        int sum = 0;
        for (int n : nums) {
            sum += n;
        }
        if (sum < target) {
            return 0;
        }
        int memo[][] = new int[2 * sum + 1][nums.length+1];
        this.helper(nums, memo, nums.length, sum + target);
        return Math.max(0, memo[sum + target][memo[0].length - 1]);
    }

    private boolean helper(int[] nums, int memo[][], int pos, int y) {
        if (pos == 0) {
            if (y == (memo.length - 1) / 2) {
                memo[y][pos] = 1;
                return true;
            } else {
                memo[y][pos] = -1;
                return false;
            }
        }

        if (memo[y][pos] != 0) {
            return (memo[y][pos] > 0);
        } else {
            if ((y - nums[pos-1] >= 0) && (this.helper(nums, memo, pos - 1, y - nums[pos-1]))) {
                memo[y][pos] += memo[y - nums[pos-1]][pos - 1];
            }
            if ((y + nums[pos-1] < memo.length) && (this.helper(nums, memo, pos - 1, y + nums[pos-1]))) {
                memo[y][pos] += memo[y + nums[pos-1]][pos - 1];
            }

            if (memo[y][pos] < 1) {
                memo[y][pos] = -1;
            }
        }
        return (memo[y][pos] > 0);
    }
}
```
不同于Partition Equal Subset Sum, 由于要找到所有的满足条件的选项，因此top-down dfs不能在找到一个满足条件结果立刻返回，但是对于特别稀疏的矩阵，是“以终为始”在递归树进行搜索，还是能跳过很多无关的计算。额外代价还是递归栈。






### 类型3: LCS问题，两个字符串逐个增加后缀进行比较
这类问题关键是从1到n进行递推，找出`(y-1, x), (y, x-1), (y-1, x-1)`到`(y,x)`的关系

#### [Edit Distance](https://leetcode.com/problems/edit-distance/submissions/)
Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.

You have the following 3 operations permitted on a word:

1. Insert a character
2. Delete a character
3. Replace a character

>Example 1: 
<code><pre>
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
</code></pre>

>Example 2: 
<code><pre>
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
</code></pre>

>算法：
1. 先把两边（上边、左边）赋值
2. (y,x)由(y-1, x), (y, x-1), (y-1, x-1)递推

>状态递推:
<pre><code>
if word1[y][x] == word2[y][x]
    dp[y][x] = min(dp[y-1][x-1], dp[y-1][x]+1, dp[y][x-1]+1) 
else
    dp[y][x] = min(dp[y-1][x-1], dp[y-1][x], dp[y][x-1]) + 1
(y,x), word1和word2前缀的Edit Distance
</code></pre>
+ 时间复杂度O(m*n)
+ 空间复杂度O(m*n), 可以优化为O(m)

bottle-up

```java
public class Solution {
      int minDistance(String word1, String word2) {
  
  
          if ((word1.length() == 0) || (word2.length() == 0)){
              return Math.max(word1.length(), word2.length());
          }
  
          char[] wordChars1 = word1.toCharArray();
          char[] wordChars2 = word2.toCharArray();
  
          int[][] dp = new int[word1.length()][word2.length()];
  
          dp[0][0] = (wordChars1[0] == wordChars2[0]) ? 0 : 1;
          boolean first = (dp[0][0] == 1) ? true : false;
          for (int i = 1; i < word1.length(); i++) {
              if ((first) && (wordChars1[i] == wordChars2[0])) {
                  dp[i][0] = dp[i - 1][0];
                  first = false;
              } else {
                  dp[i][0] = dp[i - 1][0] + 1;
              }
          }
  
          first = (dp[0][0] == 1) ? true : false;
          for (int i = 1; i < word2.length(); i++) {
              if ((first) && (wordChars2[i] == wordChars1[0])) {
                  dp[0][i] = dp[0][i - 1];
                  first = false;
              } else {
                  dp[0][i] = dp[0][i - 1] + 1;
              }
          }
  
  
          for (int y = 1; y < word1.length(); y++) {
              for (int x = 1; x < word2.length(); x++) {
                  if (wordChars1[y] == wordChars2[x]) {
                      dp[y][x] = Math.min(Math.min(dp[y][x - 1] + 1, dp[y - 1][x] + 1), dp[y - 1][x - 1]);
                  } else {
                      dp[y][x] = Math.min(Math.min(dp[y][x - 1] + 1, dp[y - 1][x] + 1), dp[y - 1][x - 1]+1);
                  }
              }
          }
          return dp[dp.length - 1][dp[0].length - 1];
  
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



[TOC]