[TOC]
# Lib Review
这里总结一些库函数的简化实现题目，这些题目通常没有算法技巧，有一些工程实现技巧和边界条件。

### String Related

#### [8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/)
mplement atoi which converts a string to an integer.

The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.

If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

If no valid conversion could be performed, a zero value is returned.

Note:
1. Only the space character ' ' is considered as whitespace character.
2. Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: `[−231,  231 − 1]`. If the numerical value is out of the range of representable values, INT_MAX $(2^31 − 1)$ or INT_MIN $(−2^31)$ is returned.

>Example 1:
<code><pre>
Input: "42"
Output: 42
</code></pre>

>Example 2:
<code><pre>
Input: "   -42"
Output: -42
Explanation: The first non-whitespace character is '-', which is the minus sign.
             Then take as many numerical digits as possible, which gets 42.
</code></pre>

>算法：先trim, 再提取正负号，按正负号分别逐位计算，每次都判断是否越界
1. 注意越界不能超过了才判断，需要用没有超过（i.e., /10）的条件；
2. 判断是否已经到最后一位的条件`pos < str.length()`放在最前面，否则后面条件会越界；
+ 时间复杂度O(N)
+ 空间复杂度O(1)

```java
class Solution {
    //1ms, 100%
    public int myAtoi(String str) {

        int res = 0;
        if ((str == null) || (str.length() < 1)) {
            return res;
        }

        //init
        char[] strChars = str.toCharArray();
        boolean isPositive = true;
        int pos = 0;

        //trim empty
        while ((pos < str.length()) && (strChars[pos] == ' ')) {
            pos++;
        }

        if (pos < str.length()) {
            //extract sign
            if (this.isSign(strChars[pos])) {
                if (strChars[pos] == '-') {
                    isPositive = false;
                }
                pos++;
            }
            //extract digit
            while ((pos < str.length()) && this.isNumber(strChars[pos])) {
                int digit = (strChars[pos] - '0');
                if (isPositive) {
                    if ((Integer.MAX_VALUE - digit) / 10 < res) {
                        return Integer.MAX_VALUE;
                    }
                    res = res * 10 + digit;
                } else {
                    if ((Integer.MIN_VALUE + digit) / 10 > res) {
                        return Integer.MIN_VALUE;
                    }
                    res = res * 10 - digit;
                }
                pos++;
            }
            //trim the rest automatically
        }
        return res;
    }

    private boolean isNumber(char c) {
        return (((c - '0') >= 0) && ((c - '0') <= 9));
    }

    private boolean isSign(char c) {
        return ((c == '+') || (c == '-'));
    }

}
```

#### [166. Fraction to Recurring Decimal](https://leetcode.com/problems/fraction-to-recurring-decimal/)
Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

>Example 1:
<code><pre>
Input: numerator = 1, denominator = 2
Output: "0.5"
</code></pre>

>Example 2:
<code><pre>
Input: numerator = 2, denominator = 1
Output: "2"
</code></pre>

>Example 3:
<code><pre>
Input: numerator = 2, denominator = 3
Output: "0.(6)"
</code></pre>

>算法： 记住产生小数的被除数的位置(hash map)，如果重复，证明小数也开始重复。
另外为了搞边界条件，只能用long；
+ 时间复杂度O(N)
+ 空间复杂度O(M), m是循环小数位数

```java
class Solution {
 //1ms, 100%
    public String fractionToDecimal(int num, int den) {

        StringBuilder res = new StringBuilder();
        if (den != 0) {
            if (((num < 0) && (den > 0))
                    || ((num > 0) && (den < 0))) {
                res.append("-");
            }
            long numerator = Math.abs((long) num);
            long denominator = Math.abs((long) den);

            // integral part
            long rest;
            if (numerator >= denominator) {
                rest = numerator % denominator;
                res.append((numerator - rest) / denominator);
            } else {
                rest = numerator;
                res.append("0");
            }

            HashMap<Long, Integer> map = new HashMap<>();
            // fractional part
            if (rest != 0) {
                res.append(".");
                map.put(rest, res.length());//remember here
                rest = rest * 10;
                long current;
                while (rest != 0) {
                    current = (rest - rest % denominator) / denominator;// current = 0 when rest < denominator
                    rest = (rest - denominator * current);
                    res.append(current);
                    if (map.containsKey(rest)) {
                        String repeats = res.substring(map.get(rest), res.length());
                        res.delete(map.get(rest), res.length());
                        res.append("(");
                        res.append(repeats);
                        res.append(")");
                        break;
                    } else {
                        map.put(rest, res.length());
                    }
                    rest = rest * 10;
                }
            }
        }
        return res.toString();
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