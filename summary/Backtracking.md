# Backtracking Review

[TOC]

## Summary

Backtracking的原理是利用递归来遍历访问所有的解空间，通常用在以排列组合方式进行搜索的问题，本质是dfs+剪枝，好处是利用递归函数可以简洁直接地进行深度优先搜索（既然是dfs，那也可以用stack来实现非递归算法）。Backtracking本身并无优化，类似于暴力搜索，比起暴力的好处是：
+ 每次发现解空间不合格的时候不用从头开始；
+ 可以根据特性进行剪枝；

[这里](https://www.cis.upenn.edu/~matuszek/cit594-2012/Pages/backtracking.html)对Backtracking有一个比较好的介绍。

Backtracking的通用模板是
```
void helper(rawArray, select, status, results){
    if (select satisfies the expected result){
        results.add(new select instance);
    }else{
        for each next element e in rawArray{
            add e to select;
            helper(rawArray, select, update(status), results);
            remove e from select; // backtracking here, the status are backed as well.
        }
    }
}
```
回溯算法设计的关键是先设计出回溯树，然后根据递归的规则，设计出解空间select, 状态status，剩下的手到擒来。

具体地，对于从一个数组取成员进行组合类的问题，模板如下：
```
public void helper(int[] nums, int selectNum, List<Integer> selectList, List<List<Integer>> result) {
    if (selectNum == nums.length) {
        result.add(new ArrayList<>(selectList)); //已经选出全部的组合，递归终止；
    } else {
        for (int i = 0; i < nums.length; i++) { //从未选择（通过下面contains控制）的成员中选择一个；
            if (!selectList.contains(nums[i])) {
                selectList.add(nums[i]); // 加入已经选择集合；
                this.helper(nums, selectNum + 1, selectList, result); //递归进入下一层，选择剩余节点；
                selectList.remove(select.size() - 1); // 回溯，已经选择集合恢复递归调用前状态，等待选择其他成员；
            }
        }
    }
}
```
Backtracking模板如上所示，关键的状态量有：
1. 选项池数组
2. 已经选择的列表（一组潜在合法的解，或者解的未完成子集）
3. 判断是否要继续的状态，包括剪枝条件（例如start, count, sum等）
4. 结果集合，保存合法的解；
算法设计的关键：根据选项池，设计排列方法得到回溯树，然后设计状态量以支持回溯树的遍历和剪枝。

## Problems
### [Permutations](https://leetcode.com/problems/permutations/)
Given a collection of distinct integers, return all possible permutations.
>Example 1:
<code><pre>
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
</code></pre>

>算法：通过backtracking，每次取一个元素，然后分别把没有取过的作为下一层级访问。注意需要判断是否取过。
+ 时间复杂度O(n!), 遍历nums排除已select开销不计。
+ 空间复杂度O(n!)
```java
public class Solution{
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if ((nums != null) && (nums.length > 0)) {
            List<Integer> select = new ArrayList<>();
            this.helper(nums, 0, select, res);
        }
        return res;
    }

    public void helper(int[] nums, int selectNum, List<Integer> selectList, List<List<Integer>> result) {
        if (selectNum == nums.length) {
            result.add(new ArrayList<>(selectList));
            //System.out.println(select.toString());
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (!selectList.contains(nums[i])) {
                    selectList.add(nums[i]);
                    this.helper(nums, selectNum + 1, selectList, result);
                    selectList.remove(selectList.size() - 1);
                }
            }
        }
    }
}
```
上述递归过程是：
1. 递归树的第一级分别选1, 2, 3，分别代表第一位的三个选项；
2. 先把1加入select，已经选择成员的计数器selectNum加1，递归helper；
3. helper进入后，判断1已经在select，因此下一位的选项是2,3; 下一个把2加入select，selectNUm++, 继续递归helper
4. helper进入后，判断1/2都已经select，下一个把3加入select，selectNUm++, 继续递归helper
5. helper判断已经满足出口条件(selectNum判断选完了），输出[1,2,3]
6. 完成输出后，退到上一层helper，select移除3, selectNum恢复到2；
7. 当时selectList=[1, 2],selectNum=3, i=2的时候，由于i选到3已经选完，因此继续退到一层helper, select移除2, selectNum恢复到1；
8. 回溯恢复状态到selectList=[1], i=1, 循环下一次i++后，selectList=[1, 3]，调用helper；
9. 同理，找到没有被访问的2, 继续递归；
10. helper判断已经满足出口条件，输出[1,3,2]
11. 继续[2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]
>回溯树:
<pre><code>
        root
  １      2        3       selectNum=1
２　３   1   3    1   2     selectNum=2
３  2   3   1    2   1     selectNum=3
</code></pre>
可见，到了叶子节点，退一步后，因为没有未被选的，因此会继续再退一步（还剩两个元素，选了一个自然也就选剩下那个了)，这也是为什么[1, 2, 3]会直接回溯到[1]。

进一步改善代码concise, selectNum是一个冗余的状态，直接用selectList.size()即可：
```java
public class Solution{
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if ((nums != null) && (nums.length > 0)) {
            List<Integer> select = new ArrayList<>();
            this.helper(nums, select, res);
        }
        return res;
    }

    public void helper(int[] nums, List<Integer> selectList, List<List<Integer>> result) {
        if (selectList.size() == nums.length) {
            result.add(new ArrayList<>(selectList));
            //System.out.println(select.toString());
        } else {
            for (int i = 0; i < nums.length; i++) {
                if (!selectList.contains(nums[i])) {
                    selectList.add(nums[i]);
                    this.helper(nums, selectList, result);
                    selectList.remove(selectList.size() - 1);
                }
            }
        }
    }
}
```
### [Subset](https://leetcode.com/problems/subsets/)
Given a set of distinct integers, nums, return all possible subsets (the power set).
Note: The solution set must not contain duplicate subsets.

>Example 1:
<code><pre>
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
</code></pre>

>算法：通过backtracking，每次取一个元素，然后分别把比自己靠后的作为下一级访问（靠前的被剪枝）。注意需要判断是否取过。
+ 时间复杂度O(n!), 遍历nums排除已select开销不计。
+ 空间复杂度O(n!)

```java
public class Solution{
    
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<>());
        if ((nums != null) && (nums.length > 0)) {
            List<Integer> select = new ArrayList<>();
            this.helper(nums, select, 0, res);
        }
        return res;
    }

    private void helper(int[] nums, List<Integer> select, int start, List<List<Integer>> subset) {
       if (select.size() == nums.length){
           //do nothing here
       }else{
           for (int i=start; i<nums.length; i++){
               if (!select.contains(nums[i])){
                   select.add(nums[i]);
                   subset.add(new ArrayList<>(select));
                   this.helper(nums, select, i+1, subset);
                   select.remove(select.size()-1);
               }
           }
       }
    }
}
```
比起Permutation，回溯过程多增加了一个start状态量，用来剪枝（标志当前选到的位置，只找自己后面的元素进行组合）。
>回溯树:
<pre><code>
       root
  １     2    3       selectNum=1
２　３    3            selectNum=2
３                    selectNum=3
</code></pre>

### [Combination Sum](https://leetcode.com/problems/combination-sum/)
Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.
The same repeated number may be chosen from candidates unlimited number of times.

Note:
+ All numbers (including target) will be positive integers.
+ The solution set must not contain duplicate combinations.

>Example 1:
<code><pre>
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
</code></pre>

>Example 2:
<code><pre>
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
</code></pre>

>算法：通过backtracking，每次取一个元素，然后分别把比自己及其靠后的作为下一级访问。
时间复杂度O(?)
空间复杂度O(?)

```java
public class Solution{
    public List<List<Integer>> combinationSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if ((nums != null) && (nums.length > 0)) {
            List<Integer> select = new ArrayList<>();
            this.helper(nums, 0, 0, select, res, target);
        }
        return res;
    }

    public void helper(int[] nums, int start, int sum, List<Integer> select, List<List<Integer>> result, int target) {
        if (sum == target) {
            result.add(new ArrayList<>(select));
            //System.out.println(select.toString());
        } else {
            if (sum < target) {
                for (int i = start; i < nums.length; i++) {
                    select.add(nums[i]);
                    this.helper(nums, i, sum + nums[i], select, result, target);
                    select.remove(select.size() - 1);
                }
            }
        }
    }
}
```
比起Subset，回溯过程也是用start状态量来剪枝，区别仅是多了一个包括自己的选项。
>回溯树:
<pre><code>
         root
   １         2    3
 1    2     ２ 3    3
1 2  2 3    2 3   3    3
</code></pre>


### [Phone Number Combination](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
>Example 1:
<code><pre>
Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
</code></pre>

Although the above answer is in lexicographical order, your answer could be in any order you want.

>算法：通过backtracking，逐个遍历数字对应的所有字母组合。
+ 时间复杂度O(3^N*4^M), N是3位个数, M是4位个数
+ 空间复杂度O(3^N*4^M)

```java
public class Solution{
    private static final String[] KEYS = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if ((digits!=null)&&(digits.length()>0)){
            int[] nums = new int[digits.length()];
            char[] digitArray = digits.toCharArray();
            for (int i = 0; i < nums.length; i++) {
                nums[i] = digitArray[i] - '0';
            }
            this.helper(nums, new StringBuilder(), res);
        }
        return res;
    }

    private void helper(int[] nums, StringBuilder select, List<String> result) {
        if (select.length() == nums.length) {
            result.add(select.toString());
        } else {
            String charSets = KEYS[nums[select.length()]];
            for (char c : charSets.toCharArray()) {
                select.append(c);
                this.helper(nums, select, result);
                select.deleteCharAt(select.length()-1);
            }
        }
    }
}
```
通过select.length()来往后选取数字，选完退出。

>回溯树:
<pre><code>
            root
       a      b       c
   d    e   d   e   d   e
  f g  f g f g f g f g f g
</code></pre>

### [Generate Parenthesis](https://leetcode.com/problems/generate-parentheses/)
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is: 

>Example 1:
<code><pre>
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
</code></pre>

>算法：通过backtracking，产生所有合法的(),如果有左括号比右括号多，则选择右括号，如果左括号还有余额，则选择左括号。
+ 时间O(4^n/sqrt(n))
+ 空间O(4^n/sqrt(n))

```java
public class Solution{
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n > 0) {
            this.helper(new ArrayList<>(), 0, 0, n, res);
        }
        return res;
    }

    private void helper(List<Character> select, int leftNum, int rightNum, int n, List<String> result) {
        if (select.size() == 2 * n) {
            result.add(this.charListToString(select));
        }else{
            if (leftNum < n){
                select.add('(');
                this.helper(select, leftNum+1, rightNum, n, result);
                select.remove(select.size()-1);
            }
            if (leftNum > rightNum){
                select.add(')');
                this.helper(select, leftNum, rightNum+1, n, result);
                select.remove(select.size()-1);
            }
        }
    }

    private String charListToString(List<Character> select){
        StringBuilder sb = new StringBuilder();
        for (char c: select){
            sb.append(c);
        }
        return sb.toString();
    }
}
```
通过select.length()来判断是否完成递归退出，通过leftNum和rightNum来判断是否能选择左括号、右括号。

>回溯树:
<pre><code>
             (
         ()    ((
        ()(    (()
        ()()   (())
</code></pre>

### [Word Search](https://leetcode.com/problems/word-search/)
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

>Example 1:
<code><pre>
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
</code></pre>

>算法：通过backtracking，搜索矩阵上下左右，已经搜索过的用一个boolean矩阵存标志（另外一个简单方法是加一个*)，退回的时候恢复标志。
注意边界条件。
+ 时间复杂度　O(N*4^K), N是board大小, K为word大小
+ 空间复杂度　O(N+K)

```java
public class Solution{
    public boolean exist(char[][] board, String word) {
        if ((board != null) && (board.length > 0) && (board[0].length > 0)
                && (word != null) && (word.length() > 0)) {
            boolean[][] visited = new boolean[board.length][board[0].length];
            for (int y = 0; y < board.length; y++) {
                for (int x = 0; x < board[0].length; x++) {
                    visited[y][x] = true;
                    if (this.helper(word.toCharArray(), 0, board, y, x, visited)) {
                        return true;
                    }
                    visited[y][x] = false;
                }
            }
        }
        return false;
    }

    private boolean helper(char[] word, int pos, char[][] board, int y, int x, boolean[][] visited) {

        //System.out.println("pos="+pos+", char: "+word[pos]+", y="+y+", x="+x+", word.length"+word.length+", board[y][x]: "+board[y][x]);
        if (word[pos] == board[y][x]) {
            if (pos == word.length - 1) {
                return true;
            } else {
                if (((x - 1) >= 0) && (!visited[y][x - 1])) {  //move left
                    visited[y][x - 1] = true;
                    if (this.helper(word, pos + 1, board, y, x - 1, visited)) {
                        return true;
                    }
                    visited[y][x - 1] = false;
                }

                if (((x + 1) < board[0].length) && (!visited[y][x + 1])) {  //move right
                    visited[y][x + 1] = true;
                    if (this.helper(word, pos + 1, board, y, x + 1, visited)) {
                        return true;
                    }
                    visited[y][x + 1] = false;
                }

                if (((y - 1) >= 0) && (!visited[y - 1][x])) {  //move up
                    visited[y - 1][x] = true;
                    if (this.helper(word, pos + 1, board, y - 1, x, visited)) {
                        return true;
                    }
                    visited[y - 1][x] = false;
                }

                if (((y + 1) < board.length) && (!visited[y + 1][x])) {  //move down
                    visited[y + 1][x] = true;
                    if (this.helper(word, pos + 1, board, y + 1, x, visited)) {
                        return true;
                    }
                    visited[y + 1][x] = false;
                }
            }
        }
        return false;
    }
}
```
通过pos来判断是否完成递归退出，通过标志位`visited[][]`来确定是否已经访问过。
>回溯树:
<pre><code>
　　　　　　 A
        B     S
       C E   E A
</code></pre>

### [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)
Given an input string (s) and a pattern (p), implement regular expression matching with support for `.` and `*`.
1. `.` Matches any single character.
2. `*` Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

Note:
1. `s` could be empty and contains only lowercase letters `a-z`.
2. `p` could be empty and contains only lowercase letters `a-z`, and characters like `.` or `*`.
>Example 1: 
<code><pre>
Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
</code></pre>

>Example 2: 
<code><pre>
Input:
s = "aa"
p = "a*"
Output: true
Explanation: '*' means zero or more of the precedeng element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
</code></pre>

>Example 3: 
<code><pre>
Input:
s = "ab"
p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".
</code></pre>

>Example 4: 
<code><pre>
Input:
s = "aab"
p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore it matches "aab".
</code></pre>

>Example 5: 
<code><pre>
Input:
s = "mississippi"
p = "mis*is*p*."
Output: false
</code></pre>

>算法：回溯，排列所有pattern的可能，如果遇到*，则0, 1, 2, 3, etc.
+ 时间复杂度O(SP), S, P are the length of String and Pattern. 
+ 空间复杂度O(SP)

```java
public class Solution{
        public boolean isMatch(String s, String p) {
            if ((s == null) || (p == null)) {
                return false;
            }
            return this.helper(s.toCharArray(), p.toCharArray(), 0, 0);
    
        }
    
        private boolean helper(char[] str, char[] pattern, int strPos, int patternPos) {
            if ((strPos == str.length) && (patternPos == pattern.length)) {
                return true;
            }
    
    
            if ((strPos <= str.length) && (patternPos < pattern.length)) {
                if (((patternPos + 1) < pattern.length) && (pattern[patternPos + 1] == '*')) {
                    //key permutation here
                    // for case * is 0
                    if (this.helper(str, pattern, strPos, patternPos + 2)) {
                        return true;
                    }
    
                    // for case * is 1, 2, 3...
                    while ((strPos < str.length) && (this.isCharMatch(str[strPos], pattern[patternPos]))) {
                        if (this.helper(str, pattern, strPos + 1, patternPos + 2)) {
                            return true;
                        }
                        strPos++;
                    }
                } else {
    
                    if ((strPos < str.length) && (this.isCharMatch(str[strPos], pattern[patternPos]))) {
                        if (this.helper(str, pattern, strPos + 1, patternPos + 1)) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
    
        private boolean isCharMatch(char sChar, char pChar) {
            if (pChar == '.') {
                return true;
            }
            return (sChar == pChar);
        }    
}
```
>回溯树(pattern): c*a
<pre><code>
　　　　　root
     a    c   cc  ccc
          a    a   a 
</code></pre>

进一步, s和p的子串如果已经比较过，可以存内存记忆（实际情况效果可能不明显）
```java
public class Solution {

    public boolean isMatch(String s, String p) {
        if ((s == null) || (p == null)) {
            return false;
        }
        boolean memo[][] = new boolean[s.length() + 1][p.length() + 1];
        return this.helper(s.toCharArray(), p.toCharArray(), 0, 0, memo);

    }

    private boolean helper(char[] str, char[] pattern, int strPos, int patternPos, boolean[][] memo) {
        if ((strPos == str.length) && (patternPos == pattern.length)) {
            return true;
        }

        if (memo[strPos][patternPos]) {
            return false;
        }

        if ((strPos <= str.length) && (patternPos < pattern.length)) {
            if (((patternPos + 1) < pattern.length) && (pattern[patternPos + 1] == '*')) {
                //key permutation here
                // for case * is 0
                if (this.helper(str, pattern, strPos, patternPos + 2, memo)) {
                    return true;
                }

                // for case * is 1, 2, 3...
                while ((strPos < str.length) && (this.isCharMatch(str[strPos], pattern[patternPos]))) {
                    if (this.helper(str, pattern, strPos + 1, patternPos + 2, memo)) {
                        return true;
                    }
                    strPos++;
                }
            } else {

                if ((strPos < str.length) && (this.isCharMatch(str[strPos], pattern[patternPos]))) {
                    if (this.helper(str, pattern, strPos + 1, patternPos + 1, memo)) {
                        return true;
                    }
                }
            }
        }
        memo[strPos][patternPos] = true;
        return false;
    }

    private boolean isCharMatch(char sChar, char pChar) {
        if (pChar == '.') {
            return true;
        }
        return (sChar == pChar);
    }
}
```
### [Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)
Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

Note: The input string may contain letters other than the parentheses ( and ).

>Example 1:
<code><pre>
Input: "()())()"
Output: ["()()()", "(())()"]
</code></pre>

>Example 2:
<code><pre>
Input: "(a)())()"
Output: ["(a)()()", "(a())()"]
</code></pre>

>Example 3:
<code><pre>
Input: ")("
Output: [""]
</code></pre>

>算法：回溯，选或者不选
+ 时间复杂度O(2^N)
+ 空间复杂度O(2^N)

```java
class Solution {
    
    private int maxLen = 0;

    public List<String> removeInvalidParentheses(String s) {
        ArrayList<String> res = new ArrayList<>();
        if ((s == null) || (s.length() < 1)) {
            res.add("");
            return res;
        }
        this.helper(s.toCharArray(), new StringBuilder(), 0, 0, 0, res);
        return res;
    }

    private void helper(char[] sChar, StringBuilder select, int pos, int left, int right, ArrayList<String> res) {

        if ((pos == sChar.length) && (left == right)) {
            if (this.maxLen < select.length()) {
                res.clear();
                this.maxLen = select.length();
            }
            if ((this.maxLen == select.length())) {
                if (!res.contains(select.toString())) {
                    res.add((select.length() == 0) ? "" : select.toString());
                }
            }
        } else if (select.length() + (sChar.length - pos) < this.maxLen) {
            return;
        } else if (pos < sChar.length) {

            // not skip
            if ((sChar[pos] == ')') && (left > right)) {
                select.append(")");
                this.helper(sChar, select, pos + 1, left, right + 1, res);
                select.deleteCharAt(select.length() - 1);
            } else if (sChar[pos] == '(') {
                select.append("(");
                this.helper(sChar, select, pos + 1, left + 1, right, res);
                select.deleteCharAt(select.length() - 1);
            } else if ((sChar[pos] != '(') && (sChar[pos] != ')')) {
                select.append(sChar[pos]);
                this.helper(sChar, select, pos + 1, left, right, res);
                select.deleteCharAt(select.length() - 1);
            }

            //skip, latter is better
            this.helper(sChar, select, pos + 1, left, right, res);

        }
    }
}
```

### [Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)
Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.
>Example 1:
<code><pre>
Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
</code></pre>

>算法：回溯，选择所有可能的第一个partition组合。
+ 时间复杂度O(N!)?
+ 空间复杂度O(N!)

```java
class Solution {

     public List<List<String>> partition(String s) {
        ArrayList<List<String>> res = new ArrayList<>();
        if ((s == null) || (s.length() < 1)) {
            return res;
        }
        this.helper(s.toCharArray(), 0, new ArrayList<>(), res);
        return res;
    }

    private void helper(char[] raw, int start, List<String> select, List<List<String>> res) {
        if (start == raw.length) {
            res.add(new ArrayList<>(select));
        } else {
            for (int partition = start; partition < raw.length; partition++) {
                if (this.isPalindrome(raw, start, partition)) {
                    select.add(String.valueOf(raw, start, partition - start + 1));
                    this.helper(raw, partition+1, select, res);
                    select.remove(select.size() - 1);
                }
            }
        }
    }

    private boolean isPalindrome(char[] raw, int start, int end) {

        while (start <= end) {
            if (raw[start++] != raw[end--]) {
                return false;
            }
        }
        return true;
    }

}
```
实现的时候需要注意partition的边界条件。
>回溯树(pattern):
<pre><code>
　　　　　aab
     a          aa
   a   ab(x)     b
  b
</code></pre>

### [51. N-Queens](https://leetcode.com/problems/n-queens/)
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
Given an integer n, return all distinct solutions to the n-queens puzzle.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space respectively.

>Example 1:
<code><pre>
Input: 4
Output: [
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],
 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above.
</code></pre>

>算法：通过递归回溯，每次选或不选
+ 时间复杂度O(N!)
+ 空间复杂度O(N!)

```java
class Solution {

    //3ms, 83.19%
    public List<List<String>> solveNQueens(int n) {
        ArrayList<List<String>> res = new ArrayList<>();
        if (n > 0) {
            boolean[][] queen = new boolean[n][n];
            this.helper(queen, 0, 0, res, n);
        }
        return res;
    }

    private void helper(boolean[][] queen, int y, int x, ArrayList<List<String>> res, int n) {
        if (y == n) {
            this.output(queen, res, n);
        } else {
            //select and move to the next line
            if (this.isValid(queen, y, x)) { //check if (y,x) is valid 
                queen[y][x] = true;
                this.helper(queen, y + 1, 0, res, n);
                queen[y][x] = false;
            }
            //not select (i.e., skip)
            if (x < n - 1) {
                this.helper(queen, y, x + 1, res, n);
            }
        }
    }

    private void output(boolean[][] queen, ArrayList<List<String>> res, int n) {
        ArrayList<String> solution = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            StringBuilder line = new StringBuilder();
            for (int i = 0; i < n; i++) {
                if (queen[j][i]) {
                    line.append("Q");
                } else
                    line.append(".");
            }
            solution.add(line.toString());
        }
        res.add(solution);
    }

    private boolean isValid(boolean[][] queen, int y, int x) {
        if (!checkLine(queen, y, x)) {
            return false;
        }
        if ((!checkSlash(queen, y, x, true))
                || (!checkSlash(queen, y, x, false))) {
            return false;
        }
        return true;
    }

    private boolean checkLine(boolean[][] queen, int y, int x) {
        for (int i = 0; i < y; i++) {
            if (queen[i][x]) {
                return false;
            }
        }
        return true;
    }

    private boolean checkSlash(boolean[][] queen, int y, int x, boolean isLeft) {
        while ((x >= 0) && (x < queen.length) && (y >= 0)) {
            if (queen[y][x]) {
                return false;
            }
            y--;
            if (isLeft) {
                x--;
            } else {
                x++;
            }
        }
        return true;
    }
}
```
>回溯树(pattern):
<pre><code>
　　　　　                 root
　　　    select(0,0)                    skip(0,0)
select(1,0)   skip(1,0)       select(0,1)     skip(0,1)
</code></pre>



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
>回溯树(pattern):
<pre><code>
　　　　　root
     a    c   cc  ccc
          a    a   a
</code></pre>





