# Tree Search Review

## Summary
树结果是图结构的变种，本质也是图操作的简化。例如，pre-order, in-order, post-order本质是dfs, 因此都可以用递归和迭代(stack)来实现。而level遍历本质则是bfs, 用迭代(queue)实现。
和回溯类似，树的搜索过程一定要想清楚递归/迭代的路径、状态和边界条件。

通常, 在递归遍历树节点过程中，状态传递的设计包括：
1. 通过递归函数返回值，比如统计和等；
2. 参数回溯；
3. 参数set, list等，等同于全局变量；
4. 全局变量；


### 树的DFS遍历
dfs遍历可以用递归和迭代两种方式，以中序为例(最常见)

>递归
```
helper(node){
    if (node == null){
        //end of dfs
    }else{
        helper (node.left);
        visit(node)
        helper (node.right);
    }
}

```
node的位置根据pre/in/post的不同而不同。

>迭代
```
while ((current is not null) || (stack is not empty)){
    while (current should be cached in stack){
        stack.push(current);
        current = current.left;
    }
    current = stack.pop();
    visit(current);
    current = current.right;
}


```
迭代路径类似链表的操作current=current.next，只不过多一个stack来辅助存储当前暂时不能访问，后续才能访问的节点。
一般情况下可以就选择递归，除非要求你迭代，后续会以递归为主。

#### [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
Given a binary tree, return the inorder traversal of its nodes' values.

>Example:
<code><pre>
Input: [1,null,2,3]
   1
     \\
      2
     /
   3
Output: [1,3,2]
</code></pre>

>算法1：递归，出口条件是root==null。
+ 时间复杂度O(N)
+ 空间复杂度O(N)，平均O(logN)。

```java
public class Solution {

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        this.helper(root, result);
        return result;
    }

    private void helper(TreeNode root, List<Integer> res){
        if (root==null){
            return; // end of dfs
        }else{
            this.helper(root.left, res);
            res.add(root.val);
            this.helper(root.right,res);
        }
    }
}
```

>算法2：迭代，把自己先放入stack，直到找不到left，然后弹出处理后，转向right。
+ 时间复杂度O(N)
+ 空间复杂度O(N)，平均O(logN)。

```java
public class Solution {

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        LinkedList<TreeNode> stack = new LinkedList<>();

        TreeNode current = root;
        while ((current!=null) || (!stack.isEmpty())){ // there is any right node or ancestor
            while (current!=null){ // push all the ancestor to stack until the leaf
                stack.push(current);
                current = current.left;
            }
            current = stack.pop(); // go to process previous pushed non-null node
            res.add(current.val);
            current = current.right; // go right
        }
        return res;
    }
}
```

#### [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
Given a binary tree, return the preorder traversal of its nodes' values.

>Example:
<code><pre>
Input: [1,null,2,3]
   1
    \\
     2
    /
   3
Output: [1,2,3]
</code></pre>

>算法1：递归，出口条件是root==null。
+ 时间复杂度O(N)
+ 空间复杂度O(N)，平均O(logN)。

```java
public class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {

        List<Integer> res = new ArrayList<>();
        this.helper(root, res);
        return res;

    }

    private void helper(TreeNode root, List<Integer> res){
        if (root == null){
            return ;
        }else{
            res.add(root.val);
            this.helper(root.left, res);
            this.helper(root.right, res);
        }
    }
}
```

>算法2：迭代，
+ 时间复杂度O(N)
+ 空间复杂度O(N)，平均O(logN)。

```java
public class Solution {


}
```

#### [Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
Given a binary tree, flatten it to a linked list in-place.

For example, given the following tree:

>Example 1:
<code><pre>
    1
   / \\
  2   5
 / \\  \\
3   4   6
The flattened tree should look like:
1
 \\
  2
  \\
    3
     \\
      4
       \\
        5
         \\
          6
</code></pre>

>算法：前序遍历，用全局变量缓存prev
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    private TreeNode prev = null;

    public void flatten(TreeNode root) {
        if (root == null){
            return;
        }
        this.flatten(root.right);
        this.flatten(root.left);
        root.right = this.prev;
        root.left = null;
        this.prev = root;
    }
}
```


#### [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
Given a binary tree, return the postorder traversal of its nodes' values.

>Example:
<code><pre>
Input: [1,null,2,3]
   1
    \\
     2
    /
   3
Output: [3,2,1]
</code></pre>

>算法1：递归，出口条件是root==null。
+ 时间复杂度O(N)
+ 空间复杂度O(N)，平均O(logN)。

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        this.helper(root, res);
        return res;

    }

    private void helper(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        } else {
            this.helper(root.left, res);
            this.helper(root.right, res);
            res.add(root.val);
        }
    }
}
```

>算法2：迭代，
+ 时间复杂度O(N)
+ 空间复杂度O(N)，平均O(logN)。

```java
public class Solution {


}
```

### 树的BFS遍历
树的BFS也是层序遍历，所有BFS都是通过queue来实现先入先出的顺序，以保证广度优先。

#### [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

>Example:
<code><pre>
Given binary tree [3,9,20,null,null,15,7],
    3
   / \\
  9  20
    /  \\
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
</code></pre>

>算法1：用queue进行bfs, hash表存每个节点的level
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {

        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        LinkedList<TreeNode> queue = new LinkedList<>();
        HashMap<TreeNode, Integer> map = new HashMap<>();
        queue.addLast(root);
        map.put(root, 0);
        while (!queue.isEmpty()) {
            TreeNode current = queue.removeFirst();
            int level = map.get(current);
            if (res.size() == level) {
                res.add(new ArrayList<>());
            }
            res.get(level).add(current.val);
            if (current.left != null) {
                queue.addLast(current.left);
                map.put(current.left, level + 1);
            }
            if (current.right != null) {
                queue.addLast(current.right);
                map.put(current.right, level + 1);
            }
        }
        return res;
    }
}
```
不同于dfs，这里`root==null`一开始需要判断

>算法2：递归，level状态可以回溯
```java
public class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        helper(res, root, 0);
        return res;
    }

    private void helper(List<List<Integer>> res, TreeNode t, int level) {
        if (t == null) return;
        if (res.size() == level) {
            res.add(new ArrayList<>());
        }
        res.get(level).add(t.val);
        helper(res, t.left, level + 1);
        helper(res, t.right, level + 1);
    }
}
```
+ 时间复杂度O(N)
+ 空间复杂度O(N)。

#### [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

>Example 1:
<code><pre>
You may serialize the following tree:
    1
   / \\
  2   3
     / \\
    4   5
as "[1,2,3,null,null,4,5]"
</code></pre>

Clarification: The above format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

Note: Do not use class member/global/static variables to store states. Your serialize and deserialize algorithms should be stateless.

>算法：用队列进行bfs
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {

    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        if (root!=null){
            LinkedList<TreeNode> queue = new LinkedList<>();
            queue.addLast(root);
            while (!queue.isEmpty()){
                TreeNode node = queue.removeFirst();
                if (node == null){
                    sb.append("N,");
                }else{
                    sb.append(node.val+",");
                    queue.addLast(node.left);
                    queue.addLast(node.right);
                }
            }
            return sb.substring(0, sb.length()-1);
        }
        return "";
    }

    public TreeNode deserialize(String data) {
        TreeNode root = null;
        if ((data!=null) && (!data.equals(""))){
            String[] nodeStrs = data.split(",");
            root = new TreeNode(Integer.parseInt(nodeStrs[0]));
            LinkedList<TreeNode> queue = new LinkedList<>();
            queue.addLast(root);
            int pos = 0;
            while (!queue.isEmpty()){
                TreeNode node = queue.removeFirst();

                String leftCodec = nodeStrs[++pos];
                if ( !leftCodec.equals("N")){
                    node.left = new TreeNode(Integer.parseInt(leftCodec));
                    queue.addLast(node.left);
                }

                String rightCodec = nodeStrs[++pos];
                if ( !rightCodec.equals("N")){
                    node.right = new TreeNode(Integer.parseInt(rightCodec));
                    queue.addLast(node.right);
                }
            }
        }
        return root;
    }
}
```

#### [Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:
```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

>Example 1:
<code><pre>
Input: {"$id":"1","left":{"$id":"2","left":{"$id":"3","left":null,"next":null,"right":null,"val":4},"next":null,"right":{"$id":"4","left":null,"next":null,"right":null,"val":5},"val":2},"next":null,"right":{"$id":"5","left":{"$id":"6","left":null,"next":null,"right":null,"val":6},"next":null,"right":{"$id":"7","left":null,"next":null,"right":null,"val":7},"val":3},"val":1}

Output: {"$id":"1","left":{"$id":"2","left":{"$id":"3","left":null,"next":{"$id":"4","left":null,"next":{"$id":"5","left":null,"next":{"$id":"6","left":null,"next":null,"right":null,"val":7},"right":null,"val":6},"right":null,"val":5},"right":null,"val":4},"next":{"$id":"7","left":{"$ref":"5"},"next":null,"right":{"$ref":"6"},"val":3},"right":{"$ref":"4"},"val":2},"next":null,"right":{"$ref":"7"},"val":1}

Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B.
</code></pre>

>算法1：递归，连接两个地方:(1) `root.left.next = root.right`; (2)`root.right.next = root.next.left`;
+ 时间复杂度O(N)
+ 空间复杂度O(1)

```java
class Solution {
    public Node connect(Node root) {
        if (root != null) {
            if (root.left != null) {
                root.left.next = root.right;
            }

            if ((root.next != null) && (root.right != null)) {
                root.right.next = root.next.left;
            }

            connect(root.left);
            connect(root.right);
        }
        return root;
    }
}
```
>算法2：用queue进行dfs，然后把最右边路径所有节点next置null;
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public Node connect(Node root) {
        if (root != null) {
            LinkedList<Node> queue = new LinkedList<>();
            queue.addLast(root);
            Node prev = null;
            while (!queue.isEmpty()) {
                Node node = queue.removeFirst();
                if (prev != null) {
                    prev.next = node;
                }
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
                prev = node;
            }

            Node current = root;
            while (current != null) {
                current.next = null;
                current = current.right;
            }
        }
        return root;
    }
}
```

### BinaryTree搜索和变换
搜索和变换的本质是进行遍历，可能需要辅助一些数据结构进行缓存，更好的是不需要Hash。树的搜索，本质是想清楚从当前root出发，left和right的状态情况。

#### [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Given the following binary tree:  root = [3,5,1,6,2,0,8,null,null,7,4]

>Example 1:
<code><pre>
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
</code></pre>

>Example 2:
<code><pre>
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
</code></pre>

>算法1：通过左右递归，如果左边和右边都有返回p, q，那么当前节点就是LCA；否则，如果只在左边，那么搜索左边，如果只在右边，那么搜索右边。
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

        if (root == null){
            return null; // not find
        }

        if ((root == p) || (root == q)){
            return root; // find p or q
        }

        TreeNode leftLca = this.lowestCommonAncestor(root.left, p, q);
        TreeNode rightLca = this.lowestCommonAncestor(root.right, p, q);

        if ((leftLca!=null) && (rightLca!=null)){
            return root; // this is the LCA
        }

        if (leftLca!=null){
            return leftLca;
        }else{
            return rightLca;
        }
    }
}
```
这里有个隐含的技巧是，一开始用null, p, q作为返回状态，当遇到LCA的时候，返回值在递归回溯过程中会被改为LCA。

注：LCA的思路和模板可以扩展若干Binary Tree搜索问题。

#### [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
>Example 1:
<code><pre>
    1
   / \\
  2   2
 / \\ / \\
3  4 4  3
</code></pre>

But the following [1,2,2,null,3,null,3] is not:
<code><pre>
    1
   / \\
  2   2
   \\   \\
   3    3
</code></pre>

>算法1：递归，如果是对称，则`{t1.left, t2.right}`, `{t1.right, t2.left}`分别对称。
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.val == t2.val)
            && isMirror(t1.right, t2.left)
            && isMirror(t1.left, t2.right);
    }
}
```

>算法2：bfs迭代
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        q.add(root);
        while (!q.isEmpty()) {
            TreeNode t1 = q.poll();
            TreeNode t2 = q.poll();
            if (t1 == null && t2 == null) continue;
            if (t1 == null || t2 == null) return false;
            if (t1.val != t2.val) return false;
            q.add(t1.left);
            q.add(t2.right);
            q.add(t1.right);
            q.add(t2.left);
        }
        return true;
    }
}
```

#### [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
Invert a binary tree.

>Example 1:
<code><pre>
Input:
     4
   /   \\
  2     7
 / \\   / \\
1   3 6   9
Output:
     4
   /   \\
  7     2
 / \\   / \\
9   6 3   1
</code></pre>

>算法1：递归，左边右边交换
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode right = invertTree(root.right);
        TreeNode left = invertTree(root.left);
        root.left = right;
        root.right = left;
        return root;
    }
}
```

>算法2：迭代，队列辅助
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode current = queue.poll();
            TreeNode temp = current.left;
            current.left = current.right;
            current.right = temp;
            if (current.left != null) queue.add(current.left);
            if (current.right != null) queue.add(current.right);
        }
        return root;
    }
}
```

#### [House Robber III](https://leetcode.com/problems/house-robber-iii/)

The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

>Example 1:
<code><pre>
Input: [3,2,3,null,3,null,1]
     3
    / \\
   2   3
    \\   \\
     3   1
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
</code></pre>

>Example 2:
<code><pre>
Input: [3,4,5,1,3,null,1]
     3
    / \\
   4   5
  / \\   \\
 1   3   1
Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
</code></pre>

>算法：rob or not rob, 进行递归
`if rob:
       then rob left.left, left.right, right.left, right.right
 else not rob:
        then rob left and right
 value = max(rob, notRob);
```
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {

       private HashMap<TreeNode, Integer> cache = new HashMap<>();

       public int rob(TreeNode root) {

           int maxRob = 0;
           if (root == null) {
               return 0;
           }

           if (this.cache.containsKey(root)) {
               return this.cache.get(root);
           }

           int valueRob = root.val;
           int valueNotRob = 0;
           if (root.left != null) {
               valueRob += this.rob(root.left.left);
               valueRob += this.rob(root.left.right);
               valueNotRob += this.rob(root.left);
           }

           if (root.right != null) {
               valueRob += this.rob(root.right.left);
               valueRob += this.rob(root.right.right);
               valueNotRob += this.rob(root.right);
           }
           maxRob = Math.max(valueRob, valueNotRob);
           this.cache.put(root, maxRob);
           return maxRob;
       }

}
```

### BinaryTree Path
递归过程中状态的缓存和传递，如果结果要返回具体path, 使用回溯。一个常用技巧是，递归函数返回结果可能和要求的目标是两个变量，这个时候可以用一个全局遍历来存求解目标。

#### [Path Sum](https://leetcode.com/problems/path-sum/)
Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

Note: A leaf is a node with no children.

>Example 1:
<code><pre>
Given the below binary tree and sum = 22,
      5
     / \\
    4   8
   /   / \\
  11  13  4
 /  \      \\
7    2      1
return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
</code></pre>

>算法：
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {

    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null){
            return false; //end condition
        }

        if ((root.left == null) && (root.right == null)) {
            return (root.val == sum); // leaf node
        }
        return this.hasPathSum(root.left, sum - root.val)
                || this.hasPathSum(root.right, sum - root.val);
    }
}
```
注意这里的leaf node有定义，左右都没有叶子，因此不能用条件`root==null`作为判断。
sum状态用`sum-root.val`即可递推。

#### [Path Sum II](https://leetcode.com/problems/path-sum-ii/)
Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Note: A leaf is a node with no children.

>Example 1:
<code><pre>
Given the below binary tree and sum = 22,
      5
     / \\
    4   8
   /   / \\
  11  13  4
 /  \    / \\
7    2  5   1
Return:
[
   [5,4,11,2],
   [5,8,4,5]
]
</code></pre>

>算法：前序遍历，递归过程保留一个path，通过回溯加减节点。
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        this.helper(root, new ArrayList<>(), sum, res);
        return res;
    }

    private void helper(TreeNode root, List<Integer> path, int sum, List<List<Integer>> res) {
        if (root == null) {
            return;
        }

        path.add(root.val);
        if ((root.left == null) && (root.right == null)) {
            if (root.val == sum) {
                res.add(new ArrayList<>(path));
            }
        }
        this.helper(root.left, path, sum-root.val, res);
        this.helper(root.right, path, sum-root.val, res);
        path.remove(path.size()-1);

    }
}
```


#### [Path Sum III](https://leetcode.com/problems/path-sum-iii/)
You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

>Example 1:
<code><pre>
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
      10
     /  \\
    5   -3
   / \\    \\
  3   2   11
 / \\   \\
3  -2   1
Return 3. The paths that sum to 8 are:
1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
</code></pre>

>算法1： 递归，sum(root) = sumToRoot(root) + sum(root.left) + sum(root.right)
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public int pathSum(TreeNode root, int sum) {
        if (root == null){
            return 0;
        }
        return pathSumFrom(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }

    private int pathSumFrom(TreeNode node, int sum) {
        if (node == null){
            return 0;
        }
        return (node.val == sum ? 1 : 0)
                + pathSumFrom(node.left, sum - node.val) + pathSumFrom(node.right, sum - node.val);
    }
}
```

>算法2： 递归一次，缓存所有前序的结果（回溯加减），注意用数组不要用ArrayList
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    private int count = 0;

    public int pathSum2(TreeNode root, int sum) {
        int[] cache = new int[1001];
        this.helper(root, sum, cache, 0);
        return count;
    }

    private void helper(TreeNode root, int sum, int[] cache, int pos) {
        if (root == null) {
            return;
        } else {
            for (int i = 0; i <= pos; i++) {
                if (cache[i] + root.val == sum) { // i==pos时候，包括root.val
                    count++;
                }
                cache[i] = cache[i] + root.val;
            }
            this.helper(root.left, sum, cache, pos + 1);
            this.helper(root.right, sum, cache, pos + 1);
            for (int i = 0; i <= pos; i++) {
                cache[i] = cache[i] - root.val;
            }

        }
    }
}
```

#### [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

>Example 1:
<code><pre>
Input: [1,2,3]
       1
      / \\
     2   3
Output: 6
</code></pre>

>Example 2:
<code><pre>
Input: [-10,9,20,null,null,15,7]
   -10
   / \\
  9  20
    /  \\
   15   7
Output: 42
</code></pre>

>算法：逐个节点找包括该节点的maxSum，某个节点的`maxSum=max(0, maxSum(node.left))+max(0, maxSum(node.right))+node.val`
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
private int res = Integer.MIN_VALUE;

    //1ms, 99.86%
    public int maxPathSum(TreeNode root) {
        maxPathDown(root);
        return res;
    }

    private int maxPathDown(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left = Math.max(0, this.maxPathDown(node.left));
        int right = Math.max(0, this.maxPathDown(node.right));
        res = Math.max(res, left + right + node.val);
        return Math.max(left, right) + node.val;
    }
}
```


### Binary Search Tree (BST)
bst由于有顺序，因此遍历的顺序也非常重要。
#### [Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/)
Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.
>Example:
<code><pre>
Input: The root of a Binary Search Tree like this:
              5
            /   \\
           2     13
Output: The root of a Greater Tree like this:
             18
            /   \\
          20     13
</code></pre>

>算法1：通过in-order dfs, 先右后左，就是从大到小访问bst。
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    private int sum = 0;

    public TreeNode convertBST(TreeNode root) {
        if (root == null) {
            //do nothing
        } else {
            this.convertBST(root.right);
            sum += root.val;
            root.val = sum;
            this.convertBST(root.left);
        }
        return root;
    }
}
```
这个问题的本质是如何按顺序访问BST. 一点tips是从最大的开始去sum.

#### [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

>Example 1:
<code><pre>
Input:
    2
   / \\
  1   3
Output: true
</code></pre>

>算法： 前序遍历, 注意边界条件,LONG
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {

    public boolean isValidBST(TreeNode root) {
        return this.helper(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean helper(TreeNode root, long min, long max) {

        if (root == null) {
            return true; //end
        }

        if ((root.val <= min) || (root.val >= max)) {
            return false; //in-valid
        }

        return this.helper(root.left, min, root.val)
                && this.helper(root.right, root.val, max);
    }
}
```

### Binary Tree 构建
数的构建关键是顺序，顺序可能会依赖辅助的数据结构。

#### [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
Given preorder and inorder traversal of a tree, construct the binary tree.

You may assume that duplicates do not exist in the tree.

For example, given
>Example 1:
<code><pre>
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:
    3
   / \\
  9  20
    /  \\
   15   7
</code></pre>

>算法：通过stack来判断顺序:
```
//p是preorder数组下标, i为inorder数组下标,都从０开始
  while (p is valid){
    if (preorder[p] != inorder[p]){
        p++加入左边节点
        左边节点push to stack
    }else{
        在stack中弹出节点，如果==inorder[i]中，则i++，这样pop和i++直到不等；//目的是找到应用加入右边节点的之前push的节点
        p++加入右边节点
        右边节点push to stack
    }
  }
```
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        TreeNode root = null;
        if ((preorder != null) && (preorder.length > 0)) {
            int p = 0, i = 0;
            root = new TreeNode(preorder[p]);
            TreeNode current = root;
            LinkedList<TreeNode> stack = new LinkedList<>();
            stack.add(current);

            while (p < preorder.length - 1) {
                if (preorder[p] != inorder[i]) {
                    current.left = new TreeNode(preorder[++p]);
                    current = current.left;
                } else {
                    while ((stack.size() > 0) && (stack.getLast().val == inorder[i])) {
                        current = stack.removeLast();
                        i++;
                    }
                    current.right = new TreeNode(preorder[++p]);
                    current = current.right;
                }
                stack.addLast(current);
            }
        }
        return root;
    }
}
```


### Trie
#### [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

Implement a trie with insert, search, and startsWith methods.

>Example 1:
<code><pre>
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false
trie.startsWith("app"); // returns true
trie.insert("app");
trie.search("app");     // returns true
</code></pre>

Note:
You may assume that all inputs are consist of lowercase letters a-z.
All inputs are guaranteed to be non-empty strings.

>算法：
+ 时间复杂度O(N)
+ 空间复杂度O(N)

```java
class Trie {

    private TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        this.root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        if (word != null) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                if (!cur.containsKey(c)) {
                    cur.put(c, new TrieNode());
                }
                cur = cur.get(c);
            }
            cur.setEnd();
        }
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode cur = root;
        if (word != null) {
            for (char c : word.toCharArray()) {
                if (!cur.containsKey(c)) {
                    return false;
                }
                cur = cur.get(c);
            }
        }
        return cur.isEnd();
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode cur = root;
        if (prefix != null) {
            for (char c : prefix.toCharArray()) {
                if (!cur.containsKey(c)) {
                    return false;
                }
                cur = cur.get(c);
            }
        }
        return true;
    }

    class TrieNode {
        // R links to node children
        private TrieNode[] links;

        private final int R = 26;

        private boolean isEnd;

        public TrieNode() {
            links = new TrieNode[R];
        }

        public boolean containsKey(char ch) {
            return links[ch -'a'] != null;
        }

        public TrieNode get(char ch) {
            return links[ch -'a'];
        }

        public void put(char ch, TrieNode node) {
            links[ch -'a'] = node;
        }

        public void setEnd() {
            isEnd = true;
        }

        public boolean isEnd() {
            return isEnd;
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
