#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>


using namespace std;

// 979. Distribute Coins In Binary Tree

/*

You are given the root of a binary tree with n nodes where each node in the tree has node.val coins. There are n coins in total throughout the whole tree.
In one move, we may choose two adjacent nodes and move one coin from one node to another. A move may be from parent to child, or from child to parent.
Return the minimum number of moves required to make every node have exactly one coin.

 

Example 1:


Input: root = [3,0,0]
Output: 2
Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.
Example 2:


Input: root = [0,3,0]
Output: 3
Explanation: From the left child of the root, we move two coins to the root [taking two moves]. Then, we move one coin from the root of the tree to the right child.
 

Constraints:

The number of nodes in the tree is n.
1 <= n <= 100
0 <= Node.val <= n
The sum of all Node.val is n.


*/



/*
The idea is that the child give x=(y-1) coins to parent, if he has y coins. (if y=0=> x=-1 then parent should give the child 1 coin)
Add an extra parameter TreeNode* parent= NULL instead of a helper function.
The number of moves comes from its left & right subtrees &
When x>0 is to give, when x<0 is to obtain. So moves+=abs(x)



Why not Pre-Order? Why Post-order?
  As the root of a subtree, it's unkown how many descendants it has before performing any kind of transversal. The thinking of parent-gives-coins-to-children will not work.
  Use post-order traversal, because every node has a parent except for the root.
  
*/


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};




static int distributeCoins(TreeNode* root, TreeNode* parent= NULL) {
  if (!root) return 0;
  int  moves = 0;
  moves += distributeCoins(root->left, root);
  moves += distributeCoins(root->right, root);
  int gift = root->val-1;// Give x coins to parent node or take x coins from them if node->val < 1
  if (parent) parent->val += gift; // now parent's coins
  // When val < 0 then it needs to get coin which also takes moves so abs(-x) -> x moves
  moves+=abs(gift);

  return moves;
}


class IterativeSolution {
public:

using node2 = tuple<TreeNode*, TreeNode*, bool>; // Node, parent & visited flag
static int distributeCoins(TreeNode* root) {
  if (!root) return 0;
  vector<node2> stack = {{root, NULL, 0}};
  int moves=0;
  while (!stack.empty()) {
    auto [node, parent, visited] = stack.back();
    stack.pop_back();
      if (!visited) {
        // Mark the node as visited & repush
        stack.emplace_back(node, parent, 1);
        // Push right and left children onto the stack
        if (node->right) stack.emplace_back(node->right, node, 0);
        if (node->left)  stack.emplace_back(node->left, node, 0);
      
      } else {
        int x=node->val-1;// give x coins to parent node
        if (parent) parent->val += x; // now parent's coins
        moves+=abs(x);
      }
  }
  return moves;
}

};

int main(){

  TreeNode *root = new TreeNode(7);
  root->left = new TreeNode(0);
  root->left->left = new TreeNode(0);
  root->left->right = new TreeNode(0);
  root->right = new TreeNode(0);
  root->right->left = new TreeNode(0);
  root->right->right = new TreeNode(0);


  std::cout << distributeCoins(root);


  int x = 66;



  return 0;
}






