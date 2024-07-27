#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <random>
#include <atomic>
#include <csignal>
#include <optional>
#include <fstream>

using namespace std;

// 1382. Balance a Binary Search Tree
/*

Given the root of a binary search tree, return a balanced binary search tree with the same node values. If there is more than one answer, return any of them.

A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than 1.

 

Example 1:


Input: root = [1,null,2,null,3,null,4,null,null]
Output: [2,1,3,null,null,null,4]
Explanation: This is not the only correct answer, [3,1,4,null,2] is also correct.
Example 2:


Input: root = [2,1,3]
Output: [2,1,3]
 

Constraints:

The number of nodes in the tree is in the range [1, 104].
1 <= Node.val <= 105mpty string reads the same forward and backward, it is a palindrome.

*/
// Hint 
// Convert the tree to a sorted array using an in-order traversal.
// Construct a new balanced tree from the sorted array recursively.
//



struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};



void createArray(TreeNode * node, std::vector<int>& array){

  if(!node) return;
  
  createArray(node->left,array);
  array.push_back(node->val);
  createArray(node->right,array);

}

TreeNode* newTree(std::vector<int> array, int left, int right){
  
  // Binary search
  if(left > right) {   
    return nullptr;
  };
  int mid = left + (right - left)/2;

  TreeNode *node = new TreeNode(array[mid]);
  node->left = newTree(array,left, mid - 1);
  node->right = newTree(array, mid + 1, right);

  return node;
}


TreeNode* balanceBST(TreeNode* root) {

  std::vector<int> sortedArray;
       
  createArray(root,sortedArray);
  int left = 0;
  int right = sortedArray.size() - 1;

  return newTree(sortedArray,left,right);
}


/* ChatGPT DSW algorithm solution... notworking right now... ToDo

void treeToVine(TreeNode* root) {
        TreeNode* grandParent = nullptr;
        TreeNode* node = root;
        while (node != nullptr) {
            if (node->left != nullptr) {
                TreeNode* oldNode = node;
                node = node->left;
                oldNode->left = node->right;
                node->right = oldNode;
                if (grandParent != nullptr) {
                    grandParent->right = node;
                }
            } else {
                grandParent = node;
                node = node->right;
            }
        }
    }

    void compress(TreeNode* root, int count) {
        TreeNode* node = root;
        for (int i = 0; i < count; i++) {
            TreeNode* temp = node->right;
            node->right = temp->right;
            node = node->right;
            temp->right = node->left;
            node->left = temp;
        }
    }

    void vineToTree(TreeNode* root, int size) {
        int fullCount = log2(size + 1) - 1;
        int leaves = size + 1 - pow(2, fullCount);
        compress(root, leaves);
        size = size - leaves;
        while (size > 1) {
            compress(root, size / 2);
            size /= 2;
        }
    }

    TreeNode* balanceBST(TreeNode* root) {
        TreeNode dummy(0);
        dummy.right = root;
        treeToVine(&dummy);
        int nodeCount = 0;
        TreeNode* temp = dummy.right;
        while (temp != nullptr) {
            nodeCount++;
            temp = temp->right;
        }
        vineToTree(&dummy, nodeCount);
        return dummy.right;
    }
*/


int main() {

  TreeNode *root = new TreeNode(2);
  root->left = new TreeNode(1);
  
  root->right = new TreeNode(3);


  auto jojo = balanceBST(root);
  std::cout << "juuuj";
  return 0;
}




