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

// 2487. Removes Nodes From Linked List

/*

You are given the head of a linked list.
Remove every node which has a node with a greater value anywhere to the right side of it.
Return the head of the modified linked list.

Example 1:

Input: head = [5,2,13,3,8]
Output: [13,8]
Explanation: The nodes that should be removed are 5, 2 and 3.
- Node 13 is to the right of node 5.
- Node 13 is to the right of node 2.
- Node 8 is to the right of node 3.
Example 2:

Input: head = [1,1,1,1]
Output: [1,1,1,1]
Explanation: Every node has value 1, so no nodes are removed.
 
Constraints:

The number of the nodes in the given list is in the range [1, 105].
1 <= Node.val <= 105

*/

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// Recursion 
// T: O(N)
// S: O(N)

ListNode* removeNodes(ListNode* head) {

  if(!head) return nullptr;
  ListNode *node = head;
  ListNode *greater = removeNodes(node->next);
  node->next = greater;

  if (greater == nullptr || node->val >= greater->val) {
    return node;
  }
  
  return greater;
}

// Stack for storage
// T: O(N)
// S: O(N)

// We can also use stack
// Initialize a stack to store nodes in non-decreasing order of their values.
// Traverse the linked list:
// If the current node's value is greater than the top element of the stack, pop elements from the stack until the condition is met.
// Push the current node onto the stack.
// Reverse the stack to obtain the modified linked list.
// Return the head of the modified linked list.

// To Do


// Reversing list 
// T: O(N)
// S: O(1)
// Reverse the given linked list.
// Initialize a dummy node to hold the result.
// Traverse the reversed list, keeping nodes whose values are greater than or equal to the previous node's value.
// Reverse the resulting list to obtain the modified linked list.
// Return the head of the modified linked list.

// To Do

int main(){

  ListNode *head = new ListNode(5);
  head->next = new ListNode(2);
  head->next->next = new ListNode(13);
  head->next->next->next = new ListNode(3);
  head->next->next->next->next = new ListNode(8);

  removeNodes(head);

  int jojo = 42;
  return 0;
}


