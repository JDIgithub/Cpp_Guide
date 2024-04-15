#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
using namespace std;

// LeetCode 2. Add Two Numbers
// You are given two non-empty linked lists representing two non-negative integers. 
// The digits are stored in reverse order, and each of their nodes contains a single digit. 
// Add the two numbers and return the sum as a linked list.
// You may assume the two numbers do not contain any leading zero, except the number 0 itself.

// Input: l1 = [2,4,3], l2 = [5,6,4]
// Output: [7,0,8]
// Explanation: 342 + 465 = 807.

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};
 
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        
  ListNode dummy(0); // Use an automatic storage duration object instead of dynamic.
  ListNode* tail = &dummy; // Pointer to the dummy node.
  
  int overflow = 0;
  while (l1 != nullptr || l2 != nullptr || overflow != 0) {
    int sum = overflow;
    if (l1 != nullptr) {
      sum += l1->val;
      l1 = l1->next; // Move to next node in l1.
    }
    if (l2 != nullptr) {
      sum += l2->val;
      l2 = l2->next; // Move to next node in l2.
    }
    overflow = sum / 10; // Calculate new overflow.
    tail->next = new ListNode(sum % 10); // Create new node and link it.
    tail = tail->next; // Move tail to point to the last node.
  }
  
  return dummy.next; // Return the node after the dummy, which is the real head.
}


int main(){


  return 0;
}


