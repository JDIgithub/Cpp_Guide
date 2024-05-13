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

// 2816. Double a Number Represented as a Linked List

/*

You are given the head of a non-empty linked list representing a non-negative integer without leading zeroes.
Return the head of the linked list after doubling it.

Example 1:

Input: head = [1,8,9] -> 189
Output: [3,7,8]       <- 189 * 2 = 378  
Explanation: The figure above corresponds to the given linked list which represents the number 189. Hence, the returned linked list represents the number 189 * 2 = 378.

Example 2:

Input: head = [9,9,9]
Output: [1,9,9,8]
Explanation: The figure above corresponds to the given linked list which represents the number 999. Hence, the returned linked list reprersents the number 999 * 2 = 1998. 
 

Constraints:

The number of nodes in the list is in the range [1, 104]
0 <= Node.val <= 9
The input is generated such that the list represents a number that does not have leading zeros, except the number 0 itself.

*/

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// Recursion My Solution 
// Ot(N)
// Os(N)
void doubleNode(ListNode * head){

  if(!head) return;
  doubleNode(head->next);
  head->val = head->val * 2;
  if(head->next != nullptr && head->next->val > 9){
    head->next->val -= 10; 
    head->val += 1;
  }
}

ListNode* doubleIt(ListNode* head) {

  if(!head) return {};
  doubleNode(head);
  if(head->val > 9){
    head->val -= 10; 
    ListNode *newHead = new ListNode(1,head);
    return newHead;
  }
  
  return head;
}

// Using Two Pointers - One for previous node so we can go in order and still add carry-on to the previous node
// Ot(n)
// Space Complexity Os(1)
ListNode* doubleIt(ListNode* head) {
  ListNode* curr = head;
  ListNode* prev = nullptr;
  // Traverse the linked list
  while (curr != nullptr) {
    int twiceOfVal = curr->val * 2;
    // If the doubled value is less than 10
    if (twiceOfVal < 10) {
      curr->val = twiceOfVal;
    } 
    // If doubled value is 10 or greater
    else if (prev != nullptr) { // other than first node
      // Update current node's value with units digit of the doubled value
      curr->val = twiceOfVal % 10;
      // Add the carry to the previous node's value
      prev->val = prev->val + 1;
    } 
    // If it's the first node and doubled value is 10 or greater
    else { // first node
      // Create a new node with carry as value and link it to the current node
      head = new ListNode(1, curr);
      // Update current node's value with units digit of the doubled value
      curr->val = twiceOfVal % 10;
    }
    // Update prev and curr pointers
    prev = curr;
    curr = curr->next;
  }

  return head;
}


int main(){

  ListNode *head = new ListNode(1);
  head->next = new ListNode(8);
  head->next->next = new ListNode(9);

  doubleIt(head);

  int jojo = 42;
  return 0;
}


