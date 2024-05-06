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

// 61. Rotate List

/*

You are given the heads of two sorted linked lists.
Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.

Example 1:

Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:

Input: list1 = [], list2 = []
Output: []

Example 3:

Input: list1 = [], list2 = [0]
Output: [0]
 
Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both list1 and list2 are sorted in non-decreasing order.
*/

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// My Solution
ListNode* myRotateRight(ListNode* head, int k) {
  
  if(head == nullptr || head->next == nullptr) return head;

  
  ListNode *tail = head;
  ListNode *newHead;
  ListNode *newTail = head;

  int size = 1;
  while(tail->next != nullptr){
    tail = tail->next;
    size++;
  }
  k = k%size;
  if(k == 0) return head; 

  for(int i = 0; i < size - k - 1;i++){
    newTail = newTail->next;
  }

  newHead = newTail->next;
  newTail->next = nullptr;
  tail->next = head;

  return newHead;
}


// Using circular Linked List
ListNode* rotateRight(ListNode* head, int k) {
  
  if(head == nullptr || head->next == nullptr) return head;
  ListNode *tail = head;

  int size = 1;
  while(tail->next != nullptr){
    tail = tail->next;
    size++;
  }

  k = k%size;               // We do not need to rotate more than size-times
  if(k == 0) return head;   // No rotation needed;

  tail->next = head;        // Making it circular Linked List
  tail = head;              // To locate new tail
  for(int i = 0; i < size - k - 1;i++){
    tail = tail->next;
  }
  head = tail->next;        // Move head to its new position
  tail->next = nullptr;     // Cut the circular Linked List into normal one

  return head;
}

int main(){

  ListNode *head = new ListNode(1);
  head->next = new ListNode(2);
  head->next->next = new ListNode(3);
  head->next->next->next = new ListNode(4);
  head->next->next->next->next = new ListNode(5);


  ListNode *newHead = rotateRight(head,20002);
  while(newHead != nullptr){

    std::cout <<  newHead->val << " ";
    newHead = newHead->next;

  }

  return 0;
}


