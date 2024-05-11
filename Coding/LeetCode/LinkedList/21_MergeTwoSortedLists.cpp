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

// 21. Merge Two Sorted Lists

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


ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {

  if(list1 == nullptr) return list2;
  if(list2 == nullptr) return list1;

  ListNode *mergedHead;     // Need to create head that will be returned
  ListNode *mergedTail;     // Pointer for going through the newly created merged list

  if(list1->val < list2->val) {
    mergedHead = list1;
    list1 = list1->next;
  } else {
    mergedHead = list2;
    list2 = list2->next;
  }
  mergedTail = mergedHead;

  while(list1 != nullptr && list2 != nullptr){

    if(list1->val < list2->val){
      mergedTail->next = list1;
      list1 = list1->next;
    } else {
      mergedTail->next = list2;
      list2 = list2->next;
    }
    mergedTail = mergedTail->next;
  }

  // Adding the remaining elements from the list that is left
  if(list1 != nullptr){
    mergedTail->next = list1;
  } else {
    mergedTail->next = list2;
  }

  return mergedHead;
}









int main(){

  ListNode *head1 = new ListNode(1);
  head1->next = new ListNode(2);
  head1->next->next = new ListNode(4);

  ListNode *head2 = new ListNode(1);
  head2->next = new ListNode(3);
  head2->next->next = new ListNode(4);


  
  std::cout << mergeTwoLists(head1,head2) << std::endl;
  

  return 0;
}


