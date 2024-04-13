#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>


//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
//  Problem: Merge k Sorted Lists
//    You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
//
//    Merge all the linked-lists into one sorted linked-list and return it.
//
//  Example 1:
//    Input: lists = [[1,4,5],[1,3,4],[2,6]]
//    Output: [1,1,2,3,4,4,5,6]
//    Explanation: The linked-lists are:
//    [1->4->5,1->3->4,2->6]
//    merging them into one sorted list:
//      1->1->2->3->4->4->5->6
//
//  Example 2:
//    Input: lists = []
//    Output: []
//
//  Example 3:
//    Input: lists = [[]]
//    Output: []
//
//  Constraints:
//    k == lists.length
//    0 <= k <= 10^4
//    0 <= lists[i].length <= 500
//    -10^4 <= lists[i][j] <= 10^4
//    lists[i] is sorted in ascending order.
//    The sum of lists[i].length won't exceed 10^4.
//  Approach:
//    This problem can be approached in several ways, one of the most common being using a min-heap (or priority queue) to efficiently find the next node to be added 
//    to the merged list at each step.
//
//  Implementation Steps:
//    Min-Heap: 
//      Create a min-heap to keep track of the head of each list. The heap will order the nodes by their value, ensuring we always have access to the smallest current node.
//    Initialization: 
//      Initialize the heap by adding the head of each list to it.
//    Merging: 
//      Remove the smallest element from the heap and add it to the merged list. Then, if this element has a next node, add the next node to the heap. 
//      Repeat this process until the heap is empty.
//    Return the merged list.
//






// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};


ListNode* mergeKLists(std::vector<ListNode*>& lists) {

  // !!! Use of Priority Queue

  // This lambda function returns true if the value of the first node is greater than the value of the second node, making the queue prioritize nodes with smaller values.
  auto compare = [](ListNode* l1, ListNode* l2) { return l1->val > l2->val; };
  // Default Behavior: By default, a priority queue in C++ acts like a max heap, meaning without any custom comparator, 
  // the largest element (the one that would come last in an ascending sort) is given the highest priority.

  // Comparator's Role Reversed: When you supply a comparator like [](ListNode* l1, ListNode* l2) { return l1->val > l2->val; }, you're essentially defining a rule that says,
  // "An element should come later (be considered lower in priority) if it is larger." This reverses the queue's natural order, making it behave like a min heap.




  // Priority queue is implemented as heap
  // By default the largest element is on top so we need to make custom
  // ListNode * will be element typ of our priority queue
  // std::vector<ListNode*> is underlying container that the priority queue will use to store its elements
  // decltype(compare) specifies the type of the comparison function used to order the elements in the priority queue.
  std::priority_queue<ListNode*, std::vector<ListNode*>, decltype(compare)> prio_queue(compare);
  // !!!
  // Initialize the priority queue with the head of each list
  for (ListNode* list : lists) {
    if (list) prio_queue.push(list);  // this will put only 3 nodes into the priority queue
  }

  ListNode dummy(0);
  ListNode* tail = &dummy;

  while (!prio_queue.empty()) {
    ListNode* node = prio_queue.top();
    prio_queue.pop();
    tail->next = node;
    tail = tail->next;
    if (node->next) prio_queue.push(node->next);  // We need to add the next node to the priority queue
  }

  return dummy.next;

  /* Another aproach with head and tail

  ListNode* head = nullptr; // Start with no head
  ListNode* tail = nullptr; // Tail to keep track of the last node

  while (!pq.empty()) {
    ListNode* node = pq.top();
    pq.pop();
    // If head is null, set it to the node
    if (!head) {
      head = node;
      tail = head; // Tail starts at the head
    } else {
      tail->next = node; // Append node to the list
      tail = node; // Move tail to the last node
    }
    if (node->next) pq.push(node->next); // If there's more in the list, add it to the priority queue
  }
  return head; // Return the head of the merged list
  */ 


}

// Helper function to create a linked list from a vector of values
ListNode* createLinkedList(const std::vector<int>& nums) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    for (int num : nums) {
        tail->next = new ListNode(num);
        tail = tail->next;
    }
    return dummy.next;
}

// Helper function to delete a linked list to prevent memory leaks
void deleteLinkedList(ListNode* head) {
    while (head) {
        ListNode* temp = head;
        head = head->next;
        delete temp;
    }
}

int main() {
  
  // Create the input lists
  std::vector<ListNode*> lists;
  lists.push_back(createLinkedList({1, 4, 5}));
  lists.push_back(createLinkedList({1, 3, 4}));
  lists.push_back(createLinkedList({2, 6}));

  // Merge the lists
  ListNode* mergedList = mergeKLists(lists);




  return 0;
}


