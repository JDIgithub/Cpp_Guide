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

using namespace std::literals;

bool WINDOW_CONDITION_BROKEN;
class ListNode{

public: 
  ListNode * next;

};


// Linked list: fast and slow pointer
int fn(ListNode* head) {
  ListNode* slow = head;
  ListNode* fast = head;
  int ans = 0;

  while (fast != nullptr && fast->next != nullptr) {
    // do logic
    slow = slow->next;
    fast = fast->next->next;
  }

  return ans;
}

// Reverse Linked List
ListNode* fn2(ListNode* head) {
  ListNode* curr = head;
  ListNode* prev = nullptr;
  while (curr != nullptr) {
    ListNode* nextNode = curr->next;
    curr->next = prev;
    prev = curr;
    curr = nextNode;
  }

  return prev;
}

int main(){

  std::vector<int> nums {1,1,1,2,2,2,3,3};

  // auto xx = removeDuplicates(nums);

  // std::cout << xx;

  return 0;
}