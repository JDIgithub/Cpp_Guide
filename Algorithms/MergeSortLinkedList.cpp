
#include <iostream>
#include <vector>




struct Node {
  
  int data;
  Node* next;

  Node(int val) : data(val), next(nullptr) {}

};

Node* mergeSort(Node* head);
Node* findMiddle(Node* head);
Node* mergeTwoLists(Node* left, Node* right);

int main() {
  
  Node* head = new Node(4);
  head->next = new Node(2);
  head->next->next = new Node(1);
  head->next->next->next = new Node(3);

  head = mergeSort(head);

  // Print the sorted list
  Node* current = head;
  while (current) {
     std::cout << current->data << " ";
    current = current->next;
  }

  return 0;
}

Node* findMiddle(Node* head) {
  
  if (head == nullptr || head->next == nullptr) {
    return head;
  }

  Node *slow = head, *fast = head->next;
  while (fast && fast->next) {
    slow = slow->next;
    fast = fast->next->next;
  }

  return slow;
}

Node* mergeTwoLists(Node* left, Node* right) {
  
  if (!left) return right;
  if (!right) return left;

  if (left->data <= right->data) {
    left->next = mergeTwoLists(left->next, right);
    return left;
  } else {
    right->next = mergeTwoLists(left, right->next);
    return right;
  }
}

Node* mergeSort(Node* head) {
  
  if (!head || !head->next) {
    return head;
  }

  Node* middle = findMiddle(head);
  Node* nextOfMiddle = middle->next;
  middle->next = nullptr;

  Node* left = mergeSort(head);
  Node* right = mergeSort(nextOfMiddle);

  return mergeTwoLists(left, right);
}





