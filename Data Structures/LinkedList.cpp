#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <stack>
#include <cassert>

#include <iostream>
#include <limits>
#include <cstring>
#include <climits>


class Node{

public:
  Node(int value): m_value(value), m_next(nullptr) { 
  }
  int m_value;
  Node* m_next;
};

class LinkedList {

public:
  // Create new Node
  LinkedList(int value){
    Node *newNode = new Node(value);
    m_head = newNode;
    m_tail = newNode;
    m_length = 1;
  }
  // We need destructor to go through list and delete all nodes
  ~LinkedList(){
    Node* temp = m_head;
    while(m_head != nullptr){
      m_head = m_head->m_next;
      delete temp;
    }
  }

  // Create new Node and add it to the end
  void append(int value){
    Node *newNode = new Node(value);

    if(m_tail == nullptr){      // If the list is empty
      m_head = newNode;
      m_tail = newNode;     
    } else {
      m_tail->m_next = newNode;
      m_tail = newNode;
    }
    m_length++;
  }


  // Create new Node and add it to the begging
  void prepend(int value){
    
    Node *newNode = new Node(value);
    if(m_head == nullptr){
      m_head = newNode;
      m_tail = newNode;
    } else{
      newNode->m_next = m_head;
      m_head = newNode;
    }
    m_length++;
  
  }
  // Set value of already existing node
  bool set(int index, int value){
    
    Node *temp = get(index);
    if(temp != nullptr){
      temp->m_value = value;
      return true;
    }
    return false;
  }

  // Insert new node to the list at the given index
  bool insert(int index, int value){
    
    if (index == 0){
      prepend(value);
      return true;
    } else if(index == m_length){
      append(value);
      return true;
    }
    Node *temp = get(index-1);
    if(temp != nullptr){
      Node *newNode = new Node(value);
      newNode->m_next = temp->m_next;
      temp->m_next = newNode;
      m_length++;
      return true;
    }

    return false;
  }
  


  void deleteLast(){

    if(m_head == nullptr){  // If list is empty
      return;
    } else if (m_head->m_next == nullptr) { // If list has only 1 element
      delete m_head;
      m_head = nullptr;
      m_tail = nullptr;
    } else {
      Node *temp = m_head;
      while(temp->m_next != nullptr){
          m_tail = temp;
          temp = temp->m_next;
      }
      delete temp;
      // It is now pointing to garbage so we need to assign nullptr instead
      m_tail->m_next = nullptr;   
    }
    m_length--;
    return;
  }


  void deleteFirst(){
    if(m_head == nullptr){  // If list is empty
      return;
    } 
    Node* temp = m_head;
    if (m_head->m_next == nullptr) { // If list has only 1 element
      m_head = nullptr;
      m_tail = nullptr;
    } else { 
      m_head = m_head->m_next;
    }
    delete temp;
    m_length--;
    return;
  }

  void deleteNode(int index){

    if(index < 0 || index >= m_length)
    if(index == 0){
      return deleteFirst();
    } else if (index == (m_length-1)){
      return deleteLast();
    }

    Node *prev = get(index-1);
    Node *temp = prev->m_next;
    prev->m_next = temp->m_next;
    delete temp;
    m_length--;
    return;
  }

  void printList(){
    Node *printNode = m_head;
    while(printNode != nullptr){
      std::cout << printNode->m_value << ' ';
      printNode = printNode->m_next;
    }
    std::cout << std::endl;
  }

  Node *get(int index) {

    if (index < 0 || index >= m_length){
      return nullptr;
    }
    Node *temp = m_head;
    for(size_t i{0}; i < index; ++i){
      temp = temp->m_next;
    }
    return temp;
  } 

  Node *getHead(){
    return m_head;
  }
  
  Node *getTail(){
    return m_tail;
  }
  
  int getLength(){
    return m_length;
  }

  // !! Very Common Interview Question !! ----------------------------------------------------

  // reverse the linked list
  void reverse() {

    // Switch head and tail 
    Node *temp = m_head;
    m_head = m_tail;
    m_tail = temp;
    // We need before and after the temp node
    Node *beforeTemp = nullptr;     // There is no before now so it is nullptr
    Node *afterTemp;    

    for(size_t i{0}; i < m_length; ++i){
      afterTemp = temp->m_next;   // setting after temp
      temp->m_next = beforeTemp;  // Reversing link from the next node to the previous node
      beforeTemp = temp;          // shift beforeTemp to temp for the next cycle
      temp = afterTemp;           // shift temp to after temp for the next cycle
    }    
  }
  // -----------------------------------------------------------------------------------------


  // LL coding interview 1 -------------------------------------------------------------------

  // Find the middle node of the linked list
  // Return type Node*
  // Tips:
  //  Use two pointers: slow and fast
  //  slow moves one step, fast moves two
  //  When fast reaches end the slow is in the middle
  //  Return slow as the middle node 


  Node* findMiddleNode(){

    if(m_head == nullptr){          // Empty list
      return nullptr;
    } else {

      Node *slow = m_head;          //
      Node *fast = m_head;
      while (fast != nullptr && fast->m_next != nullptr){ // If it will be even length of LL this will return point to the right of the middle
         slow = slow->m_next;
         fast = fast->m_next->m_next; 
      }
      return slow;

    }
  }

  // -----------------------------------------------------------------------------------------

  // -----------------------------------------------------------------------------------------


  // LL coding interview 2 -------------------------------------------------------------------

  // Implement function called hasLoop() to detect if a given LL contains loop or not



private:
  Node* m_head;
  Node* m_tail;
  int m_length;

};

int main() {

  LinkedList *myLinkedList = new LinkedList(4);
  myLinkedList->printList();
  myLinkedList->append(2);
  myLinkedList->printList();
  myLinkedList->prepend(1);
  myLinkedList->printList();
  myLinkedList->insert(1,8);
  myLinkedList->printList();
//  myLinkedList->deleteNode(2);
//  myLinkedList->printList();
  myLinkedList->reverse();
  myLinkedList->printList();

  return 0;
}


