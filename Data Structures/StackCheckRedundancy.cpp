#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <stack>


#include <iostream>

class Node{

public:
  Node(int value): m_value(value), m_next(nullptr) { 
  }
  int m_value;
  Node* m_next;
};

class Stack {

public:
  // Create new Node
  Stack(int value){
    Node *newNode = new Node(value);
    m_top = newNode;
    m_height = 1;
  }

  // We need destructor to go through stack and delete all nodes
  ~Stack(){
    Node* temp = m_top;
    while(m_top != nullptr){
      m_top = m_top->m_next;
      delete temp;
    }
  }


  void printStack(){
    Node *temp = m_top;
    while(temp != nullptr){
      std::cout << temp->m_value << std::endl;
      temp = temp->m_next;
    }

  }

  // We are using linked list so this method is the same as prepend in the LL class
  // But because we do need to worry about tail we do not need the if check there
  void push(int value){
    Node *newNode = new Node(value);
    newNode->m_next = m_top;
    m_top = newNode;
    m_height++;
  }

  // Very similar to deleteFirst with LinkedList but here we will return the value of the popped node
  int pop(){

    if (m_height == 0) { return INT_MIN;} // We need to return the least probable number that could be in the stack
    Node *temp = m_top;
    int poppedValue = m_top->m_value;
    m_top = m_top->m_next;
    m_height--;
    delete temp;
    return poppedValue;
  }


private:
  Node* m_top;
  size_t m_height;

};


template <typename TValue>
class MyStack {
public:
  using TValuePtr = TValue*;

private:
  TValuePtr* mElements;
  std::size_t mSize;
  std::size_t mCapacity;

public:
  MyStack(std::size_t initialCapacity): mElements(new TValuePtr[initialCapacity]), mSize(0), mCapacity(initialCapacity) {}
  ~MyStack() {
    // Potential memory leak here as individual elements are not deleted
    delete[] mElements;
  }
// Correction:
  ~MyStack() {
    for (std::size_t i = 0; i < mSize; ++i) {
        delete mElements[i]; // Delete each dynamically allocated element
    }
    delete[] mElements; // Delete the array of pointers
  }

  void push(TValue value) {
    if (mSize >= mCapacity) {
      // Reallocate memory if capacity is reached
      std::size_t newCapacity = mCapacity * 2;
      TValuePtr* newElements = new TValuePtr[newCapacity];
      for (std::size_t i = 0; i < mSize; ++i) { newElements[i] = mElements[i]; }
      delete[] mElements;
      mElements = newElements;
      mCapacity = newCapacity;
    }
    mElements[mSize++] = new TValue(value); // Memory allocated here
  }

  TValue pop() {
    if (mSize == 0) throw std::out_of_range("Stack is empty");
    TValuePtr elem = mElements[--mSize];
    TValue value = *elem;
    delete elem; // Fix memory leak by deleting the element
    return value;
  }

  bool isEmpty() const {
    return mSize == 0;
  }
    
    // Other methods like top(), size() could be here
};

int main() {


  return 0;
}


