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


int main() {

  Stack *myStack = new Stack(4);
  myStack->printStack();
  myStack->push(1);
  myStack->printStack();

  std::cout << myStack->pop() << ' ' << myStack->pop() << ' ' << myStack->pop() << std::endl;


  return 0;
}


