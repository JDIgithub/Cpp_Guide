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

class Queue {

private:
  Node* m_first;
  Node* m_last;
  int m_length;

public:
  
  Queue(int value){
    Node *newNode = new Node(value); 
    m_first = newNode;
    m_last = newNode;
    m_length = 1;
  }

  void printQueue(){
    Node *temp = m_first;
    while(temp){
      std::cout << temp->m_value << ' ';
      temp = temp->m_next;
    }
    std::cout << std::endl;
  }

  void enqueue(int value) {
    Node *newNode = new Node(value);
    if(m_first == nullptr){
      m_first = newNode;
      m_last = newNode;
    } else {
      m_last->m_next = newNode;
      m_last = newNode;
    }
    m_length++;

  }

  int dequeue(){
    if(m_first == nullptr){
      return INT_MIN;
    }  
    Node *temp = m_first;
    int dequeuedValue = m_first->m_value;
    if(m_first == m_last){
      m_first = nullptr;
      m_last = nullptr;
    } else {
      m_first = m_first->m_next;
    }
    delete temp;
    m_length--;
    return dequeuedValue;

  }

};


class QueueUsingTwoStacks {
private:
  std::stack<int> stack1, stack2;

public:
  int front() {
    if (stack2.empty()) {
      while (!stack1.empty()) {
        stack2.push(stack1.top());
        stack1.pop();
      }
    }

    if (stack2.empty()) {
      return INT_MIN;
    }

    return stack2.top();
  }

  bool isEmpty() {
    return stack1.empty() && stack2.empty();
  }


  //   | Description:                                        |
  //   | - This method adds a new value to the end of the    |
  //   |   queue (enqueue) using 'stack1'.                   |
  //   | - Return type: void                                 |
  //   |                                                     |
  //   | Tips:                                               |
  //   | - Use 'stack1' to enqueue a new value.              |
  //   | - Simply push the new value onto 'stack1'.          |

  void enqueue(int value){
    stack1.push(value);
  }

  //   | Description:                                        |
  //   | - This method removes a value from the front of the |
  //   |   queue (dequeue) using 'stack2'.                   |
  //   | - If 'stack2' is empty, it fills 'stack2' from      |
  //   |   'stack1'.                                         |
  //   | - Return type: void                                 |
  //   |                                                     |
  //   | Tips:                                               |
  //   | - If 'stack2' is empty, pop all elements from       |
  //   |   'stack1' into 'stack2'.                           |
  //   | - Then, pop the top element of 'stack2'.            |
  void dequeue(){
    if(isEmpty()) return;
    if(stack2.empty()){
      while(!stack1.empty()){
        stack2.push(stack1.top());
        stack1.pop();
      }
    }
    stack2.pop();
  }

    
};



int main() {

  Queue *myQueue = new Queue(4);

  myQueue->printQueue();
  myQueue->enqueue(2);
  myQueue->printQueue();

  return 0;
}


