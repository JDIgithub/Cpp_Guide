#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <stack>


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




class StackVector {

private:
  std::vector<int> stackVector;
    
public:
  std::vector<int>& getStackVector() {
    return stackVector;
  }
    
  void printStack() {
    for (int i = stackVector.size() - 1; i >= 0; i--) {
      std::cout << stackVector[i] << std::endl;
    }
  }
    
  bool isEmpty() {
    return stackVector.size() == 0;
  }
    
  int peek() {
    if (isEmpty()) {
      return int();
    } else {
      return stackVector[stackVector.size() - 1];
    }
  }
    
  int size() {
    return stackVector.size();
  }
    
  void push(int value){
    stackVector.push_back(value); 
  }

  void pop(){
    if(!stackVector.empty()){
      stackVector.pop_back(); 
    }
  }
};

//   | Description:                                        |
//   | - This function reverses the input string 'str'.    |
//   | - Uses a stack to hold the characters.              |
//   | - Pops from stack and appends to 'reversedString'.  |
//   | - Return type: string                               |
//   |                                                     |
//   | Tips:                                               |
//   | - Use a stack to hold each character of the string. |
//   | - Push each character of 'str' onto the stack.      |
//   | - Pop from the stack and append to 'reversedString' |
//   |   until the stack is empty.                         |
//   | - Return the reversed string.                       |

std::string reverseString(const std::string& str) {
  
  std::stack<char> charStack;
  std::string reverseStr;
    
  for(char c: str){
    charStack.push(c);
  }  
  while(!charStack.empty()){
    reverseStr += charStack.top();
    charStack.pop();
  }  
  return reverseStr;
}

//   | Description:                                        |
//   | - This function checks if the input string          |
//   |   'parentheses' has balanced parentheses.           |
//   | - Uses a stack to hold the open parentheses.        |
//   | - Return type: bool                                 |
//   |                                                     |
//   | Tips:                                               |
//   | - Use a stack to hold open parentheses.             |
//   | - Push '(' onto the stack.                          |
//   | - When encountering ')', check if stack is empty    |
//   |   or top of stack is not '('. If so, return false.  |
//   | - Otherwise, pop from the stack.                    |
//   | - At the end, if stack is empty, return true.       |
//   | - Otherwise, return false.                          |

bool isBalancedParentheses(const std::string& parentheses) {

  std::stack<char> parStack;
  std::unordered_map<char, char> brackets = {{')', '('}, {'}', '{'}, {']', '['}};
  
  for (char c : parentheses) {
    // If it's an opening bracket, push it onto the stack.
    if (c == '(' || c == '{' || c == '[') {
      parStack.push(c);
    } else if (c == ')' || c == '}' || c == ']') {
      // If it's a closing bracket, check for the corresponding opening bracket.
      if (parStack.empty() || parStack.top() != brackets[c]) {
        return false;
      }
      parStack.pop();
    }
  }
  return parStack.empty();
}

//   | Description:                                         |
//   | - This function sorts the input stack 'inputStack'.  |
//   | - Uses an additional stack for sorting.              |
//   | - Return type: void                                  |
//   |                                                      |
//   | Tips:                                                |
//   | - Create an additional stack.                        |
//   | - Pop elements from 'inputStack' and push them       |
//   |   into 'additionalStack' in sorted order.            |
//   | - Use a temporary variable to hold the top element   |
//   |   of 'inputStack'.                                   |
//   | - Move elements back to 'additionalStack' if needed. |
//   | - Finally, move all elements back to 'inputStack'.   |

void sortStack(std::stack<int>& inputStack) {
  // Additional stack to temporarily store and sort elements
  std::stack<int> additionalStack;
  // Iterate until the input stack becomes empty
  while (!inputStack.empty()) {
      // Store the top element of the input stack in temp
      int temp = inputStack.top();
      inputStack.pop();
      // Move elements from additionalStack to inputStack while
      // they are greater than temp
      while (!additionalStack.empty() && additionalStack.top() > temp) {
          inputStack.push(additionalStack.top());
          additionalStack.pop();
      }
      // Push temp onto the additionalStack
      additionalStack.push(temp);
  }
  // Move sorted elements back to inputStack
  while (!additionalStack.empty()) {
      inputStack.push(additionalStack.top());
      additionalStack.pop();
  }
}








int main() {

  std::stack<int> s;
  s.push(1);
  s.push(7);
  s.push(2);
  s.push(5);
  sortStack(s);

  std::cout << s.top() << std::endl;
  s.pop();
  std::cout << s.top() << std::endl;
  s.pop();
  std::cout << s.top() << std::endl;
  s.pop();
  std::cout << s.top() << std::endl;
  s.pop();

  return 0;
}


