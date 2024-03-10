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





class MaxStack {
private:
  std::stack<int> mainStack;
  std::stack<int> maxStack;
public:
  MaxStack() { }
  void push(int x) {
    mainStack.push(x);
    if (maxStack.empty() || x >= maxStack.top()) {
      maxStack.push(x);
    }
  }
  int pop() {
    assert(!mainStack.empty()); // Ensure the stack is not empty
    int topElement = mainStack.top();
    mainStack.pop();
    if (topElement == maxStack.top()) {
      maxStack.pop();
    }
    return topElement;
  }
  int top() {
    assert(!mainStack.empty()); // Ensure the stack is not empty
    return mainStack.top();
  }
  int getMax() {
    assert(!maxStack.empty()); // Ensure the stack is not empty
    return maxStack.top();
  }
};

int main() {
 
  MaxStack maxStack;
  maxStack.push(5);
  maxStack.push(1);
  maxStack.push(5);
  std::cout << "Top element: " << maxStack.top() << std::endl;     // Outputs: 5
  std::cout << "Maximum element: " << maxStack.getMax() << std::endl; // Outputs: 5
  maxStack.pop();
  std::cout << "Maximum element after pop: " << maxStack.getMax() << std::endl; // Outputs: 5

  return 0;
}