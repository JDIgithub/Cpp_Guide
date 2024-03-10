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

class StackLeaking {
  
  int* arr;
  int capacity;
  int top;

public:
  StackLeaking(int size) : capacity(size), top(-1) {
    arr = new int[size];
  }

  void push(int value) {
    if (top < capacity - 1) {
      arr[++top] = value;
    } else {
      std::cout << "Stack overflow" << std::endl;
    }
  }

  int pop() {
    if (top >= 0) {
      return arr[top--];
    } else {
      std::cout << "Stack underflow" << std::endl;
      return -1; // Assuming -1 is used to indicate an error
    }
  }

  // Destructor is missing here, causing a memory leak


};

class StackSolved {
  
  int* arr;
  int capacity;
  int top;

public:
  
  StackSolved(int size) : capacity(size), top(-1) {
    arr = new int[size];
  }

  // Copy constructor for deep copying
  StackSolved(const StackSolved& other) : capacity(other.capacity), top(other.top) {
    arr = new int[capacity];
    std::copy(other.arr, other.arr + capacity, arr);
  }

  // Copy assignment operator for deep copying
  StackSolved& operator=(const StackSolved& other) {
    if (this != &other) {
      delete[] arr;
      capacity = other.capacity;
      top = other.top;
      arr = new int[capacity];
      std::copy(other.arr, other.arr + capacity, arr);
    }
    return *this;
  }

  // Destructor for freeing allocated memory
  ~StackSolved() {
    delete[] arr;
  }

  void push(int value) {
    if (top < capacity - 1) {
      arr[++top] = value;
    } else {
      std::cout << "Stack overflow" << std::endl;
    }
  }

  int pop() {
    if (top >= 0) {
      return arr[top--];
    } else {
      std::cout << "Stack underflow" << std::endl;
      return -1;
    }
  }

};


template<typename TValue>
class MyStack {
public:
  using TValuePtr = TValue*;

  MyStack(std::size_t initialCapacity) : mCapacity(initialCapacity), mSize(0), mElements(new TValuePtr[initialCapacity]) {}
  ~MyStack() {
    // Memory leak: Missing deallocation of mElements
    // delete[] mElements;
  }

  void push(const TValue& value) {
    if (mSize >= mCapacity) {
      // Code for resizing the array is not included to focus on memory leak issue
    }
    mElements[mSize++] = new TValue(value);
  }

  void pop() {
    if (mSize > 0) {
      delete mElements[--mSize];
    }
  }

  const TValue& top() const {
    if (mSize > 0) {
      return *mElements[mSize - 1];
     } else {
      throw std::out_of_range("Stack is empty");
     }
  }

  bool empty() const {
    return mSize == 0;
  }

  std::size_t size() const {
    return mSize;
  }

private:
  std::size_t mCapacity;
  std::size_t mSize;
  TValuePtr* mElements;
};