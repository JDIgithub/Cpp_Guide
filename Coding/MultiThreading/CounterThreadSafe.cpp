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

/* Multithreaded Counter

You are given a class Counter that needs to be thread-safe. 
The Counter class should have the following member functions:

    void increment(): Increments the counter by 1.
    int get() const: Returns the current value of the counter.

You need to implement the Counter class such that multiple threads can safely increment the counter concurrently.

Counter counter;

// Thread 1
counter.increment();
counter.increment();
counter.increment();

// Thread 2
counter.increment();
counter.increment();

int value = counter.get(); // Expected value: 5

Requirements:

    Implement the Counter class.
    Ensure that the class is thread-safe.
    Use appropriate synchronization mechanisms to protect the shared data.

Constraints:

    The counter should always start from 0.
    The increment() function should be called by multiple threads concurrently.

Implementation Details:

You may use any synchronization primitives available in the C++ Standard Library (e.g., std::mutex, std::atomic, etc.).
Ensure that your implementation avoids race conditions.


*/

class Counter {

private:

  std::mutex mtx;
  int count;

public:

  Counter(): count{0}{}
  int get(){
    std::lock_guard<std::mutex> lock(mtx);
    return count;
  }

  void increment(){
    std::lock_guard<std::mutex> lock(mtx);
    count++;
}

};


int main() {

  Counter counter;

  std::thread thr1([&counter](){
    counter.increment();
    counter.increment();
    counter.increment();  
  });

  std::thread thr2([&counter](){
    counter.increment();
    counter.increment();  
  });


  thr1.join();
  thr2.join();

  std::cout << "Counter is " << counter.get(); // Expected value: 5



  return 0;
}
