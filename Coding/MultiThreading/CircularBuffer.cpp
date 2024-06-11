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
#include <csignal>
#include <optional>

using namespace std;

template <typename T>
class CircularBuffer{

private:

  std::vector<T> m_buffer;
  int m_head,m_tail;
  std::mutex mtx;
  std::condition_variable cv;
  bool m_isFull;

public:

  CircularBuffer(int size): m_buffer(size), m_head(0), m_tail(0), m_isFull(false) {

  }
  
  void insert(T newElement){

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this]() { return !m_isFull; });

    m_buffer[m_tail] = newElement;
    m_tail = (m_tail + 1) % m_buffer.size();
    if(m_isFull){
      m_head = (m_head + 1) % m_buffer.size();
    }
    if(m_tail == m_head){
      m_isFull = true;
    }
    
    cv.notify_one();
  }

  std::optional<T> getFront(){

    std::unique_lock<std::mutex> lock(mtx); // Unique lock must be used with CV because it supports unlock while waiting
    cv.wait(lock, [this]() { return !isEmpty(); });  

    /*    If we need time out so the producer wont wait indefinitely for the consumer 

        if (!cv.wait_for(lock, timeout, [this]() { return !m_isFull; })) {
            return false; // Timeout occurred
        }
    */

    if(isEmpty()) return std::nullopt;  // This wont ever happened with the cv wait above so this does not need to be std::optional but whatever

    T result = m_buffer[m_head];
    m_head = (m_head + 1) % m_buffer.size();
    m_isFull = false;

    cv.notify_one();
    return result;
  }

  bool isEmpty() const { return (!m_isFull && m_tail == m_head); }
  bool isFull() const { return m_isFull; }

};


std::mutex coutMTX;

int main() {


  CircularBuffer<int> cb(5);


/*
  cb.insert(1);
  cb.insert(2);
  cb.insert(3);
  cb.insert(4);
  cb.insert(5);
  cb.insert(6);
  cb.insert(7);
  std::optional<int> result = cb.getFront();
*/

  std::thread producer([&cb](){

    for(int i = 0; i < 10; i++){
      cb.insert(i);
      std::lock_guard<std::mutex> lock(coutMTX);
      std::cout<< "Produced: " << i << '\n';
    }


  });

  std::thread consumer([&cb](){
    
    for(int i = 0; i < 10; i++){
      std::optional<int> item = cb.getFront();
      std::lock_guard<std::mutex> lock(coutMTX);
      if(item){
        std::cout << "Consumed: " << *item << '\n';
      } else {
        std::cout << "Nothing to be consumed \n";
      }
    }

  }); 

  producer.join();
  consumer.join();

  return 0;
}





