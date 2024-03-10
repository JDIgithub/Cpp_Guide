#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>



class CircularBuffer {
private:
  std::vector<int> buffer;
  int m_head;
  int m_tail;
  int m_capacity;
  bool m_isFullFlag;

public:
  CircularBuffer(int size) : buffer(size), m_head(0), m_tail(0), m_capacity(size), m_isFullFlag(false) {}

  void add(int item) {
    buffer[m_tail] = item;
    m_tail = (m_tail + 1) % m_capacity; // m_tail interval is <0; m_capacity)
      
    if (m_isFullFlag) { m_head = (m_head + 1) % m_capacity; }  // Overwrite the oldest data
    if (m_tail == m_head) { m_isFullFlag = true; }
  }

  int get() {
    if (isEmpty()) { return -1; } // or throw an exception, or handle underflow as needed

    int item = buffer[m_head];
    m_head = (m_head + 1) % m_capacity;
    m_isFullFlag = false;  // Buffer is not full anymore
    return item;
  }

  bool isEmpty() const { return (!m_isFullFlag && m_tail == m_head); }
  bool isFull() const { return m_isFullFlag; }

};

int main() {
   
  CircularBuffer cb(5); // Create a circular buffer of size 5
  cb.add(1);
  cb.add(2);
  cb.add(3);
  //std::cout << cb.get() << std::endl;  // Should print 1
  //std::cout << cb.get() << std::endl;  // Should print 2

  cb.add(4);
  cb.add(5); // 1 2 3 4 5
  cb.add(6);
  cb.add(7); // 6 7 3 4 5

  // buffer is 6 7 3 4 5 but according to head and tail position it should print 3 4 5 6 7
  while (!cb.isEmpty()) { std::cout << cb.get() << std::endl; }  // Should print 3, 4, 5, 6, 7

  return 0;
}




