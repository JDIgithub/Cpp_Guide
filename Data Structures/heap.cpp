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

class Heap{

private:
  std::vector<int> heap;
  int parent(int index){ return index/2; }
  int leftChild(int index){ return 2 * index; }
  int rightChild(int index){ return 2 * index + 1; }
  void sinkDown(){
    int maxIndex = 1;
    int index = 1;
    while(true){
      int leftIndex = leftChild(index);
      int rightIndex = rightChild(index);
      if(heap[leftIndex] > heap[maxIndex]){
        maxIndex = leftIndex;
      }
      if(heap[rightIndex] > heap[maxIndex]){
        maxIndex = rightIndex;
      }
      if(maxIndex != index){
        std::swap(heap[index], heap[maxIndex]);
        index = maxIndex;
      } else {
        return;
      }
    }  
  }
public:
  Heap(){ heap.push_back(0); }
  void insert(int value){
    heap.push_back(value);
    int currentIndex = heap.size() - 1;

    while( currentIndex > 1 && (heap[currentIndex] > heap[parent(currentIndex)]) ){
      std::swap(heap[currentIndex], heap[parent(currentIndex)]);
      currentIndex = parent(currentIndex);
    }
  }

  int remove(){
    if(heap.size() == 1) return INT_MIN;
    int maxValue = heap[1];
    if(heap.size() == 2) {
      heap.pop_back();
    } else {
      heap[1] = heap.back(); // Moves last element to the top
      heap.pop_back();
      sinkDown();
    } 
    return maxValue;
  }

};

int main() {

  Heap * myHeap = new Heap();
  myHeap->insert(99);
  myHeap->insert(72);
  myHeap->insert(61);
  myHeap->insert(58);
  myHeap->insert(100);

  int x = myHeap->remove();

  return 0;
}




