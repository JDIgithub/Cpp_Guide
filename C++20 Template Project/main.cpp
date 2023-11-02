#include <iostream>
#include <concepts>
#include <vector>
#include <algorithm>
#include "class.h"
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <list>


#include <functional>

using namespace std;




// Uniform traversal of containers
template <typename T> void printCollection (const T& collection){

  auto it = collection.begin();
  std::cout << "[";
  while(it != collection.end()){
    std::cout << " " << *it;
    ++it;
  }
  std::cout << "]" << std::endl;

}


template <typename T, typename K> void printMap (const std::multimap<T,K> & map){

  auto it = map.begin();
  std::cout << "[";
  while(it != map.end()){
    std::cout << " " << it->first << "," << it->second;
    ++it;
  }
  std::cout << " ]" << std::endl;

}





// We need to have second parameter that tell us which container is being used. Deque is default one
template <typename T, typename Container = std::deque<T> > void printStack (std::stack<T, Container> stack){

  // We are working on copy here so it doesnt matter that we are deleting elements from stack
  std::cout << "[";
  while(!stack.empty()){
    T item = stack.top();
    std::cout << " " << item;
    stack.pop(); // Removes the top element so the next iteration it will read the next one.
  }
  std::cout << " ]" << std::endl;

}


template <typename T, typename Container = std::deque<T>> void clearStack (std::stack<T,Container>& stack){
  while(!stack.empty()){
    stack.pop(); 
  }
}


// We need to have second parameter that tell us which container is being used. Deque is default one
template <typename T, typename Container = std::vector<T> > void printPriorityQueue (std::priority_queue<T, Container> priorityQueue){

  // We are working on copy here so it doesnt matter that we are deleting elements from stack
  std::cout << "[";
  while(!priorityQueue.empty()){
    std::cout << " " << priorityQueue.top();
    priorityQueue.pop(); // Removes the front element so the next iteration it will read the next one.
  }
  std::cout << " ]" << std::endl;
}

template <typename T, typename Container = std::vector<T>> void clearPriorityQueue (std::priority_queue<T,Container>& priorityQueue){
  while(!priorityQueue.empty()){
    priorityQueue.pop(); 
  }
}

struct Sum
{
  int sum{0};
  void operator()(int n) { sum += n; }
};


int main()
{

  std::vector<int> input {5,7,4,2,8,6,1,9,0,3,11,45,6,23};
  std::vector<int> output {11,22,33};
  
  std::cout << "Output size: " << output.size() << std::endl;             // 3
  std::cout << "Output capacity: " << output.capacity() << std::endl;     // 3

  // Uses whatever space there is. Does not extend the capacity
  std::transform(std::begin(input),std::end(input),std::begin(output),[](int n){ return n*2;}); 
  printCollection(output);  // [ 10 14 8 ]   <- 5*2, 7*2, 4*2

  std::cout << "Output size: " << output.size() << std::endl;             // 3
  std::cout << "Output capacity: " << output.capacity() << std::endl;     // 3


}



































    
void swapPtr(int *a, int *b)
{
   int tmp = *a;  
   *a = *b;         // Dereferencing pointers to get value
   *b = tmp;
}

void swapRef(int &a, int &b)
{
  int tmp = a;      
  a = b;            // We have values and when we change them, the original is changing too
  b = tmp;
}

int PointersReferences()
{
    int i = 10;
    int &r = i;         // Reference
    int *p = &i;        // Pointer
    
    //          Name      Address
    //           i          0x20
    //           r          0x20
    //           p          0x24
    
    int var = 90;
    
  //  r = var;         // r = 90 -> i = 90    
  //  p = &var;        // *p = 90 
  //  *p = 60;         // *p = 60  and var = 60  (shares address)    
  
    std::cout << "i: " << &i << " r: " <<  &r <<  " p: " << &p << std::endl;  
	// Reference and Original value have the same address but pointer has its own even if he stores address of the value
        
    // Swap with pointers:
    int a = 5;
    int b = 10;
    std::cout << "a: " << a << " b: " <<  b << std::endl;     
    swapPtr(&a,&b);                                    // We have to give address for pointer init                 
  	std::cout << "a: " << a << " b: " <<  b << std::endl;         
    swapRef(a,b);                                      // We have to give values for reference to make alias of
    std::cout << "a: " << a << " b: " <<  b << std::endl;      
  
        
	return 0;
}