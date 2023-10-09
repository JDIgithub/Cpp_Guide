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

int main()
{
  std::priority_queue<int> numbers; // The greatest has higher priority
  numbers.push(10);
  numbers.push(8);
  numbers.push(12);
  printPriorityQueue(numbers);  // [ 12 10 8 ]
  numbers.push(11);
  numbers.push(3);
  printPriorityQueue(numbers);  // [ 12  11 10 8 3 ]
  
  // Access
  numbers.top();  // Highest priority element access
 // numbers.top() = 55;  // top() returns const reference so we can not modify it

  // Erasing
  numbers.pop();  // Highest priority element will be removed

  std::priority_queue<int, std::vector<int>, std::less<int>> numbers2; // Default
  std::priority_queue<int, std::vector<int>, std::greater<int>> numbers3; // Non-Default
  // Using our own functor
  auto cmp = [](int left, int right){ return left < right;};
  std::priority_queue<int, std::vector<int>, decltype(cmp)> numbers4(cmp);






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