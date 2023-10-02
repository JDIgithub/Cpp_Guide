#include <iostream>
#include <concepts>
#include <vector>
#include <array>
#include <deque>
#include <algorithm>
#include "class.h"

#include <optional>
#include <cstring>


using namespace std;
#include <memory>



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


int main()
{

  int num{0};

  std::deque<int> numbers {1,2,3,4,5,6}; 
  printCollection(numbers);

  // Accessing elements
  num = numbers[3];     // No bound check
  num = numbers[30];    // Undefined behavior
  num = numbers.at(3);  // Bound check
  num = numbers.front();// First element
  num = numbers.back(); // Last element

  // Clear
  numbers.clear();  // Deletes all the elements in the collection 
  // Assign
  numbers = {10,20,30,40,50,60}; 

  // Insertion
  auto iterator = numbers.begin() + 2;
  /*
  std::cout << *iterator <<std::endl; // 30
  // Elements are inserted at position in front of the *iterator element
  numbers.insert(iterator,300);   // [10 20 300 30 40 50 60] 
  std::cout << *iterator <<std::endl; // 30  
  numbers.insert(iterator,400);   // [10 20 300 400 30 40 50 60] -> iterator moves as we add elements because it keeps pointing to the same element (but not always)
  std::cout << *iterator <<std::endl; // 400 - Now the iterator points to newly inserted element to maintain its relative position in its internal block.
  numbers.insert(iterator,500);   // [10 20 300 500 400 30 40 50 60]
  std::cout << *iterator <<std::endl; // Still 400
*/
  // Emplace
  numbers.emplace(iterator,45);  // Parameters following the iterator are passed to constructor of the type stored in the vector
  numbers.emplace_back(15);      // Will emplace element at the end of the collection 
 
  // Erase
  numbers.erase(iterator);
  numbers.erase(numbers.begin() + 1, numbers.begin() + 4); // Will erase numbers[1] - numbers[3]

  // Pop_back
  //numbers.pop_back()



/*

  // Supports regular forward iterators
  auto it = numbers.begin();
  while(it != numbers.end()){
    std::cout << " " << *it;
    ++it;
  }

  // Supports reverse iterators
  auto rit = numbers.rbegin();
  while(rit != numbers.rend()){
    std::cout << " " << *rit;
    ++rit;
  }
*/
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