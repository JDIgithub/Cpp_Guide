#include <iostream>
#include <concepts>
#include <vector>

#include <algorithm>
#include "class.h"

#include <list>


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


int main()
{

  int num{0};

  std::vector<int> numbers {1,2,3,4,5,6}; 

  // Insertion
  auto it = numbers.begin() + 2;
  printCollection(numbers);
  std::cout << *it << std::endl;    // 3
  numbers.insert(it, 15);
  // Iterator wont move. It will still point to the third element
   printCollection(numbers);
  std::cout << *it << std::endl;    // 15

  


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
  }*/

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