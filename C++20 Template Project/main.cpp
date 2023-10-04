#include <iostream>
#include <concepts>
#include <vector>
#include <algorithm>
#include "class.h"
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>



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
  std::cout << "]" << std::endl;

}

int main()
{

 
  std::unordered_set<int> numbers = {1,2,1,6,2,8,9,24,6,2};  // Not ordered
  std::unordered_map<int,int> numbersMap = {  {1,11}, {0,12}, {4,13}, {2,14}, {3,15} }; // Not ordered





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