#include <iostream>
#include <concepts>
#include <vector>
#include <algorithm>
#include "class.h"

#include <optional>
#include <cstring>


using namespace std;
#include <memory>

int main()
{

  // Constructing Vector
  std::vector<std::string> vec_str{"The","sky", "is", "blue"};
  std::vector<int> ints {1, 2, 3, 4};
  std::vector<int> ints2 (20, 55);  // Vector with 20 items, all initialized to 55

  // Accessing elements

  std::cout << vec_str[2] << std::endl; // Prints "is"
  std::cout << vec_str.at(3) << std::endl; // Prints "blue"
  std::cout << vec_str.front() << std::endl; // Prints first element
  std::cout << vec_str.back() << std::endl; // Prints last element

  // Pointer to the first element:
  auto *vec_data = vec_str.data();
  // To get vector size:
  size_t vec_size = vec_str.size();
  
  // Adding elements:
  vec_str.push_back("new"); // Adds the element at the end of the vector
  vec_str.at(3) = "another"; // Will insert element at the specific index and the previous element will be removed

  // Removing elements: 
  vec_str.pop_back();       // Removes the last element of the vector


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