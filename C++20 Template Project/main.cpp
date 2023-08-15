#include <iostream>
//#include <utility>
#include "class.h"




int main(){

  int n1{5};
  int n2{5};

  // <=> does not return an int like std::string::compare() but a type whose value is comparable to literal 0
  auto result = ( n1 <=> n2);   
  std::cout << std::boolalpha;
  std::cout << "n1 > n2: " << ((n1 <=> n2) > 0) << std::endl;
  std::cout << "n1 >= n2: " << ((n1 <=> n2) >= 0) << std::endl;
  std::cout << "n1 == n2: " << ((n1 <=> n2) == 0) << std::endl;
  std::cout << "n1 != n2: " << ((n1 <=> n2) != 0) << std::endl;
  std::cout << "n1 < n2: " << ((n1 <=> n2) < 0) << std::endl;
  std::cout << "n1 <= n2: " << ((n1 <=> n2) <= 0) << std::endl;
  
  return 0;
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