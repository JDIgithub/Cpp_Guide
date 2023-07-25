#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>



#include <optional>
#include <cstring>






template <typename T> concept MyIntegral = std::is_integral_v<T>;
template <typename T> concept Multipliable = requires (T a, T b) {
  a * b;
};
template <typename T> concept Incrementable = requires (T a) {
  a+=1;
  ++a;
  a++;
};


// Syntax 1
template <typename T> requires MyIntegral<T> T sum(T a, T b){
  return a + b;
};

// Syntax 2
template <MyIntegral T> T sum(T a, T b){
  return a + b;
};

// Syntax 3
auto sum(MyIntegral auto a, MyIntegral auto b){
  return a + b;
};




int main(){

  int a {5};
  int b {10};
  auto result1 = sum(a,b);

  double c {11.1};
  double d {15.4};
  auto result2= sum(c,d); // Error
    
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