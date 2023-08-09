#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>
#include <vector>
#include "class.h"

#include <optional>
#include <cstring>

#define PI 3.14


struct Person{
  std::string name;
  int age;
};

double adjustment{0.0};




namespace NoAdjustment{
  double add(double x, double y){
    return x + y;
  }
  double sub(double x, double y){
    return x - y;
  }
}


double add(double x, double y){
  return x + y;
}


namespace WithAdjustment{
  double add(double x, double y){
    return x + y - adjustment;
  }
  
  void do_something(){
    double result = ::add(5,6); // 5 + 6 function from The Default Global Namespace is called
  }


  double sub(double x, double y){
    return x - y - adjustment;
  }
}

namespace WithAdjustment{
  double mult(double x, double y);
  double div(double x, double y);
}


#include <iostream>
using namespace std;  // Brings in the entire namespace
int main() {

  cout << "Namespace std" << endl;



  double result1 = NoAdjustment::add(4,2);    // 4 + 2
  double result2 = WithAdjustment::add(4,2);  // 4 + 2 - adjustment



  return 0;
}

namespace WithAdjustment{
  double mult(double x, double y){
    return x*y - adjustment;
  }
  double div(double x, double y){
    return x/y - adjustment;
  }
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