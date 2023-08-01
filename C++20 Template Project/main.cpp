#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>
#include <vector>


#include <optional>
#include <cstring>

#define PI 3.14


class IntHolder {
  int* data;

public:
  IntHolder(int value) : data(new int(value)) {
  }
  ~IntHolder() {
    delete data;
  }

  //Copy constructor
  IntHolder(const IntHolder& other) : IntHolder(*other.data) {
    // Allocate new memory and copy the integer value
  }

  void setData(int value){
    *data = value;
  }
  void print() const {
    std::cout << *data << std::endl;
  }

};

int main() {

  IntHolder a(5);
 
  // Copy constructor is invoked for b
  IntHolder b = a;
  b.setData(8);
  b.print();  // Output: 8
  a.print();  // Output: 5 
  // But with the default copy constructor it would be 8 as well


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