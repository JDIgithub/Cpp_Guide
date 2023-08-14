#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>
#include <vector>
//#include "class.h"

#include <optional>
#include <cstring>
#include <cassert>

using namespace std;
#include <memory>
 
class Number {

public:
  Number() = default;
  Number(int value);

  // Type conversion. double() is the operator here
  // function has non return type because the return type is hidden in the operator so there is no need to be redundant
   operator double() const{
    return static_cast<double>(m_wrapped_int);
  }
  // We can mark function explicit if we want to forbid implicit conversion
  explicit operator Point() const {
    return Point(static_cast<double>(m_wrapped_int), static_cast<double>(m_wrapped_int));
  }
private:
  int m_wrapped_int{0};
};

class Point {
 
public:
  Point() = default;
  Point(double x, double y) : m_x(x), m_y(y){
    some_data = new int(0);
  }
  Point(const Point& p){ // Copy constructor 
    if(this != &p){
      delete some_data;                                // Deep copy
      some_data = new int(*(p.some_data));
      m_x = p.m_x;
      m_y = p.m_y;
    }
  }
  ~Point(){
    delete some_data;
  }

  Point& operator=(const Point& right_operand){        
    if(this != &right_operand){
      delete some_data;                                // Deep copy
      some_data = new int(*(right_operand.some_data)); // Without the self assignment check we would release memory and then we couldnt copy 
                                                       // the same data that we just released
      m_x = right_operand.m_x;
      m_y = right_operand.m_y;
    }
    return *this;
  }
private:
  double m_x{};
  double m_y{};
  int * some_data;
};









class Print{

public:

  void operator()(std::string name){
    std::cout << "The name is: " << name << std::endl;
  }
  std::string operator()(std::string first_name, std::string last_name){
    return first_name + " " + last_name;
  }

};



int main()
{
  Print print;
  print("Duncan");
  std::cout << print("Jon", "Snow") << std::endl;




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