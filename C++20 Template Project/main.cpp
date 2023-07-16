#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>


int function_name (int & x, const int & y){ 
  // This function uses reference to the original variable 
  // So if we change thus reference the original variable will be changed as well

  ++x; // Original is modified
  ++y; // Original can not be modified so with const * it is pure input parameter
  
  return 0;
}



enum class Month {
    Jan = 1, Feb, Mar, Apr, May, Jun,
    Jul, Aug, Sep, Oct, Nov, Dec
};


int main(){

Month month {Month::Jul};
std::cout << "Month: " << static_cast<int>(month) << std::endl; // Will print 6











  int arg1{5};
  int arg2{10};
  int result_var{0};


  std::cout<< "arg1 " << arg1 << std::endl;
std::cout<< "arg2 " << arg2 << std::endl;

  result_var = function_name(arg1,arg2);
  

  std::cout<< "arg1 " << arg1 << std::endl;
std::cout<< "arg2 " << arg2 << std::endl;

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