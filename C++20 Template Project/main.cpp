#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>



#include <optional>

int main(){
  
  // Declaration and initialization
  std::optional<int> items{3};
  std::optional<std::string> name {"Jan Novak"};
  std::optional<std::string> dog_name{};  // Initializes to std::nullopt
  std::optional<int> age {std::nullopt};  // null equivalent for std::optional

  // Setting values
  age = 25;

  // Reading
  std::cout << "items: " << items.value() << std::endl;
  std::cout << "items: " << *items << std::endl;  // Kinda confusing because its not a pointer.
  // Trying to use std::nullopt variable will throw an exception
  std::cout << "nullopt: " << dog_name.value() << std::endl; // Throws exception and crashes program
  // We can do a check
  if(dog_name.has_value()){
    std::cout << "Dog does have a name: " << dog_name.value() << std::endl;
  } else {
    std::cout << "Dog does not have a name." << std::endl;
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