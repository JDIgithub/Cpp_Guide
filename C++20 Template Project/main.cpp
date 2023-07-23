#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>



#include <optional>
#include <cstring>




// We can of course create template that takes arguments of multiple data types
template <typename T, typename P> problematic_maximum (T a, P b); // But how to define the return type here?

// BAD design for the return type:
template <typename T, typename P> P problematic_maximum(T a, P b);  // Return type depends on order of parameters
// Better approach:
template <typename returnType, typename T, typename P> returnType maximum(T a, P b){  // Separate parameter for return type
  return (a > b) ? a : b;
}

template <typename T, typename P> auto maximum(T a, P b){
  return (a > b) ? a : b;
}

// To avoid repetition of the return expression we can use decltype(auto) are return type
template <typename T, typename P> decltype(auto) maximum(T a, P b) {
  return (a > b) ? a : b;
}



int main(){

  auto max1 = maximum(12.5,33); // double return type deduced
  auto max2 = maximum('b',90 ); // int return type deduced

  // Explicit arguments: Forces return type on compiler
  auto max3 = maximum<char,char>('b',90);  // It will deduce type from the explicit arguments -> char




  int a {9};
  double b {5.5};

  std::cout << "size: " << sizeof(decltype((a > b)? a : b)) << std::endl;

  decltype(((a > b)? a : b)) c {67};  // Declaring c as a type of that expression








	int a{10};
	int b{23};
	double c{34.7};
	double d{23.4};
	std::string e{"hello"};
	std::string f{"world"};
	
	auto max_int = maximum(a,b); // int type deduced
	auto max_double = maximum(c,d);// double type deduced
	auto max_str = maximum(e,f) ;// string type deduced

	const char* g{"wild"};
	const char* h{"animal"};

  const char* result = maximum(g,h);  // This will call the template specialization function
	std::cout << "max(const char*) : " << result << std::endl;
   
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