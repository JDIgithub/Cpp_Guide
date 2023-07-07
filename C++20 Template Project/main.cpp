#include <iostream>
#include <bitset>
#include <stdint.h>


int main(){
   
  std::cout << "jojoooo";

  int * min_address;

  // Dynamically allocate memory at run time and make pointer point to it
  int *p_number{nullptr}; 
  p_number = new int;     // Dynamically allocates space for a single int on the heap
  *p_number = 77;         // Writing into dynamically allocated memory   
    
  {
    int *p_num1 { new int {55} }; 
  } // memory with int {55} leaked

  

  int scores [] {1,2,3,4,5,6,7,8,9,10};

  for( auto score: scores){
     score = score*10; // Modifies only copy of value not the original
  }

  for( auto &score: scores){
     score = score*10; // Modifies original as well
  }



  for( auto score: scores){

    score = score*10; // Modifies copy of value in scores
     
  }








  //p_num1 = new int{44}; // memory with int{55} leaked


  size_t size{10};

  double *p_salaries{ new double[size] };          // Will contain garbage values
  int *p_students   { new(std::nothrow) int[size]{} }; // All values initialized to 0
  double *p_scores  { new(std::nothrow) double[size]{1,2,3,4} }; // First 4 init to these numbers
                                                                // Rest will be 0  

  delete[] p_scores;
  p_scores = nullptr;

  




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