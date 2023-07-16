#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>


// Function that will returns sums of all digits of the input variable 
unsigned int digit_sum (unsigned int num){
    
    int dig_sum{0};
    
    if (num < 0){
        num = -num;
    }
    while(num > 0){
        int digit = num % 10;
        dig_sum += digit;
        num /=10;    // Equivalent: num = num/10; 
    }
    return dig_sum;
}







int function_name (int x, int y){ 
  // This function uses copies of the original that was passed as input
  // So if we change the copy the original stays the same
  // We can use references if we dont want to work with copies
  x++;
  y++;
  
  return x;
}




int main(){
   


 std::cout << "a: " << std::endl;      
  
  


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