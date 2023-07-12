#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>

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

  

  /*
  std::string str1 {"I amaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa string"};
  std::cout << " String: " << str1 << std::endl;
  std::cout << " String size: " << str1.size() << std::endl;
  std::cout << " String capacity: " << str1.capacity() << std::endl;

  str1.clear(); // Deletes stored string


  std::cout << " String: " << str1 << std::endl;
  std::cout << " String size: " << str1.size() << std::endl;
  std::cout << " String capacity: " << str1.capacity() << std::endl;




  std::string str1 {"012"};
  str1.insert(1,2, 'A');  // Inserts 2 characters at index 1. => "0AA12"


  std::cout << " String: " << str1 << std::endl;                       
*/

  std::string str1 {"Finding Nemo"};
  std::string str2 {"Searching for"};
  str1.replace(0,7,str2); // Replace 7 characters, starting at 0 index, with str2
  str1.replace(0,7,str2,0,9); // The same but only 9 characters starting at 0 index from str2



  std::cout << " Result: " << str1;





  str1.push_back('!');  // Appends '!' at the end of the string
  str1.pop_back();      // Removes the last character from the string
  

/*

  // trYYYYYYYYYY
  for(char &chr: str1){
    chr='X';
  }
  // ? Will it change str1?

*/



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