#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>
#include <vector>


#include <optional>
#include <cstring>

#define PI 3.14



class Cylinder {

private:
  double baseRadius {1.0};
  double height {1.0};
  int * random {nullptr};
public:

  Cylinder(){
    random = new int;
    *random = 42;
  }
  // Destructor
  ~Cylinder(){
    delete random;
  }

  // Getters
  double getBaseRadius(){
    return baseRadius;
  }
  double getHeight(){
    return height;
  }

  // Setters
  void setBaseRadius(double radiusParam){
    baseRadius = radiusParam;
  }

  void setHeight(double heightParam){
    height = heightParam;
  }





  double volume(){
    return PI * baseRadius * height;
  }

};









// Driver Code
int main()
{

  // Stack object:
  Cylinder c1(10,2);
  auto volume1 = c1.volume();

  // Heap object
  Cylinder * c2  = new Cylinder(11,20); // Create object on the heap
  auto volume2 = (*c2).volume();
  auto volume3 = c2->volume();

  delete c2;


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