#include <iostream>
#include <bitset>
#include <stdint.h>
#include <string>
#include <vector>
#include "class.h"

#include <optional>
#include <cstring>


using namespace std;
#include <memory>
 
// A generic smart pointer class
template <class T>
class SmartPtr {
    T* ptr; // Actual pointer
public:
    // Constructor
    explicit SmartPtr(T* p = NULL) { ptr = p; }
 
    // Destructor
    ~SmartPtr() { delete (ptr); }
 
    // Overloading dereferencing operator
    T& operator*() { return *ptr; }
 
    // Overloading arrow operator so that
    // members of T can be accessed
    // like a pointer (useful if T represents
    // a class or struct or union type)
    T* operator->() { return ptr; }
};
 

/*
class Node {
private:
    int key;
    Node* next;
 
    // Other members of Node Class 
    friend int LinkedList::search();
    // Only search() of linkedList
    // can access internal members
};
*/

/*
int main()
{
  Player p1("Basketball");
  p1.set_first_name("John");  // We do not have access to private members of Person class 
  p1.set_last_name("Snow");   // so we have to use setters
  std::cout << "player: " << p1 << std::endl;

  return 0;
}*/



void draw_shape(Shape * s){
    s->draw();
}


class base {
public:
    virtual void print() { cout << "print base class" << endl; }
    void show() { cout << "show base class" << endl; }
};
 
class derived : public base {
public:
    // We dont have to use word "virtual" because if we have virtual function in the base class 
    // then the same function in the derived class is already considered virtual but it is good practice to type it anyway
    void print() { cout << "print derived class" << endl; } 
    void show() { cout << "show derived class" << endl; }
};
 
int main()
{
  base* bptr;
  derived d;
  bptr = &d;
 
  // Virtual function, bound at runtime -> Derived print will be used
  bptr->print();  

  // Non-virtual function, bound at compile time -> Base show will be used
  bptr->show();

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