#include "class.h"

size_t Point::m_point_count {}; // Initialize static member of Point class to 0



Outer::Outer(int int_param, double double_param) 
  : m_var1(int_param), m_var2(double_param) {
}

Outer::Outer() : Outer (0,0.0){
}

Outer::~Outer(){
}

void Outer::createInnerClassObject(){
  Inner inner1(10.0); // Inner is private class so we can create object only inside of the Outer class
}

// Inner constructor
Outer::Inner::Inner(double double_param) : inner_var(double_param){
}


