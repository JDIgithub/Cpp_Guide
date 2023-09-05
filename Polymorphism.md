# Polymorphism

- Managing derived objects in memory through base pointers or references and getting right method called on the base pointer or reference
- Base class pointer or reference can takes multiple forms
- When we have different classes inheriting from the same base class we can do something like this:

![](Images/polymorphism1.png)

- Circle, Rectangle and Oval all inherit from shape
- We can then set up function that will be same for all of them:

![](Images/polymorphism2.png)

- Another benefit of polymorphism is that we can store different kind of objects in the same collection:

![](Images/polymorphism3.png)

- Polymorphism does not need to start always at the top base class. We can start with polymorphism in derived class and it will work with its derived class but then of course we can not use the top base object pointer but pointer at that object where the polymorphism starts



## Static Binding with Inheritance

- We would like this behavior:

![](Images/staticBinding.png)

- But the compiler just looks at the pointer type to decide which version of draw() to call
- It sees Shape* and calls Shape::draw() in all three cases in the picture 

## Dynamic Binding (Dynamic Polymorphism)

- We need to mark methods that are needed to be dynamically resolved as 'virtual'
- With dynamic binding the compiler is not looking at the type of the pointer but at the type of the actual object that the pointer is managing

![](Images/dynamicBinding.png)
![](Images/dynamicBinding2.png)

- Dynamic binding is not free. Size of the object will increase because the program needs to keep track of information that allows us to resolve function calls dynamically.


## Slicing

![](Images/slicing.png)

- Once our data are sliced off, they are gone and there is no way to get them back

![](Images/slicing2.png)


## Override (C++11)

- Introduced to avoid errors with typos when we are declaring virtual functions in derived classes
- With 'Override' compiler will inforce that our virtual function really exists in the parent class
  
![](Images/override.png)



## Overloading

### Function Hiding

- If the parent class has some overloads of some function and we override (does not matter if we use 'override' or not) any of the variant in the derived class, all of the other overloads will now be hidden for that derived object so we need to override all of the overloads that we want to use with that object.

![](Images/functionHiding.png)

- Function Hiding with polymorphism:

![](Images/functionHiding2.png)


### Overloading downstream

- If we create new overload in the derived class that is not in the base class, we can not use it with polymorphism through the base class pointer because that function is not known to the base class


## Polymorphism and Inheritance with static members


- If we have some static member in the parent class, the derived class will inherit it but it will be shared with the parent:

![](Images/staticMemberInheritance.png)

- If we want to have the same static member in the derived class but separated from the base class, we need to declare it in the derived class as well

![](Images/staticMemberInheritance2.png)
![](Images/staticMemberInheritance3.png)


## Final (C++11)

- Restricts how we override methods in the derived classes
- Restricts how we can derive from a base class

- If we want to prohibit further overrides in derived classes we can use 'final' like this:
  
![](Images/final.png)

- Or if we want to prohibit any further inheritance we can use 'final' this way:

![](Images/final2.png)

## Fun Fact

- 'Override' and 'Final' are not really C++ keywords but rather context-sensitive identifiers that act as keywords in specific context but can be used as names in other context

![](Images/finalOverride.png)

- It is like that because when they introduced them in C++11 these words could already be used in many codes as names so they did not want to break old codes but in modern C++ we should not use them as names

## Access Specifiers 

### With Dynamic Binding

- The base class access specifier wins when we are calling virtual functions through the base class pointer so this can lead to wierd scenarios
- So when we call the virtual function through the base class pointer the access specifier in the base class determines whether the function is accessible, regardless of the access specifier in the derived class
- In general, when the function call is done through dynamic binding, the access specifier of the base class is applied but if the call is done through static binding then the acces specifier of the derived class is applied

- Guideline:
  - Except for the base class, it is good to mark all other derived overrides as private, unless the specific problem requires otherwise

### With Static Binding

![](Images/accessSpeficierStaticBinding.png)










- l
- l
- l
- l
- l
- l

- Two types:
  - Run-Time Polymorphism: Virtual Functions
  - Compile-Time Polymorphism: Function Overloading, Operator Overloading


















// ------ Polymorphism ---------------------------------------------------------
//
// Two types:
//  - Run-Time polymorphism (virtual functions)
//  - Compile-Time polymorphism (function overloading, operator overloading)

// Compile-Time polymorphism
//
//  Function overloading
//      - When there are multiple functions with the same name but different parameters
//      - Function can be overloaded by change in number of arguments or/and change in type of arguments
//
class Geeks
{
    public:
      
    // function with 1 int parameter
    void func(int x) { cout << "value of x is " << x << endl; }
      
    // function with same name but 1 double parameter
    void func(double x) { cout << "value of x is " << x << endl; }
      
    // function with same name and 2 int parameters
    void func(int x, int y) { cout << "value of x and y is " << x << ", " << y << endl; }
};
  
int main() {
      
    Geeks obj1;
      
    // Which function is called will depend on the parameters passed
    // The first 'func' is called 
    obj1.func(7);
      
    // The second 'func' is called
    obj1.func(9.132);
      
    // The third 'func' is called
    obj1.func(85,64);
    return 0;
} 

//  Operator overloading
//      - For example: We can make operator + for string class to concatenate two strings.
//      - ?? add something interesting

class Complex {
private:
    int real, imag;
public:
    Complex(int r = 0, int i =0)  {real = r;   imag = i;}
       
    // This is automatically called when '+' is used with
    // between two Complex objects
    Complex operator + (Complex const &obj) {
         Complex res;
         res.real = real + obj.real;
         res.imag = imag + obj.imag;
         return res;
    }
    void print() { cout << real << " + i" << imag << endl; }
};
   
int main()
{
    Complex c1(10, 5), c2(2, 4);
    Complex c3 = c1 + c2; // An example call to "operator+"
    c3.print();
}



// ------ Polymorphic objects ---------------------------------------------------------
//
//  - Objects of a class type that declares or inherits atleast one virtual function.
//  - Within each polymorphic object, the implementation stores additional information which is used
//    by virtual function calls and by the RTTI features (dynamic_cast and typeid) to determine, at run-time,
//    the type with which the object was created.
//  - For non-polymorphic object, the interpretation of the value is determined from the expression in which the object is used
//    and decided at compile-time
//

#include <iostream>
#include <typeinfo>
struct Base1 {
    // polymorphic type: declares a virtual member
    virtual ~Base1() {}
};
struct Derived1 : Base1 {
     // polymorphic type: inherits a virtual member
};
 
struct Base2 {
     // non-polymorphic type
};
struct Derived2 : Base2 {
     // non-polymorphic type
};
 
int main()
{
    Derived1 obj1; // object1 created with type Derived1
    Derived2 obj2; // object2 created with type Derived2
 
    Base1& b1 = obj1; // b1 refers to the object obj1
    Base2& b2 = obj2; // b2 refers to the object obj2
 
    std::cout << "Expression type of b1: " << typeid(decltype(b1)).name() << '\n'
              << "Expression type of b2: " << typeid(decltype(b2)).name() << '\n'
              << "Object type of b1: " << typeid(b1).name() << '\n'
              << "Object type of b2: " << typeid(b2).name() << '\n'
              << "Size of b1: " << sizeof b1 << '\n'
              << "Size of b2: " << sizeof b2 << '\n';
}