# Classes

- Mechanism to build our own types
- It is a blueprint for creating objects. 
- It is a user-defined data type, which holds its own data members and member functions, which can be accessed and used by creating instances (objects) of that class. 
- The class defines what type of data it contains and the functions or methods that can manipulate that data.
  
![](Images/classDeclaration.png)

- Members of class are private by default
- Members can be raw stack variables, pointers or references
- Class methods (member function of a class) have access to the member variables even if they are  private
- Private members of classes are not accessible from the outside of the class definition

- Size of the class is sum of the size of its member variables but there could also be some padding that the compiler may add for memory alignment
- Memory alignment is used to ensure that data is efficiently accessed by the CPU and it typically involves adding padding bytes between data members to align them
  on memory addresses that are multiples of their size
- The functions are not counted into that because they are not stored as part if each individual object. Instead member functions are stored in a 
  separate code section of memory shared among all instances of the class

## Good practice

- For bigger classes it is better to create their own source and header file for them
- In the header there should be declaration of class methods and members
- In the source there should be bodies of the class methods


## Constructors

- Special class method that is called when an instance of a class is declared
- It has no return type and the same name as the class
- Initializes objects. It is automatically called when object is being created
- Usually initialize member variables of a class
- Types
  - Default constructor - Does not take any argument
  - Parametrized constructor - Passes arguments to the constructor
  - Copy constructor - Initializes object using another object of the same class
  - Move constructor - Allows a r-value object to be moved into a new object
 
![](Images/basicConstructors.png)

### Parametrized Constructor

- Constructor that uses its parameters to initilize object variables
- It can also use default parameters

![](Images/parametrizedConstructor.png)

- But if we call parametrized constructor without some of his parameters that are defaulted to some value and there is also another constructor that has less parameters and it will be match for him as well, then the compiler will be confused because he does not know if he should call constructor with less parameters or the one that has some of the parameters defaulted.
- So we must be careful about that and use default parameters only when it will not colide with another constructor to avoid compiler error
- We should also be careful that we can declare default parameters only in class definition. When the function body is separated elsewhere we can not repeat the default parameters there again.

### Defaulted Constructor

- The moment we set up our own constructor (even only the Parametrized one), the compiler will NOT generate default constructor
- So if we still want to construct objects without parameters with the default constructor we need to put in our own default constructor

![](Images/defaultedConstructor.png)


### Copy Constructor

- Initializes object using another object of the same class 
- Works only with the l-value references and copy semantics
- **Has to have const reference of the original object to copy as input parameter**
- Called when:
  - An object of the class is returned by value
  - An object of the class is passed (to a function) by value as an argument
  - An object is constructed based on another object of the same class
  - A compiler generates a temporary object
- It is however not guaranteed that a copy constructor will be called in all these cases, because the C++
  Standard allows the compiler to optimize the copy away in certain cases
- If we do not define our own copy constructor, C++ compiler creates default copy constructor for each
  class which does a member-wise copy between objects.
- We need to define our own copy constructor only if an object has pointers or run-time allocation for the resource like
  file handle, network connection, etc...
- The problem with pointers is that the default member wise copy will copy the address so both the original and the copied objects pointers 
  are now pointing to the same address and if we change the variable through copied object pointer, the original will change as well  
- Default copy constructor does only shallow copy (Shallow copy copies pointers itself not the content it points to)
- In user defined deep copy we make sure that pointers of copied object point to new memory locations    
- Copy constructor can be made private
- When we make him private in a class, objects of that class become non-copyable. (or rectangle(const rectangle &r) = delete;)
 
![](Images/copyConstructor.png)

- We can also use delegating with the copy constructor:

![](Images/copyConstructorDelegate.png)


### Move Constructor

- Works only with the r-value references and move semantics (pointing to already existing object in memory)
- On declaring the new object and assigning it with the r-value, firstly a temporary object is created, and then that temporary 
  object is used to assign the values to the object. Due to this the copy constructor is called several times and increases the overhead 
  and decreases the computational power of the code. To avoid this overhead and make the code more efficient we use move constructors.
- They moves resources in the heap unlike copy constructor which copy the data of the existing object and assigning it to the new object, move constructor
  just makes the pointers of the declared object to point to the same place as pointers of temporary object and nulls out the pointers of the temporary objects -> prevents unnecessarily copying data in the memory
- Prevents more than one object to point to same memory location

![](Images/moveConstructor.png)
![](Images/moveConstructor2.png)

#### std::move

- Making sure we have a temporary objects
- It is needed because sometimes the compilers are trying to do some optimalizations which could ruin our temporaries

![](Images/stdMove.png)


### Explicit Constructors

- If we are using normal parametrized constructor with one parameter then the compiler can implictily convert for example some number into the object
- If we want to forbid this implicit conversion we need to specify our constructor as 'explicit'

![](Images/explicitConstructor.png)

### Constructor delegation (C++11)

- Constructor calls another constructor of the same class to perform common initiaization tasks.
- Way to avoid code duplication

![](Images/delegatingConstructors.png)

- Only way to delegate constructor is through the initializer list
- If we will use initializer list for constructor delegation, We can not use the same list for anything else
- But we can still use the body of that constructor but be careful about the sequence

- Event sequence:
  - Selected constructor will be called
  - Before we get into the body of this constructor, the compiler will delegate and calls delegated constructor
  - Delegated constructor will construct object
  - Control reaches body of the delegated constructor
  - Control reaches body of the originally selected constructor 

### Deleted Constructors

- A way to explicitly disable constructor 
- Mostly used to forbid the default constructor but it can forbid any type of constructor to be used when creating an object.
- We can even disable move or copy constructors

![](Images/deletedConstructor.png)


## Destructors

- Special methods that are called when an object dies
- They are needed when the object needs to release some dynamic memory or for some other kind of clean up

![](Images/destructor.png)

- Called when local stack object goes out of scope and when heap object is released with delete
- The destructors are also called in places that may not be obvious:
  - When object is passed by value to a function
  - When a local object is returned from a function

## Order of Constructors/Destructors Calls ToDo

- When we create objects like in the next screen, the destructor will actually be called in the reverse order
- Object that was constructed last will be destructed first
  
![](Images/constrDestrOrder.png)

- It is because there could be some other objects that depends on object that was created before
- If we destroy the last object first we can avoid this problem


## Setters and Getters

- Methods to modify or read member variables of a class

![](Images/settersGetters.png)

### Getters and Setters Combined

- We can combine getter and setter into one function using references

![](Images/settersGettersCombined.png)


## Managing Objects With Pointers

![](Images/pointerObject.png)

## This Pointer

- Each class member function contains a hidden pointer called 'this'
- That pointer contains the address of the current object, for which the method is being executed
- This also applies to constructor and destructors
- It can also help when solving names conflicts

![](Images/thisConflicts.png)

- We can also chain the class method calls if they are returning 'this' pointer. Could be useful for setters

- Chaining using references
![](Images/thisPointerChain.png)

- Chaining using pointers
![](Images/thisPointerChain2.png)

## Struct

- User defined data type that groups together multiple related variables under one name
- It is very similar to a class but with some differences in default access control and inheritance
- Useful when we want to organize related data together
- Default access control here is public while in 'Class' they are private by default
![](Images/structClass.png)
- Another difference is in default inheritance type which is public by default here as well
- But of course we can override these defaults same like with Classes

## Structured Bindings

- Introduced in C++17
- Provides a convenient and readable way to unpack tuple-like objects into individual named variables
- This feature enhances the readability and conciseness of the code especially when dealing with tuples, pairs, arrays or structs that return multiple values from a function or require decomposition

![](Images/structuredBindingsTuple.png)
![](Images/structuredBindings.png)

- By default, the unpacked variables are immutable (const). If you need to modify them, you should declare the structured binding as auto& (for lvalues) or auto&& (for rvalues) depending on the context.


## Objects

- It is an instance of a class or struct
- It is concrete realization of the blueprint provided by the class or struct
- An object represents a specific piece of data and encapsulates both its attributes(member variables) and behaviors(member functions)

### Const Objects

![](Images/constObject.png)

- Problem is that the compiler does not know that the getter functions or the print function will not modify the class
- We need to tell the compiler that the getters will not modify the class

![](Images/constGetter.png)

- We can also overload the function so for const object it will use const function but for normal object it will use normal function

![](Images/constOverload.png)

- Summary
  - For const objects we can only call const member functions
  - Const objects are completely non-modifiable
  - Any attempt to modify an objects member will result into compile error
  - We can not call any non-const function within a const function

### Dangling Pointers and References in Objects

- A pointer or reference is said to be dangling if it is pointing to or referencing invalid data
- A simple example for pointers is a pointer pointing to a deleted piece of memory
- Good practice is that after we delete our objects from memory we will set pointers to 'nullptr'


## Initialization

### Initializer List

- Instead of Member wise copy initialization:

![](Images/constructorNormalInit.png)

- We can use Initializer list:

![](Images/initializerList.png)

- Initializer list avoids unnecessary copies
- In some cases, they are the only way to initialize an object

![](Images/InitListvsCopyInit.png)

- Non-static constants could be initialized only through the initializer list. It is not possible inside of the constructor body.

### Initializer List Constructors

- It can be used on aggregate types like a struct or an array
- Aggregate type is a type that represents a collection of individual elements that can be different type (struct) or same type (array)

- By default:

![](Images/structByDefault.png)

- The same but via initializer list:

![](Images/initializerListStruct.png)

- This way we can change the order
- Or we can do stuff like this:

![](Images/initListStructSum.png)

- Another benefit of using initializer_list is that we can use '{}' to create objects


### Designated Initializers (C++20)

- Borrowed from C, allows the initialization of data members
- It is useful when we want to init only some of the members

![](Images/designatedInitializers.png)

- Members that are not initialized explicitly are implicitly initialized to 0

### In-class Member Initialization (C++11)

![](Images/inClassInit.png)

- Before C++11 in-class member initialization was only possible for
  - Static constants of integral type
  - Static constants of enum type


## Mutable Member Variables

- Variable which value can be changed even if the object or method it is within is declared as 'const'
- It can only be applied to non-static and non-const class member variables

![](Images/mutableMember.png)


## Member variables of the same class type

![](Images/memberSelfType.png)

- We can make it work through pointers

![](Images/memberSelfTypePointer.png)

- But we need to initialize it as nullptr:

![](Images/memberSelfTypePointerNull.png)

- Also be careful

![](Images/memberSelfTypeRecursive.png)

- We can also set up static variables of the self type

![](Images/staticSelfType.png)

## Nested Classes

- We can declare class inside of another class
- Header:
![](Images/nestedClasses.png)
- CPP:
![](Images/nestedClassesCPP.png)
- Main:
![](Images/nestedClassesMain.png)

- It can be useful if we want to limit how to create the object of that inner class
- If Inner is private, its object can be created only from the outer class

- Outer does not have access to private section of inner
- Inner has access to private section of Outer
- Inner can directly access static members of Outer but can not access member variables without going through an object

- If Inner is public we can create inner object like this:

![](Images/nestedClasses2.png)

- **We can create object of the nested class like this only if it is public class**


- **ToDo Rules of 3: If we have explicit destructor then we should have copy and move constructor as well!!   why?**
  


## Singleton

- Design pattern that restricts the instantiation of a class to one single instance
- This is useful when exactly one object is needed to coordinate actions across the system
- The singleton pattern is often used in scenarios such as managing a connection to a database or in setting where having more than one instance of the class would lead to problems

- **Basic Implementation**
  - **Private Constructor:** Constructor is made private to prevent direct construction calls with the **new** operator
  - **Deleted Copy Constructor and Assignment Operator**
  - **Static Method for Access:** A public static method allows the class instance to be accessed
  - **Static Instance:** Static Instance of the class itself is maintained within the class

  ![](Images/classSingleton.png)

  - This basic implementation is not thread-safe. To make it thread-safe mechanisms like mutexes would need to be used to lock the code that initializes the instance
- **Static Initialization:** 
  - Another approach to ensure that the instance is created only once is to use the magic static feature of C++11, which guarantees that a static local variable is initialized only once, even when called from multiple threads.