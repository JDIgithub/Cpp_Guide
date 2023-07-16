# Functions

![](Images/functionScheme.png)

- Function signatures are its name and parameters
- Calling a function
  
![](Images/callingFunction.png)


## Passing By Parameter

- We must be aware that without references or pointers, the function will make a copy of the input parameters

![](Images/functionCopyParameter.png)
![](Images/functionPointer.png)
![](Images/functionReference.png)

- Also if we do not want to change value passed by input parameter, we can always declare the input parameter as **const**
- If we are passing pointers or references via the function parameter then we should always declare them as **const** if they
are meant as input only parameter because otherwise the function could change the original.

- If we use array as input parameter, it will decay to the pointer so we can not access to its size information so we need to pass
another parameter with that information or use vector, etc.
- But if we pass the array as reference then we can still get to the information about size etc.

### Pass parameter by Value

- It is kinda OK if parameters are fundamental types (int, double, etc...)
- Not recommended for the relatively large types (user defined) because of the copy creations
- It makes copies so it can waste memory

### Pass parameter by Reference

- Does not make copies
- Changes to the parameter are reflected on the argument outside the scope of the function
- Saves memory
- Recommended for passing around large types
- When to use reference:
  - When we want to modify value outside of the function scope
  - We do not need to represent null or empty state
  - We want to avoid syntactic noise of pointer dereferencing

### Pass parameter by Pointer

- The pointer address itself is passed by value
- Can go through dereferencing the parameter and make the changes reflect outside the scope of the function
- Avoids copies (Pointer is very cheap to copy)
- Bad syntax
- When to use pointer:
  - We want to allocate memory dynamically inside the function
  - We want to pass an array or range of values
  - We want to represent nullable parameter (by passing a null pointer)
  - We want to explicitly convey the idea of ownership or shared ownership

### std::string_view parameters

- If we will use string reference as function parameter then we can not call it with string literal because
references does not support implicit conversion
- We can pass the string by value but then it will create copy
- Best way is to use string_view 

![](Images/stringViewParameter.png)


### Default Arguments

- If we want to have some default values, then we can specify them in the function declaration
- If we call this function without specific values then it will use the default values

![](Images/defaultArguments.png)
  
- If function declaration is separated from the function definition then we can specify the default values only in the declaration
  


## Separating function definition and declaration

- Sometimes it is more flexible to split the function into its header and body and keep the code for each in different places
- For example if we do not want to expose how to function works (Libraries etc.) so we share just the header file
- Or even in the same file but we want to just declare before main and put function body below main

![](Images/separatedDeclaration.png) 


- Often is good to declare function in separate file (header)

![](Images/functionSeparation.png)

- The linker searches for definitions in all '.cpp' files in the project so the name of the file
does not need be the same as header file name.

## Implicit conversions

- When you pass data of a type different than what the function takes, the compiler will try to insert an implicit conversion from the type we pass 
to the type the compiler takes
- If the conversion fails, We will get a compiler error

![](Images/functionImplicitConversion.png)

### Implicit conversions with references

- We can use them for read-only purposes
- IF we want to write to the reference it will be compile error even if it is not supposed to be const reference
- When we are trying to write, compiler is confused if it should modify the temporary (conversed) value or the original value so it will throw error

### Implicit conversion with pointers

- Compiler can not implicitly converse pointers from for example int* to double* 
- There are some exception when implicit conversion can be used on pointers

![](Images/implicitConversionPointers.png)


## constexpr Functions (C++11)

- Enables computations to be performed at compile time rather than at runtime (If possible)
- Allows the evaluation of expressions during compilation if the arguments are also compile time constants
- Can improve the performance of the code
- Conditions:
  - Function must have non-void return type
  - Function body cannot declare variables or define new types
  - Function body cannot contain 'try' block or 'throw' expressions
  - Function body must contain only one 'return' statement  (Not true since C++14)

![](Images/constexprFunction.png)

## consteval Functions (C++20)

- Enforces that a function must be executed during compile time
- Any function declared as 'consteval' must produce a compile time constant
- Unlike 'constexpr' functions, 'consteval' can not be evaluated at runtime
- Must have non-void return type 
- All its arguments must be literal types (constants)
- **Not all constants can be evaluated at compile time** (For example when they are initialized by runtime variables)

![](Images/constevalFunction.png)