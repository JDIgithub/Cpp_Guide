# Theory

## Introduction ToDo
    
- Some C++ or programming in general theory sumup



## Compilers ToDo

- Piece of software that turns our code (C++ or other) into binary code that is executed by the CPU instruction to run our program







## Preprocessing Compilation Linking ToDo


- ToDo

![](Images/preprocessingCompilation.png)


- When we have the object files ready they will be process by linker
- Linker will stich them together and outputs single binary file

![](Images/linking.png)







![](Images/compilationModel.png)




## auto deduction

- If auto will try to deduce type from reference that it wont be reference to the type but just the type that is being referenced

![](Images/autoDeduction.png)

- But it is possible to deduce reference with 'auto&':

![](Images/autoReference.png)

- 'auto&' will also preserve the constness of reference (const reference will stay const reference)
- for naked 'auto' constness does not matter because it will create copy and the copy will not modify original anyway 

## Static variables

- We can define global variable 'static' when we want to use it just within that source file
- Or we can define local variable 'static' when we want to use it more then once without initializing it again when function would be call again
  - Local static variable wont be destroyed when it scopes ends and is initialized only once the first time control passes through their declaration
  - This is known as "lazy initialization" or "on-demand initialization" 
  - That is useful for example if we want to know how many times is some function called etc.

![](Images/staticVariables.png)

- Both global and static variables have static storage duration
- They live throughout the entire lifetime of the program
- Local static variables can not be used outside of their scope even tho their lifetime goes outside that scope


## Not sure where to put it yet - Random stuff

- Initialization with {} is safer because it will prevent data loss from implicit conversions or at least warn us
  according to used compiler. int x {20.7}; -> warn or error    vs.  int x = 20.7; -> Will convert to x = 20 without warning

