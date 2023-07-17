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