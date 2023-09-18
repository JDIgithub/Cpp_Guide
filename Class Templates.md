# Class Templates

- It is a class blueprint that can be used by compiler to generate more specific class instances
- It is not real C++ code just blueprint

![](Images/classTemplates.png)
![](Images/classTemplates2.png)
ss
- The definition should show up in the header file because compiler will look only into header files to look for what it should generate template instances 
- All member function definitions moved into the header file, the compiler needs to see them there to generate proper template instances
- A template is only instantiated once, it is reused every time the type is needed in our code
- All the class members are inline by default so we are safe from ODR issues

## Constructors

![](Images/constructorsClassTemplate.png)

## Destructors

![](Images/destructorClassTemplate.png)


## Instances of Class Templates

- Template instance is created for a given type only once
- Only Methods that are used are instantiated

## Non Type Template Parameters

