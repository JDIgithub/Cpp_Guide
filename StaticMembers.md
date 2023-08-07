# Static and Const Members


## Static members

- They are not tied to any object
- They are tied to the blueprint itself
- They are created even before a single class object has been created
- Each instance of class has its own variables and functions
- However all the instances of the same class can share static members

![](Images/staticMember.png)

### Declaration


![](Images/staticMemberDeclaration.png)

- But we can initialize it like this:

![](Images/staticMemberInit.png)

- We can not initialize non const static variables "in-class"
- But we can initialize const integral static variables "in-class"
- We can also initialize constexpr static variables "in-class"
- **It must also be in the public section of the class**