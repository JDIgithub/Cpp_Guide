# Namespaces

- Feature of C++ programming language to avoid name conflicts
- This way we can create functions with same name just under different namespace

![](Images/namespaces.png)

- We can even separate the namespaces into more blocks

![](Images/namespacesSeparated.png)

- The functions are called through their namespaces like this:

![](Images/callingNamespaces.png)

## Namespaces across multiple files

- We can use namespaces across multiple files
- But we should be careful when we want to use the same namespace 

![](Images/namespacesFiles1.png)
![](Images/namespacesFiles2.png)

![](Images/namespacesFiles3.png)
![](Images/namespacesFiles4.png)

- Main:

![](Images/namespacesFiles5.png)

## Default Global Namespace

- Anything outside of any specified namespace lives in the default global namespace
- We can call functions and variables from the default global namespace through the '::' operator
- It can be useful when we are inside of some namespace and we want to specify to call something from the outside

![](Images/defaultGlobalNamespace.png)

## Built in Namespaces

- There are already lot of build in namespaces from different libraries so we should be aware of that
and try to not named our namespaces with the same name

### Namespace std

- The most used one is 'std' from the C++ standard libraries

![](Images/namespaceStd.png)

## Using

- To call things from specified namespaces without specifying the namespace name

![](Images/usingNamespace.png)

