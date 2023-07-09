
# Strings

## Character Manipulation

- By using \<cctype\> library We can check if character is:
  - alphanumeric 
  - alphabetic
  - etc...
- We can see all the functions [here](https://en.cppreference.com/w/cpp/header/cctype) 

## C-String manipulation

- We can use [\<cstring\>](https://en.cppreference.com/w/cpp/header/cstring) library to work with C-strings
- So check all the functions there
- We can check length of the string, compare strings, copy strings, merge two string into one and much more



### strlen

- Unlike std::size and ranged base for loop, we can use std::strlen even with array decayed into a pointer

![](Images/cString.png)


## C++ String (std::string)

- Included by \<string> library

### Initializing std::string

![](Images/stringInit.png)

### Concatenating std::string

![](Images/concatenatingStrings.png)

### Accessing characters in the std::string

![](Images/accessingCharsString.png)

#### c_str() 

- returns const char *, so we should not use it to modify data in the string

![](Images/stringCstr.png)

### data()

- returns char *, so we can modify string with it

![](Images/dataString.png)


