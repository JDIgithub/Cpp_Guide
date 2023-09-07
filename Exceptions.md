# Exceptions









- l
- l
- l
- l
- l
- l
- l
- l
- l
- l
- l
- l
- l
- l












// Exceptions
//  - One of the advantages of C++ over the C
//  - Run-Time anomalies or abnormal conditions that a program encounters during its execution.
//  - There are two types of exceptions:
//      - Synchronous
//      - Asynchronous (Beyond the program control (disc failure etc.))
//  - C++ keyword for exceptions handling:
//      - Try   - Block of code that can throw an exception
//              - Allows you to define a block of code to be tested for errors while it is being executed    
//      - Catch - Block of code that is executed when a particular exception is thrown
//              - Try and Catch keyword come in pairs
//      - Throw - Throws exception. Also used to list the exceptions that a function throws but does not handle itself
//              - Lets us create a custom error
//  - Why exception handling?
//      - Separation of error handling code from normal code
//      - Functions can handle any exceptions they choose: 
//          - Function can throw many exceptions but we may choose to handle only some of them
//          - Exceptions that are thrown but not catched are handled by the caller
//      - Grouping of Error Types
//          - Both basic types and objects can be thrown as exception 
//
//  - There is special catch block called 'catch all' ( catch(...) ) that can be used to catch all types of exceptions.
//  - If an exception is thrown but it is not caught anywhere, the program terminates abnormally
//  - All exceptions are unchecked -> compiler doesnt check whether an exception is caught or not but its better practice to list unchecked exceptions: void fun(int *ptr, int x) throw (int *, int) {


#include <iostream>
using namespace std;
 
int main()
{
   int x = -1;
 
   // Some code
   cout << "Before try \n";
   try {
      cout << "Inside try \n";
      if (x < 0)
      {
         throw x;
         cout << "After throw (Never executed) \n";
      }
   }
   catch (int x ) {
      cout << "Exception Caught \n";
   }
 
   cout << "After catch (Will be executed) \n";
   return 0;
}



#include <iostream>
using namespace std;
 
int main()
{
    try  {
       throw 10;
    }
    catch (char *excp)  {
        cout << "Caught " << excp;
    }
    catch (...)  {
        cout << "Default Exception\n";
    }
    return 0;
}


void mightGoWrong() {

    bool error = true;

    if(error){
        throw 8; 
    }

 
}

int maiin()
{
    mightGoWrong();
    
    
    
    return 0;
}


//  - A derived class exception should be caught before a base class exception
//


#include<iostream>
using namespace std;
 
class Base {};
class Derived: public Base {};
int main()
{
   Derived d;
   // some other stuff
   try {
       // Some monitored code
       throw d;
   }
   catch(Base b) {
        cout<<"Caught Base Exception";
   }
   catch(Derived d) {  //This catch block is NEVER executed
        cout<<"Caught Derived Exception";
   }
   getchar();
   return 0;
}

// Correct way:

#include<iostream>
using namespace std;
 
class Base {};
class Derived: public Base {};
int main()
{
   Derived d;
   // some other stuff
   try {
       // Some monitored code
       throw d;
   }
   catch(Derived d) {
        cout<<"Caught Derived Exception";
   }
   catch(Base b) {
        cout<<"Caught Base Exception";   
   }
   getchar();
   return 0;
}

// C++ library has a standard exception class which is base class for all standard exceptions -> All standard exceptions can be caught by catching this type

class exception {
public:
  exception () throw();
  exception (const exception&) throw();
  exception& operator= (const exception&) throw();
  virtual ~exception() throw();
  virtual const char* what() const throw();
}


// exception example
#include <iostream>       // std::cerr
#include <typeinfo>       // operator typeid
#include <exception>      // std::exception

class Polymorphic {virtual void member(){}};

int main () {
  try
  {
    Polymorphic * pb = 0;
    typeid(*pb);  // throws a bad_typeid exception
  }
  catch (std::exception& e)
  {
    std::cerr << "exception caught: " << e.what() << '\n';
  }
  return 0;
}


#include <iostream>
using namespace std;
 
// Here we specify the exceptions that this function
// throws.
void fun(int *ptr, int x) throw (int *, int) // Dynamic Exception specification
{
    if (ptr == NULL)
        throw ptr;
    if (x == 0)
        throw x;
    /* Some functionality */
}
 
int main()
{
    try {
       fun(NULL, 0);
    }
    catch(...) {
        cout << "Caught exception from fun()";
    }
    return 0;
}




// Try-catch block can be nested. Also an exception can be re-thrown using throw



#include <iostream>
using namespace std;
 
int main()
{
    try {
        try {
            throw 20;
        }
        catch (int n) {
            cout << "Handle Partially ";
            throw; // Re-throwing an exception
        }
    }
    catch (int n) {
        cout << "Handle remaining ";
    }
    return 0;
}
