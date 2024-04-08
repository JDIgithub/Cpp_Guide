#include <iostream>
#include <thread>

std::thread thread_object(callable);

// Launching thread using function pointer -----------------
void fun(params)
{
    // Do something
}
  
// The parameters to the function are put after the comma
std::thread thread_obj(fun, params); 



// Launching thread using function objects -----------------
 
// Define the class of function object
class fn_object_class {
    // Overload () operator
    void operator()(params)
    {
        // Do Something
    }
}
  
// Create thread object
std::thread thread_object(fn_object_class(), params)




//  Launching thread using lambda expression ---------------

auto lambda = [](params) {
    // Do Something
};
  
// Pass f and its parameters to thread 
// object constructor as
std::thread thread_object(lambda, params);



// Waiting for threads to finish
//      - Once a thread has started we may need to wait for the thread to finish before we can take some action.
//      - To wait for thread us the std::thread::join() function. 
//      - This function makes the current thread wait until the thread identified by *this has finished executing.
//      - Example to block main thread until thread t1 has finished:


int main()
{
    // Start thread t1
    std::thread t1(callable);
  
    // Wait for t1 to finish
    t1.join();
  
    // t1 has finished do other stuff
  
    ...
}




