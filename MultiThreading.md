# MultiThreading

- Some classes here are move-only to support RAII (Resource Acquisition Is Initialization) idiom
  - Object acquires ownership in the constructor
  - Object releases ownership in the destructor
  - The resource instance can be moved from one object to another

- Each thread has its own entry point function
  - When the thread starts, it executes the code in this function
  - When the function returns the thread ends
  - The main thread continues to execute its own code
  - It does not wait for the other threads, unless we explicitly tell it to

- **Thread**
  - Software thread
  - For example an object of the C++ std::thread class

- **Task**
  - Higher level abstraction
  - Some work that should be performed concurrently

## Concurrency

- Performing two ore more activities at the same time
- For example if there is some long task and we want to see feedback during the processing of that task (For example Download progress)

### Hardware Concurrency

- Modern computers have multiple processors
- Different processors can perform different activities at the same time (Even within the same program)
- They are known as "hardware threads"
- Each processor follows its own thread of execution through the code

### Software Concurrency

- Modern operating systems support "software threading"
- A program can perform multiple activities at the same time
  - These activities are managed by the operating system
- Typically there are more software thread than hardware threads


## std::thread (C++11)

- The base level of concurrency
- Rather low level implementation
- Maps onto a software thread
- Managed by the operating system
- Similar to Boost threads, but with some important differences:
  - No thread cancellation
  - Different argument passing semantics
  - Different behavior on thread destruction

### Launching a Thread

- We need to create an std::thread object defined in [\<thread\>](https://en.cppreference.com/w/cpp/thread/thread) header
- The constructor starts a new execution thread
- The parent thread will continue its own execution
- std::thread constructor takes a callable object - Thread's entry point function
  - The execution thread will invoke this function
- The entry point function
  - Can be any callable object
  - Can not be overloaded
  - Any return value is ignored

![](Images/startingThread.png) 