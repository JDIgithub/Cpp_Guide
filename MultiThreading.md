# MultiThreading

- Some classes here are move-only to support RAII (Resource Acquisition Is Initialization) idiom
  - Object acquires ownership in the constructor
  - Object releases ownership in the destructor
  - The resource instance can be moved from one object to another



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