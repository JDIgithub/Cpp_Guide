# Standard Template Library STL

- Containers
- Algorithms
- Iterators

## Containers

### std::vector

- Storing stuff contiguously in memory and providing helper methods to manipulate the data
- Defined in the [\<vector\>](https://en.cppreference.com/w/cpp/container/vector) header
- Some basic functions:
  
![](Images/vectorFunctions.png)


### std::array

- Storing stuff in a fixed size container










### std::list
### std::deque
### std::stack
### std::queue



## Iterators

![](Images/iterators.png)


## Algorithms

- The algorithms library defines functions for a variety of purposes that operate on ranges of elements
- Range is defined as (first, last) where last refers to the element past the last element to inspect or modify
  
- Sorting
- Finding
- Copying
- Filling
- Generating
- Transforming
- etc.

### Constrained Algorithms (C++20)

- C++20 provides constrained versions of most algorithms in hte namespace std::range
- In these algorithms a rance can be specified as either an iterator-sentinel pair or as a single range argument
- Projections and pointer to member callables are supported
- Additionally, the return types of most algorithms have been changed to return all potentially useful information