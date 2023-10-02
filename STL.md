# Standard Template Library STL

- Containers
- Algorithms
- Iterators

## Containers

- Containers store elements in different ways
- The STL offers different containers for use in different scenarios to fit our particular use case in the best way possible
- Choosing the right container for the job is very strong skill
- If we do not know beforehand which container to choose, we can start with std::vector
- More info [here](https://en.cppreference.com/w/cpp/container)

### Sequence Containers

- For elements stored in a sequence
  
#### std::vector

- Storing stuff contiguously in memory and providing helper methods to manipulate the data
- Defined in the [\<vector\>](https://en.cppreference.com/w/cpp/container/vector) header
- Some basic functions:
  
![](Images/vectorFunctions.png)

- Very fast lookup by index with '[]'

#### std::array

- Storing stuff in a fixed size container
- Defined in the [\<array\>](https://en.cppreference.com/w/cpp/container/array) header
- Because this is fixed size container there is not that much of flexibility when inserting items
- Usually the data is inserted when array is created or with fill() method

![](Images/stdArray.png)

#### std::deque

- Double Ended queue
- Defined in the [\<deque\>](https://en.cppreference.com/w/cpp/container/deque) header
- Very fast insertions and removals from both ends of the container
- Supports random access operators like '[]'
- The elements are not stored contiguously in memory however, they are stored and organized in such a way that insertions and removals from both ends of the deque:
  - Happens very fast
  - Do not require any element to be copied or moved
  - They never invalidate pointers or references to any other elements in the collection -> adding or removing elements and the both ends wont invalidate pointers or references to existing data

![](Images/deque.png)

- When we insert at the iterator's position, the iterator will still point to the same element it was pointing to before the insertion
- But this is guaranteed only when we are inserting at the start or at the end
- When we are inserting at different position, due to the internal structure of std::deque, sometimes after an insertion, the iterator might point to the newly inserted element, especially when it causes some internal reallocation or movement of the chunks/blocks in the deque:

![](Images/dequeInsertion.png)



#### std::list
#### std::forward_list


### Associative Containers

- Elements stored by key

#### std::set
#### std::map
#### std::multiset
#### std::multimap

### Container Adaptors

- Specialization of sequence containers

#### std::stack
#### std::queue
#### std::priority_queue



## Iterators

![](Images/iterators.png)

- Traversing containers in a unified way, regardless of the internal structure of the container
- Each C++ container usually also defines iterators that traverse it

![](Images/vectorIterators.png)

- We can use iterators in universal templates for containers:

![](Images/templateIterators.png)

- More info [here](https://www.cplusplus.com/reference/iterator/)
  


### Reverse Iterators

- To travel through containers backwards

![](Images/reverseIterators.png)

- If we increment reverse iterator it will move to the left (backwards)

![](Images/reverseIteratorsPrint.png)

### Constant Iterators

- We can not change the element that we are pointing to through this iterator

![](Images/constIterator.png)

- There are even constant reverse iterators:

![](Images/constReverseIterator.png)

- When we are getting iterator with begin() or end() it will also return const iterator when the container itself is const:

![](Images/constIterator2.png)


### Comparing Iterators

- We can not compare iterators of different types

![](Images/iteratorsComparing.png)

### Iterator Types

- We have different types of iterators based on what they guarantee

![](Images/iteratorsTypes.png)

#### Input Iterators

- We can use input iterator to read from container
- If we dereference it we will get Rvalue -> We can not assign to it
- We can not use increment or decrement -> we can not move around with this iterator
- Can be dereferenced as an Rvalue
- Single pass from begining to end
- Supports: '++', '*' (read), '->' (read), '==', '!='

#### Output Iterators

- We can use ouput iterator to insert data into container
- If we dereference it we will get Lvalue so we get the original data from the container
- We can not use increment or decrement -> we can not move around with this iterator
- Can be dereferenced as an Lvalue
- Supports: '++', '*' (write), '->' (write), '==', '!='
  
#### Forward

- They can move forward in the collection ( '++' )
- It has also features of Input operator but if it is mutable then it can also be used to insert data
- Basically combination of input and output iterator
- Supports: '++', '*' (read, write), '->' (read, write), '==', '!='

#### Bidirectional

- They can move both forward and backward in the collection
- They have both '++' and '--'operator
- It has also features of Forward operator  

#### Random Access

- We can choose any element we want in the collection and we will get it straight away without moving there from begin() or end()
- Supports arithmetic operators '+' and '-' 
- Supports inequality comparisons ( '<', '>', '<=', '>=' )
- Supports compound assignment operator '+=' and '-='
- Supports offset dereference operator '[]'
- It has also features of Bidirectional operator  

#### Contiguous Iterator (C++20)

- Refines random access iterator by providing a guarantee the denoted elements are stored contiguously in the memory


### std::begin() and std::end()

- Template functions that return the begin and end iterator respectively for the underlying container passed as parameter
- These functions are usually helpful when we want our iterator based code to work even for regular raw C arrays
- C arrays support pointers and pointers meet all the requirements for random access iterators
- The requirement for the template argument is that the collection passed in should support these begin and end iterators

![](Images/stdBeginEnd.png)


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