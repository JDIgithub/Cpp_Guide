# Algorithms





## Algorithm Complexity

- Knowledge of Big O notation is crucial. It describes the performance or complexity of an algorithm, especially in terms of time (time complexity) and space (space complexity).
- Understanding how changes in input size affect performance helps in optimizing code.

![](Images/algorithmComplexities.png)


### Big O Notation

- Mathematical notation used to describe the upper bound of the time complexity or space complexity of an algorithm
- It characterizes the performance of an algorithm in terms of the size of the input data (n)
- This notation provides a high-level understanding of the algorithm in terms of efficiency and scalability
- Crucial in comparing the efficiency of different algorithms
- It help to predict performance


**Cases**

- **Best Case Ω** 
  - For example if we are searching for element in array with for loop, the best case is when the first element is the one we need
  - For that we use symbol: **Ω**
  
- **Average Case Θ**
  - For example if we are searching for element in array with for loop, the average case is that the wanted element will be in the middle
  - For that we use symbol: **Θ**
- **Worst Case O**
  - For example if we are searching for element in array with for loop, the worst case is when the wanted element is the last one
  - For that we use symbol: **O**


**Key Concepts**

- **Upper Bound**
  - Big O provides an upper limit on the time an algorithm will take or the space it will use in the worst-case scenario
- **Input Size(n)**
- **Ignoring Constants**
  - Big O notation abstracts away constants and less significant terms
  - For example O(2n) is simplified to O(n) and O(n^2 + n) to O(n^2)

    ![](Images/Osimplification.png)
    ![](Images/Osimplification2.png)
    ![](Images/Osimplification3.png)



**Common Big O Types:**

- **O(1) - Constant Time** 
  - The algorithm takes the same amount of time regardless of the input size
  - For example: Algorithm that determines if a number is even or odd

    ![](Images/constantO.png)

- **O(log n) - Logarithmic Time**  
  - The complexity increases logarithmically with the input size
  - Algorithms that divide the problem in half each time (Binary Search,...)
  - If we have 8 sorted numbers it takes 3 steps to divide them till we get the one searched element (8->4  4->2   2->1) and 2^3 = 8 so the time complexity is logarithmic

    ![](Images/logarithmicO.png)

- **O(n) - Linear Time**
  - The complexity is directly proportional to the input size
  - For example: Iterating through an array to find an element and so on

    ![](Images/linearO.png)


- **O(n * log n) - Log-Linear Time**
  -  Common in algorithms that break problems down into smaller parts, solve them individually and often combine solutions (Merge Sort,...)
  -  The most efficient we can make sorting algorithm (If we want to sort various data types not just numbers)
  

- **O(n^2) - Quadratic Time**
  - Time complexity increases quadratically with the input size
  - It is common in algorithms with nested iterations over the data set

    ![](Images/quadraticO.png)

- **O(2^n) - Exponential Time**
  - The time complexity doubles with each adition to the input data set
  - Common in brute-force algorithms for solving problems
- **O(n!) - Factorial Time**
  - Extremely slow with growing n
  - Common in algorithms that generate all permutations of a dataset 


![](Images/bigOgraph.png)

**Different Terms for Inputs**

- If we have multiple different terms for input, We can not simplify the big O notation so easily
- For example if we have 2 for loops that goes to n it is n + n = 2n which is simplified to O(n)
- But if one loops goes to a and second goes to b that we can not do this and O is O(a + b)

  ![](Images/OdifferentTerms.png)

- The same goes for multiplication so we get O(a*b) instead of O(n^2)

  ![](Images/OdifferentTerms2.png)

### Space Complexity
 
- Creating array of size n -> O(n) space complexity
- Creating Matrix n x n -> O(n^2) space complexity
- When we are using recursive calls, each function call is added to the call stack and picks up actual space in memory

### Time Complexity






## Common Algorithms

### Sorting Algorithms

#### Bubble Sort

- The algorithm gets it name because smaller elements "bubble" to the top of the list
- It is stable sort meaning that it maintains the relative order of equal sort items
- It can detect if the list is already sorted and stop early
- It is not very efficient on large lists and is generally used for educational purposes to introduce the concept of sorting algorithms

**Steps**

1. The algorithm compares the first two elements
2. If the first element is greater than the second element, they are swapped
3. Move to the next pair of elements and repeat the comparison and swap if necessary
4. Continue this process for the entire array. By the end of this first pass the largest element will "bubble up" to the end of the array
5. Repeat the entire process for the remaining elements (excluding the last element which is already on the correct position)
6. The algorithm stops when a pass through the array results in no swaps, indicating that the array is sorted

  ![](Images/bubbleSort.png)


**Complexity**

- **Time Complexity:** 
  - O(n^2)
- **Space Complexity:**
  - O(1) as it only requires a single additional memory space for swapping


**Examples**

![](Images/algorithmsSortingBubble.png)




#### Insertion Sort

- Simple and intuitive sorting algorithm
- It builds the final sorted array one item at a time -> much less efficient than more advanced algorithms
- **Advantages:**
  - Simple to understand and implement
  - Efficient for small data sets and it is more efficient than other simple quadratic algorithms such as selection sort or bubble sort
  - It is a stable sort
  - It is a good choice when we know that the data set is almost sorted (We can get to O(n))
  
**Steps**

1. Considering the first element to be a sorted sub-list of one element -> Starts with the second element
2. Take this element and compare it to the elements in the sorted sub-list (everything to the left of the current position)
3. Shift all the elements in the sorted sub-list that are greater than the current element to the right by one position and insert the current element at its correct position
4. Move to the next element and repeat the process until the entire list is sorted

  ![](Images/insertionSort.png)

**Complexity**

- **Time Complexity:**
  - O(n^2) 
- **Space Complexity:**
  - O(1) as it only requires a single additional memory space for the value being inserted

**Examples**

![](Images/algorithmsSortingInsertion.png)

#### Selection Sort

- The selection sort algorithm sorts an array by repeatedly finding the minimum element from the unsorted part and moving it to the beginning. 

**Steps**

1. Start with the first element as the minimum
2. Scan the remaining array to find the minimum element
3. Swap the minimum element with the first element of the unsorted part
4. Move the boundary of the unsorted array by one element to the right
5. Repeat steps 2-4 until the array is sorted

**Complexity**

- **Time Complexity**
  - O(n^2) 
- **Space Complexity**
  - O(1)


**Example**

![](Images/algorithmsSortingSelection.png)






#### Quick Sort

- Highly efficient sorting algorithm and is based on a dived and conquer approach
- It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot
- The sub-arrays are then sorted recursively
- This can be done in-place requiring small additional amounts of memory to perform the sorting
- Quick Sort is not a stable sort, which means that the relative order of equal sort items is not preserved
- It is efficient for large data sets and often performs better than other O(n * log n) algorithms like merge sort or heap sort



**Steps**

1. **Choose a Pivot**

  - This can be any element from the array
  - Common strategy includes picking the first element, the last element, random element or median element

2. **Partitioning**

  - Rearrange the array so that all elements with values less than pivot come before the pivot, while all elements with values greater than pivot come after it
  - After partitioning the pivot is in its final position
  - This is called the partition operation

3. Recursively apply the above steps to the sub array of elements with smaller values and separately to the sub-array of elements with greater values
4. The recursion stops when the sub-array has less than or equal to one element ( as a singe element or empty array is always sorted)


- In this image we can see that 4 was chosen as the pivot:

![](Images/algorithmSortingPivotOrig.png)  
![](Images/algorithmSortingPivot.png)
![](Images/algorithmSortingPivot2.png)


**Complexity**

- **Average Time Complexity:** 
  - O(n * log n)
- **Worst Time Complexity:** 
  - O(n^2)
  - This happens when the smallest or largest element is always chosen as the pivot. However this is rare especially if the pivot selection is randomized
- **Space Complexity:**     
  - O(log n)
  - This is because of the recursion stack. The actual sorting is done in-place
  

**Example**

![](Images/algorithmSortingQuick.png)

- **Function quickSort** 
  - Takes the array or portion of it and recursively sorts it
  - It does it by partitioning the array around the pivot (selected by the partition function) then sorting sub-arrays before and after the pivot
- **Function pivot** 
  - Rearranges the elements in th array so that all elements less than pivot come before it and all elements greater come after it
  - Pivot is then placed to its correct position
- This implementation uses the first element as the pivot


#### Merge Sort 

- Efficient, stable, divide-and-conquer comparison based sorting algorithm
- It is particularly good for sorting large data sets and performs well on linked lists
- Merge Sort is particularly useful for sorting linked lists in O(n * log n) time
- It is stable sort, which means it preserves the input order of equal elements in the sorted output
- Merge Sort is often preferred for sorting a linked list whereas Quick Sort is preferred for arrays
- The additional space complexity of O(n) can be a disadvantage compared to algorithms like Quick Sort which sorts in place


**Concept**

1. **Divide:** The array is recursively split into two halves until each sub-array contains a single element or is empty.
2. **Conquer:** Each sub-array is then merged back together in a sorted manner
3. **Combine:** The sorted sub arrays are combined to form the final sorted array

**Steps**

1. If the array has more than one element we divide it in half. We keep dividing the resulting sub arrays until we have sub arrays that are empty or contain only one element   
2. The merging process involves comparing elements of the sub-arrays and combining them in sorted order. This is done by creating a new array and filling it by choosing the smallest or largest element from the sub-arrays in each comparison
3. This process of dividing and merging continues recursively witch each level of recursion merging two sub-arrays at a time
4. The final step is when all sub-arrays are merged back int a single sorted array
  
  ![](Images/mergeSort.png)


**Complexity**

- **Time Complexity:**
  - With Arrays: 
    - O(n * log n) 
    - Division of array in half (log n) divisions
    - Merging arrays in linear time (n)
  - With Linked Lists:
    - O(n * log n) 
    - Same division and merging
- **Space Complexity:**
  - With Arrays: 
    - O(n) 
    - Due to the temporary arrays used for merging
  - With Linked Lists:
    - O(log n) for recursive version
    - O(1) for iterative version


**Examples**

- MergeSort() function;
  - There is wrong if condition to end recursion... ToDo
  
![](Images/algorithmSortingMerge.png)

- Merge() function:

![](Images/algorithmSortingMerge2.png)


- With LinkedList:

![](Images/algorithmSortingMergeLinkedList.png)
![](Images/algorithmSortingMergeLinkedList2.png)

- In the example above the recursive approach is used but be aware of stack overflow for very large lists, in which case an iterative approach might be better


#### Heap Sort

1. **Heap Data Structure**

- A binary heap is complete binary tree that satisfies the heap property
- In max heap, every parent node is greater than or equal to its child nodes
- In min heap, every parent is less than or equal to its child nodes

2. **Building Heap**

- The first step in Heap Sort is to transform the list into a max heap or min heap (if sorting in descending order)
- This is done using a process known as "heapifying" 

3. **Sorting the Array**

- Once the heap is built, the root of the heap is guaranteed to be the first element of the array
- Swap this root element with the last element of the array and reduce the heap size by one. The last element now is at its final position
- "Heapify" the root of the tree again so that the largest element is at the root
- Repeat this process until all elements are sorted

**Complexity**

- **Time Complexity**
  - O(n * log n) - Heapify process is O(log n) and it is called n times
- **Space Complexity**
  - O(1) - Heap Sort sorts the array in place and requires a constant amount of extra space

**Characteristics**

- In-place sorting
- Not stable - The relative order of equal elements might no be preserved
- Good for large data sets:
  - Particularly efficient for data sets that are too large to fit into memory
  - Effective when we need to sort large data sets with minimal space complexity.
- However it is less efficient than other O(n * log n) sorting algorithms in practical scenarios

**Examples**

![](Images/algorithmsSortingHeap.png)


### Searching Algorithms

#### Binary Search

- Efficient algorithm for finding an item from a sorted list of items.
- It works by repeatedly dividing in half the portion of the list that could contain the item, until we have narrowed down the possible locations to just one


**Concept**

1.  Compare the target value to the middle element of the array
2.  If they are not equal the half in which the target cannot lie is eliminated and the search continues on the remaining half
3.  If the elements are equal, the position of the middle element is returned as the result
4.  This process is repeated until the target value is found or the remaining array to be searched is empty

**Conditions**

- The list must be sorted
- Binary search is used on a list, not on linked list because random access is slover in linked list

**Steps**

1.  Find the middle element
2.  Compare the middle element with the target value
    - If the target value equals to the middle element, return index
    - If the target value is less than the middle element, repeat the search for the left half
    - If the target vallue is greater than the middle element, repeat the search for the right half
3.  If the target is not found and the array can not be split further, the item is not in the array
  
  ![](Images/binarySearchConcept.png)


**Complexity**

- **Time Complexity:**
  - Best Case: O(1) - This occurs when the central index is the target to be found
  - Other Cases: O(log n) - Witch each comparison, half of the remaining elements are eliminated, leading to the logarithmic time complexity    
- **Space Complexity:**
  - Iterative Implementation: O(1)
  - Recursive Implementation: O(log n) - consumes space on the call stack

**Advantages**

- Highly efficient for larga datasets as long as the dataset is sorted
- Much faster than linear search (which is O(n)) in most cases

**Limitations**

- Only for sorted arrays
- For very small arrays, the overhead of the algorithm might not be worth it compared to a simple linear search
  
**Examples**

![](Images/algorithmSearchingBinary.png)



### Tree Traversal

#### Breadth First Search (BFS)

![](Images/algorithmsBFS.png)

  - It uses additional memory because it is necessary to track the child nodes of all the nodes on a given level while searching that level

  ![](Images/bfsIterativelyExample.png)

#### Depth First Search

- **Pre-Order**

  - root-left-right

    ![](Images/algorithmsDFSpreorder.png)

- **Post-Order**
  
  - left-right-root

    ![](Images/algorithmsDFSpostorder.png)

- **In-Order**

  - left-root-right

    ![](Images/algorithmsDFSinorder.png)



#### DFS vs BFS

- **BFS**
  - **Pros**
    - Shortest Path for Closer Nodes
  - **Cons**
    - More Memory

  - If we have info that the desired node is not too deep that DFS is better
  - Good to find shortest path

- **DFS**
  - **Pros**
    - Less Memory
  - **Cons**
    - Can get Slow
  - Good to check if something exists

- If we know a solution is not far from the root -> **BFS**
- If the tree is very deep and solutions are rare -> **BFS** because DFS would take really long time
  - Early Stopping in BFS: BFS, by its nature, stops as soon as it finds the first solution, which is guaranteed to be the shortest path. This can be much earlier than DFS, which might only find the solution after exploring many deep paths.

- If the tree is very wide -> **DFS** because BFS would need too much memory

- If solutions are frequent but located deep in the tree -> **DFS**
- Determining whether path exists between two nodes -> **DFS** because in BFS we have sequence of nodes that are not connected to each other
- Finding shortest path -> **BFS**





### Graph Traversal

- **BFS** 
  - To find shortest path
- **DFS**
  - Find if path exist

### Graph Shortest Path

- We can use BFS for that if we have unweighted graph
- But if some edge has different weight than other we need special algorithms because BFS ignores the edges weights

#### Dijkstra Algorithm

- Can be faster than Bellman-Ford



#### Bellman-Ford

- Can work even with negative weights unlike the Dijkstra
- Worst case O(n^2)

## Algorithm Design Techniques

- Familiarity with techniques like recursion, divide and conquer, dynamic programming, and greedy algorithms.
- Data Storage Structures


### Divide and Conquer

- Merge Sort, Quick Sort, ...


### Dynamic Programming

- Optimization technique using "caching"


- **Requirements**

  - **Overlapping Sub-problems**

    - If we have some problems that repeat

      ![](Images/overlappingSubproblems.png)

    - We can than store the result for the first unique sub-problem into the array and then if the same sub-problem happens again just get the result from the array
    - This is called Memoization

      ![](Images/dynamicProgrammingMemoization.png)


  - **Optimized Sub-structure**

    - The sub-problems also have to have optimized sub-structure

      ![](Images/dynamicProgrammingOpti.png)

    - Here the optimal way from A to C is edge A,C and optimal way from C to D is edge C,D but A-C-D is not optimal solved problem... optimal is A-B-D so the criteria here is not met


- **Memoization (Caching)**
  - Use to hold some piece of data in simple accessible variable
  - Memoization is a specific form of caching
  - Downside of memoization is that it makes code more complex
  - Also vector that we use to memoize needs to stay permanently in memory

    ![](Images/memoization.png)

- Dynamic programming is **Divide and Conquer** + **Memoization**



- **Top Down vs Bottom Up**

  - See Fibonacci example below

**Examples**

- **Fibonacci Sequence**

  ![](Images/dynamicProgramingFibonacci.png)

  - We can see that we have overlapping sub-problems in the code above
  - Also if we have optimal solution to the sub-problems it gives us the optimal solution for the whole problem
  - To compute fibonacci with the code above it is O(2^n) which is super slow
  - But with memoization and dynamic programming we can do it with O(2n - 1) which is O(n)

  ![](Images/dynamicProgramingFibonacci2.png)
  ![](Images/FibonacciMemoization.png)

  - We did it with recursion and that was top down
  - But we can do it iteratively and that is bottom up

  ![](Images/FibonacciBottomUpCode.png)
  ![](Images/FibonacciBotomUp.png)



### Greedy Algorithms ToDo

### Recursion

- Function that calls itself until it does not

- **Rules**

  - The process is the same each recursion step
  - Each time recursion step runs the problem gets smaller



  ![](Images/callStack.png)



- **Application**

  - **Factorial**
    - The most famous problem that is solved by recursion is factorial
      
      ![](Images/recursionFactorial.png)
      ![](Images/recursionFactorialCallstack.png)


- **Recursive vs Iterative**

  - Anything that can be implemented recursively can be implemented iteratively 
  - But for some problem it is actual easier to write recursive solution
  - Recursion can lead to stack overflow when too many function calls are loaded to the stack





- Every time we are using a tree or converting something into a tree, consider recursion
- Divided into a number of sub-problems that are smaller instances of the same problem
- Each instance of the sup-problem is identical in nature
- The solutions of each sub-problem can be combined to solve the problem at hand
- Divide and Conquer using Recursion

### Backtracking

- General algorithmic technique used for solving problems recursively by building a solution incrementally
- It is often used to solve combinatorial problems, where we need to find combination of permutation of elements that satisfy certain constraints
- **Concept**
  1.  **Choose** - Choose an option from the available choices
  2.  **Explore** - Recursively explore this choice
  3.  **Un-Choose** - If the choice does not lead to a solution, undo the choice and try another option

- **Key Points**
  - **Recursive Nature**: Backtracking is inherently recursive since it involves exploring all possible choices at each step
  - **State Space Tree**: The problem is often represented as a tree where each node represents a state and each edge represents a decision or choice
  - **Pruning**: IF at any point it is determined that the current path will not lead to a valid solution, the algorithm abandons that path and backtrack to the previous step

- **Example Problems**
  - **N-Queens** - Placing N queens on an NxN chessboard so that no two queens threaten each other
  - **Sudoku Solver** - Filling 9x9 grid so that each row column and 3x3 sub-grid contains the digits 1-9
  - **Permutations and Combinations** - Generating all permutations or combinations of a given set of numbers




## Standard Algorithms

- Set of functions in the standard library
- Implement classic algorithms such as searching and sorting, populating, copying, reordering etc...
- Operate on containers and sequences of data
- Most are in \<[algorithm](https://en.cppreference.com/w/cpp/algorithm)\>
- Some in \<[numeric](https://en.cppreference.com/w/cpp/numeric)\>

- **Algorithm Execution**

  - Function call which takes an iterator range
    - Usually corresponds to sequence of elements in a container
    - Often **begin()** and **end()** to process the entire container
  - Iterates over the range of elements
  - Performs an operation on the elements
  - Returns either:
    - An iterator representing an element of interest
    - The result of the operation on the elements

- Many algorithms use a **predicate** - a function which returns bool
- Easiest way to create a predicate is with the **lambda** expression

  ![](Images/stdFindIf.png)



- We can se more in [STL chapter](STL.MD)
  




## Kadane Algorithm

- Popular algorithm used for finding the maximum sum of a contiguous sub-array within a one-dimensional numeric array. 
- It operates in linear time, making it very efficient for this purpose.