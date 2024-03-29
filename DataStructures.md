# Data Structures

## Introduction

- These are ways of organizing and storing data so it can be accessed and modified efficiently.
- Examples include arrays, linked lists, stacks, queues, trees, and graphs.
- Each structure has its unique properties and use-cases, like trees for hierarchical data or graphs for networked data.
- They are critical for implementing algorithms effectively
- Choosing the right algorithm and data structure can drastically improve the performance of an application, especially in data-intensive tasks



![](Images/dataStructuresOperations.png)



## Linear

- Like arrays, linked lists, stacks and queues
- They are fundamental for basic operations

### Array ToDo

- Collection of items stored in contiguous block of memory
- For storing multiple items of the same type 


### Linked List

![](Images/linkedList.png)
![](Images/linkedList2.png)

**Big O**

- Adding element at the end of the list is **O(1)**
- But deleting the last element is more complicated because we need to move tail to the previous node but we cat get to it only from the head and iterate so it is **O(n)**
- Adding element at the start of the list is as well **O(1)**
- Removing element from the start is **O(1)** because head si moving forward which is no problem
- Adding and removing and even accessing elements in the middle is **O(n)** because we must iterate to them from the head

  ![](Images/linkedListComparedToVector.png)


**Code ToDo**
 
- Add some Link to the code here or at least screens.
- My implementation of LinkedList: ToDo 


### Stack

- LIFO
- We can implemented stack on vector but also on linked list
- When implementing on Linked List we always want the tail to be at the bottom of the stack because both Adding and removing element from the start of the Linked List is constant so we want m_head to be at top of the stack

**Code ToDo**
 
- Add some Link to the code here or at least screens.
- My implementation of Stack: ToDo 

### Queue

- FIFO
- We can also implement Queue on vector or on linked list
- With vector the start of the vector will have O(n) for both insertion and deletion so the Queue will have O(n) for insertion or deletion (depends on the Queue direction) 
- But on linked list we can insert to the tail->next and delete from head which is both constant so the Queue will have O(1) insertion and deletion
- So the Queue direction will be opposite to the Linked List direction

**Code ToDo**
 
- Add some Link to the code here or at least screens.
- My implementation of Queue: ToDo 


## Tree

- Such as Binary trees, AVL trees, Red-Black trees and heaps
- Trees are essential for hierarchical data representation and efficient data access patterns
- Use cases:
  - Implementing priority queues
  - Syntax trees in compilers
  - Hierarchical data representation like file system directories
  - Database indices
- In C++ trees are typically implemented using classes
- For Binary tree class **Node** would contain the data and pointer to the left and right children
- STL does not have a direct tree implementation but **set**, **map**, **multiset** and **multimap** are typically implemented using Red-Black trees



**Terminology**

- **Node:** 
  - The fundamental part of a tree, which contains data and references (or links) to other nodes
  - **Parent Node**
  
    ![](Images/parentNode.png)

  - **Children Nodes**

    ![](Images/childrenNodes.png)

  - Every Node can have only one parent

- **Root:** The top node of the tree with no parent
- **Leaf:** Node with no children
- **Child and Parent:** A node directly connected to another node when moving away from the root
- **Siblings:** Nodes that share the same parent
- **Subtree:** A tree formed by a node and its descendants
- **Depth of Node:** The length of the path from the root to the node
- **Height of a Tree:** The depth of the deepest node

### Traversal

1. **In-Order Traversal (Left,Root,Right)**  

   - Particularly useful in BST for getting sorted order

2. **Pre-Order Traversal (Root,Left,Right)**

   - Useful for copying a tree

3. **Post-Order Traversal (Left,Right,Root)**

   - Useful for deleting a tree

4. **Level-Order Traversal**

   - Traverses the tree level by level


### Complexity

- The time complexity for operations lie search, insert and delete depends on the type of tree and its balance
- For perfectly balanced BST these operations are O(log n)
- For skewed tree they can degrade to O(n)


### Binary Tree

- Each node has at most two children, commonly referred to as the left and right child
  
  ![](Images/binaryTree.png)


**Terminology** 

- **Full Binary Tree**
  - Every node has 0 or 2 children

    ![](Images/fullTree.png)

- **Perfect Binary Tree**
  - A full binary tree in which all leaves have the same depth

  ![](Image/perfectTree.png)

- **Complete Binary Tree**
  - All levels are fully filled except possible the last level which is filled from left to right
  
    ![](Images/completeTree.png)

- **Balanced Binary Tree**
  - The height of the left and right subtrees of any node differ by no more than 1

- **Degenerate Tree**
  - A tree where every internal node has one child
  - This essentially becomes a linked list


### Binary Search Tree (BST)

- A binary tree where the left child contains only nodes with values less than the parent node and the right child only nodes with values greater

  ![](Images/binarySearchTree.png)


- This property provides efficient lookup, insertion and deletion operations
- Used for efficient data storage and retrieval
- Fast look up, insertion and deletion operations

**Big O**

- Insertion, Deletion and Look up are all O(log n)
- Well the worst case is O(n) so technically it is not O(log n) but this is rare
- This happens when the tree never forks so it is basically a linked list

  ![](Images/worstBST.png)

  ![](Images/vectorLLvsBST.png)


**Basic Concepts**
  - The left subtree of a node contains only nodes with keys lesser than the node's key
  - The right subtree of a node contains only nodes with keys greater than the node's key
  - The left and right subtree each must also be a binary search tree
  - There must be no duplicate nodes

**Advantages**
  - **Efficient lookup and modification:** Searching closer to O(log n) for balanced trees
  - **Sorted data representation:** In order traversal of BST yields elements in sorted order
  - **Dynamic data structure:** BST can grow or shrink to accommodate the data as needed

**Limitations**
  - **Balancing** 
    - Can become skewed if elements are inserted in a particular order (e.g. sorted order)
    - This can degrade performance with operations approaching O(n) complexity
  - **Balanced BST Variants**
    - To overcome balancing issues, self balancing BSTs like AVL or Red-Black trees are used which maintain a balanced structure through rotations
  - **Memory Overhead**
    - Each node typically requires extra space for two pointers (left and right)

**Implementation in C++**
  - BST can be implemented using a class or struct to represent nodes with each node containing data, left child pointer and right child pointer
  - The tree itself can be managed by a class that provides functions for insertion, deletion, search and traversal

**Appications**
  - BSTs are widely used in scenarios where data is dynamically inserted and deleted an where order needs to be preserved
  - They form the basis for more complex data structures like sets, maps and multimap in the C++ Standard Library


#### Operations

1. **Search**

   - Begin at the root node
   - Compare the search key with the key of the current node
   - If they are equal the search is successful
   - If the search key is less, continue in the left subtree
   - If the search key is greater, continue in the right subtree
   - If we reach leaf node, the search is unsuccessful
   - **Time Complexity:** Average - O(log n)  Worst Case - O(n) for skewed tree 

2. **Insertion**

   - Begin at the root node
   - Traverse the tree similar to the search operation to find the correct position for the new node
   - ??Insert the new node as a leaf??
   - **Time Complexity:** Average - O(log n)  Worst Case - O(n) 
3. **Deletion**

   - **Leaf node:** Simply remove the node
   - **Node with one child:** Remove the node and replace it with its child
   - **Node with two children:** Find the node's in-order predecessor (largest node in its left subtree) or in-order successor (the smallest node in its right subtree), swap it with the node to be deleted and then remove the node
   - **Time Complexity:** Average - O(log n)  Worst Case - O(n)

4. **Traversal**

   - **In-Order (Left,Root,Right):** Produces a sorted sequence of values
   - **Pre-Order (Root,Left,Right):** Used to create a copy of the tree
   - **Post-Order (Left,Right,Root):** Useful for deleting the tree
   - **Level-Order**

#### Red-Black Tree

- Self-balancing binary search tree 
- An important data structure used for efficient data storage and retrieval
- Each node contains an extra bit for denoting the color of the node (red or black)
- The tree uses these properties, along with several rules, to ensure that it remains balanced during insertions and deletions
- As result the operations like searching, insertion, and deletion can be done in logarithmic time complexity

- **Properties**
  - **Color Property:** Every node is colored red or black
  - **Root Property:** Root of the tree is always black
  - **Leaf Property:** 
    - Every leaf (NIL node) is black. 
    - NIL are considered as a leaves even though they are not carrying data just empty mark for endpoint of a branch
  - **Red Node Property:**
    - If a red node exists, both its children must be black
  - **Black Height Property:**
    - The number of black nodes from a node to a leaf
    - Every path from a node to any of its descendant NIL nodes has the same number of black nodes

- **Benefits**
  - **Balanced Height**
    - The balancing of the tree ensures it has a height of O(log n) where n is number of nodes
    - This guarantees that basic dynamic set of operations (search,insert,delete) can be performed in logarithmic time
  - **Better Worst-Case scenario**
    - Unlike simple binary search tree, red-black trees provide good worst case guarantees for these operations

- **Use Cases**
  - Red-Black Trees are widely used in computer science 
  - Used in STL (maps,sets,multimaps,multisets)


- **Operations**
  
1. **Insertion**
   - Insert nodes as in a standard binary search tree
   - Color the new node to red
   - After insertion, restore the red-black tree properties using rotations and re-coloring if needed
2. **Deletion**
   - More complex than insertion due to the tree's need to maintain its balancing properties
   - After deleting a node, the tree might be restructured (using rotations) and re-colored to maintain red-black properties
3. **Search**
   - Same as in standard binary search tree
4. **Rotations**
   - **Left Rotation:** Pivot the node down to its right child position making the right child the parent 
   - **Right Rotation:** Pivot the node down to its left child position making the left child the parent
   - Rotations are used during insertions and deletions to maintain balance

### Heap

- Special tree-based data structure that satisfies the heap property
- For example in a max heap, for any given node the value is greater than its children

### Trie (Prefix Tree)

- A tree where each node represents a character of a string
- Tries are particularly used for efficient text searches and auto-complete functionalities

### B-Trees and B+ Trees

- Used in databases and file systems for representing sorted data in a way that allows for efficient insertion, deletion and searching






## Hash Tables

- Used for fast data retrieval through hashing

![](Images/hashTable.png)

- Structure that can map keys to values
- We can imagine hash table as a vector of linked lists but ideally we want to have only one item in every linked list
- If we have more items on the same index -> linked list have more than one element -> this is called collision
- In Vector we have O(n) for searching, inserting and deleting if we go by value but it is constant when we know the index
- In hash table value is turned into the index by hash function and therefore we can get O(1) for searching,inserting and deleting
- The number of addresses in the hash table should be prime, because then the key:value pair will be distributed more randomly -> we will have fewer collisions


**Efficiency:**

- Due to this mechanism hash tables are incredibly efficient for lookup operations
- Average time complexity of O(1) for searching, inserting and deleting elements 

**Components**

- **Keys and Values**
 
  - The primary components of a hash table are key-value pairs
  - The key is used to generate a unique hash

- **Hash Function**

  ![](Images/hashFunction.png)


  - Takes a key as input and returns an integer, which is used as the index at which the value associated with the key is stored
  - **Properties of good Hash Function**
    - **Uniform Distribution:** It should distribute keys uniformly across the hash table to minimize collisions 
    - **Fast Computation:** The function should be quick to compute
    - **Less Collision:** It should minimize collision rate
    - **One Way:** We can get output from input but we can not reverse it to get input back
    - **Deterministic:** It will always produce the same output from the same input

- **Array of Buckets or Slots**

  - The array where the key-value pairs are stored
  - The size of this array can impact the performance of the hash table

- **Collision Handling**

  - Since a hash function can map multiple keys to the same index, collisions can occur
  - There are several ways to handle collisions:
    - **Chaining:** Each slot in the array is a linked list and all key-value pairs with the same hash index are stored in the list at that slot
    - **Open addressing:** In case of a collision, a sequence of probing is done to find an empty slot 

**Operations:**

- **Insertion**
  - Apply hash function to the key to determine the index for storing the value
  - Handle any collision if the calculated index is already occupied
- **Search**
  - Use the hash function to find the index of the desired key
    - If collisions are handled via chaining, traverse the chain to find the key
  - **Deletion**
    - Similar to search, find the key and remove the key-value pair

**Advantages**

- **Efficiency:** Hash tables offer very fast data retrieval ideal for applications where rapid access to data is required
- **Lookup-Intensive Applications:** They are widely used in situations with heavy lookup operations like database indexing, caching and associative arrays

**Limitations**

- **Poor Worst Case Performance**
  - In the worst case such as when all keys hash to the same index, the performance can degrade to O(n)
- **Memory Overhead**
  - Hash Tables require extra space for the storage structure which can be significant especially for chaining

**Implementation**
  - In C++ STL provides implementations of hash tables in the form of **std::unordered_map** and **std::unordered_set**


**Unordered Set**

- Unordered set is implemented using a hast table
- Very similar to unordered map but in map we have **key:value** pairs but set does not have values, **just keys**
- Look up, Insertion and Deletion are all O(1)
- There are no duplicates



## Graphs

- Collection of nodes (also called vertices) and edges that connect pairs of nodes
- Used to represent relationships between pairs of objects
- They are incredibly versatile and applicable in various domains

![](Images/graph.png)


**Terminology**

- **Node(Vertex):** 
  - Represents an entity (for example city on map)
  
  ![](Images/graphNode.png)



- **Edge:** 
  - Represents connection or relationship between two nodes (for example road between cities)

  ![](Images/graphEdge.png)


**Directional Graphs**

- Edges have one-way direction from one vertex to the second vertex

  ![](Images/directionalGraph.png)

**Undirected Graphs or Bidirectional**

- Edges have no direction (or two-way directions)
- The edge(u,v) is identical to edge(v,u)
- **Example:** Twitter followings

  ![](Images/undirectedGraph.png)


**Weighted Graphs**

- Edges have weights associated with them
- These weights could represent distances, cost, or any other metric
- **Example:** Road maps with distances

  ![](Images/weightedEdges.png)


**Cyclic vs Acyclic Graphs**

- Cyclic graphs contain at least one graph cycle (a path of edges and vertices wherein vertex is reachable from itself)
- Acyclic graphs do not have any cycles

**Connected vs Disconnected**

- In a connected graph there is a path between every pair of vertices
- In disconnected graph some vertices cannot be reached from others

**Graph Representation**

- **Adjacency Matrix**

  - 2D array of size V*V (V is number of vertices)
  - **matrix[i][j] = 1** if there is an edge from vertex i to vertex j, else **0**
  - Simple but takes more space (O(V^2) for sparse graphs)

  ![](Images/adjacencyMatrix.png)


- **Adjency list**
  - An array of list
  - The size of the array is equal to the number of vertices
  - **list[i]** contains all the vertices that are adjacent to vertex i
  - Saves space for sparse graphs but can be less efficient for dense graphs
 

![](Images/adjecencyList.png)



- **Tree**
  - Tree is also form of directional graph

    ![](Images/binaryTree.png)

  - Linked List is special form of tree therefore Linked List is also graph
  - 



**Big O**

- Adjacency Matrix has space complexity O( Vertices^2 ) but Adjacency List only O( Vertices + Edges)
- To add vertex the time complexity is O(Vertices^2) for Matrix but only O(1) for List
- To add edge between 2 vertices it is time complexity O(1) for both
- To remove edge it is time complexity O(1) for both also
- To remove vertex it is time complexity O(Vertices^2) for Matrix and O(Vertices) for list because we must iterate trough the list and check the values if there is some edge connection to the deleted vertex from any other vertex



**Graph Algorithms**

- **Traversal**
  - Depth-First Search (DFS) and Breadth-First Search (BFS) are fundamental for exploring nodes in a graph
- **Shortest Path**
  - Algorithms like Dijkstra's, Bellman-Ford, etc.. find the shortest path between nodes in a weighted graph
- **Minimum Spanning Tree**
  - Algorithms like Prim's and Kruskal's find the minimum spanning tree of a graph
- **Topological Sorting**
  - Applies do directed acyclic graphs and results in a linear ordering of vertices

**Applications**

- **Social Networks:** Analyzing social structures
- **Google Maps:** Finding the shortest path between locations
- **Internet Routing:** Finding optimal paths for data packet transmission
- **Dependency Analysis:** In project planning or software compilation

**C++ Implementation**
  - In C++ graphs are often implemented using either adjacency lists (**std::vector** or **std::list**) or adjacency matrices (**std::vector** of vectors)




## Advanced Data Structures

- Understanding more complex structures like binary search trees, heaps, hash tables, and graph representations (like adjacency lists/matrices).
  

## STL containers:

- Provides implementations of many data structures like **std::vector** (dynamic array), **std::list** (linked list), **std::map** (balanced binary tree), **std::unordered_map** (hash table), etc...
- We should understand the underlying data structures of these containers (e.g., a std::map typically implemented as a red-black tree) and their performance implications
