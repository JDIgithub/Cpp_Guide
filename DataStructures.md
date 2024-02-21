# Data Structures

## Introduction

- These are ways of organizing and storing data so it can be accessed and modified efficiently.
- Examples include arrays, linked lists, stacks, queues, trees, and graphs.
- Each structure has its unique properties and use-cases, like trees for hierarchical data or graphs for networked data.
- They are critical for implementing algorithms effectively
- Choosing the right algorithm and data structure can drastically improve the performance of an application, especially in data-intensive tasks



## Types

### Linear

- Like arrays, linked lists, stacks and queues
- They are fundamental for basic operations


### Tree

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
- 

#### Basic Concept

1. **Node**

   - The fundamental part of a tree, which contains data and references (or links) to other nodes

2. **Root**

   - The top node of the tree with no parent

3. **Leaf**

   - Node with no children

4. **Child and Parent**

   - A node directly connected to another node when moving away from the root

5. **Siblings**

   - Nodes that share the same parent

6. **Subtree**

   - A tree formed by a node and its descendants

7. **Depth of Node**

   - The length of the path from the root to the node

8. **Height of a Tree** 

   - The depth of the deepest node

#### Traversal

1. **In-Order Traversal (Left,Root,Right)**  

   - Particularly useful in BST for getting sorted order

2. **Pre-Order Traversal (Root,Left,Right)**

   - Useful for copying a tree

3. **Post-Order Traversal (Left,Right,Root)**

   - Useful for deleting a tree

4. **Level-Order Traversal**

   - Traverses the tree level by level


#### Complexity

- The time complexity for operations lie search, insert and delete depends on the type of tree and its balance
- For perfectly balanced BST these operations are O(log n)
- For skewed tree they can degrade to O(n)


#### Binary Tree

- Each node has at most two children, commonly referred to as the left and right child

1. **Full Binary Tree**

   - Every node has 0 or 2 children

2. **Complete Binary Tree**

   - All levels are fully filled except possible the last level which is filled from left to right

3. **Perfect Binary Tree**

   - A full binary tree in which all leaves have the same depth

4. **Balanced Binary Tree**

   - The height of the left and right subtrees of any node differ by no more than 1

5. **Degenerate Tree**

   - A tree where every internal node has one child
   - This essentially becomes a linked list


#### Binary Search Tree (BST)

- A binary tree where the left child contains only nodes with values less than the parent node and the right child only nodes with values greater
- This property provides efficient lookup, insertion and deletion operations
- Used for efficient data storage and retrieval
- Fast look up, insertion and deletion operations

- **Basic Concepts**
  - The left subtree of a node contains only nodes with keys lesser than the node's key
  - The right subtree of a node contains only nodes with keys greater than the node's key
  - The left and right subtree each must also be a binary search tree
  - There must be no duplicate nodes

- **Advantages**
  - **Efficient lookup and modification:** Searching closer to O(log n) for balanced trees
  - **Sorted data representation:** In order traversal of BST yields elements in sorted order
  - **Dynamic data structure:** BST can grow or shrink to accommodate the data as needed

- **Limitations**
  - **Balancing** 
    - Can become skewed if elements are inserted in a particular order (e.g. sorted order)
    - This can degrade performance with operations approaching O(n) complexity
  - **Balanced BST Variants**
    - To overcome balancing issues, self balancing BSTs like AVL or Red-Black trees are used which maintain a balanced structure through rotations
  - **Memory Overhead**
    - Each node typically requires extra space for two pointers (left and right)

- **Implementation in C++**
  - BST can be implemented using a class or struct to represent nodes with each node containing data, left child pointer and right child pointer
  - The tree itself can be managed by a class that provides functions for insertion, deletion, search and traversal

- **Appications**
  - BSTs are widely used in scenarios where data is dynamically inserted and deleted an where order needs to be preserved
  - They form the basis for more complex data structures like sets, maps and multimap in the C++ Standard Library

##### Operations

1. **Search**

   - Begin at the root node
   - Compare the search key with the key of the current node
   - If they are equal the search is successful
   - If the search key is less, continue in the left subtree
   - If the search key is greater, continue in the right subtree
   - If we reach leaf node, the search is unsuccessful
   - **Time Complexity:** Avarage - O(log n)  Worst Case - O(n) for skewed tree 

2. **Insertion**

   - Begin at the root node
   - Traverse the tree similar to the search operation to find the correct position for the new node
   - ??Insert the new node as a leaf??
   - **Time Complexity:** Avarage - O(log n)  Worst Case - O(n) 
3. **Deletion**

   - **Leaf node:** Simply remove the node
   - **Node with one child:** Remove the node and replace it with its child
   - **Node with two children:** Find the node's in-order predecessor (largest node in its left subtree) or in-order successor (the smallest node in its right subtree), swap it with the node to be deleted and then remove the node
   - **Time Complexity:** Avarage - O(log n)  Worst Case - O(n)

4. **Traversal**

   - **In-Order (Left,Root,Right):** Produces a sorted sequence of values
   - **Pre-Order (Root,Left,Right):** Used to create a copy of the tree
   - **Post-Order (Left,Right,Root):** Useful for deleting the tree
   - **Level-Order**

##### Red-Black Tree

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

#### Heap

- Special tree-based data structure that satisfies the heap property
- For example in a max heap, for any given node the value is greater than its children

#### Trie (Prefix Tree)

- A tree where each node represents a character of a string
- Tries are particularly used for efficient text searches and auto-complete functionalities

#### B-Trees and B+ Trees

- Used in databases and file systems for representing sorted data in a way that allows for efficient insertion, deletion and searching






### Hash Tables

- Used for fast data retrieval through hashing

- **Basic Concepts**
  - Data structure that implements an associative array
  - Structure that can map keys to values
  - It uses hash function to compute an index into an array of slots from which the desired value can be fetched
- **Efficiency:**
  - Due to this mechanism hash tables are incredibly efficient for lookup operations
  - Average time complexity of O(1) for searching, inserting and deleting elements 

- **Components**
  - **Keys and Values**
    - 




### Graphs

- Representing and working with networks of nodes and edges




## Advanced Data Structures:

- Understanding more complex structures like binary search trees, heaps, hash tables, and graph representations (like adjacency lists/matrices).
  

## STL containers:

- Provides implementations of many data structures like **std::vector** (dynamic array), **std::list** (linked list), **std::map** (balanced binary tree), **std::unordered_map** (hash table), etc...
- We should understand the underlying data structures of these containers (e.g., a std::map typically implemented as a red-black tree) and their performance implications
