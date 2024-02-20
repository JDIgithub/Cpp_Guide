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

#### Types

##### Binary Tree

- Each node has at most two children, commonly referred to as the left and right child

##### Binary Search Tree (BST)

- A binary tree where the left child contains only nodes with values less than the parent node and the right child only nodes with values greater
- This property provides efficient lookup, insertion and deletion operations

##### Heap

- Special tree-based data structure that satisfies the heap property
- For example in a max heap, for any given node the value is greater than its children

##### Trie (Prefix Tree)

- A tree where each node represents a character of a string
- Tries are particularly used for efficient text searches and auto-complete functionalities

##### B-Trees and B+ Trees

- Used in databases and file systems for representing sorted data in a way that allows for efficient insertion, deletion and searching

#### Traversal

- **In-Order Traversal**  










### Hash Tables

- Used for fast data retrieval through hashing






### Graphs

- Representing and working with networks of nodes and edges




## Advanced Data Structures:

- Understanding more complex structures like binary search trees, heaps, hash tables, and graph representations (like adjacency lists/matrices).
  

## STL containers:

- Provides implementations of many data structures like **std::vector** (dynamic array), **std::list** (linked list), **std::map** (balanced binary tree), **std::unordered_map** (hash table), etc...
- We should understand the underlying data structures of these containers (e.g., a std::map typically implemented as a red-black tree) and their performance implications
