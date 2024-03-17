#include <iostream>
#include <queue>


struct Node{

  Node(int value): m_value(value), m_left(nullptr), m_right(nullptr) { 
  }
  int m_value;
  Node* m_left;
  Node* m_right;
};

class BinarySearchTree {

private:
  Node* m_root;
  Node* m_last;
  int m_length;
 
public:

  BinarySearchTree(){
    m_root = nullptr;
  }
  BinarySearchTree(int value){
    Node* newNode = new Node(value);
    m_root = newNode;
  }

  bool insert(int value){

    Node* newNode = new Node(value);
    if(m_root == nullptr) {
      m_root = newNode;
      return true;
    }
    Node *temp = m_root;
    while(true){
      if(value == temp->m_value){ 
        return false;
      }
      if(value < temp->m_value){
        if(temp->m_left == nullptr){
          temp->m_left = newNode;
          return true;
        } 
        temp = temp->m_left;
      } else {
        if(temp->m_right == nullptr){
          temp->m_right = newNode;
          return true;
        } 
        temp = temp->m_right;
      }
    }
  }

  bool contains(int value) {
    if(m_root == nullptr){
      return false;
    }
    Node *temp = m_root;
    while(temp != nullptr){
      if(value < temp->m_value){
        temp = temp->m_left;
      } else if(value > temp->m_value){
        temp = temp->m_right;
      } else {
        return true;
      } 
    }
    return false;

  }


// Tree Traversal ----------------------------------------------------------------------------------------------


  void BFS(){
    std::queue<Node*> myQueue;
    myQueue.push(m_root);

    while (myQueue.size() > 0) {
      Node *currentNode = myQueue.front();
      myQueue.pop();
      std::cout << currentNode->m_value << " ";
      if(currentNode->m_left != nullptr){   // Add left first if exists
        myQueue.push(currentNode->m_left);
      }
      if(currentNode->m_right != nullptr){   // Then right if exists
        myQueue.push(currentNode->m_right);
      }
    }
    std::cout << std::endl;
  }

  // root-left-right
  // We need input Node parameter because we will do it with recursion so we need to move the node each call
  void DFSpreOrder(Node *currentNode){
    std::cout << currentNode->m_value << " ";     // root first
    if(currentNode->m_left){                      // then left    
      DFSpreOrder(currentNode->m_left);
    } 
    if (currentNode->m_right){                    // then right
      DFSpreOrder(currentNode->m_right);
    }
  }

  // Work around when root is private -> we can do it through overloading the function
  void DFSpreOrder (){
    DFSpreOrder(m_root);    // Now we can call it with the private root
    std::cout << std::endl;
  }


  // left-right-root
  // We need input Node parameter because we will do it with recursion so we need to move the node each call
  void DFSinOrder(Node *currentNode){
    
    if(currentNode->m_left){                      // first left    
      DFSinOrder(currentNode->m_left);
    }
    std::cout << currentNode->m_value << " ";     // then root 
    if (currentNode->m_right){                    // then right
      DFSinOrder(currentNode->m_right);
    }

  }

  // Work around when root is private -> we can do it through overloading the function
  void DFSinOrder (){
    DFSinOrder(m_root);    // Now we can call it with the private root
    std::cout << std::endl;
  }

  // left-right-root
  // We need input Node parameter because we will do it with recursion so we need to move the node each call
  void DFSpostOrder(Node *currentNode){
    if(currentNode->m_left){                      // first left    
      DFSpostOrder(currentNode->m_left);
    }
    if (currentNode->m_right){                    // then right
      DFSpostOrder(currentNode->m_right);
    }
    std::cout << currentNode->m_value << " ";     // then root 
  }

  // Work around when root is private -> we can do it through overloading the function
  void DFSpostOrder (){
    DFSpostOrder(m_root);    // Now we can call it with the private root
    std::cout << std::endl;
  }

// -------------------------------------------------------------------------------------------------------------

};




int main() {

  BinarySearchTree* myBST = new BinarySearchTree();

  myBST->insert(47);
  myBST->insert(21);
  myBST->insert(76);
  myBST->insert(18);
  myBST->insert(52);
  myBST->insert(82);
  myBST->insert(27);


  myBST->BFS();
  myBST->DFSpreOrder();
  myBST->DFSpostOrder();
  myBST->DFSinOrder();
  
  return 0;
}





