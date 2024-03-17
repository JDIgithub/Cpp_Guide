#include <iostream>
#include <vector>
#include <unordered_map>



struct Node{

  Node(const std::string& key, int value): m_key(key),m_value(value), m_next(nullptr) { 
  }
  std::string m_key;
  int m_value;
  Node* m_next;

};

class HashTable {

private:
  static const int SIZE = 7;
  Node* dataMap[SIZE];

public:
  HashTable(){
  for(int i = 0; i < SIZE; i++) {
      dataMap[i] = nullptr;
    }
  }
  void printTable(){
    for(size_t i{0}; i < SIZE; ++i){
      std::cout << i << ":" << std::endl;
      if(dataMap[i] != nullptr){
        Node *temp = dataMap[i];
        while(temp){
          std::cout << " {" << temp->m_key << ", " << temp->m_value << "}" << std::endl;
          temp = temp->m_next;
        }    
      }
    }
  }
  // Hash function that turns key into index
  int hash(const std::string &key){
    int hash {0};
    for(size_t i {0}; i < key.length(); ++i){
      int asciiValue = int(key[i]);
      hash = (hash + asciiValue * 23) % SIZE;  // If we multiple by prime number (23) it makes result more random
                                               // % SIZE because we can generate only number smaller than size   
    }
    return hash;
  }
  void set(const std::string &key, int value){
    int index = hash(key);
    Node *newNode = new Node(key, value);
    if(dataMap[index] == nullptr){  // If bucket is empty
      dataMap[index] = newNode; 
    } else {                        // else    
      Node *temp = dataMap[index];  // dataMap[index] hear is basically head of the linked list that is at that index
      while (temp->m_next != nullptr){
        temp = temp->m_next;
      }
      temp->m_next = newNode;       // put new element at the end of the linked list
    }
  }

  int get(const std::string &key){
    
    int index = hash(key);
    Node *temp = dataMap[index];
    while(temp != nullptr){
      if(temp->m_key == key){
        return temp->m_value;
      }
      temp = temp->m_next;
    }
    return INT_MIN;
  }



  std::vector<std::string> keys(){
    std::vector<std::string> allKeys;
    for(size_t i{0}; i < SIZE; ++i){
      Node *temp = dataMap[i];
      while(temp!= nullptr){
        allKeys.push_back(temp->m_key);
        temp = temp->m_next;
      }
    }
    return allKeys;
  }

};


// !! Interview Question for Hash Table !! ----------------------------------------------------------------------------

// Check if these two vectors have an item in common:

// V1  1 3 5
// V2  2 4 5  

// We can do it with nested loops check every element of V1 against every element of V2 but this is O(n^2)
bool itemInCommon(const std::vector<int>& v1, const std::vector<int>& v2){
  for(auto i: v1){
    for(auto j: v2){
      if(i == j) {return true;}
    }
  }
  return false;
}

// !! BUT !! we can do it with Hast tables and improve O to O(2n) -> O(n)

bool itemInCommonHT(const std::vector<int>& v1, const std::vector<int>& v2){
  std::unordered_map<int, bool> myMap;  // Hash table under the hood

  for (auto i : v1){
    //myMap[i] = true; // this approach can overwrite the value when the key already exist
    myMap.insert({i,true}); // This is safer. Can not overwrite value
  }
  for (auto i : v2){
    if(myMap[i]){
      return true;
    }
  }
  return false;
}
// --------------------------------------------------------------------------------------------------------------------



int main() {

  HashTable  *myHashTable = new HashTable();


  myHashTable->set("nails", 100);
  myHashTable->set("tile", 50);
  myHashTable->set("lumber", 80);
  myHashTable->set("bolts", 200);
  myHashTable->set("screws", 140);

  myHashTable->printTable();

  std::cout << myHashTable->get("lumber") << " " << myHashTable->get("pillow") << std::endl;
  for(const auto &key: myHashTable->keys()){
    std::cout << key << std::endl;
  }

  return 0;
}


