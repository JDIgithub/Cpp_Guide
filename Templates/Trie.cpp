#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <random>
#include <atomic>


/* Template
// note: using a class is only necessary if you want to store data at each node.
// otherwise, you can implement a trie using only hash maps.
struct TrieNode {
  int data;
  std::unordered_map<char, TrieNode*> children;
  TrieNode() : data(0), children(std::unordered_map<char, TrieNode*>()) {}
};

TrieNode* buildTrie(std::vector<std::string> words) {

  TrieNode* root = new TrieNode();
  for (std::string word: words) {
    TrieNode* curr = root;
    for (char c: word) {
      if (curr->children.find(c) == curr->children.end()) {
        curr->children[c] = new TrieNode();
      }
      curr = curr->children[c];
    }
    // at this point, you have a full word at curr
    // you can perform more logic here to give curr an attribute if you want
  }
 return root;
}
*/

// Example


class TrieNode {
public:
  std::unordered_map<char, TrieNode*> children;
  bool isEndOfWord;
  TrieNode() {
    isEndOfWord = false;
  }
};

class Trie {
private:
  TrieNode* root;

public:
  Trie() {
    root = new TrieNode();
  }

  void insert(std::string word) {
    TrieNode* node = root;
    for (char ch : word) {
      if (node->children.find(ch) == node->children.end()) {
        node->children[ch] = new TrieNode();
      }
      node = node->children[ch];
    }
    node->isEndOfWord = true;
  }

  bool search(std::string word) {
    TrieNode* node = root;
    for (char ch : word) {
      if (node->children.find(ch) == node->children.end()) {
        return false;
      }
      node = node->children[ch];
    }
    return node->isEndOfWord;
  }

  bool startsWith(std::string prefix) {
    TrieNode* node = root;
    for (char ch : prefix) {
      if (node->children.find(ch) == node->children.end()) {
        return false;
      }
      node = node->children[ch];
    }
    return true;
  }
};

int main() {
    Trie trie;
    
    trie.insert("hello");
    trie.insert("helium");
    
    std::cout << trie.search("hello") << std::endl; // 1 (true)
    std::cout << trie.search("hell") << std::endl;  // 0 (false)
    std::cout << trie.startsWith("hell") << std::endl; // 1 (true)
    std::cout << trie.startsWith("heli") << std::endl; // 1 (true)
    std::cout << trie.startsWith("hero") << std::endl; // 0 (false)
    
    return 0;
}
