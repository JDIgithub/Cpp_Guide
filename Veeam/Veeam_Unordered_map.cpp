#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <stack>
#include <cassert>

#include <iostream>
#include <limits>

//what will be the result of using Key class as a key in std::unordered_map<Key,std::string, KeyHasher, KeyEqual> unordered_map?

struct Key{
  int value;
   Key(int v) : value(v) {} 
};

struct KeyEqual{
  bool operator()(const Key& lhs, const Key& rhs) const{
  return lhs.value == rhs.value;    
}
};

struct KeyHasher{
 std::size_t operator()(const Key& k) const{
    return 1;
 }
};

int main() {

  std::unordered_map<Key,std::string, KeyHasher, KeyEqual> unordered_map;


  // Create a key
  Key myKey{5}; // Let's say the key is an integer 5

  // Insert a new element into the unordered_map
  unordered_map.insert({myKey, "Hello, World!"});

  // Alternatively, using the subscript operator
  unordered_map[myKey] = "Hello, World!";


  return 0;
}

